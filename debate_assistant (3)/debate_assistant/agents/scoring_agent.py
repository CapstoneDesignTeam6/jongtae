"""
agents/scoring_agent.py — 턴별 사용자 발언 평가

[평가 지표 5개]
  1. 발언 구체성   specificity          : 수치·사례·출처의 정밀도
  2. 인과 연결     causality            : 원인-결과-함의 연결 깊이
  3. 도메인 폭     domain_breadth       : 한 발언 안에서 넘나드는 영역의 수
  4. 정보 자립도   information_autonomy : 스스로 구성한 정보 비율
  5. 개념 정확도   conceptual_accuracy  : 전문용어·고유명사의 정확한 사용

[반환 구조]
  turn_score:
    {
      "turn": 1,
      "scores": {
        "specificity":           { "score": 1~5, "reason": "...", "evidence": "..." },
        "causality":             { "score": 1~5, "reason": "...", "evidence": "..." },
        "domain_breadth":        { "score": 1~5, "reason": "...", "evidence": "...",
                                   "domain_keywords": [...] },
        "information_autonomy":  { "score": 1~5, "reason": "...", "evidence": "..." },
        "conceptual_accuracy":   { "score": 1~5, "reason": "...", "evidence": "...",
                                   "errors": null | "설명" }
      },
      "total": 5~25
    }

  final_score:
    {
      "turns": [ turn_score, ... ],
      "summary": {
        "specificity":          { "avg": float, "trend": "상승"|"하락"|"유지", "scores_per_turn": [...] },
        "causality":            { ... },
        "domain_breadth":       { ..., "all_domain_keywords": [...] },
        "information_autonomy": { ... },
        "conceptual_accuracy":  { ..., "all_errors": [...] }
      },
      "total_avg": float
    }
"""

import re
import json

from data.evidence import build_history_block
from agents.llm import call_ollama


METRICS = [
    "specificity",
    "causality",
    "domain_breadth",
    "information_autonomy",
    "conceptual_accuracy",
]


# ── 유틸 ─────────────────────────────────────────────────────────

def _extract_user_turns(history: list[dict]) -> list[str]:
    return [h["content"] for h in history if h.get("role") == "user"]


def _trend(scores: list[float]) -> str:
    if len(scores) < 2:
        return "유지"
    diff = scores[-1] - scores[0]
    if diff >= 1:
        return "상승"
    if diff <= -1:
        return "하락"
    return "유지"


# ── ScoringAgent ─────────────────────────────────────────────────

class ScoringAgent:

    def __init__(self, user_label: str = "찬성", ai_label: str = "반대"):
        self.user_label   = user_label
        self.ai_label     = ai_label
        self._turn_scores: list[dict] = []

    # ── 공개 API ─────────────────────────────────────────────────

    def score_turn(self, history: list[dict], topic: str) -> dict:
        """턴 종료 직후 호출. turn_number는 내부에서 자동 계산."""
        turn_number = len(self._turn_scores) + 1
        print(f"\n[ScoringAgent] 턴 {turn_number} 평가 시작")

        user_turns = _extract_user_turns(history)
        if not user_turns or turn_number > len(user_turns):
            return {"error": f"턴 {turn_number}에 해당하는 유저 발언 없음"}

        current_user_utterance = user_turns[turn_number - 1]
        prev_ai_utterance      = self._get_prev_ai(history, turn_number)
        history_block          = build_history_block(history)

        result = self._evaluate(
            topic=topic,
            turn_number=turn_number,
            current_user_utterance=current_user_utterance,
            prev_ai_utterance=prev_ai_utterance,
            history_block=history_block,
        )

        self._turn_scores.append(result)
        print(f"  └─ [ScoringAgent] 턴 {turn_number} 완료 | 총점: {result.get('total', 0)}/25")
        return result

    def score_final(self) -> dict:
        """전체 턴 통계 집계."""
        if not self._turn_scores:
            return {"error": "평가된 턴이 없음"}

        summary = {}
        for m in METRICS:
            scores = [
                t["scores"][m]["score"]
                for t in self._turn_scores
                if "scores" in t and m in t["scores"]
            ]
            entry = {
                "avg":             round(sum(scores) / len(scores), 2) if scores else 0,
                "trend":           _trend(scores),
                "scores_per_turn": scores,
            }
            # 도메인 키워드 누적
            if m == "domain_breadth":
                all_kw: list[str] = []
                for t in self._turn_scores:
                    all_kw.extend(
                        t["scores"].get("domain_breadth", {}).get("domain_keywords", [])
                    )
                entry["all_domain_keywords"] = list(dict.fromkeys(all_kw))
            # 개념 오류 누적
            if m == "conceptual_accuracy":
                all_errors: list[str] = []
                for t in self._turn_scores:
                    err = t["scores"].get("conceptual_accuracy", {}).get("errors")
                    if err:
                        all_errors.append(err)
                entry["all_errors"] = all_errors
            summary[m] = entry

        total_avgs = [t.get("total", 0) for t in self._turn_scores]
        total_avg  = round(sum(total_avgs) / len(total_avgs), 2)

        return {
            "turns":     self._turn_scores,
            "summary":   summary,
            "total_avg": total_avg,
        }

    def reset(self):
        self._turn_scores = []

    # ── 내부 헬퍼 ────────────────────────────────────────────────

    def _get_prev_ai(self, history: list[dict], turn_number: int) -> str:
        user_count = 0
        for i, h in enumerate(history):
            if h.get("role") == "user":
                user_count += 1
                if user_count == turn_number:
                    for j in range(i - 1, -1, -1):
                        if history[j].get("role") == "ai":
                            return history[j]["content"]
                    return "(없음)"
        return "(없음)"

    # ── LLM 평가 ─────────────────────────────────────────────────

    def _evaluate(
        self,
        topic: str,
        turn_number: int,
        current_user_utterance: str,
        prev_ai_utterance: str,
        history_block: str,
    ) -> dict:

        prompt = f"""당신은 시사 토론 평가 전문가입니다.
아래 정보를 바탕으로 유저의 이번 턴 발언을 5가지 지표로 평가하세요.

[토론 주제]
{topic}

[유저 입장]
{self.user_label}

[이번 턴 직전 AI 발언]
{prev_ai_utterance}

[이번 턴 유저 발언 — 평가 대상]
{current_user_utterance}

[전체 토론 기록]
{history_block}

━━━ 평가 기준 (각 지표 1~5점) ━━━

점수는 반드시 1~5 사이 정수만 사용하세요.
1점과 5점은 극단적인 경우에만 부여하고, 대부분의 발언은 2~4점 사이로 평가하세요.
각 기준은 독립적으로 평가하세요 (다른 기준 점수에 영향받지 않음).

[1. 발언 구체성 specificity]
정보의 정밀도를 측정한다.

1점: 전부 "~인 것 같다", "~라고 들었다", "아마" 같은 불확실 표현만
2점: 주장은 있지만 근거가 막연하고 수치·출처 없음
3점: 일부 구체적 사례나 수치가 있으나 출처 불명확
4점: 구체적 사례와 수치 포함, 출처 언급
5점: 수치·사례·기관명·출처까지 명확하게 제시

[2. 인과 연결 causality]
사례를 단순 나열하는 데 그치지 않고, 원인-결과-함의를 연결해 이해하고 있는지 측정한다.

1점: 사례 이름만 던짐, 설명 없음
2점: 사례 + 결과만 서술, 원인 연결 없음
3점: 사례 + 원인 또는 결과 중 하나만 연결
4점: 사례 + 원인 + 결과 연결
5점: 사례 + 원인 + 결과 + 자신의 주장과의 연결까지

[3. 도메인 폭 domain_breadth]
이 발언 하나에서 얼마나 다양한 영역을 넘나들며 논점을 구성하는지 측정한다.
도메인 예시: 경제, 환경, 외교, 군사, 사회, 기술, 인권, 역사 등

1점: 단일 도메인 안에서 같은 논거만 반복
2점: 단일 도메인, 한 가지 각도만 사용
3점: 단일 도메인이지만 두 가지 이상의 세부 논점을 구분해 사용
4점: 두 개 도메인을 넘나들며 논점을 구성
5점: 세 개 이상의 도메인을 연결해 복합적 논점을 구성

domain_keywords 필드에 이번 발언에서 등장한 도메인 키워드를 1~3개 추출할 것.

[4. 정보 자립도 information_autonomy]
발언자가 외부 발언에 기대지 않고 스스로 정보를 생산·구성하는 비율을 측정한다.

1점: 타인의 말·자료를 그대로 재인용하거나 단순 동의·부정만
2점: 외부 발언을 자신의 언어로 바꿔 말하는 수준 (재구성에 그침)
3점: 외부 정보를 활용하되 자신의 사례·수치를 1개 이상 추가
4점: 자신이 직접 수집·선택한 정보가 발언의 절반 이상을 구성
5점: 발언 전체가 발언자 스스로 구성한 정보·논리로 이루어짐

[5. 개념 정확도 conceptual_accuracy]
전문 용어, 고유명사, 제도·정책 명칭 등이 맥락에 맞게 쓰였는지 평가한다.

1점: 핵심 개념을 명백히 잘못 사용하거나 혼동함
2점: 개념을 대략적으로만 이해하고 부정확하게 사용
3점: 개념을 대체로 올바르게 사용하나 일부 부정밀함
4점: 개념을 정확하게 사용하며 맥락에도 적합함
5점: 개념을 정확히 사용하고 그 개념의 한계나 세부 조건까지 인식

전문 용어나 고유명사가 등장하지 않는 발언은 3점으로 처리한다.
errors 필드에 오용된 개념이 있으면 해당 단어와 간단한 설명을 기재하고, 없으면 null로 처리한다.

━━━ 출력 형식 ━━━
아래 JSON만 출력. 다른 텍스트 없이.

{{
  "turn": {turn_number},
  "scores": {{
    "specificity": {{
      "score": 1~5 정수,
      "reason": "점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)"
    }},
    "causality": {{
      "score": 1~5 정수,
      "reason": "점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)"
    }},
    "domain_breadth": {{
      "score": 1~5 정수,
      "reason": "점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)",
      "domain_keywords": ["도메인1", "도메인2"]
    }},
    "information_autonomy": {{
      "score": 1~5 정수,
      "reason": "점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)"
    }},
    "conceptual_accuracy": {{
      "score": 1~5 정수,
      "reason": "점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)",
      "errors": null
    }}
  }},
  "total": 위 5개 score 합계 정수
}}"""

        raw = call_ollama(prompt, "")
        print(f"  └─ [evaluate] LLM 원문: {raw[:200]}")
        return self._parse_result(raw, turn_number)

    def _parse_result(self, raw: str, turn_number: int) -> dict:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                scores = parsed.get("scores", {})
                total = 0
                for m in METRICS:
                    s = scores.get(m, {}).get("score", 1)
                    s = max(1, min(5, int(s)))
                    scores.setdefault(m, {})["score"] = s
                    total += s
                parsed["scores"] = scores
                parsed["total"]  = total
                parsed["turn"]   = turn_number
                return parsed
        except Exception as e:
            print(f"  └─ [parse_result] JSON 파싱 실패: {e} | 원문: {raw[:300]}")

        # fallback
        return {
            "turn": turn_number,
            "scores": {
                "specificity":          {"score": 1, "reason": "평가 실패", "evidence": ""},
                "causality":            {"score": 1, "reason": "평가 실패", "evidence": ""},
                "domain_breadth":       {"score": 1, "reason": "평가 실패", "evidence": "",
                                         "domain_keywords": []},
                "information_autonomy": {"score": 1, "reason": "평가 실패", "evidence": ""},
                "conceptual_accuracy":  {"score": 1, "reason": "평가 실패", "evidence": "",
                                         "errors": None},
            },
            "total": 5,
        }
