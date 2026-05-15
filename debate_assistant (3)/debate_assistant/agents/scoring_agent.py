"""
agents/scoring_agent.py — 턴별 사용자 발언 평가

[평가 지표 5개]
  1. 발언 구체성   : ~인 것 같다 / ~라고 들었다 같은 불확실 표현 vs 구체적 사례·수치
  2. 인과 연결     : 사례만 나열 vs 사례 + 상황 설명 + 결과 연결
  3. 도메인 다양성 : 토론 전체에서 다룬 도메인 영역의 폭
  4. 정보 주도성   : AI 발언에만 반응 vs 사용자가 새 시사 이슈·주제를 스스로 꺼내는지
  5. 편향도        : 유리한 통계만 사용 / 반례 무시 / 감정 선동 여부를 종합 평가
                    (점수가 높을수록 편향이 낮고 균형 잡힌 논증)

[반환 구조]
  turn_score:
    {
      "turn": 1,
      "scores": {
        "specificity": { "score": 1~5, "reason": "...", "evidence": "..." },
        "causality":   { "score": 1~5, "reason": "...", "evidence": "..." },
        "domain":      { "score": 1~5, "reason": "...", "evidence": "...", "domains": [...] },
        "initiative":  { "score": 1~5, "reason": "...", "evidence": "..." },
        "bias":        {
                         "score": 1~5, "reason": "...", "evidence": "...",
                         "details": {
                           "stat_bias":      "유리한 통계만 사용했는지 한 줄 평",
                           "counterarg":     "반례를 무시했는지 한 줄 평",
                           "emotional_bias": "감정 선동에 의존했는지 한 줄 평"
                         }
                       },
      },
      "total": 5~25
    }

  final_score:
    {
      "turns": [ turn_score, ... ],
      "summary": {
        "specificity": { "avg": float, "trend": "상승"|"하락"|"유지", "scores_per_turn": [...] },
        "causality":   { ... },
        "domain":      { ..., "all_domains": [...] },
        "initiative":  { ... },
        "bias":        { ... },
      },
      "total_avg": float
    }
"""

import re
import json

from data.evidence import build_history_block
from agents.llm import call_ollama


METRICS = ["specificity", "causality", "domain", "initiative", "bias"]


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
        prior_domains          = self._collect_prior_domains()

        result = self._evaluate(
            topic=topic,
            turn_number=turn_number,
            current_user_utterance=current_user_utterance,
            prev_ai_utterance=prev_ai_utterance,
            history_block=history_block,
            prior_domains=prior_domains,
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
            if m == "domain":
                all_domains: list[str] = []
                for t in self._turn_scores:
                    all_domains.extend(t["scores"].get("domain", {}).get("domains", []))
                entry["all_domains"] = list(dict.fromkeys(all_domains))
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

    def _collect_prior_domains(self) -> list[str]:
        domains: list[str] = []
        for t in self._turn_scores:
            domains.extend(t.get("scores", {}).get("domain", {}).get("domains", []))
        return list(dict.fromkeys(domains))

    # ── LLM 평가 ─────────────────────────────────────────────────

    def _evaluate(
        self,
        topic: str,
        turn_number: int,
        current_user_utterance: str,
        prev_ai_utterance: str,
        history_block: str,
        prior_domains: list[str],
    ) -> dict:

        prior_domains_str = ", ".join(prior_domains) if prior_domains else "없음"

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

[전체 토론 기록 (도메인 파악용)]
{history_block}

[이전 턴들에서 이미 등장한 도메인]
{prior_domains_str}

━━━ 평가 기준 (각 지표 1~5점) ━━━

점수는 반드시 1~5 사이 정수만 사용하세요.
1점과 5점은 극단적인 경우에만 부여하고, 대부분의 발언은 2~4점 사이로 평가하세요.

[1. 발언 구체성 specificity]
1점: 전부 "~인 것 같다", "~라고 들었다", "아마" 같은 불확실 표현만
2점: 주장은 있지만 근거가 막연하고 수치·출처 없음
3점: 일부 구체적 사례나 수치가 있으나 출처 불명확
4점: 구체적 사례와 수치 포함, 출처 언급
5점: 수치·사례·기관명·출처까지 명확하게 제시

[2. 인과 연결 causality]
1점: 사례 이름만 던짐
2점: 사례 + 결과만 서술, 원인 연결 없음
3점: 사례 + 원인 또는 결과 중 하나만 연결
4점: 사례 + 원인 + 결과 연결
5점: 사례 + 원인 + 결과 + 자신의 주장과의 연결까지

[3. 도메인 다양성 domain]
이번 발언이 이전 턴들과 얼마나 다른 영역을 다루는지 평가.
도메인 예시: 경제, 환경, 외교, 군사, 사회, 기술, 인권, 역사 등
1점: 이전 턴과 완전히 같은 도메인·같은 논거 반복
2점: 같은 도메인에서 거의 같은 각도
3점: 같은 도메인이지만 새로운 각도나 세부 논점
4점: 기존 도메인 + 새 도메인 1개 추가
5점: 이전에 없던 새로운 도메인을 주도적으로 개척

이번 발언에서 등장한 도메인 키워드도 함께 추출하세요 (1~3개).

[4. 정보 주도성 initiative]
1점: AI 발언 그대로 재인용하거나 단순 동의/부정만
2점: AI 발언에 반응하되 자신의 언어로 바꿔 말하는 수준
3점: AI 발언에 반응하면서 새 사례나 수치 1개 추가
4점: AI가 언급하지 않은 새로운 시사 사례나 관점을 꺼냄
5점: AI가 다루지 않은 새로운 시사 이슈·주제를 유저가 먼저 도입

[5. 편향도 bias]
아래 세 가지 편향 요소를 종합해 하나의 점수로 평가.
점수가 높을수록 편향이 낮고 균형 잡힌 논증.

  ① 통계 편향 (stat_bias): 자신에게 유리한 통계·사례만 골라 쓰고 불리한 데이터는 언급하지 않는지
  ② 반례 무시 (counterarg): 상대방이 제시한 반례·반박을 무시하거나 화제를 돌려 회피하는지
     - 이번 턴이 첫 번째 턴이거나 직전 AI 발언에 반례가 없으면 이 요소는 중립(감점 없음)으로 처리
  ③ 감정 선동 (emotional_bias): 논리·근거 없이 공포·혐오·과장·동정 호소만으로 주장하는지

1점: 세 요소 모두 심각 — 유리한 수치만 인용, 반례 완전 무시, 감정 선동 위주
2점: 두 요소 이상에서 편향이 뚜렷함
3점: 일부 편향이 있으나 논리적 근거도 병행
4점: 편향 요소가 경미하며 전반적으로 균형 잡힌 논증
5점: 세 요소 모두 없음 — 유불리 통계 균형, 반례 성실히 수용, 감정 선동 없음

details 필드에 각 요소별 한 줄 평을 작성하세요.

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
    "domain": {{
      "score": 1~5 정수,
      "reason": "점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)",
      "domains": ["도메인1", "도메인2"]
    }},
    "initiative": {{
      "score": 1~5 정수,
      "reason": "점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)"
    }},
    "bias": {{
      "score": 1~5 정수,
      "reason": "종합 점수 이유 1문장",
      "evidence": "발언에서 근거가 된 실제 문구 (없으면 빈 문자열)",
      "details": {{
        "stat_bias":      "유리한 통계만 사용했는지 한 줄 평",
        "counterarg":     "반례를 무시했는지 한 줄 평",
        "emotional_bias": "감정 선동에 의존했는지 한 줄 평"
      }}
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

        return {
            "turn": turn_number,
            "scores": {
                "specificity": {"score": 1, "reason": "평가 실패", "evidence": ""},
                "causality":   {"score": 1, "reason": "평가 실패", "evidence": ""},
                "domain":      {"score": 1, "reason": "평가 실패", "evidence": "", "domains": []},
                "initiative":  {"score": 1, "reason": "평가 실패", "evidence": ""},
                "bias":        {
                    "score": 1, "reason": "평가 실패", "evidence": "",
                    "details": {
                        "stat_bias":      "",
                        "counterarg":     "",
                        "emotional_bias": "",
                    },
                },
            },
            "total": 5,
        }