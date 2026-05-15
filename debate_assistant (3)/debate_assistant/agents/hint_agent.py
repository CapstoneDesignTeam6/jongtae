"""
agents/hint_agent.py — 반박/재반박 힌트 생성

[v2 최적화 — 2025-05]
  ① stance 생성      : num_predict=30 토큰 제한          → ~19s 절약
  ② step0+step1+step2 → _step_combined 단일 LLM 호출
                        num_predict=500 토큰 제한          → LLM 호출 3→1회
  ③ Tavily 검색       : 순차 루프 → ThreadPoolExecutor 병렬 → ~4s 절약
  전체 효과: 108s → 72s  (약 34% 단축, step3 제외 기준)

[로직 유지]
  - 내용 반복 차단: _used_logics(반박 각도) + _used_conclusions(결론 귀결 패턴)
  - 전략 순환: _used_strategies (직전 2개 제외)
  - 도메인 추적: _used_domains (같은 도메인이어도 허점 각도 교체)
"""

import re
import json
import os
from concurrent.futures import ThreadPoolExecutor

from langchain_tavily import TavilySearch

from data.evidence import build_evidence_block, build_history_block
from agents.llm import call_ollama

os.environ["TAVILY_API_KEY"] = "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"
_tavily = TavilySearch(max_results=2)

STRATEGY_MAP = {
    "1": "반증 사례: 상대 주장의 일반화를 단 하나의 구체적 반례로 무너뜨린다.",
    "2": "비용-편익: 상대 주장이 가져올 이득보다 비용·부작용이 훨씬 크다는 점을 부각한다.",
    "3": "가치 우선순위: 상대 논리보다 지금 더 중요한 가치가 있음을 주장한다.",
    "4": "귀류법: 상대 논리를 끝까지 밀면 모순적 결론에 도달함을 보여준다.",
    "5": "역이용: 상대가 제시한 사실을 오히려 내 주장의 근거로 전환한다.",
}


# ── 유틸 ─────────────────────────────────────────────────────────

def _last_user_claim(history: list[dict]) -> str:
    for turn in reversed(history):
        if turn.get("role") in ("user", "human"):
            return turn.get("content", "")[:400]
    return ""


def _last_ai_claims(history: list[dict], n: int = 2) -> list[str]:
    turns = [t.get("content", "") for t in history if t.get("role") in ("assistant", "ai")]
    return [turns[-(i + 1)][:400] for i in range(n) if len(turns) > i]


# ── HintAgent ────────────────────────────────────────────────────

class HintAgent:

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        user_label: str = "찬성",
        ai_label: str = "반대",
        user_stance: str = "",
    ):
        self.evidence            = evidence_items or []
        self.user_label          = user_label
        self.ai_label            = ai_label
        self._user_stance_fixed  = user_stance or None
        self._stance_cache: dict[str, str] = {}

        self._used_domains:     list[str] = []
        self._used_strategies:  list[str] = []
        self._used_logics:      list[str] = []
        self._used_conclusions: list[str] = []

    # ── stance 생성 ───────────────────────────────────────────────
    # [v2] num_predict=30 으로 토큰 제한 → 불필요한 장문 생성 차단

    def _resolve_stance(self, topic: str) -> str:
        topic = re.sub(r'([,?!])(\S)', r'\1 \2', topic)
        if self._user_stance_fixed:
            return self._user_stance_fixed
        if topic in self._stance_cache:
            return self._stance_cache[topic]

        prompt = f"""토론에서 유저의 입장을 한 문장으로 표현하세요.

토론 주제: {topic}
유저 입장: {self.user_label}

위 주제에서 "{self.user_label}" 측이 주장하는 결론을 한 문장으로 만드세요.

예시:
- 주제: "A vs B, 누가 더 나쁜가?" / 입장: A → "A가 더 나쁘다"
- 주제: "A vs B, 누가 더 손해인가?" / 입장: A → "A가 더 손해다"
- 주제: "A vs B, 누가 더 이득인가?" / 입장: A → "A가 더 이득이다"
- 주제: "A vs B, 누가 더 위험한가?" / 입장: A → "A가 더 위험하다"
- 주제: "A vs B, 누가 더 유리한가?" / 입장: A → "A가 더 유리하다"
- 주제: "A vs B, 누가 더 강한가?" / 입장: A → "A가 더 강하다"
- 주제: "A vs B, 누가 더 옳은가?" / 입장: A → "A가 더 옳다"
- 주제: "A vs B, 누가 더 문제인가?" / 입장: A → "A가 더 문제다"
- 주제: "A vs B 중 무엇이 더 효과적인가?" / 입장: A → "A가 더 효과적이다"
- 주제: "A vs B 중 무엇이 더 나은가?" / 입장: A → "A가 더 낫다"
- 주제: "A vs B 중 무엇이 더 안전한가?" / 입장: A → "A가 더 안전하다"
- 주제: "A vs B 중 무엇이 더 위험한가?" / 입장: A → "A가 더 위험하다"
- 주제: "A의 정책은 옳은가?" / 입장: 찬성 → "A의 정책은 옳다"
- 주제: "A의 정책은 옳은가?" / 입장: 반대 → "A의 정책은 옳지 않다"
- 주제: "A는 성공적인가?" / 입장: 찬성 → "A는 성공적이다"
- 주제: "A는 성공적인가?" / 입장: 반대 → "A는 성공적이지 않다"
- 주제: "A는 필요한가?" / 입장: 찬성 → "A는 필요하다"
- 주제: "A는 필요한가?" / 입장: 반대 → "A는 필요하지 않다"
- 주제: "A는 정당한가?" / 입장: 찬성 → "A는 정당하다"
- 주제: "A는 정당한가?" / 입장: 반대 → "A는 정당하지 않다"
- 주제: "A 전쟁, 누가 더 손해인가?" / 입장: A → "A가 더 손해다"
- 주제: "A 전쟁, 누가 더 이득인가?" / 입장: B → "B가 더 이득이다"
- 주제: "A와 B, 어느 쪽이 더 책임이 큰가?" / 입장: A → "A의 책임이 더 크다"
- 주제: "A와 B, 어느 쪽이 더 피해가 큰가?" / 입장: B → "B의 피해가 더 크다"
- 주제: "A와 B 중 누가 더 도덕적인가?" / 입장: A → "A가 더 도덕적이다"
- 주제: "A와 B 중 누가 더 비도덕적인가?" / 입장: B → "B가 더 비도덕적이다"
- 주제: "A 정책 도입, 찬성인가 반대인가?" / 입장: 찬성 → "A 정책 도입은 옳다"
- 주제: "A 정책 도입, 찬성인가 반대인가?" / 입장: 반대 → "A 정책 도입은 옳지 않다"
- 주제: "A와 B, 누가 더 잘했는가?" / 입장: A → "A가 더 잘했다"
- 주제: "A와 B, 누가 더 잘못했는가?" / 입장: B → "B가 더 잘못했다"

지금 적용:
- 주제: "{topic}" / 입장: {self.user_label} → ?

규칙:
- 30자 이내
- 결론 문장만 출력. 다른 말 없이.

출력:"""

        # [v2] num_predict=30
        raw    = call_ollama(prompt, "", num_predict=30).strip().strip('"\'「」').rstrip(".")
        stance = raw

        print(f"  └─ [stance] '{stance}'")
        self._stance_cache[topic] = stance
        return stance

    # ── 공개 API ─────────────────────────────────────────────────

    def rebuttal_hint(self, history: list[dict], topic: str) -> dict:
        return self._generate_hint(history, topic, mode="rebuttal")

    def counter_hint(self, history: list[dict], topic: str) -> dict:
        return self._generate_hint(history, topic, mode="counter")

    # ── 공통 파이프라인 ──────────────────────────────────────────

    def _generate_hint(self, history: list[dict], topic: str, mode: str) -> dict:
        stance = self._resolve_stance(topic)

        evidence_block = build_evidence_block(self.evidence, max_chars=5000)
        history_block  = build_history_block(history)

        ai_turns    = _last_ai_claims(history, n=2)
        ai_latest   = ai_turns[0] if ai_turns else ""
        user_latest = _last_user_claim(history)

        if mode == "rebuttal":
            situation = (
                f"상대({self.ai_label})가 방금 새 주장을 펼쳤습니다.\n"
                f"유저({self.user_label})는 그 주장을 반박해야 합니다."
            )
        else:
            situation = (
                f"유저({self.user_label})가 주장했습니다: \"{user_latest[:150]}\"\n"
                f"그러자 상대({self.ai_label})가 유저 주장을 공격했습니다.\n"
                f"유저는 공격을 막아내고 자신의 원래 입장을 복원해야 합니다."
            )

        # [v2] step0 + step1 + step2_llm → 단일 호출
        parsed     = self._step_combined(ai_latest, topic, stance, situation)
        domain     = parsed["domain"]
        compressed = parsed["compressed"]
        logic      = parsed["logic"]
        strategy   = STRATEGY_MAP.get(parsed["strategy"], STRATEGY_MAP["1"])
        queries    = parsed["queries"]

        # [v2] Tavily 병렬 검색
        search_block = self._run_tavily(queries)

        hint = self._step3_hint(
            compressed, logic, strategy, search_block,
            evidence_block, history_block, topic, stance, situation, domain
        )

        return {"hint": hint, "raw_response": hint}

    # ── [v2] combined: step0 + step1 + step2_llm 통합 ────────────
    # LLM 호출 3번 → 1번, num_predict=500

    def _step_combined(
        self, ai_claim: str, topic: str, stance: str, situation: str
    ) -> dict:
        recent_domains    = self._used_domains[-2:]
        recent_strats     = self._used_strategies[-2:]
        prior_logics      = " / ".join(self._used_logics[-2:])      if self._used_logics      else "없음"
        prior_conclusions = " / ".join(self._used_conclusions[-2:]) if self._used_conclusions else "없음"

        domain_notice = (
            f"직전 힌트 도메인: {', '.join(recent_domains)} — 같은 도메인이더라도 반드시 다른 허점 각도로 공격할 것."
            if recent_domains else ""
        )
        avoid_strat_str = (
            f"다음 전략은 직전에 이미 사용했으므로 반드시 제외: {', '.join(recent_strats)}"
            if recent_strats else ""
        )
        available_strategies = {k: v for k, v in STRATEGY_MAP.items() if k not in recent_strats}
        if not available_strategies:
            available_strategies = STRATEGY_MAP
        strat_list = "\n".join(f"{k}. {v}" for k, v in available_strategies.items())

        prompt = f"""당신은 토론 분석가 겸 코치입니다. 아래 지시를 순서대로 수행하고 지정된 형식으로만 출력하세요.

[상황]
{situation}

[토론 주제]
{topic}

[유저가 증명해야 할 결론]
"{stance}"

[상대의 주장]
{ai_claim}

{domain_notice}

[직전 힌트의 반박 각도 — 다른 각도로 공격할 것]
{prior_logics}

[직전 힌트의 결론 귀결 패턴 — 이 패턴으로 끝내지 말 것]
{prior_conclusions}

{avoid_strat_str}

[사용 가능한 전략]
{strat_list}

━━━ 출력 형식 (아래 키만, 추가 설명 없이) ━━━

도메인: [상대 주장이 주로 근거하는 논거 영역. 이 토론 주제와 상대 주장에서 실제로 사용된 개념 안에서 한 단어로.]
핵심_주장: [상대의 결론 한 문장]
가장_약한_지점: ["{stance}"를 증명하는 데 위 도메인 안에서 가장 유리하게 공격할 수 있는 논리적 허점. 직전과 같은 도메인이라면 반드시 다른 허점.]
공격_이유: [위 도메인 관점에서 왜 그 허점이 "{stance}"를 뒷받침하는가]
전략번호: [위 사용 가능한 전략 번호 중 하나]
반박논리: [도메인 영역 안에서 "{stance}"를 증명하는 한국어 2문장. 마지막 문장은 "{stance}"를 직접 긍정하는 결론. 직전 반박 각도와 논리 구조가 같으면 다른 측면으로 바꿀 것.]
결론_패턴: [반박논리 결론부를 10자 이내로 요약]
검색쿼리: ["영어쿼리1", "영어쿼리2", "영어쿼리3", "영어쿼리4"]

규칙:
- 검색쿼리 1·2번: 상대 주장의 피해가 제한적·관리 가능함을 보여주는 증거 쿼리
- 검색쿼리 3·4번: 유저 측 주장이 더 심각·결정적임을 보여주는 증거 쿼리
- 검색쿼리는 영어만, ASCII만, 연도(2024/2025/2026) 포함, 각각 달라야 함
- 검색쿼리는 반드시 유효한 JSON 배열 형식으로
- 위 키 외에 다른 텍스트 출력 금지

출력:"""

        # [v2] num_predict=500
        raw = call_ollama(prompt, "", num_predict=500).strip()

        # ── 파싱 ─────────────────────────────────────────────────
        result = {
            "domain":             "일반",
            "compressed":         raw,   # 전체 출력을 step3 의 [상대 주장 약점] 으로 전달
            "logic":              stance,
            "strategy":           "1",
            "conclusion_pattern": "",
            "queries":            [],
        }

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("도메인:"):
                result["domain"] = line.split(":", 1)[-1].strip()
            elif line.startswith("전략번호:"):
                num = line.split(":", 1)[-1].strip()[:1]
                if num in STRATEGY_MAP:
                    result["strategy"] = num
            elif line.startswith("반박논리:"):
                parsed = line.split(":", 1)[-1].strip()
                if parsed:
                    result["logic"] = parsed
            elif line.startswith("결론_패턴:"):
                result["conclusion_pattern"] = line.split(":", 1)[-1].strip()[:20]
            elif line.startswith("검색쿼리:"):
                raw_q = line.split(":", 1)[-1].strip()
                try:
                    pq = json.loads(raw_q)
                    result["queries"] = [q for q in pq if isinstance(q, str) and q.isascii()][:4]
                except Exception:
                    pass

        # 검색쿼리 파싱 실패 시 전체 raw 에서 JSON 배열 탐색
        if not result["queries"]:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                try:
                    pq = json.loads(match.group())
                    result["queries"] = [q for q in pq if isinstance(q, str) and q.isascii()][:4]
                except Exception:
                    pass

        # 전략 강제 교체 (직전과 같은 전략이 나온 경우)
        recent_strats = self._used_strategies[-2:]
        if result["strategy"] in recent_strats:
            counts = {k: self._used_strategies.count(k) for k in available_strategies}
            result["strategy"] = min(counts, key=counts.get)
            print(f"  └─ [combined] 전략 강제 교체 → {result['strategy']}")

        # 상태 업데이트
        self._used_domains.append(result["domain"])
        self._used_strategies.append(result["strategy"])
        self._used_logics.append(result["logic"][:150])
        if result["conclusion_pattern"]:
            self._used_conclusions.append(result["conclusion_pattern"])

        print(f"  └─ [combined] 도메인='{result['domain']}' 전략={result['strategy']} 쿼리={len(result['queries'])}개")
        print(f"  └─ [combined] 반박논리: '{result['logic'][:80]}'")

        return result

    # ── 3단계: 힌트 생성 (원본과 동일) ──────────────────────────

    def _step3_hint(
        self,
        compressed: str,
        logic: str,
        strategy: str,
        search_block: str,
        evidence_block: str,
        history_block: str,
        topic: str,
        stance: str,
        situation: str,
        domain: str,
    ) -> str:
        prior_conclusions = " / ".join(self._used_conclusions[:-1]) if len(self._used_conclusions) > 1 else "없음"

        prompt = f"""당신은 한국어 토론 코치입니다. 유저에게 반박 힌트를 작성합니다.

══════════════════════════════════════
[상황]
{situation}

[토론 주제]
{topic}

[유저가 증명해야 할 결론]
"{stance}"

[이번 힌트의 논거 도메인] ← 4개 문장 전체가 반드시 이 영역 안에서만 전개될 것
{domain}

[반박 방향]
{logic}

[전략]
{strategy}

[상대 주장의 약점]
{compressed}

══════════════════════════════════════
[증거 A: 상대 격파용 ({domain} 영역)]
[검색 결과]
{search_block}

[팀 뉴스]
{evidence_block}

[증거 B: 유저 강화용 ({domain} 영역)]
(위 증거 중 유저 입장을 뒷받침하는 내용 선택)

[이미 사용한 내용 — 반복 금지]
{history_block}

[직전 힌트의 결론 귀결 패턴 — 이 패턴으로 끝내지 말 것]
{prior_conclusions}
══════════════════════════════════════

아래 4개 문장을 작성하세요.
⚠️ 4개 문장 모두 "{domain}" 영역의 논거만 사용할 것. 다른 영역으로 빠지지 말 것.
⚠️ 길이는 신경 쓰지 말고, 사례 설명과 논리 연결이 충분히 될 때까지 써야 함.

--- 문장1: 상대 약점 공격 ---
상대 주장을 인정하되 그 한계를 드러내는 문장. 아래 중 이번 증거와 논리에 가장 잘 맞는 방식으로 쓸 것. 매번 "해결 가능한 문제입니다"로 끝내지 말 것.

공격 방식 예시 (이 중 하나를 선택하되, 주제에 맞게 자유롭게 변형):
  ※ 반드시 아래처럼 [인정 문장] + [반격 문장] 두 개로 나눠 쓸 것.

  예시1)
    [인정] "~은 사실이에요."
    [반격] "그런데 실제로는 [누가] [언제] [어떻게] 했더니 [결과]가 나왔어요."

  예시2)
    [인정] "~을 주장하는 건 이해해요."
    [반격] "그런데 [증거]를 보면 그 영향은 [이유] 때문에 [범위] 안에서 그쳤어요."

  예시3)
    [인정] "~이 문제인 건 맞아요."
    [반격] "그런데 [증거]를 보면 [비교 대상]에 비해 그 규모가 훨씬 작아요."

필수 요소:
  ① 상대 주장 인정
  ② 증거 A의 사례를 충분히 설명: 언제, 누가, 무슨 일이 있었는지, 그 결과가 어떻게 됐는지를 구체적으로 서술하고 수치를 붙일 것. 출처(기관명, 보고서명, 매체명 등)도 반드시 함께 밝힐 것. 사례 이름만 던지거나 한 단어로 요약하면 안 됨.
  ③ 그 사례가 왜 상대 주장의 한계를 보여주는지 — 어떤 수단·주체·방법으로 어떻게 대응됐는지 또는 왜 영향이 제한적이었는지 — 인과관계를 이어서 설명할 것
금지: "관리 가능하다", "해결할 수 있는 문제입니다"만 쓰고 구체적 수단을 생략하는 것

--- 문장2: 유저 주장 강화 ---
필수 요소:
  ① 증거 B의 사례를 충분히 설명: 언제, 누가, 무슨 일이 있었는지, 그 결과가 어떻게 됐는지를 구체적으로 서술하고 수치를 붙일 것. 출처(기관명, 보고서명, 매체명 등)도 반드시 함께 밝힐 것. 사례 이름만 던지거나 한 단어로 요약하면 안 됨.
  ② 그 사례가 왜 유저 측 피해가 더 심각한지를 인과관계로 설명: 무엇이 무너졌고, 그것이 왜 다시 되돌리기 어려운지, 그 결과로 어떤 상황이 이어지는지를 단계적으로 써야 함. 결론만 쓰면 안 됨.
금지: "훨씬 더 심각하다"는 결론만 쓰고 이유를 생략하는 것

--- 문장3: 논리 연결 + 결론 ---
문장1과 문장2에서 나온 사례와 근거를 명시적으로 연결해서 결론을 도출하는 문장.
필수:
  ① 문장1의 핵심 근거(어떤 사례에서 어떤 점이 드러났는지)와 문장2의 핵심 근거(어떤 사례에서 어떤 점이 드러났는지)를 함께 언급
  ② 둘을 비교했을 때 왜 유저 측 주장이 더 결정적인지 설명
  ③ "{stance}"로 자연스럽게 결론 도출
  ④ 직전 결론 패턴({prior_conclusions})과 다른 방식으로 결론 낼 것
금지: "따라서 우리가 옳다"처럼 근거 없이 결론만 쓰는 것 / 이미 쓴 귀결 패턴 반복

--- 문장4: 실전 가이드 ---
구조: "[문장1 또는 문장2의 수치·사례]를 근거로 [상대의 {domain} 관련 핵심 약점]을 짚은 뒤, [구체적 주장 방향]으로 펼쳐보는 건 어떨까요?"
필수: ① 앞에서 쓴 실제 수치/사례 언급 ② {domain} 안에서 유저가 할 수 있는 구체적 말하기 방향

══════════════════════════════════════
출력 규칙:
- 한국어만 사용
- 문장 전체를 "~요", "~어요", "~죠" 체로 통일할 것. "~다", "~습니다" 체 금지.
- 번호·기호·헤더 없이 4개 문장만 출력 (빈 줄 없이 이어서)
- 길이 제한 없음. 사례 설명과 논리 연결이 충분하면 됨.
- ⚠️ 모든 문장의 결론은 반드시 "{stance}"를 지지해야 함. "{stance}"와 반대되는 결론이 나오면 그 문장 전체를 다시 써야 함. 출력 전에 각 문장이 "{stance}"를 지지하는지 스스로 확인할 것.
- [이미 사용한 내용] 반복 금지
- 문장1~4 모두 어려운 한자어·전문용어 사용 금지.
- 전문 용어 최소화. 중학생도 이해할 수 있는 수준으로
- 딱딱한 논문체 금지. 대화하듯 자연스럽게 쓸 것.

출력:"""

        raw    = call_ollama(prompt, "").strip()
        lines  = [line.strip() for line in raw.splitlines() if line.strip()]
        result = " ".join(lines)

        print(f"  └─ [3단계] 힌트 생성 완료 ({len(result)}자)")
        return result

    # ── [v2] Tavily 병렬 실행 ────────────────────────────────────
    # 순차 루프 → ThreadPoolExecutor (max_workers=4)

    def _run_tavily(self, queries: list[str]) -> str:
        if not queries:
            print("  └─ [tavily] 쿼리 없음 — 건너뜀")
            return "(검색 결과 없음)"

        def _fetch(q: str) -> list[dict]:
            safe_q = "".join(c for c in q if ord(c) < 128).strip()
            if not safe_q:
                return []
            results = []
            try:
                raw = _tavily.invoke(safe_q)
                if isinstance(raw, str) and raw.strip():
                    results.append({"query": safe_q, "title": "", "url": "", "content": raw})
                elif isinstance(raw, dict):
                    for r in raw.get("results", [raw]):
                        if isinstance(r, dict):
                            r.setdefault("query", safe_q)
                            results.append(r)
                elif isinstance(raw, list):
                    for r in raw:
                        if isinstance(r, dict):
                            r.setdefault("query", safe_q)
                            results.append(r)
            except Exception as e:
                print(f"  └─ 검색 오류 ({safe_q[:30]}...): {e}")
            return results

        all_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for partial in ex.map(_fetch, queries):
                all_results.extend(partial)

        if not all_results:
            return "(검색 결과 없음)"

        first_two = set(queries[:2]) if len(queries) >= 2 else set()

        block = ""
        for i, r in enumerate(all_results[:6], 1):
            content = (r.get("full_content") or r.get("content") or "")[:500]
            if not content:
                continue
            role = "격파용 (상대 한계 증거)" if r.get("query", "") in first_two else "강화용 (유저 주장 증거)"
            block += (
                f"[검색{i} | {role}]\n"
                f"쿼리: {r.get('query', '')}\n"
                f"제목: {r.get('title', '')}\n"
                f"출처: {r.get('url', '')}\n"
                f"내용: {content}\n\n"
            )

        print(f"  └─ [tavily] 결과 {len(all_results)}건")
        return block.strip() or "(검색 결과 없음)"