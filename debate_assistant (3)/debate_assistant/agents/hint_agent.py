"""
agents/hint_agent.py — 반박/재반박 힌트 생성 (일반화판)

핵심 변경:
  1. _parse_stance_from_topic 제거 → LLM 단독으로 stance 생성
  2. 0단계 도메인 예시 제거 → LLM이 주제 보고 자유롭게 결정
  3. 3단계 금지 어휘 목록 제거 (특정 토픽 편향)
  그 외 프롬프트·전략·출력 언어는 한국어 그대로 유지
"""

import re
import json
import os

from langchain_tavily import TavilySearch

from data.evidence import build_evidence_block, build_history_block
from agents.llm import call_ollama

os.environ["TAVILY_API_KEY"] = "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"
_tavily = TavilySearch(max_results=2)

# 전략 목록
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
        self._used_domains: list[str]    = []
        self._used_strategies: list[str] = []
        self._used_logics: list[str]     = []

    # ── stance 생성 (LLM 단독) ───────────────────────────────────

    def _resolve_stance(self, topic: str) -> str:
        """
        규칙 기반 파싱 없이 LLM만으로 stance를 생성한다.
        어떤 포맷·주제의 토론에도 동작한다.
        """
        if self._user_stance_fixed:
            return self._user_stance_fixed
        if topic in self._stance_cache:
            return self._stance_cache[topic]

        prompt = f"""토론에서 유저가 증명해야 할 핵심 결론 한 문장을 만들어야 합니다.

토론 주제: {topic}
유저가 지지하는 쪽: {self.user_label}

[검토 기준 — 모두 만족해야 함]
1. 토론 주제의 핵심 질문에 "{self.user_label}"이 답이 되는 형태인가?
2. 토론 주제에 없는 새 개념이 추가되지 않았는가?
3. 30자 이내인가?

[출력 형식]
결론 문장만 출력. 설명·부연·따옴표 없이.

출력:"""

        raw    = call_ollama(prompt, "").strip().strip('"\'「」').rstrip(".")
        stance = raw if (raw and len(raw) <= 40) else f"{self.user_label}의 입장이 옳다"

        print(f"  └─ [stance] '{stance}'")
        self._stance_cache[topic] = stance
        return stance

    # ── 공개 API ─────────────────────────────────────────────────

    def rebuttal_hint(self, history: list[dict], topic: str) -> dict:
        """상대의 새 주장 직후 → 반박 힌트"""
        return self._generate_hint(history, topic, mode="rebuttal")

    def counter_hint(self, history: list[dict], topic: str) -> dict:
        """상대가 내 주장을 공격한 직후 → 재반박 힌트"""
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

        ai_target = ai_latest

        compressed, domain = self._step0_compress(ai_target, topic, stance, situation)
        logic, strat       = self._step1_logic(compressed, domain, topic, stance, situation, history_block)
        search_block       = self._step2_search(compressed, logic, topic, stance)
        hint               = self._step3_hint(
            compressed, logic, strat, search_block,
            evidence_block, history_block, topic, stance, situation
        )

        return {"hint": hint, "raw_response": hint}

    # ── 0단계: 상대 주장 압축 + 도메인 감지 ─────────────────────

    def _step0_compress(
        self, ai_claim: str, topic: str, stance: str, situation: str
    ) -> tuple[str, str]:
        """
        도메인 예시를 주입하지 않는다.
        LLM이 실제 주제를 읽고 맞는 도메인을 스스로 결정한다.
        """
        recent_domains   = self._used_domains[-2:]
        avoid_domain_str = (
            f"단, 다음 도메인은 직전 힌트에서 이미 사용했으므로 이번에는 반드시 피할 것: {', '.join(recent_domains)}"
            if recent_domains else ""
        )

        prompt = f"""당신은 토론 분석가입니다.

[상황]
{situation}

[토론 주제]
{topic}

[유저가 증명해야 할 결론]
"{stance}"

[상대의 주장]
{ai_claim}

{avoid_domain_str}

아래 항목을 각각 한 문장으로 작성하세요. 다른 내용은 절대 출력하지 마세요.

도메인: [상대 주장이 주로 근거하는 논거 영역을 한 단어로. 이 토론 주제와 상대 주장에서 실제로 사용된 개념 안에서 골라야 하며, 위에서 피하라고 한 도메인은 선택하지 말 것.]
핵심_주장: [상대의 결론을 한 문장으로]
가장_약한_지점: ["{stance}"를 증명하는 데 위 도메인 안에서 가장 유리하게 공격할 수 있는 논리적 허점]
공격_이유: [위 도메인의 관점에서 왜 그 허점이 "{stance}"를 뒷받침하는가]

출력:"""

        raw    = call_ollama(prompt, "").strip()
        domain = "일반"
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("도메인:"):
                domain = line.split(":", 1)[-1].strip()
                break

        self._used_domains.append(domain)
        print(f"  └─ [0단계] 도메인='{domain}' (누적: {self._used_domains})")
        return raw or ai_claim[:150], domain

    # ── 1단계: 반박 논리 + 전략 선택 ────────────────────────────

    def _step1_logic(
        self,
        compressed: str,
        domain: str,
        topic: str,
        stance: str,
        situation: str,
        history_block: str,
    ) -> tuple[str, str]:
        prior_logics    = " / ".join(self._used_logics[-2:]) if self._used_logics else "없음"
        recent_strats   = self._used_strategies[-2:]
        avoid_strat_str = (
            f"다음 전략은 직전 힌트에서 이미 사용했으므로 반드시 제외: {', '.join(recent_strats)}"
            if recent_strats else ""
        )
        available_strategies = {k: v for k, v in STRATEGY_MAP.items() if k not in recent_strats}
        if not available_strategies:
            available_strategies = STRATEGY_MAP
        strat_list = "\n".join(f"{k}. {v}" for k, v in available_strategies.items())

        prompt = f"""당신은 토론 코치입니다.

[상황]
{situation}

[토론 주제]
{topic}

[유저가 증명해야 할 결론]
"{stance}"

[상대 주장의 약점 분석]
{compressed}

[이번 반박의 논거 도메인] ← 반박논리는 반드시 이 영역 안에서만 전개할 것
{domain}

[직전 힌트에서 사용한 논리 방향 — 이번에는 반드시 다른 각도로]
{prior_logics}

{avoid_strat_str}

[사용 가능한 전략]
{strat_list}

규칙:
- 반박논리는 반드시 "{domain}" 영역의 구체적 논거로만 구성할 것
- 직전 논리 방향과 다른 각도에서 공격할 것
- "{stance}"에 반하는 내용 절대 금지

아래 형식으로만 출력하세요. 다른 내용 없이.

전략번호: [위 사용 가능한 전략 번호 중 하나]
반박논리: ["{domain}" 영역 안에서 "{stance}"를 증명하는 한국어 2문장. 마지막 문장은 "{stance}"를 직접 긍정하는 결론이어야 함.]

출력:"""

        raw       = call_ollama(prompt, "")
        strat_num = None
        logic     = stance

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("전략번호:"):
                num = line.split(":", 1)[-1].strip()[:1]
                if num in STRATEGY_MAP:
                    strat_num = num
            elif line.startswith("반박논리:"):
                parsed = line.split(":", 1)[-1].strip()
                if parsed:
                    logic = parsed

        if strat_num is None or strat_num in recent_strats:
            counts    = {k: self._used_strategies.count(k) for k in available_strategies}
            strat_num = min(counts, key=counts.get)
            print(f"  └─ [1단계] 전략 강제 교체 → {strat_num}")

        self._used_strategies.append(strat_num)
        self._used_logics.append(logic[:60])

        strategy = STRATEGY_MAP[strat_num]
        print(f"  └─ [1단계] 전략: {strat_num}. {strategy[:50]}...")
        print(f"  └─ [1단계] 반박논리: {logic[:150]}...")
        return logic, strategy

    # ── 2단계: 증거 검색 ─────────────────────────────────────────

    def _step2_search(
        self, compressed: str, logic: str, topic: str, stance: str
    ) -> str:
        prompt = f"""Output ONLY a valid JSON array of 4 English search queries. No markdown, no explanation.

Debate topic: {topic}
User must prove: "{stance}"
Opponent's weak point: {compressed[:200]}
Rebuttal direction: {logic[:200]}

Generate exactly 4 queries:
- Query 1 & 2: Find evidence that the OPPONENT's claimed damage is LIMITED, MANAGEABLE, or RECOVERABLE.
- Query 3 & 4: Find evidence that the USER's claimed damage is MORE SEVERE, IRREVERSIBLE, or DECISIVE.

Rules:
- English only, no special characters
- Include year 2024, 2025, or 2026
- Prefer queries that yield concrete numbers or named cases
- Each query must be clearly different from the others

Output format: ["query1", "query2", "query3", "query4"]

Output:"""

        raw     = call_ollama(prompt, "")
        queries: list[str] = []
        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                parsed  = json.loads(match.group())
                queries = [q for q in parsed if isinstance(q, str) and q.isascii()][:4]
        except Exception as e:
            print(f"  └─ [2단계] 쿼리 파싱 실패: {e}")

        if not queries:
            print("  └─ [2단계] 쿼리 없음 — 검색 건너뜀")
            return "(검색 결과 없음)"

        print(f"  └─ [2단계] 검색 쿼리: {queries}")
        return self._run_tavily(queries)

    # ── 3단계: 힌트 생성 ─────────────────────────────────────────

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
    ) -> str:
        """
        특정 토픽 편향 금지 어휘 목록 제거.
        LLM이 증거와 논리에 맞는 표현을 자유롭게 선택한다.
        """
        prompt = f"""당신은 한국어 토론 코치입니다. 유저에게 반박 힌트를 작성합니다.

══════════════════════════════════════
[상황]
{situation}

[토론 주제]
{topic}

[유저가 증명해야 할 결론] ← 모든 문장이 이것을 지지해야 함
"{stance}"

[반박 방향]
{logic}

[전략]
{strategy}

[상대 주장의 약점] ← 문장1에서 공격할 재료
{compressed}

══════════════════════════════════════
[증거 A: 상대 격파용] ← 상대 측 피해가 제한적/관리 가능하다는 증거
[검색 결과]
{search_block}

[팀 뉴스]
{evidence_block}

[증거 B: 유저 강화용] ← 위 증거 중 유저 입장을 뒷받침하는 내용을 선택
(위 검색 결과 및 팀 뉴스 중 유저 입장을 뒷받침하는 내용을 선택하여 사용)

[이미 사용한 내용 — 반복 금지]
{history_block}
══════════════════════════════════════

아래 4개 문장을 작성하세요.

--- 문장1: 상대 약점 공격 ---
구조: "[상대 주장]은 사실이지만, [증거 A의 구체적 수치나 사례]를 보면 [왜 그 영향이 제한적인지 이유]입니다."

필수 요소:
  ① 상대 주장 인정 ("~은 사실이지만,")
  ② [증거 A]에서 가져온 수치 또는 실제 사례 (출처·연도·숫자 포함)
  ③ 그 영향이 왜 일시적/극복 가능/제한적인지 구체적 이유
금지: 이유 없이 "일시적", "관리 가능"만 쓰는 것 / 증거 없는 주장

--- 문장2: 유저 주장 강화 ---
구조: "반면 [유저 측 핵심 피해/강점]은 [증거 B의 구체적 사례·수치]에서 보듯 훨씬 [크거나 심각하거나 결정적]입니다."

필수 요소:
  ① [증거 B]에서 가져온 구체적 사례 (주체+사건+결과 형태)
  ② 유저 측이 왜 더 큰 피해를 입거나 더 유리한 논거인지 설명
금지: 증거 없이 유저 측이 더 낫다고만 주장하는 것

--- 문장3: 논리 연결 + 결론 ---
구조: "[문장1 근거]와 [문장2 근거]를 함께 보면, [상대 측 한계]와 달리 [유저 측 강점]이 더 결정적이므로 {stance}."

필수 요소:
  ① 앞 두 문장의 근거를 명시적으로 연결
  ② "{stance}" 결론을 자연스럽게 도출
금지: 단순히 "따라서 우리가 옳다"로 끝내는 것

--- 문장4: 실전 가이드 ---
구조: "[문장1 또는 문장2의 구체적 수치·사례]를 근거로 [상대 주장의 핵심 약점]을 짚은 뒤, [구체적 주장 방향]으로 펼쳐보는 건 어떨까요?"

필수 요소:
  ① 앞에서 쓴 실제 수치나 사례를 언급
  ② 유저가 실제로 할 수 있는 구체적인 말하기 방향 제시
금지: "~가 옳다고 주장해보세요"처럼 막연한 지시

══════════════════════════════════════
출력 규칙:
- 한국어만 사용
- 번호·기호·헤더 없이 4개 문장만 출력 (문장 사이 빈 줄 없이 이어서 출력할 것)
- 총 600자 이내
- "{stance}"에 반하는 문장 절대 금지
- [이미 사용한 내용]에 있는 사실 반복 금지
- 쉽고 자연스러운 한국어로 (친구에게 설명하듯)

출력:"""

        raw    = call_ollama(prompt, "").strip()
        lines  = [line.strip() for line in raw.splitlines() if line.strip()]
        result = " ".join(lines)

        print(f"  └─ [3단계] 힌트 생성 완료 ({len(result)}자)")
        return result

    # ── Tavily 실행 ──────────────────────────────────────────────

    def _run_tavily(self, queries: list[str]) -> str:
        all_results: list[dict] = []
        for q in queries:
            safe_q = "".join(c for c in q if ord(c) < 128).strip()
            if not safe_q:
                continue
            try:
                raw = _tavily.invoke(safe_q)
                if isinstance(raw, str) and raw.strip():
                    all_results.append({"query": safe_q, "title": "", "url": "", "content": raw})
                elif isinstance(raw, dict):
                    for r in raw.get("results", [raw]):
                        if isinstance(r, dict):
                            r.setdefault("query", safe_q)
                            all_results.append(r)
                elif isinstance(raw, list):
                    for r in raw:
                        if isinstance(r, dict):
                            r.setdefault("query", safe_q)
                            all_results.append(r)
            except Exception as e:
                print(f"  └─ 검색 오류 ({safe_q[:30]}...): {e}")

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

        return block.strip() or "(검색 결과 없음)"