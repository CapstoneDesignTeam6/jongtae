"""
agents/hint_agent.py — 반박/재반박 힌트 생성
gemma4:26b 최적화: 반박/재반박 컨텍스트 분리
"""

import re
import json
import os

from langchain_tavily import TavilySearch

from data.evidence import build_evidence_block, build_history_block, extract_last_ai_claim
from agents.llm import call_ollama

os.environ["TAVILY_API_KEY"] = "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"
_tavily = TavilySearch(max_results=2)

STRATEGY_MAP = {
    "1": "반증 사례 (Counter-example): 상대 주장의 일반화가 틀렸음을 단 하나의 반례로 증명.",
    "2": "비용-편익 공략 (Cost-Benefit): 상대 주장이 가져올 이익보다 비용·부작용이 훨씬 크다는 점 부각.",
    "3": "가치 서열 공격 (Hierarchy of Values): 상대 논리보다 지금 더 중요한 가치가 있음을 주장.",
    "4": "자가당착/귀류법 (Reductio ad Absurdum): 상대 논리를 끝까지 밀면 모순적 결론에 도달함을 보여줌.",
    "5": "포괄적 포섭 (Turning the Tables): 상대가 제시한 사실을 오히려 내 주장의 근거로 역이용.",
}

STRATEGY_MAP_EN = {
    "1": "Counter-example: Disprove the opponent's generalization with a single counterexample.",
    "2": "Cost-Benefit: Show that the costs/side-effects of the opponent's claim outweigh the benefits.",
    "3": "Hierarchy of Values: Argue that a more important value overrides the opponent's logic.",
    "4": "Reductio ad Absurdum: Show that following the opponent's logic leads to absurd conclusions.",
    "5": "Turning the Tables: Use the facts the opponent presented as evidence FOR your own claim.",
}


def extract_last_user_claim(history: list[dict]) -> str:
    for turn in reversed(history):
        if turn.get("role") in ("user", "human"):
            return turn.get("content", "")[:400]
    return ""


def extract_last_two_ai_claims(history: list[dict]) -> tuple[str, str]:
    ai_turns = [
        t.get("content", "") for t in history
        if t.get("role") in ("assistant", "ai")
    ]
    latest   = ai_turns[-1][:400] if len(ai_turns) >= 1 else ""
    previous = ai_turns[-2][:400] if len(ai_turns) >= 2 else ""
    return latest, previous


class HintAgent:

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        user_label: str = "찬성",
        ai_label: str = "반대",
        user_stance: str = "",
    ):
        self.evidence    = evidence_items or []
        self.user_label  = user_label
        self.ai_label    = ai_label
        self.user_stance = user_stance

    # ── 공개 API ──────────────────────────────────────────────────

    def rebuttal_hint(self, history: list[dict], topic: str) -> dict:
        return self._generate_hint(history, topic, mode="rebuttal", max_chars=2500*2)

    def counter_hint(self, history: list[dict], topic: str) -> dict:
        return self._generate_hint(history, topic, mode="counter", max_chars=2500 * 4)

    # ── 공통 힌트 생성 ───────────────────────────────────────────

    def _generate_hint(self, history, topic, mode, max_chars) -> dict:
        effective_stance = self.user_stance or f"'{self.user_label}' 입장이 옳다"

        json_block    = build_evidence_block(self.evidence, max_chars=max_chars)
        history_block = build_history_block(history)

        if mode == "rebuttal":
            ai_target    = extract_last_ai_claim(history)
            user_claim   = ""
            mode_context = (
                f"Situation: The opponent ({self.ai_label}) just made a new argument. "
                f"The user ({self.user_label}) needs to rebut it."
            )
        else:
            ai_target, ai_previous = extract_last_two_ai_claims(history)
            user_claim   = extract_last_user_claim(history)
            mode_context = (
                f"Situation: The user ({self.user_label}) argued: \"{user_claim[:150]}\"\n"
                f"Then the opponent ({self.ai_label}) attacked that argument.\n"
                f"Now the user needs to counter-attack and RESTORE their original position.\n"
                f"The user must defend \"{effective_stance}\" — not explain the opponent's costs."
            )

        ai_claim_compressed = self._compress_ai_claim(ai_target, topic, effective_stance, mode_context)
        rebuttal_logic, strategy = self._build_rebuttal_logic(ai_claim_compressed, topic, effective_stance, mode_context)
        tavily_block = self._search_by_logic(rebuttal_logic, topic)

        raw = self._build_hint(
            ai_claim_compressed, topic,
            rebuttal_logic, strategy,
            tavily_block, json_block, history_block,
            effective_stance, mode_context,
        )
        return {"hint": raw, "raw_response": raw}

    # ── 0단계: AI 주장 압축 ──────────────────────────────────────

    def _compress_ai_claim(self, ai_claim, topic, effective_stance, mode_context) -> str:
        system = f"""You are a debate analyst.

{mode_context}

Read the opponent's argument below and extract ONLY its weaknesses.
The goal is to help the user prove: "{effective_stance}"

Opponent's argument:
{ai_claim}

Fill in this template (Korean, one sentence each, no extra text):
Core claim: [opponent's main conclusion]
Weakest point: [most attackable logical flaw]
Why it's wrong: [why that flaw fails, from the perspective of "{effective_stance}"]

Output:"""

        raw = call_ollama(system, "")
        compressed = raw.strip() or ai_claim[:120]
        print(f"  └─ AI 주장 압축: {compressed[:150]}...")
        return compressed

    # ── 1단계: 반박 논리 + 전략 ──────────────────────────────────

    def _build_rebuttal_logic(self, ai_claim_compressed, topic, effective_stance, mode_context) -> tuple[str, str]:
        strategy_list = "\n".join(f"{k}. {v}" for k, v in STRATEGY_MAP_EN.items())

        system = f"""You are a debate coach.

{mode_context}

Debate topic: {topic}
Conclusion to prove: "{effective_stance}"
Opponent's flaws to attack:
{ai_claim_compressed}

Fill in the template below. Output the template only.

Strategy number: [1-5]
Rebuttal logic: [2 Korean sentences that prove "{effective_stance}". Last sentence MUST directly affirm "{effective_stance}". Never support the opponent.]

Strategy options:
{strategy_list}

IMPORTANT: The rebuttal logic must make "{effective_stance}" the final conclusion.
Do NOT end with anything that sounds like the opponent's position.

Output:
Strategy number: """

        raw = call_ollama(system, "")

        strategy_num = "1"
        logic = effective_stance
        for line in raw.split("\n"):
            line = line.strip()
            if line.lower().startswith("strategy number:"):
                num = line.split(":")[-1].strip()[:1]
                if num in STRATEGY_MAP:
                    strategy_num = num
            if line.lower().startswith("rebuttal logic:"):
                parsed = line.split(":", 1)[-1].strip()
                if parsed:
                    logic = parsed
            if line.startswith("전략번호:"):
                num = line.replace("전략번호:", "").strip()[:1]
                if num in STRATEGY_MAP:
                    strategy_num = num
            if line.startswith("반박논리:"):
                parsed = line.replace("반박논리:", "").strip()
                if parsed:
                    logic = parsed

        strategy = STRATEGY_MAP.get(strategy_num, STRATEGY_MAP["1"])
        print(f"  └─ 전략: {strategy_num}. {strategy[:50]}...")
        print(f"  └─ 반박논리: {logic[:150]}...")
        return logic, strategy

    # ── 2단계: 뉴스 검색 ─────────────────────────────────────────

    def _search_by_logic(self, rebuttal_logic: str, topic: str) -> str:
        system = f"""Output ONLY a JSON array of 4 English search queries. No explanation. No markdown.

Two types of queries needed:
1. TWO queries finding evidence that the opponent's loss is manageable/recoverable
   (e.g. US economic resilience, US GDP growth, US historical war spending recovery)
2. TWO queries finding evidence that the user's side faces unrecoverable destruction
   (e.g. infrastructure destruction long-term impact, physical damage non-recovery cases)

All queries must relate to:
Rebuttal logic: {rebuttal_logic}
Debate topic: {topic}

Include year: 2025 or 2026. Prefer concrete numbers.
Format: ["query1", "query2", "query3", "query4"]

Output:"""

        raw_queries = call_ollama(system, "")

        queries = []
        try:
            match = re.search(r"\[.*?\]", raw_queries, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                queries = [
                    q for q in queries
                    if isinstance(q, str) and all(ord(c) < 128 for c in q)
                ][:4]
        except Exception as e:
            print(f"  └─ 검색 쿼리 파싱 실패: {e}")

        if not queries:
            return "(검색 결과 없음)"

        print(f"  └─ 검색 쿼리: {queries}")
        return self._run_tavily(queries)

    # ── 3단계: 힌트 완성 ─────────────────────────────────────────

    def _build_hint(
        self,
        ai_claim_compressed, topic,
        rebuttal_logic, strategy,
        tavily_block, json_block, history_block,
        effective_stance, mode_context,
    ) -> str:
        system = f"""You are a Korean debate coach writing a hint.

        {mode_context}

        === USER'S CONCLUSION (every sentence must support this) ===
        "{effective_stance}"

        === REBUTTAL DIRECTION ===
        {rebuttal_logic}

        === STRATEGY ===
        {strategy}

        === OPPONENT'S FLAWS (reference for Sentence 1) ===
        {ai_claim_compressed}

        === EVIDENCE (use facts from here) ===
        [Team news]
        {json_block}

        [Search results]
        {tavily_block}

        === DO NOT REPEAT (already used) ===
        {history_block}

        === WRITE 4 KOREAN SENTENCES ===

        --- STYLE ---
        - Plain, conversational Korean. Like explaining to a friend.
        - Short clauses. One idea per clause.
        - Forbidden words: 상쇄, 불가역적, 패러다임, 지정학적 함의, 실존적, 국가 체제
        - Use "~을 생각하면" instead of "~을 고려할 때"
        - Use "~일 뿐이다" instead of "~에 불과하다"

        --- SENTENCE 1 ---
        Two connected sentences that together make ONE argument block.
        First sentence — acknowledge opponent's point but show it's limited or manageable:
          "[상대 주장]은 사실이지만, [구체적 수치 또는 실제 사례] + [왜 극복 가능한지 구체적 이유]입니다."
        
          MANDATORY structure:
          1. Acknowledge: "[상대 주장]은 사실이지만,"
          2. Evidence: [Evidence]에서 가져온 수치 또는 실제 사례
          3. WHY it's limited: 그것이 왜 일시적/극복 가능한지 구체적 이유
             - 회복 사례가 있는가? (~년 만에 회복했다)
             - 감당할 수 있는 규모인가? (전체의 ~%에 불과하다)
             - 대응 수단이 있는가? (~로 상쇄 가능하다)
        
          FORBIDDEN:
          - "일시적입니다"만 쓰고 왜 일시적인지 이유 없이 끝내는 것
          - "관리 가능합니다"만 쓰고 왜 관리 가능한지 이유 없이 끝내는 것
          - 수치·사례 없이 주장만 하는 것
          - [Evidence]에 없는 내용으로 이유를 지어내는 것
      These phrases MUST always be followed by a specific number or named case from [Evidence].

        Second sentence — contrast with the user side's stronger point:
          "반면 [유저측]의 [구체적 강점 또는 피해 유형]은 [왜 더 중요하거나 심각한지], 실제로 [구체적 사례 + 수치 또는 연도]가 이를 보여줍니다."

          The real-world example MUST name: subject + event/situation + specific outcome.
          Use from [Evidence] if available; otherwise use a well-known real case relevant to the topic.

        BAD Sentence 1 (no evidence, no specific example):
        "상대 주장은 일부 사실이지만 우리 측 주장이 더 중요합니다."

        GOOD Sentence 1 (evidence + example both present):
        "상대의 주장은 사실이지만, [구체적 수치나 사례]라 그 영향은 제한적입니다. 반면 [유저측]의 [강점]은 [구체적 사례]가 보여주듯 훨씬 결정적입니다."

        --- SENTENCE 2 ---
        One specific fact from [Evidence]:
          source name + number/% + what it means in context
        BAD: "보고서에 따르면 영향이 클 것입니다."
        GOOD: "[출처] [연도]에 따르면 [수치]로, [이것이 토론 주제에서 갖는 의미]입니다."

        --- SENTENCE 3 ---
        Connect Sentence 2's fact to prove "{effective_stance}".
        Name the SPECIFIC REASON why the user's conclusion is correct.
        BAD: "따라서 우리 측 입장이 옳습니다."
        GOOD: "[Sentence 2의 수치/사실]을 생각하면, [상대측 한계]와 달리 [유저측 강점]이 더 결정적이므로 {effective_stance}."

        --- SENTENCE 4 ---
        Tell the user HOW to argue. Pattern:
        "[Sentence 1의 사례 또는 Sentence 2 출처/수치]를 근거로, [상대 주장의 핵심 약점]을 짚으며 [구체적인 주장 방향]을 펼쳐보는 건 어떨까요?"
        FORBIDDEN: "~가 옳다고 반박해보는 건 어떨까요?"

        --- OUTPUT RULES ---
        - Korean only. No English.
        - No bullets, numbers, headers. Four plain sentences only.
          (Sentence 1 = two connected sentences counted as one block)
        - Under 420 Korean characters total.
        - Never support the opponent's side.
        - Never repeat facts from [DO NOT REPEAT].
        - Use [Evidence] first for examples; if unavailable, use well-known real cases relevant to the debate topic.

        Output:"""
        return call_ollama(system, "").strip()

    # ── Tavily 실행 ───────────────────────────────────────────────

    def _run_tavily(self, queries: list[str]) -> str:
        all_results = []
        for q in queries:
            safe_q = "".join(c for c in q if ord(c) < 128).strip()
            if not safe_q:
                continue
            try:
                raw = _tavily.invoke(safe_q)
                if isinstance(raw, str) and raw.strip():
                    all_results.append({
                        "query": safe_q, "title": "", "url": "",
                        "content": raw, "full_content": raw,
                    })
                elif isinstance(raw, dict):
                    if "results" in raw:
                        for r in raw["results"]:
                            if isinstance(r, dict):
                                r["query"] = safe_q
                                r.setdefault("full_content", r.get("content", ""))
                                all_results.append(r)
                    else:
                        raw["query"] = safe_q
                        raw.setdefault("full_content", raw.get("content", ""))
                        all_results.append(raw)
                elif isinstance(raw, list):
                    for r in raw:
                        if isinstance(r, dict):
                            r["query"] = safe_q
                            r.setdefault("full_content", r.get("content", ""))
                            all_results.append(r)
            except Exception as e:
                print(f"  └─ 힌트 검색 오류: {e}")

        if not all_results:
            return "(검색 결과 없음)"

        results_block = ""
        for i, r in enumerate(all_results[:6], 1):
            content = (r.get("full_content") or r.get("content") or "")[:500]
            if not content:
                continue
            results_block += (
                f"[검색결과 {i}] 쿼리: {r.get('query', '')}\n"
                f"제목: {r.get('title', '')}\n"
                f"출처: {r.get('url', '')}\n"
                f"내용: {content}\n\n"
            )

        return results_block.strip() or "(검색 결과 없음)"