"""
agents/summary_agent.py — 토론 정리 + 피드백 생성
"""

import re
import json
import os
import datetime

from langchain_tavily import TavilySearch

from data.evidence import build_evidence_block, build_history_block
from agents.llm import call_ollama

os.environ["TAVILY_API_KEY"] = "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"
_tavily = TavilySearch(max_results=2)


class SummaryAgent:

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        user_label: str = "찬성",
        ai_label: str = "반대",
    ):
        self.evidence   = evidence_items or []
        self.user_label = user_label
        self.ai_label   = ai_label

    def summarize(
        self,
        history: list[dict],
        topic: str,
    ) -> dict:
        turns = sum(1 for h in history if h["role"] == "user")

        json_block    = build_evidence_block(self.evidence, max_chars=2500)
        history_block = build_history_block(history)

        invalid_contents, clean_history_block = self._filter_invalid_turns(history, topic)

        summary, raw_summary = self._summarize_core(
            history_block, topic, json_block, turns, invalid_contents
        )
        logic_feedback, raw_feedback = self._generate_feedback_with_search(
            clean_history_block, topic, json_block, turns,
            summary_context=summary
        )
        extra_info, raw_extra = self._generate_extra_info(
            clean_history_block, topic, json_block
        )

        return {
            "summary":        summary,
            "logic_feedback": logic_feedback,
            "extra_info":     extra_info,
            "raw_summary":    raw_summary,
            "raw_feedback":   raw_feedback,
            "raw_extra":      raw_extra,
            "invalid_turns":  invalid_contents,
        }

    # ── 무효 발언 필터 ────────────────────────────────────────────

    def _filter_invalid_turns(
        self,
        history: list[dict],
        topic: str,
    ) -> tuple[list[str], str]:

        user_turns = []
        user_turn_num = 0
        for h in history:
            if h["role"] == "user":
                user_turn_num += 1
                user_turns.append({"turn": user_turn_num, "content": h["content"]})

        if not user_turns:
            return [], build_history_block(history)

        turns_block = "\n".join(
            f"[유저 발언 {t['turn']}번째] {t['content']}"
            for t in user_turns
        )

        check_system = """당신은 토론 심판입니다.
아래 유저 발언들이 주어진 토론 주제와 관련된 논거를 포함하는지 판단하세요.

판단 기준:
- 토론 주제와 관련된 사실, 수치, 사례, 논리적 주장이 포함되면 유효
- 감탄사, 일상 대화, 주제와 무관한 내용, 단순 감정 표현이면 무효

출력 규칙:
- 무효 발언이 있으면 해당 번째 번호만 JSON 배열로 출력. 예: [1, 3]
- 모든 발언이 유효하면 빈 배열 출력. 예: []
- JSON 배열 외 아무것도 출력하지 말 것"""

        raw = call_ollama(check_system, f"토론 주제: {topic}\n\n{turns_block}")
        print(f"  └─ 유효성 검사 원문: {raw[:200]}")

        invalid_contents = []
        clean_history = history[:]

        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                invalid_turn_numbers = set(json.loads(match.group()))
                print(f"  └─ 무효 턴: {invalid_turn_numbers}")
                if invalid_turn_numbers:
                    invalid_contents = [
                        t["content"] for t in user_turns
                        if t["turn"] in invalid_turn_numbers
                    ]
                    invalid_set = {
                        t["content"] for t in user_turns
                        if t["turn"] in invalid_turn_numbers
                    }
                    clean_history = [
                        h for h in history
                        if not (h["role"] == "user" and h["content"] in invalid_set)
                    ]
        except Exception as e:
            print(f"  └─ 유효성 검사 파싱 실패: {e} | 원문: {raw[:200]}")

        return invalid_contents, build_history_block(clean_history)

    def _summarize_core(
            self,
            history_block: str,
            topic: str,
            json_block: str,
            turns: int,
            invalid_contents: list[str],
    ) -> tuple[str, str]:

        invalid_note = ""
        if invalid_contents:
            previews = []
            for c in invalid_contents:
                preview = c.strip()[:15]
                if len(c.strip()) > 15:
                    preview += "..."
                previews.append(f'"{preview}"')
            invalid_note = (
                f"\n\n[참고] {', '.join(previews)} 같은 발언은 "
                f"토론 논거로 보기 어려워 요약에서 제외했어요."
            )

        # ── 1단계: 주장·근거 추출 ──────────────────────────────────
        extract_system = f"""토론 기록에서 각 측 발언을 문장 단위로 그대로 복사하세요.
        절대 요약·추측·추가·창작 금지. 토론 기록에 있는 문장만 그대로.

        [{self.user_label}]
        (유저 발언에서 주장·근거 문장을 그대로 복사. 줄 나눔.)

        [{self.ai_label}]
        (AI 발언에서 주장·근거 문장을 그대로 복사. 줄 나눔.)"""

        extracted = call_ollama(
            extract_system,
            f"토론 기록:\n{history_block}"
        )
        print(f"  └─ 추출 결과:\n{extracted[:300]}")

        # ── 2단계: 문체 정리 ───────────────────────────────────────
        polish_system = f"""[추출된 문장들]을 자연스럽고 읽기 쉬운 문어체 한 문단으로 이어주세요.

        규칙:
        - [추출된 문장들]에 있는 내용만 사용. 새로운 내용·단어 추가 절대 금지.
        - 문장들을 자연스럽게 연결하되 내용은 바꾸지 말 것.
        - 추측·단정 표현("~할 것입니다", "~가능성이 큽니다") 절대 금지.
        - 대신 "~라고 주장했습니다", "~라고 말했습니다", "~가능성이 크다고 언급했습니다",
          "~될 것이라고 강조했습니다" 처럼 발언을 인용하는 형식으로 마무리.
        - 딱딱한 나열 금지. 문장과 문장이 자연스럽게 이어지도록.
        - 각 문단 240자 이내.

        아래 형식으로 출력:

        [{self.user_label}]
        (정리된 문단)

        [{self.ai_label}]
        (정리된 문단)"""

        raw = call_ollama(
            polish_system,
            f"추출된 내용:\n{extracted}"
        )

        # 파싱
        user_match = re.search(
            rf"\[{re.escape(self.user_label)}\]\s*(.*?)(?=\[{re.escape(self.ai_label)}\]|$)",
            raw, re.DOTALL
        )
        ai_match = re.search(
            rf"\[{re.escape(self.ai_label)}\]\s*(.*?)$",
            raw, re.DOTALL
        )

        if user_match and ai_match:
            user_block = user_match.group(1).strip()
            ai_block = ai_match.group(1).strip()
            summary = (
                f"[{self.user_label}]\n{user_block}\n\n"
                f"[{self.ai_label}]\n{ai_block}"
            )
        else:
            summary = raw.strip()

        if invalid_note:
            summary = summary + invalid_note

        return summary, raw

    # ── 2차: 논리 피드백 + Tavily 보완 검색 ─────────────────────

    def _generate_feedback_with_search(
        self,
        history_block: str,
        topic: str,
        json_block: str,
        turns: int,
        summary_context: str,
    ) -> tuple[str, str]:

        feedback_system = f"""당신은 시사 토론 코치입니다.
유저={self.user_label}, {turns}라운드.

한 문단만 출력. 제목·번호·불릿 없이 바로 본문.

[논리 피드백, 200자 이내]
유저({self.user_label}) 논증에서 강했던 점 1가지, 보완이 필요한 논리 구조 1가지를 구체적으로.
쉽고 자연스러운 말로. 지시형 금지. 관찰·분석만."""

        raw_feedback = call_ollama(
            feedback_system,
            f"주제: {topic}\n\n토론 요약:\n{summary_context}\n\n"
            f"전체 토론 기록:\n{history_block}\n\n기본 증거:\n{json_block}"
        )
        logic_feedback = raw_feedback.strip()

        queries = self._generate_queries_with_ollama(logic_feedback, topic)
        if not queries:
            return logic_feedback, raw_feedback

        search_results = self._run_tavily_search(queries)
        if not search_results:
            return logic_feedback, raw_feedback

        supplement = self._build_supplement(
            search_results, logic_feedback, topic, history_block
        )
        final_feedback = f"{logic_feedback}\n\n{supplement}" if supplement else logic_feedback
        return final_feedback, raw_feedback

    def _generate_queries_with_ollama(self, logic_feedback, topic) -> list[str]:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        system = """You are a debate researcher.
Read the [Logic Feedback] and identify the logical weakness that needs to be supplemented.
Generate 5 English search queries to find real case studies, statistics, or historical examples
that can strengthen that weakness.

Rules:
- Output ONLY a JSON array of 5 English strings. No explanation, no markdown.
- Include specific keywords: statistics, case study, historical example, 2025, 2026
- Example format: ["query1", "query2", "query3", "query4", "query5"]"""

        raw = call_ollama(
            system,
            f"Date: {current_date}\nDebate topic: {topic}\nLogic feedback: {logic_feedback}"
        )

        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                return [
                    q for q in queries
                    if isinstance(q, str) and all(ord(c) < 128 for c in q)
                ][:5]
        except Exception as e:
            print(f"  └─ 쿼리 파싱 실패: {e}")
        return []

    def _run_tavily_search(self, queries: list[str]) -> list[dict]:
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
                    if "error" in raw:
                        continue
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
                        if isinstance(r, str) and r.strip():
                            all_results.append({
                                "query": safe_q, "title": "", "url": "",
                                "content": r, "full_content": r,
                            })
                        elif isinstance(r, dict):
                            r["query"] = safe_q
                            r.setdefault("full_content", r.get("content", ""))
                            all_results.append(r)
            except Exception as e:
                print(f"  └─ 검색 오류: {e}")
        return all_results

    def _build_supplement(
        self,
        search_results: list[dict],
        logic_feedback: str,
        topic: str,
        history_block: str,
    ) -> str:
        results_block = ""
        for i, r in enumerate(search_results[:8], 1):
            content = (r.get("full_content") or r.get("content") or "")[:400]
            if not content:
                continue
            results_block += (
                f"[검색결과 {i}] 쿼리: {r.get('query', '')}\n"
                f"제목: {r.get('title', '')}\n출처: {r.get('url', '')}\n"
                f"내용: {content}\n\n"
            )

        if not results_block.strip():
            return ""

        system = """당신은 시사 토론 코치입니다.

        [논리 피드백]에서 지적된 보완 포인트를 [Tavily 검색결과]의 실제 사례나 수치로 뒷받침하세요.
        단, 사례를 나열하는 데 그치지 말고, 그 사례가 토론에서 어떤 역할을 할 수 있는지
        쉽고 자연스러운 말로 친구처럼 알려주세요.

        규칙:
        - [전체 토론 기록]에서 이미 언급된 사건·수치·사례는 절대 반복 금지
        - 검색결과에 없는 내용 사용 금지. 확인되지 않은 수치 금지
        - 출처는 "~에 따르면" 형식으로 자연스럽게 한 번만 언급
        - 수치나 사례를 소개한 뒤, "이걸 쓰면 ~을 보여줄 수 있어요" 또는
          "이 사례를 들면 ~가 훨씬 설득력 있게 들릴 거예요" 같은 식으로 마무리
        - 전문 용어 최소화. 중학생도 이해할 수 있는 수준으로
        - 200자 이내, 2~3문장, 번호·불릿 없이 바로 본문만 출력"""

        raw = call_ollama(
            system,
            f"논리 피드백:\n{logic_feedback}\n\n"
            f"전체 토론 기록 (중복 금지):\n{history_block}\n\n"
            f"Tavily 검색결과:\n{results_block}"
        )
        return raw.strip()

    # ── 3차: news_data 미언급 추가 사례 ──────────────────────────

    def _generate_extra_info(
        self,
        history_block: str,
        topic: str,
        json_block: str,
    ) -> tuple[str, str]:

        system = """당신은 시사 정보 제공자입니다.

규칙:
- [전체 토론 기록]에서 이미 언급된 사건명·수치·사례·기관명은 절대 반복 금지
- [기본 증거]에서만 새 정보 추출. 증거에 없는 내용 사용 금지
- 출처를 "~에 따르면" 형식으로 자연스럽게 언급
- 200자 이내, 번호·불릿 없이 바로 본문만 출력
- 확인되지 않은 수치 금지
- 쉽고 자연스러운 말로"""

        raw = call_ollama(
            system,
            f"주제: {topic}\n\n"
            f"전체 토론 기록 (중복 금지):\n{history_block}\n\n"
            f"기본 증거 (news_data - 새 정보 출처):\n{json_block}"
        )
        return raw.strip(), raw