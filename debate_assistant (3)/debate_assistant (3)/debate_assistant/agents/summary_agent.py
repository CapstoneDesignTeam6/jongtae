"""
agents/summary_agent.py — 토론 정리 + 피드백 생성

LLM 호출 구조 (Ollama + Tavily):
    1차 Ollama : 토론 요약 + 쟁점 분석
    2차 Ollama : 논리 피드백 (강점 + 약점)
                 → Ollama로 영문 Tavily 쿼리 생성
                 → Tavily 검색 (ASCII 쿼리만 전송)
                 → 검색 결과를 피드백 뒤에 추가
    3차 Ollama : news_data.json 미언급 추가 사례 추출
"""

import re
import json
import os
import datetime

from langchain_tavily import TavilySearch

from data.evidence import build_evidence_block, build_history_block
from agents.llm import call_ollama

os.environ["TAVILY_API_KEY"] =""
_tavily = TavilySearch(max_results=2)


class SummaryAgent:

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        stance: int = 1,
    ):
        self.evidence   = evidence_items or []
        self.user_label = "찬성" if stance == 1 else "반대"
        self.ai_label   = "반대" if stance == 1 else "찬성"

    def summarize(
        self,
        history: list[dict],
        topic: str,
        turns: int = 1,
    ) -> dict:
        json_block    = build_evidence_block(self.evidence, max_chars=2500)
        history_block = build_history_block(history)

        print("  [토론 정리 1/3] 요약·쟁점 생성 중...")
        summary, issues, raw_summary = self._summarize_core(
            history_block, topic, json_block, turns
        )

        print("  [토론 정리 2/3] 논리 피드백 + 보완 검색 중...")
        logic_feedback, raw_feedback = self._generate_feedback_with_search(
            history_block, topic, json_block, turns,
            summary_context=f"{summary}\n\n{issues}".strip()
        )

        print("  [토론 정리 3/3] 추가 사례 추출 중...")
        extra_info, raw_extra = self._generate_extra_info(
            history_block, topic, json_block
        )

        return {
            "summary":        summary,
            "issues":         issues,
            "logic_feedback": logic_feedback,
            "extra_info":     extra_info,
            "raw_summary":    raw_summary,
            "raw_feedback":   raw_feedback,
            "raw_extra":      raw_extra,
        }

    # ── 1차: 토론 요약 + 쟁점 ────────────────────────────────────

    def _summarize_core(
        self,
        history_block: str,
        topic: str,
        json_block: str,
        turns: int,
    ) -> tuple[str, str, str]:
        system = f"""당신은 시사 토론 기록자입니다. 승패 판정 금지.
유저={self.user_label}, AI={self.ai_label}, {turns}라운드 진행.

오직 두 문단만 출력. 제목 없이 문단 사이 빈 줄 하나.

[문단 1 - 토론 요약, 200자 이내]
양측이 내세운 핵심 논점과 근거(수치·사례)를 중립적으로 정리.

[문단 2 - 쟁점 분석, 150자 이내]
양측이 가장 첨예하게 부딪힌 지점과 서로 충분히 답하지 못한 부분.

공통: 번호·불릿·제목 없이 바로 본문. 확인되지 않은 수치 금지."""

        raw   = call_ollama(system,
            f"주제: {topic}\n\n{history_block}\n\n"
            f"기본 증거 (news_data.json):\n{json_block}"
        )
        paras   = [p.strip() for p in raw.split("\n\n") if p.strip()]
        summary = paras[0] if paras else raw
        issues  = paras[1] if len(paras) >= 2 else ""
        return summary, issues, raw

    # ── 2차: 논리 피드백 + Tavily 보완 검색 ─────────────────────

    def _generate_feedback_with_search(
        self,
        history_block: str,
        topic: str,
        json_block: str,
        turns: int,
        summary_context: str,
    ) -> tuple[str, str]:

        # Step 1: 논리 피드백 생성
        feedback_system = f"""당신은 시사 토론 코치입니다.
유저={self.user_label}, {turns}라운드.

한 문단만 출력. 제목·번호·불릿 없이 바로 본문.

[논리 피드백, 200자 이내]
유저 논증에서 강했던 점 1가지, 보완이 필요한 논리 구조 1가지를 구체적으로.
지시형 금지. 관찰·분석만."""

        raw_feedback = call_ollama(feedback_system,
            f"주제: {topic}\n\n"
            f"토론 요약:\n{summary_context}\n\n"
            f"전체 토론 기록:\n{history_block}\n\n"
            f"기본 증거 (news_data.json):\n{json_block}"
        )
        logic_feedback = raw_feedback.strip()
        print(f"  └─ 피드백 생성 완료")

        # Step 2: 영문 쿼리 생성 (Ollama)
        queries = self._generate_queries_with_ollama(logic_feedback, topic)
        if not queries:
            return logic_feedback, raw_feedback

        # Step 3: Tavily 검색
        search_results = self._run_tavily_search(queries)
        if not search_results:
            print("  └─ 검색 결과 없음. 피드백만 반환.")
            return logic_feedback, raw_feedback

        # Step 4: 검색 결과 → 보완 문단 생성 후 피드백 뒤에 붙임
        supplement = self._build_supplement(
            search_results, logic_feedback, topic, history_block
        )
        final_feedback = f"{logic_feedback}\n\n{supplement}" if supplement else logic_feedback
        return final_feedback, raw_feedback

    def _generate_queries_with_ollama(
        self,
        logic_feedback: str,
        topic: str,
    ) -> list[str]:
        """Ollama로 영문 전용 검색 쿼리 생성."""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        system = """You are a debate researcher.
Read the [Logic Feedback] and identify the logical weakness that needs to be supplemented.
Generate 5 English search queries to find real case studies, statistics, or historical examples
that can strengthen that weakness.

Rules:
- Output ONLY a JSON array of 5 English strings. No explanation, no markdown.
- Include specific keywords: statistics, case study, historical example, 2025, 2026
- Example format: ["query1", "query2", "query3", "query4", "query5"]"""

        raw = call_ollama(system,
            f"Date: {current_date}\n"
            f"Debate topic: {topic}\n"
            f"Logic feedback: {logic_feedback}"
        )

        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                # ASCII만 허용 (latin-1 인코딩 오류 방지)
                queries = [
                    q for q in queries
                    if isinstance(q, str) and all(ord(c) < 128 for c in q)
                ][:5]
                print(f"  └─ 검색 쿼리 {len(queries)}개 생성")
                for i, q in enumerate(queries, 1):
                    print(f"     [{i}] {q}")
                return queries
        except Exception as e:
            print(f"  └─ 쿼리 파싱 실패: {e} | 원문: {raw[:200]}")
        return []

    def _run_tavily_search(self, queries: list[str]) -> list[dict]:
        """Tavily 검색. ASCII 쿼리만 전송."""
        all_results = []
        for q in queries:
            # 혹시라도 non-ASCII가 섞였으면 제거
            safe_q = "".join(c for c in q if ord(c) < 128).strip()
            if not safe_q:
                continue
            try:
                raw = _tavily.invoke(safe_q)

                if isinstance(raw, str):
                    if raw.strip():
                        all_results.append({
                            "query": safe_q, "title": "", "url": "",
                            "content": raw, "full_content": raw,
                        })
                    print(f"  └─ 검색 완료: {safe_q[:50]!r} (텍스트)")

                elif isinstance(raw, dict):
                    if "error" in raw:
                        print(f"  └─ Tavily 에러: {raw['error']}")
                        continue
                    if "results" in raw:
                        for r in raw["results"]:
                            if isinstance(r, dict):
                                r["query"] = safe_q
                                r.setdefault("full_content", r.get("content", ""))
                                all_results.append(r)
                        print(f"  └─ 검색 완료: {safe_q[:50]!r} ({len(raw['results'])}건)")
                    else:
                        raw["query"] = safe_q
                        raw.setdefault("full_content", raw.get("content", ""))
                        all_results.append(raw)
                        print(f"  └─ 검색 완료: {safe_q[:50]!r} (dict)")

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
                    print(f"  └─ 검색 완료: {safe_q[:50]!r} ({len(raw)}건)")

            except Exception as e:
                print(f"  └─ 검색 오류: {e}")

        print(f"  └─ 총 검색 결과: {len(all_results)}건")
        return all_results

    def _build_supplement(
        self,
        search_results: list[dict],
        logic_feedback: str,
        topic: str,
        history_block: str,
    ) -> str:
        """검색 결과 → 피드백 보완 문단 (Ollama)."""
        results_block = ""
        for i, r in enumerate(search_results[:8], 1):
            content = (r.get("full_content") or r.get("content") or "")[:400]
            if not content:
                continue
            title = r.get("title", "")
            url   = r.get("url", "")
            query = r.get("query", "")
            results_block += (
                f"[검색결과 {i}] 쿼리: {query}\n"
                f"제목: {title}\n출처: {url}\n내용: {content}\n\n"
            )

        if not results_block.strip():
            print("  └─ 검색결과 내용 없음. 보완 문단 생략.")
            return ""

        system = """당신은 시사 토론 코치입니다.
 
[논리 피드백]의 보완이 필요한 부분을 [Tavily 검색결과]의 실제 사례·수치로 보강하고,
그 내용을 어떻게 논거로 활용할 수 있는지 코치의 시각으로 제안하세요.
 
규칙:
- [전체 토론 기록]에서 이미 언급된 사건·수치·사례는 절대 반복 금지
- 검색결과에 없는 내용 사용 금지. 확인되지 않은 수치 금지
- 출처를 "~에 따르면" 형식으로 자연스럽게 언급
- 마지막 문장은 "이 점을 논거로 활용하면 ~" 형식으로 AI의 전략적 제안으로 마무리
- 200자 이내, 2~3문장, 번호·불릿 없이 바로 본문만 출력"""

        raw = call_ollama(system,
            f"논리 피드백:\n{logic_feedback}\n\n"
            f"전체 토론 기록 (중복 금지):\n{history_block}\n\n"
            f"Tavily 검색결과:\n{results_block}"
        )
        return raw.strip()

    # ── 3차: news_data.json 미언급 추가 사례 ─────────────────────

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
- 200자 이내,  번호·불릿 없이 바로 본문만 출력
- 확인되지 않은 수치 금지"""

        raw = call_ollama(system,
            f"주제: {topic}\n\n"
            f"전체 토론 기록 (중복 금지):\n{history_block}\n\n"
            f"기본 증거 (news_data.json - 새 정보 출처):\n{json_block}"
        )
        return raw.strip(), raw