"""
agents/intro_agent.py — 토론 시작 전 주제 배경 정보 검색 + 요약

역할:
  - 토론 주제를 받아 배경 지식 제공
  - news_data(외부 뉴스)가 있으면 그것을 우선 사용 (Tavily 검색 스킵)
  - news_data가 없으면 Tavily로 관련 최신 정보를 검색
  - 검색/수신 결과를 LLM으로 요약해 참가자에게 배경 지식 제공
  - /intro 엔드포인트에서 호출됨

사용법:
    agent = IntroAgent()

    # 외부 뉴스 없을 때 (Tavily 검색)
    result = agent.run(topic="미국 vs 이란, 누가 더 손해인가?")

    # 외부 뉴스 있을 때 (news_data 우선)
    result = agent.run(topic="미국 vs 이란, 누가 더 손해인가?", news_data=[...])

    # result = {"summary": "...", "search_block": "..."}
"""

import re
import json
import os

from langchain_tavily import TavilySearch
from agents.llm import call_ollama

os.environ["TAVILY_API_KEY"] = "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"
_tavily = TavilySearch(max_results=3)


class IntroAgent:

    def __init__(self):
        pass

    # ── 공개 API ──────────────────────────────────────────────────

    def run(self, topic: str, news_data: list | None = None) -> dict:
        """
        Args:
            topic:     토론 주제 문자열
            news_data: 외부 서버에서 받은 뉴스 배열 (있으면 Tavily 검색 스킵)
                       각 항목은 {"title": "...", "content": "...", "url": "..."} 형태 권장

        Returns:
            {
                "summary":      "주제 배경 요약 텍스트",
                "search_block": "사용된 뉴스/검색 결과 원문 블록 (디버그용)"
            }
        """
        print(f"\n[IntroAgent] 주제: {topic}")

        if news_data:
            print(f"  └─ 외부 news_data {len(news_data)}건 사용 (Tavily 스킵)")
            search_block = self._build_block_from_news(news_data)
        else:
            print(f"  └─ news_data 없음 → Tavily 검색 시작")
            queries      = self._step1_queries(topic)
            search_block = self._step2_search(queries)

        summary = self._step3_summarize(topic, search_block)
        return {"summary": summary, "search_block": search_block}

    # ── news_data → search_block 변환 ────────────────────────────

    def _build_block_from_news(self, news_data: list) -> str:
        block = ""
        for i, item in enumerate(news_data[:8], 1):
            if not isinstance(item, dict):
                continue
            content = (
                item.get("content")
                or item.get("summary")
                or item.get("description")
                or ""
            )[:1500]
            if not content:
                continue
            block += (
                f"[뉴스{i}]\n"
                f"제목: {item.get('title', '')}\n"
                f"출처: {item.get('url', item.get('source', ''))}\n"
                f"내용: {content}\n\n"
            )

        result = block.strip() or "(뉴스 내용 없음)"
        print(f"  └─ [news_data 변환] {len(news_data)}건 → {len(result)}자")
        return result

    # ── 1단계: 검색 쿼리 생성 ────────────────────────────────────

    def _step1_queries(self, topic: str) -> list[str]:
        prompt = f"""Output ONLY a valid JSON array of 4 English search queries. No markdown, no explanation.

Debate topic: {topic}

Generate exactly 4 search queries to gather background information about this debate topic:
- Query 1: Core facts and recent news about the topic (2024, 2025, or 2026)
- Query 2: Key statistics or data related to the topic
- Query 3: Main arguments or positions from one side
- Query 4: Main arguments or positions from the other side

Rules:
- English only, no special characters
- Each query must be clearly different
- Prefer concrete, specific queries over vague ones

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
            print(f"  └─ [1단계] 쿼리 파싱 실패: {e}")

        if not queries:
            safe_topic = "".join(c for c in topic if ord(c) < 128).strip()
            queries = [safe_topic or "debate background"]
            print(f"  └─ [1단계] fallback 쿼리 사용")

        print(f"  └─ [1단계] 검색 쿼리: {queries}")
        return queries

    # ── 2단계: Tavily 검색 ───────────────────────────────────────

    def _step2_search(self, queries: list[str]) -> str:
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
            print("  └─ [2단계] 검색 결과 없음")
            return "(검색 결과 없음)"

        block = ""
        for i, r in enumerate(all_results[:8], 1):
            content = (r.get("full_content") or r.get("content") or "")[:1500]
            if not content:
                continue
            block += (
                f"[검색{i}]\n"
                f"쿼리: {r.get('query', '')}\n"
                f"제목: {r.get('title', '')}\n"
                f"출처: {r.get('url', '')}\n"
                f"내용: {content}\n\n"
            )

        result = block.strip() or "(검색 결과 없음)"
        print(f"  └─ [2단계] 검색 완료 ({len(all_results)}건)")
        return result

    # ── 3단계: 배경 요약 생성 ────────────────────────────────────

    def _step3_summarize(self, topic: str, search_block: str) -> str:
        prompt = f"""당신은 토론 진행자입니다. 토론 참가자들에게 주제에 대한 배경 지식을 제공해야 합니다.

[토론 주제]
{topic}

[참고 자료]
{search_block}

아래 형식에 맞춰 배경 정보를 작성하세요.

작성 규칙:
- 한국어로 작성
- 길이 제한 없음. 내용이 충분히 전달될 때까지 작성
- 어려운 전문용어 최소화, 중학생도 이해할 수 있는 수준
- "~요", "~어요", "~죠" 체 사용. "~다", "~습니다" 체 금지
- 번호·기호·헤더 없이 자연스러운 문단으로 작성
- 참고 자료에 없는 내용 지어내지 말 것

[절대 금지]
- "안녕하세요", "반갑습니다", "환영합니다" 같은 인사말로 시작하는 것
- "토론을 시작해 볼까요?", "함께 알아봐요" 같은 마무리·유도 문구로 끝내는 것
- 첫 문장은 반드시 배경 설명 내용으로 바로 시작할 것
- 마지막 문장은 반드시 배경 설명 내용으로 끝낼 것

작성 항목 (순서대로, 문단 구분 없이 이어서):
1. 이 토론 주제가 왜 지금 중요한지 (배경, 최근 상황)
2. 핵심 쟁점이 무엇인지 (양측이 다투는 핵심 포인트)
3. 각 입장의 주요 근거 (찬성/반대 또는 A측/B측)
4. 관련된 주요 사실이나 수치 (있을 경우)

출력:"""

        raw    = call_ollama(prompt, "").strip()
        lines  = [line.strip() for line in raw.splitlines() if line.strip()]
        result = " ".join(lines)

        print(f"  └─ [3단계] 요약 생성 완료 ({len(result)}자)")
        return result