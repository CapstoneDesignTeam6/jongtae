"""
agents/quiz_agent.py — 토론 후 복습 퀴즈 (4지선다 3개)

ReviewQuizAgent: 토론에서 실제 오간 논증·반론·인과관계를 얼마나 이해했는지 측정
  - 사전 퀴즈(intro_quiz_agent)와 동일한 구조, 동일한 파싱/정규화 재사용
  - 유형 3가지: argument_core / logic_gap / rebuttal_target
  - 토론 history + evidence 기반 → 배경지식이 아닌 "이 토론에서 오간 내용" 기반 출제

응답 형식:
    {
        "quizzes": [
            {
                "quiz_type": "argument_core",
                "type": "reasoning",
                "question": "...",
                "choices": ["...다.", "...다.", "...다.", "...다."],
                "correct_index": 2,
                "explanation": "..."
            },
            ...
        ],
        "selected_types": ["argument_core", "logic_gap", "rebuttal_target"]
    }

평가 응답 형식:
    {
        "results": [...],
        "total_score": 0~3,
        "detail": {"argument_core": True, "logic_gap": False, ...}
    }
"""

import json
import re

from data.evidence import build_evidence_block, build_history_block

import os
os.environ.setdefault("TAVILY_API_KEY", "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")

# 신뢰할 수 없는 출처 도메인 목록
_UNTRUSTED_DOMAINS = {
    "linkedin.com", "reddit.com", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "tiktok.com", "youtube.com",
    "quora.com", "pinterest.com", "tumblr.com", "medium.com",
}

def _tavily_search(query: str, max_results: int = 5, max_chars: int = 2000) -> str:
    """Tavily로 검색 후 신뢰 출처만 필터링해 제목+출처+내용 블록 반환."""
    try:
        from langchain_tavily import TavilySearch
        import re as _re
        result = TavilySearch(max_results=max_results).invoke(query)
        if isinstance(result, str):
            return result[:max_chars]
        items = result if isinstance(result, list) else result.get("results", [])
        blocks = []
        for r in items:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "").strip()
            # 도메인 추출
            domain_match = _re.search(r"https?://(?:www\.)?([^/]+)", url)
            domain = domain_match.group(1) if domain_match else ""
            # 비신뢰 도메인 스킵
            if any(d in domain for d in _UNTRUSTED_DOMAINS):
                continue
            title = r.get("title", "").strip()
            body  = (r.get("content") or r.get("full_content") or "").strip()[:500]
            if not body:
                continue
            blocks.append(f"[출처: {domain}]\n제목: {title}\n내용: {body}")
            if len(blocks) >= 3:
                break
        return "\n\n".join(blocks)[:max_chars]
    except Exception as e:
        print(f"    [Tavily] 검색 실패: {e}")
        return ""

GENERATE_TOKENS = 2000
MAX_RETRIES = 2


def _call_llm(prompt: str, max_tokens: int = GENERATE_TOKENS) -> str:
    try:
        from agents.llm import call_ollama
        return call_ollama(prompt, "", num_predict=max_tokens)
    except Exception as e:
        return f"[ERROR] Ollama: {e}"


# ──────────────────────────────────────────────────────────────────
# 퀴즈 유형 정의
# ──────────────────────────────────────────────────────────────────
# 사전 퀴즈: 배경지식 추론 (주제를 얼마나 아는가)
# 사후 퀴즈: 토론 이해도 (이 토론에서 오간 논증을 얼마나 이해했는가)

QUIZ_TYPES: dict[str, dict] = {
    "argument_core": {
        "name": "논거의 숨은 전제 추론",
        "measure": "토론에서 제시된 논거가 성립하려면 반드시 참이어야 하는 숨은 전제를 찾는 능력",
        "correct_criteria": (
            "이 논거가 설득력을 가지려면 반드시 받아들여야 하는 숨은 전제. "
            "이 전제가 흔들리면 논거 전체가 무너진다. "
            "토론에서 명시적으로 언급되지 않았지만 논증 구조 안에 내재된 가정이어야 한다."
        ),
        "distractor_criteria": (
            "각 오답은 다음 중 하나여야 한다:\n"
            "- 논거를 강화하는 추가 근거이지, 논거가 의존하는 전제가 아닌 것\n"
            "- 논거와 관련 있어 보이지만 이 논증 구조와 다른 차원의 가정\n"
            "- 전제처럼 들리지만 논거가 무너져도 여전히 성립하는 독립적 사실\n"
            "모든 오답은 '이것도 전제 아닌가?' 싶을 만큼 그럴듯해야 한다. "
            "토론 내용을 단순 기억하면 풀리는 문제가 되면 안 된다 — 반드시 추론이 필요해야 한다."
        ),
    },
    "argument_flaw": {
        "name": "논증 약점 찾기",
        "measure": "토론에서 제시된 주장이나 반박에 어떤 논리적 문제가 있는지 찾는 능력",
        "correct_criteria": (
            "이 주장 또는 반박이 가진 가장 치명적인 논리적 약점. "
            "전제가 과도하게 단순화되었거나, 인과관계가 성립하지 않거나, "
            "반론이 실제로는 상대 주장을 강화하는 등 논증 구조 자체의 결함이어야 한다."
        ),
        "distractor_criteria": (
            "각 오답은 다음 중 하나여야 한다:\n"
            "- 약점처럼 들리지만 실제로는 주장의 강점이나 근거를 설명하는 것\n"
            "- 논리적 문제가 있긴 하지만 이 주장의 핵심을 흔들지 못하는 부차적 결함\n"
            "- 주제와 관련 있지만 이 특정 논증과 다른 차원의 문제를 지적하는 것\n"
            "모든 오답은 '이것도 약점 아닌가?' 싶을 만큼 그럴듯해야 한다."
        ),
    },
    "news_inference": {
        "name": "추가 정보 기반 심화 추론",
        "measure": "토론의 핵심 주장을 실제 뉴스/데이터와 연결해 더 깊이 사고하는 능력",
        "correct_criteria": (
            "토론에서 제기된 주장 또는 반박을 실제 뉴스/데이터와 연결했을 때 "
            "가장 타당하게 도출되는 추론이나 판단. "
            "토론 내용만으로는 알 수 없고, 추가 정보를 함께 고려해야만 정답이 보이는 것이어야 한다."
        ),
        "distractor_criteria": (
            "각 오답은 다음 중 하나여야 한다:\n"
            "- 토론 내용과는 맞지만 추가 정보와 충돌하는 추론\n"
            "- 추가 정보와는 맞지만 토론의 핵심 맥락을 무시한 추론\n"
            "- 그럴듯해 보이지만 토론 내용과 추가 정보 모두에서 근거가 부족한 추론\n"
            "모든 오답은 '이 정보를 보면 이게 맞는 것 같은데?' 싶어야 한다."
        ),
    },
}

DEFAULT_TYPES = list(QUIZ_TYPES.keys())


# ──────────────────────────────────────────────────────────────────
# ReviewQuizAgent
# ──────────────────────────────────────────────────────────────────

class ReviewQuizAgent:
    """토론 후 복습용 4지선다 퀴즈 3개."""

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        user_label: str = "찬성",
        ai_label: str = "반대",
    ):
        self.evidence = evidence_items or []
        self.user_label = user_label
        self.ai_label = ai_label

    def generate(
        self,
        history: list[dict],
        topic: str,
        types: list[str] | None = None,
        search_block: str = "",
    ) -> dict | None:
        """
        Args:
            search_block: IntroAgent가 생성한 뉴스 검색 블록 (news_inference 유형에 사용)
        Returns:
            {"quizzes": [...], "selected_types": [...]}
        """
        selected = types if types is not None else list(DEFAULT_TYPES)
        _validate_types(selected)

        history_block = build_history_block(history)
        evidence_block = build_evidence_block(self.evidence, max_chars=2000)

        print(f"\n[ReviewQuizAgent] 주제: {topic}")
        print(f"  유형: {[QUIZ_TYPES[t]['name'] for t in selected]}")

        quizzes = []
        for qtype in selected:
            quiz = self._make_one(topic, history_block, evidence_block, qtype, search_block)
            if quiz:
                quizzes.append(quiz)
                print(f"  [{QUIZ_TYPES[qtype]['name']}] 완료")
            else:
                print(f"  [{QUIZ_TYPES[qtype]['name']}] 실패 - 건너뜀")

        print(f"  최종 퀴즈: {len(quizzes)}개")
        return {"quizzes": quizzes, "selected_types": selected} if quizzes else None

    def evaluate(self, quizzes: list[dict], user_answers: list[int]) -> dict:
        results = []
        for quiz, user_idx in zip(quizzes, user_answers):
            correct_idx = quiz.get("correct_index", -1)
            is_correct = user_idx == correct_idx
            results.append({
                "quiz_type": quiz.get("quiz_type"),
                "question": quiz.get("question"),
                "user_index": user_idx,
                "correct_index": correct_idx,
                "correct": is_correct,
                "explanation": quiz.get("explanation"),
            })
            mark = "✓" if is_correct else "✗"
            name = QUIZ_TYPES.get(quiz.get("quiz_type", ""), {}).get("name", "?")
            print(f"  [{name}] {mark}  유저:{user_idx} / 정답:{correct_idx}")

        return {
            "results": results,
            "total_score": sum(1 for r in results if r["correct"]),
            "detail": {r["quiz_type"]: r["correct"] for r in results},
        }

    # ── 문항 1개 생성 ──────────────────────────────────────────────

    def _make_one(
        self,
        topic: str,
        history_block: str,
        evidence_block: str,
        qtype: str,
        search_block: str = "",
    ) -> dict | None:
        meta = QUIZ_TYPES[qtype]

        # news_inference는 별도 3단계 흐름
        if qtype == "news_inference":
            return self._make_news_inference(topic, history_block, meta, self.user_label, self.ai_label)

        for attempt in range(1, MAX_RETRIES + 2):
            prompt = _build_prompt(
                topic, history_block, evidence_block,
                qtype, meta, self.user_label, self.ai_label,
                search_block=search_block,
            )
            raw = _call_llm(prompt)

            if not raw or "[ERROR]" in raw:
                print(f"    [{meta['name']}] 시도 {attempt} LLM 오류")
                continue

            quiz = _parse_single_quiz(raw, qtype)
            if quiz:
                return quiz

            print(f"    [{meta['name']}] 시도 {attempt} 파싱 실패, 재시도")

        return None

    def _make_news_inference(
        self, topic: str, history_block: str, meta: dict,
        user_label: str = "유저", ai_label: str = "AI"
    ) -> dict | None:
        """
        news_inference 전용 3단계:
          Step 1: 토론에서 특정 발화 하나 + 미검증 수치/전제 추출 + 검색 쿼리
          Step 2: Tavily 검색 → 3~4줄 한국어 요약
          Step 3: 그 발화 + 요약 데이터로만 판단하는 퀴즈 생성
        """
        # ── Step 1: 특정 발화 + 미검증 부분 + 검색 쿼리 ───────────
        step1_prompt = "\n".join([
            "아래 토론 기록을 읽고, 다음 순서로 분석해라.",
            "",
            "목표: 토론에서 발화 하나를 골라, 그 주장의 미검증 수치나 전제를 찾는다.",
            "",
            "우선순위:",
            "  1순위) 수치/근거 없이 주장만 한 발화 (유저 또는 AI 모두 해당)",
            "        예) '상대방은 ~에 훨씬 더 의존한다' → 실제 수치 없음",
            "  2순위) 수치는 없지만 결론에 결정적인 전제를 품고 있는 발화",
            "        예) '미국은 에너지 자립도가 높아졌다' → 실제 자립도 수준 불명확",
            "",
            "선택한 발화를 그대로 인용하고, 그 발화에서 검색으로 확인할 수 있는",
            "미검증 부분을 한 줄로 특정해라.",
            "검색 쿼리는 영어, 5단어 이내, 수치가 나올 법한 구체적인 쿼리로.",
            "",
            "[토론 기록]",
            history_block,
            "",
            "출력 형식 (다른 말 없이 다섯 줄만):",
            f"화자: [{user_label} 또는 {ai_label}]",
            "맥락: [이 발화가 나온 상황을 한 줄로 — 무엇에 대한 응답인지, 어떤 주장을 뒷받침하는 발화인지]",
            "선택발화: [발화 내용 그대로 인용]",
            "미검증부분: [이 발화에서 수치/사실이 없는 부분 한 줄]",
            "검색쿼리: [영어 검색 쿼리]",
        ])
        step1_raw = _call_llm(step1_prompt, max_tokens=400)
        print(f"    [Step1] {step1_raw[:100]}")

        # 파싱
        speaker = selected_claim = context_of_claim = gap = query = ""
        for line in step1_raw.splitlines():
            if line.startswith("화자:"):
                speaker = line.replace("화자:", "").strip()
            elif line.startswith("맥락:"):
                context_of_claim = line.replace("맥락:", "").strip()
            elif line.startswith("선택발화:"):
                selected_claim = line.replace("선택발화:", "").strip()
            elif line.startswith("미검증부분:"):
                gap = line.replace("미검증부분:", "").strip()
            elif line.startswith("검색쿼리:"):
                query = line.replace("검색쿼리:", "").strip()

        if not speaker:
            speaker = "한 참여자"
        if not query:
            query = topic[:50]

        # ── Step 2: Tavily 검색 → 관련성 확인 → 요약 ────────────
        # 쿼리가 너무 광범위하면 발화와 무관한 결과가 나오므로 재시도 포함
        context_summary = ""
        for search_attempt in range(1, 3):
            raw_search = _tavily_search(query)
            if not raw_search:
                print(f"    [Step2] 검색 결과 없음 (시도 {search_attempt})")
                break

            # 검색 결과가 발화와 관련 있는지 LLM으로 확인
            relevance_prompt = "\n".join([
                "아래 검색 결과가 주어진 발화의 미검증 부분을 보완하는 데 유용한 수치나 사실을 담고 있는가?",
                "유용하면 YES, 무관하거나 발화와 다른 주제면 NO로만 답해라.",
                "",
                f"[발화] {selected_claim}",
                f"[미검증 부분] {gap}",
                "",
                "[검색 결과 요약]",
                raw_search[:600],
                "",
                "판단 (YES 또는 NO):",
            ])
            relevance = _call_llm(relevance_prompt, max_tokens=10).strip().upper()
            print(f"    [Step2] 관련성 판단: {relevance} (쿼리: {query})")

            if "NO" in relevance:
                # 쿼리를 발화 핵심어 기반으로 재생성
                requery_prompt = "\n".join([
                    "아래 발화의 미검증 부분을 직접 검색할 수 있는 영어 쿼리를 1개만 만들어라.",
                    "쿼리는 구체적인 수치나 통계를 찾을 수 있어야 한다. 5단어 이내.",
                    "쿼리만 출력. 다른 말 없이.",
                    "",
                    f"[발화] {selected_claim}",
                    f"[미검증 부분] {gap}",
                ])
                query = _call_llm(requery_prompt, max_tokens=30).strip().strip('"').strip("'")
                print(f"    [Step2] 쿼리 재생성: {query}")
                continue

            # 관련성 있음 → 요약
            sum_prompt = "\n".join([
                "아래 검색 결과에서 발화의 미검증 부분을 보완하는 사실/수치만 골라",
                "한국어로 4~5줄 이내로 요약해라. 수치와 출처를 충분히 담을 것.",
                "",
                "규칙:",
                "- 말투: '~다', '~이다' 체. ~어요/~예요 금지",
                "- 각 검색 결과의 [출처] 도메인과 제목을 활용해 출처를 표기할 것",
                "  표기 형식: '출처명에 따르면 ~다' 또는 '~이다(출처명)'",
                "  URL 경로(예: /history, /news)를 출처로 쓰지 말 것 — 도메인 또는 기관명만 사용",
                "- 출처가 불분명한 수치는 쓰지 말 것",
                "- 불릿 없이 자연스러운 문장으로",
                "- 마지막에 결론·판단 문장 금지 ('따라서 ~', '결국 ~' 금지)",
                "- 상반된 수치나 사실이 있으면 그대로 병렬 제시할 것",
                "",
                f"[맥락] {context_of_claim}",
                f"[발화] {selected_claim}",
                f"[미검증 부분] {gap}",
                "",
                "[검색 결과]",
                raw_search,
                "",
                "요약:",
            ])
            context_summary = _call_llm(sum_prompt, max_tokens=500).strip()
            lines = [l.strip() for l in context_summary.splitlines() if l.strip()]
            context_summary = " ".join(lines)[:600]
            break

        if not context_summary:
            context_summary = "관련 추가 정보를 찾지 못했습니다."

        print(f"    [Step2] 요약: {context_summary[:80]}")

        # ── Step 3: 선택 발화 + 추가 정보 기반 퀴즈 생성 ──────────
        for attempt in range(1, MAX_RETRIES + 2):
            quiz_prompt = "\n".join([
                "당신은 토론 이해도 평가 전문가입니다.",
                "아래 토론의 특정 발화와 추가 정보를 함께 사용해 4지선다 퀴즈를 만드세요.",
                "",
                f"토론 주제: {topic}",
                "",
                "[토론에서 선택된 발화]",
                selected_claim,
                "",
                "[이 발화의 미검증 부분을 보완하는 추가 정보]",
                context_summary,
                "",
                "[출제 규칙]",
                "- 질문은 반드시 한 문장으로 작성할 것",
                f"- 포함 요소: [{context_of_claim}]라는 맥락에서 {speaker}가 [{selected_claim}]라고 주장했을 때, 추가 정보를 바탕으로 이 주장을 평가한 것으로 가장 적절한 것은?",
                "- 위 내용을 자연스러운 한 문장으로 압축할 것. 줄바꿈·번호 금지",
                "- 화자 표기: 반드시 '유저' 또는 'AI'로만. '찬성 측', '반대 측' 절대 금지",
                "",
                "[선지 설계 원칙]",
                "- 선지는 추가 정보를 그대로 확인하는 것이 아닌 추론·판단을 요구해야 함",
                "  추가 정보만 읽으면 바로 풀리는 선지는 실패",
                "  추가 정보 + 토론 맥락을 함께 고려해야 정답이 보여야 함",
                "",
                "- 선지 설계 핵심 원칙:",
                "  모든 선지(정답 포함)는 추가 정보에서 제시된 사실/상황을 토론 맥락에 적용해 판단하는 구조여야 한다",
                "  추가 정보를 단순히 요약하거나 수치를 나열하는 선지는 실패",
                "  '추가 정보가 이 상황에서 의미하는 바가 무엇인가'를 판단해야 풀리는 선지여야 한다",
                "",
                "- 정답: 추가 정보의 사실/상황을 토론의 핵심 발화와 연결했을 때 가장 타당하게 도출되는 판단",
                "  한 문장, 간결하게",
                "",
                "- 오답 설계 (각각 완전히 다른 논리 오류, 모두 그럴듯해야 함):",
                "  (A) 추가 정보의 사실은 맞게 이해했지만 발화의 맥락과 연결이 어긋난 판단",
                "  (B) 추가 정보가 시사하는 방향을 반대로 적용한 판단",
                "  (C) 추가 정보와 발화 모두 부분적으로만 고려해 전체를 왜곡한 판단",
                "  오답도 '추가 정보를 보면 이게 맞는 것 같은데?' 싶어야 함. 쉽게 걸리면 실패",
                "",
                "[형식]",
                "- 선지: ...다. 로 끝나는 평서형",
                "- correct_index: 0부터 시작, 매번 다르게 섞을 것",
                "- explanation: 정확히 4문장. (1)(2)(3)(4) 기호 사용.",
                "  각 문장에서 추가 정보의 수치를 언급하며 왜 맞고 틀린지 설명.",
                "",
                "JSON만 출력. 백틱/마크다운 금지.",
                "{",
                '  "quiz_type": "news_inference",',
                '  "type": "reasoning",',
                '  "question": "질문?",',
                '  "choices": ["선지A다.", "선지B다.", "선지C다.", "선지D다."],',
                '  "correct_index": 0,',
                '  "explanation": "(1)은 ... (2)는 ... (3)은 ... (4)는 ..."',
                "}",
            ])

            raw = _call_llm(quiz_prompt)
            if not raw or "[ERROR]" in raw:
                continue

            quiz = _parse_single_quiz(raw, "news_inference")
            if quiz:
                quiz["context_summary"] = context_summary  # 화면 표시용
                quiz["gap_point"] = gap
                return quiz

            print(f"    [news_inference] 시도 {attempt} 파싱 실패, 재시도")

        return None


# ──────────────────────────────────────────────────────────────────
# 프롬프트 빌더
# ──────────────────────────────────────────────────────────────────

def _build_prompt(
    topic: str,
    history_block: str,
    evidence_block: str,
    qtype: str,
    meta: dict,
    user_label: str,
    ai_label: str,
    search_block: str = "",
) -> str:
    # news_inference 유형: 뉴스 블록을 핵심 입력으로 사용
    is_news = qtype == "news_inference"

    if is_news:
        news_section = [
            "[실제 뉴스/데이터 - 퀴즈 출제의 핵심 근거]",
            search_block if search_block else "(뉴스 데이터 없음 — 토론 내용만으로 출제)",
            "",
            "[news_inference 유형 출제 규칙]",
            "- 토론에서 제기된 주장 하나를 골라, 위 뉴스/데이터와 연결해서 질문을 만들어라",
            "- 토론 내용만 알아도 풀리면 안 된다 — 뉴스 정보까지 함께 고려해야 정답이 보여야 한다",
            "- 질문은 '이 정보를 보면 ~에 대해 어떻게 판단해야 하는가?' 형태가 좋다",
            "",
        ]
    else:
        news_section = []

    lines = [
        "당신은 토론 이해도 평가 전문가입니다.",
        "아래 토론 기록을 꼼꼼히 읽고, 지정된 유형의 4지선다 퀴즈를 하나 만드세요.",
        "",
        "토론 주제: " + topic,
        "토론 참여자: 유저=" + user_label + " / AI=" + ai_label,
        "★ 발화자 표기 규칙 (질문·선지·해설 모두 적용):",
        "  반드시 '유저' 또는 'AI'로만 표기. '찬성 측', '반대 측', '찬성', '반대' 표현 절대 금지",
        "",
        "[토론 기록]",
        history_block,
        "",
        "[참고 자료 - 보조용]",
        evidence_block,
        "",
    ] + news_section + [
        "퀴즈 유형: " + meta["name"] + " - " + meta["measure"],
        "",
        "[질문 작성 규칙]",
        "- 토론에서 실제로 오간 주장/반론/근거를 기반으로 한 질문",
        "- 한 문장, 명확하게",
        "",
        "[정답 기준]",
        meta["correct_criteria"],
        "",
        "[오답 설계 - 가장 중요]",
        meta["distractor_criteria"],
        "",
        "[핵심 요구사항]",
        "- 4개 선지 모두 이 토론을 들은 사람에게 그럴듯하게 들려야 한다",
        "- 정답은 추론 끝에 납득되어야 하고, 오답은 '이것도 맞는 것 같은데?' 싶어야 한다",
        "",
        "[형식]",
        "- 모든 선지: ...다. 또는 ...된다. 로 끝나는 평서형",
        "- correct_index: choices 배열 0부터 시작 (정답 위치를 매번 다르게 섞을 것)",
        "- explanation: 정확히 4문장. (1)(2)(3)(4) 기호 사용.",
        "  정답 문장 예시: (1)은 토론 내용과 뉴스 데이터를 함께 고려할 때 가장 타당한 추론이므로 정답이다.",
        "  오답 문장 예시: (2)는 그럴듯하지만 뉴스 데이터와 충돌하므로 오답이다.",
        "",
        "JSON만 출력. 백틱/마크다운 금지. 다른 말 금지.",
        "",
        "{",
        '  "quiz_type": "' + qtype + '",',
        '  "type": "reasoning",',
        '  "question": "질문 텍스트?",',
        '  "choices": ["선지A다.", "선지B다.", "선지C다.", "선지D다."],',
        '  "correct_index": 0,',
        '  "explanation": "(1)은 ... (2)는 ... (3)은 ... (4)는 ..."',
        "}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# 파싱 & 정규화 (intro_quiz_agent와 동일 로직)
# ──────────────────────────────────────────────────────────────────

def _validate_types(types: list[str]) -> None:
    invalid = [t for t in types if t not in QUIZ_TYPES]
    if invalid:
        raise ValueError(f"알 수 없는 유형: {invalid}. 가능: {list(QUIZ_TYPES)}")


def _parse_single_quiz(raw: str, qtype: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    obj = _find_balanced_object(cleaned) or _find_balanced_object(raw)
    if not obj:
        return None
    try:
        item = json.loads(obj)
    except Exception:
        return None
    if not isinstance(item, dict):
        return None

    choices = None
    for key in ("choices", "options", "answers"):
        val = item.get(key)
        if isinstance(val, list) and len(val) == 4:
            choices = [str(c).strip() for c in val]
            break
    if not choices:
        return None

    question = item.get("question", "").strip()
    explanation = item.get("explanation", "").strip()
    if not question or not explanation:
        return None

    correct_idx = _parse_correct_index(item, choices)
    if correct_idx is None:
        return None

    correct_idx = _sync_correct_index(correct_idx, explanation)
    choices = _normalize_choices(choices)

    return {
        "quiz_type": qtype,
        "type": "reasoning",
        "question": question,
        "choices": choices,
        "correct_index": correct_idx,
        "explanation": explanation,
    }


def _parse_correct_index(item: dict, choices: list[str]) -> int | None:
    raw = item.get("correct_index", item.get("answer_index"))
    try:
        idx = int(raw)
        if 0 <= idx <= 3:
            return idx
    except (TypeError, ValueError):
        pass
    ans_text = item.get("correct_answer") or item.get("answer")
    if isinstance(ans_text, str):
        for i, c in enumerate(choices):
            if c.strip() == ans_text.strip():
                return i
    return None


def _sync_correct_index(correct_idx: int, explanation: str) -> int:
    label_map = {"①": 0, "②": 1, "③": 2, "④": 3}
    hits = re.findall(r"([①②③④])[^。.]*?정답|정답[^。.]*?([①②③④])", explanation)
    indices = set()
    for a, b in hits:
        label = a or b
        if label in label_map:
            indices.add(label_map[label])
    if not indices:
        paren_map = {"(1)": 0, "(2)": 1, "(3)": 2, "(4)": 3}
        hits2 = re.findall(r"(\(\d\))[^.]*?정답|정답[^.]*?(\(\d\))", explanation)
        for a, b in hits2:
            label = a or b
            if label in paren_map:
                indices.add(paren_map[label])
    if len(indices) == 1:
        synced = next(iter(indices))
        if synced != correct_idx:
            print(f"    [보정] correct_index {correct_idx} -> {synced}")
        return synced
    return correct_idx


def _normalize_choices(choices: list[str]) -> list[str]:
    result = []
    for c in choices:
        c = c.strip().rstrip(".")
        if c.endswith("?"):
            result.append(c)
        elif c.endswith("다"):
            result.append(c + ".")
        else:
            result.append(c + "다.")
    return result


def _find_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = in_str = escape = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            escape = (ch == "\\") if not escape else 0
            if not escape and ch == '"':
                in_str = 0
        elif ch == '"':
            in_str = 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# 하위 호환
QUIZ_TYPE_META = QUIZ_TYPES
ALL_TYPES = list(QUIZ_TYPES.keys())
