"""
agents/quiz_agent.py — 토론 내용 기반 퀴즈 생성 + 평가

ReviewQuizAgent  : 토론 내용 복습용 OX 3개
WeaknessQuizAgent: 유저 약점 기반 주관식 2개 + 평가

퀴즈 응답 형식:
    {
        "quizzes": [
            {"type": "ox", "question": "...", "answer": true, "explanation": "..."},
            ...
        ]
    }
    또는
    {
        "quizzes": [
            {"type": "subjective", "question": "...", "context": "..."},
            ...
        ]
    }

주관식 평가 응답 형식:
    {
        "results": [
            {"question": "...", "answer": "유저 답변", "score": 1~5, "reason": "점수 이유 1문장"},
            ...
        ],
        "total_score": 2~10
    }
"""

import re
import json

from data.evidence import build_evidence_block, build_history_block
from agents.llm import call_ollama


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def _extract_json_array(raw: str) -> list | None:
    """LLM 출력에서 JSON 배열 파싱 (마크다운 펜스 포함 대응)."""
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return None


def _parse_ox(raw: str) -> list[dict]:
    """OX 퀴즈 JSON 배열 파싱 및 검증."""
    items = _extract_json_array(raw)
    if not items:
        print("  └─ [ox] JSON 추출 실패")
        return []

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("question", "answer", "explanation")):
            continue

        ans = item["answer"]
        if isinstance(ans, str):
            item["answer"] = ans.lower() in ("true", "yes", "o")
        elif isinstance(ans, int):
            item["answer"] = bool(ans)
        elif not isinstance(ans, bool):
            continue

        item["type"] = "ox"
        result.append(item)

    print(f"  └─ [ox] 파싱 완료: {len(result)}/3개")
    return result


def _parse_subjective(raw: str) -> list[dict]:
    """주관식 퀴즈 JSON 배열 파싱 및 검증."""
    items = _extract_json_array(raw)
    if not items:
        print("  └─ [subjective] JSON 추출 실패")
        return []

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("question", "context")):
            continue
        item["type"] = "subjective"
        result.append(item)

    print(f"  └─ [subjective] 파싱 완료: {len(result)}/2개")
    return result


# ── ReviewQuizAgent: OX 3개 ──────────────────────────────────────────────────

class ReviewQuizAgent:
    """토론 내용 복습용 OX 퀴즈 3개."""

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        user_label: str = "찬성",
        ai_label: str = "반대",
    ):
        self.evidence   = evidence_items or []
        self.user_label = user_label
        self.ai_label   = ai_label

    def generate(self, history: list[dict], topic: str) -> dict | None:
        """
        Returns:
            {"quizzes": [{"type": "ox", "question": ..., "answer": bool, "explanation": ...}, ...]}
        """
        print(f"\n[ReviewQuizAgent] 주제: {topic}")

        history_block = build_history_block(history)
        json_block    = build_evidence_block(self.evidence, max_chars=2500)

        system = """당신은 시사 토론 퀴즈 출제자입니다.
아래 토론 기록을 바탕으로 핵심 사실·인과관계를 확인하는 OX 문제 3개를 만드세요.

[좋은 문제의 조건]
- 토론에서 실제로 다룬 구체적인 사실·수치·인과관계를 정확히 이해했는지 묻는 문제
- 참(O)과 거짓(X)이 토론 내용만으로 명확히 판단되어야 함
- 참·거짓을 고루 섞을 것 (전부 같은 답 금지)
- 거짓 문장은 "완전히 엉뚱한 것"이 아니라 "그럴듯하지만 핵심이 빗나간 것"으로 만들 것

[금지 사항]
- 토론 기록에 없는 내용으로 출제 금지
- 뉴스 자료에만 의존하는 문제 금지 — 토론 내용 중심으로 출제

[언어 규칙]
- 중학생도 이해할 수 있는 쉽고 친근한 말투 ("~요", "~어요" 체)
- 어려운 용어는 괄호 안에 짧게 풀어서 설명

출력 규칙:
- 반드시 아래 JSON 배열만 출력. 설명·마크다운 금지.
- answer: true(O) 또는 false(X)
- explanation: 왜 O인지 또는 왜 X인지 (1~2문장)

JSON 형식:
[
  {"question": "문제", "answer": true, "explanation": "해설"},
  {"question": "문제", "answer": false, "explanation": "해설"},
  {"question": "문제", "answer": true, "explanation": "해설"}
]"""

        raw = call_ollama(
            system,
            f"토론 주제: {topic}\n"
            f"유저={self.user_label}, AI={self.ai_label}\n\n"
            f"토론 기록:\n{history_block}\n\n"
            f"참고 자료 (news_data, 보조용):\n{json_block}",
        )

        print(f"  └─ [ox raw] {raw[:150]}")
        quizzes = _parse_ox(raw)[:3]
        return {"quizzes": quizzes} if quizzes else None


# ── WeaknessQuizAgent: 주관식 2개 + 평가 ────────────────────────────────────

class WeaknessQuizAgent:
    """유저 약점 기반 주관식 퀴즈 2개 생성 + 답변 평가."""

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        user_label: str = "찬성",
        ai_label: str = "반대",
    ):
        self.evidence   = evidence_items or []
        self.user_label = user_label
        self.ai_label   = ai_label

    def generate(self, history: list[dict], topic: str) -> dict | None:
        """
        Returns:
            {"quizzes": [{"type": "subjective", "question": ..., "context": ...}, ...]}
        """
        print(f"\n[WeaknessQuizAgent] 주제: {topic}")

        history_block = build_history_block(history)
        json_block    = build_evidence_block(self.evidence, max_chars=2500)

        # 1단계: 유저 약점 분석
        weak_concept = self._analyze_weakness(history_block, topic)
        if not weak_concept:
            return None

        # 2단계: 약점 기반 주관식 생성
        quizzes = self._generate_subjective(
            history_block, json_block, topic, weak_concept
        )
        return {"quizzes": quizzes} if quizzes else None

    def evaluate(
        self,
        topic: str,
        history: list[dict],
        qa_pairs: list[dict],
    ) -> dict:
        """
        주관식 답변 평가.

        Args:
            topic:    토론 주제
            history:  토론 기록 (배경 맥락용)
            qa_pairs: [{"question": "...", "answer": "유저 답변"}, ...]

        Returns:
            {
                "results": [
                    {"question": ..., "answer": ..., "score": 1~5, "reason": ...},
                    ...
                ],
                "total_score": 2~10
            }
        """
        print(f"\n[WeaknessQuizAgent] 주관식 평가 시작 ({len(qa_pairs)}개)")

        history_block = build_history_block(history)
        results = []

        for i, qa in enumerate(qa_pairs, 1):
            question = qa.get("question", "")
            answer   = qa.get("answer", "")
            if not question or not answer:
                continue

            scored = self._evaluate_one(topic, history_block, question, answer)
            results.append(scored)
            print(f"  └─ [{i}번] 점수: {scored['score']}/5")

        total_score = sum(r["score"] for r in results)
        return {"results": results, "total_score": total_score}

    # ── 내부 메서드 ──────────────────────────────────────────────

    def _analyze_weakness(self, history_block: str, topic: str) -> str:
        """유저 발언에서 이해가 부족한 개념·사실 1가지를 특정한다."""
        system = f"""당신은 시사 토론 분석가입니다.
유저({self.user_label})의 발언을 분석해서, 아래 조건에 맞는 약점 1가지를 찾아내세요.

[찾아야 할 약점]
- 유저가 주장은 했지만 실제 배경지식이 부족해 보이는 개념·사실·수치
- 인과관계를 잘못 연결하거나, 수치 없이 막연하게 주장한 부분
- 상대방 주장의 핵심을 제대로 이해하지 못하고 피상적으로 반응한 부분
- 토론 스킬(말하는 방법)이 아닌, 내용(사실·개념·수치) 자체의 이해 부족

[출력 규칙]
- 약점이 되는 개념·사실을 한 단어 또는 짧은 구로 먼저 명시 (예: "이란 경제제재의 실제 영향")
- 왜 그게 약점인지 2문장으로 설명
- 번호·불릿 없이 바로 본문만 출력"""

        raw = call_ollama(
            system,
            f"토론 주제: {topic}\n\n토론 기록:\n{history_block}",
        )
        return raw.strip()

    def _generate_subjective(
        self,
        history_block: str,
        json_block: str,
        topic: str,
        weak_concept: str,
    ) -> list[dict]:
        """약점으로 특정된 개념·사실을 정면으로 묻는 주관식 2개 생성."""
        system = """당신은 시사 토론 퀴즈 출제자입니다.
유저가 제대로 이해하지 못한 개념·사실을 정면으로 묻는 주관식 질문 2개를 만드세요.

[좋은 문제의 조건]
- "그 개념이 실제로 어떻게 작동하는가?", "왜 그런 결과가 나오는가?"를 묻는 열린 질문
- 찬반 어느 쪽으로도 답할 수 있는 구조
- 단순 사실 확인이 아닌, 토론 내용을 바탕으로 본인의 해석을 이끌어내는 질문
- context: 이 질문이 왜 중요한지 한 줄로 (유저에게 방향 안내용)

[금지 사항]
- "어떻게 반박할까?", "가장 논리적인 대응은?" 같은 토론 전략 문제 절대 금지
- 단답형으로 끝나는 사실 확인 문제 금지

[언어 규칙]
- 중학생도 이해할 수 있는 쉽고 친근한 말투 ("~요", "~어요", "~죠" 체)
- 어려운 용어는 괄호 안에 짧게 풀어서 설명

출력 규칙:
- 반드시 아래 JSON 배열만 출력. 설명·마크다운 금지.

JSON 형식:
[
  {"question": "문제", "context": "이 질문이 중요한 이유"},
  {"question": "문제", "context": "이 질문이 중요한 이유"}
]"""

        raw = call_ollama(
            system,
            f"토론 주제: {topic}\n\n"
            f"유저가 제대로 이해하지 못한 개념·사실:\n{weak_concept}\n\n"
            f"토론 기록 (참고용):\n{history_block}\n\n"
            f"참고 자료 (news_data, 보조용):\n{json_block}",
        )

        print(f"  └─ [subjective raw] {raw[:150]}")
        return _parse_subjective(raw)[:2]

    def _evaluate_one(
        self,
        topic: str,
        history_block: str,
        question: str,
        answer: str,
    ) -> dict:
        """주관식 답변 1개를 1~5점으로 평가."""
        prompt = f"""당신은 시사 토론 평가자입니다.
아래 주관식 답변을 1~5점으로 평가하세요.

[토론 주제]
{topic}

[토론 기록 (맥락용)]
{history_block}

[질문]
{question}

[유저 답변]
{answer}

━━━ 평가 기준 (1~5점) ━━━
1점: 질문을 이해하지 못했거나 무관한 답변
2점: 질문은 이해했지만 토론 내용과 무관하게 단순 감상만
3점: 토론 내용을 어느 정도 이해하고 본인 생각을 표현했으나 근거 없음
4점: 토론 내용을 이해하고 구체적인 근거나 사례를 들어 본인 해석을 표현
5점: 토론 내용을 정확히 이해하고, 자신만의 시각으로 깊이 있는 해석을 제시

━━━ 출력 형식 ━━━
아래 JSON만 출력. 다른 텍스트 없이.

{{
  "question": "{question}",
  "answer":   "{answer}",
  "score":    1~5 정수,
  "reason":   "점수 이유 1문장"
}}"""

        raw = call_ollama(prompt, "")
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                score  = max(1, min(5, int(parsed.get("score", 1))))
                return {
                    "question": question,
                    "answer":   answer,
                    "score":    score,
                    "reason":   parsed.get("reason", ""),
                }
        except Exception as e:
            print(f"  └─ [evaluate_one] 파싱 실패: {e}")

        return {"question": question, "answer": answer, "score": 1, "reason": "평가 실패"}