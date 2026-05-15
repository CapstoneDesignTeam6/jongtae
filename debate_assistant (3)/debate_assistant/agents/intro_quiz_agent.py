"""
agents/intro_quiz_agent.py — 토론 시작 전 배경 요약 기반 퀴즈 생성 + 주관식 평가

역할:
  - OX 문제 3개 생성
  - 주관식 문제 2개 생성 (요약 내용에 대한 본인 생각을 묻는 방식)
  - 주관식 답변을 받아 1~5점으로 평가 (이해도 + 자신의 해석)
  - /intro/quiz, /intro/quiz/evaluate 엔드포인트에서 호출됨

퀴즈 응답 형식:
    {
        "quizzes": [
            {
                "type": "ox",
                "question": "...",
                "answer": true,
                "explanation": "..."
            },
            {
                "type": "subjective",
                "question": "...",
                "context": "이 질문이 왜 중요한지 한 줄 안내"
            },
            ...
        ]
    }

주관식 평가 응답 형식:
    {
        "results": [
            {
                "question": "...",
                "answer":   "유저 답변",
                "score":    1~5,
                "reason":   "점수 이유 1문장"
            },
            ...
        ],
        "total_score": 2~10  ← 나중에 토론 후 퀴즈 점수와 비교용
    }
"""

import re
import json

from agents.llm import call_ollama


class IntroQuizAgent:

    def __init__(self):
        pass

    # ── 공개 API ──────────────────────────────────────────────────

    def run(self, topic: str, summary: str) -> dict:
        """
        OX 3개 + 주관식 2개 생성.

        Args:
            topic:   토론 주제
            summary: IntroAgent.run()의 summary 결과

        Returns:
            {"quizzes": [...]}
        """
        print(f"\n[IntroQuizAgent] 주제: {topic}")

        ox_quizzes        = self._generate_ox(topic, summary)
        subjective_quizzes = self._generate_subjective(topic, summary)

        quizzes = ox_quizzes[:3] + subjective_quizzes[:2]
        print(f"  └─ 최종 퀴즈: OX {len(ox_quizzes[:3])}개 + 주관식 {len(subjective_quizzes[:2])}개")
        return {"quizzes": quizzes}

    def evaluate_subjective(
        self,
        topic: str,
        summary: str,
        qa_pairs: list[dict],
    ) -> dict:
        """
        주관식 답변 평가.

        Args:
            topic:    토론 주제
            summary:  IntroAgent가 생성한 배경 요약
            qa_pairs: [{"question": "...", "answer": "유저 답변"}, ...]

        Returns:
            {
                "results": [
                    {"question": "...", "answer": "...", "score": 1~5, "reason": "..."},
                    ...
                ],
                "total_score": 2~10
            }
        """
        print(f"\n[IntroQuizAgent] 주관식 평가 시작 ({len(qa_pairs)}개)")
        results = []

        for i, qa in enumerate(qa_pairs, 1):
            question = qa.get("question", "")
            answer   = qa.get("answer", "")
            if not question or not answer:
                continue

            scored = self._evaluate_one(
                topic=topic,
                summary=summary,
                question=question,
                answer=answer,
            )
            results.append(scored)
            print(f"  └─ [{i}번] 점수: {scored['score']}/5")

        total_score = sum(r["score"] for r in results)
        return {
            "results":     results,
            "total_score": total_score,
        }

    # ── OX 생성 ──────────────────────────────────────────────────

    def _generate_ox(self, topic: str, summary: str) -> list[dict]:
        prompt = f"""Output ONLY a valid JSON array. No markdown, no explanation.

Debate topic: {topic}

Background summary (Korean):
{summary}

Create exactly 3 true/false (OX) quiz questions in Korean based on the summary.

Rules:
- Clearly answerable as true or false from the summary
- Mix true and false (not all same)
- "~요", "~어요" speech style
- No question numbers in the question field

Output format:
[
  {{"type": "ox", "question": "...", "answer": true, "explanation": "..."}},
  {{"type": "ox", "question": "...", "answer": false, "explanation": "..."}},
  {{"type": "ox", "question": "...", "answer": true, "explanation": "..."}}
]

Output:"""

        raw = call_ollama(prompt, "")
        print(f"  └─ [ox raw] {raw[:150]}")
        return self._parse_array(raw, "ox", expected=3)

    # ── 주관식 생성 ───────────────────────────────────────────────

    def _generate_subjective(self, topic: str, summary: str) -> list[dict]:
        prompt = f"""당신은 시사 토론 진행자입니다.
아래 배경 요약을 읽고, 참가자가 토론 전에 자신의 생각을 정리할 수 있도록
주관식 질문 2개를 만드세요.

[토론 주제]
{topic}

[배경 요약]
{summary}

[질문 조건]
- 단순 사실 확인이 아닌, 요약 내용을 바탕으로 "본인은 어떻게 생각하는지"를 묻는 질문
- 요약에 나온 핵심 쟁점이나 수치를 언급하며 질문할 것
- 찬반 어느 쪽으로도 답할 수 있는 열린 질문
- "~요", "~어요", "~죠" 체 사용
- context: 이 질문이 왜 중요한지 한 줄로 (유저에게 방향 안내용)

출력 형식 (JSON 배열만, 다른 텍스트 없이):
[
  {{"type": "subjective", "question": "...", "context": "..."}},
  {{"type": "subjective", "question": "...", "context": "..."}}
]

출력:"""

        raw = call_ollama(prompt, "")
        print(f"  └─ [subjective raw] {raw[:150]}")
        return self._parse_array(raw, "subjective", expected=2)

    # ── 주관식 단일 평가 ─────────────────────────────────────────

    def _evaluate_one(
        self,
        topic: str,
        summary: str,
        question: str,
        answer: str,
    ) -> dict:
        prompt = f"""당신은 시사 토론 평가자입니다.
아래 주관식 답변을 1~5점으로 평가하세요.

[토론 주제]
{topic}

[배경 요약]
{summary}

[질문]
{question}

[유저 답변]
{answer}

━━━ 평가 기준 (1~5점) ━━━
1점: 질문을 이해하지 못했거나 무관한 답변
2점: 질문은 이해했지만 요약 내용과 무관하게 단순 감상만
3점: 요약 내용을 어느 정도 이해하고 본인 생각을 표현했으나 근거 없음
4점: 요약 내용을 이해하고 구체적인 근거나 사례를 들어 본인 해석을 표현
5점: 요약 내용을 정확히 이해하고, 자신만의 시각으로 깊이 있는 해석을 제시

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

    # ── JSON 파싱 유틸 ────────────────────────────────────────────

    def _parse_array(self, raw: str, quiz_type: str, expected: int) -> list[dict]:
        parsed = self._extract_json_array(raw)
        if not parsed:
            print(f"  └─ [{quiz_type}] JSON 추출 실패")
            return []

        result = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            if quiz_type == "ox":
                if not all(k in item for k in ("question", "answer", "explanation")):
                    continue
                ans = item["answer"]
                if isinstance(ans, str):
                    item["answer"] = ans.lower() in ("true", "yes", "o")
                elif isinstance(ans, int):
                    item["answer"] = bool(ans)
                elif not isinstance(ans, bool):
                    continue

            elif quiz_type == "subjective":
                if not all(k in item for k in ("question", "context")):
                    continue

            item["type"] = quiz_type
            result.append(item)

        print(f"  └─ [{quiz_type}] 파싱 완료: {len(result)}/{expected}개")
        return result

    def _extract_json_array(self, raw: str) -> list | None:
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
                parsed = json.loads(raw[start:end + 1])
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        return None