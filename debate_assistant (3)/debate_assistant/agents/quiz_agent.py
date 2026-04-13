"""
agents/quiz_agent.py — 복습용 퀴즈 + 약점 퀴즈 생성

ReviewQuizAgent  : 토론 내용 단순 복습용 4지선다
WeaknessQuizAgent: 유저 약점 기반 4지선다
"""

import re
import json
import random

from data.evidence import build_evidence_block, build_history_block
from agents.llm import call_ollama


def _shuffle_options(correct: str, distractors: list[str]) -> tuple[list[str], int]:
    """
    정답 + 오답 3개를 섞어서 무작위 배치.
    반환: (options 리스트, 정답 인덱스 1~4)
    """
    options = distractors[:3] + [correct]
    random.shuffle(options)
    answer = options.index(correct) + 1  # 1-based
    return options, answer


def _parse_quiz_json(raw: str) -> dict | None:
    """LLM 출력에서 JSON 파싱."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  └─ 퀴즈 JSON 파싱 실패: {e} | 원문: {raw[:200]}")
    return None


def _build_quiz_response(parsed: dict) -> dict | None:
    """
    파싱된 JSON → 최종 퀴즈 응답 형식.
    정답 위치 무작위 배치.
    """
    try:
        correct     = parsed["correct"]
        distractors = parsed["distractors"]  # 3개
        options, answer = _shuffle_options(correct, distractors)

        option_explanations = {}
        for i, opt in enumerate(options, 1):
            if i == answer:
                option_explanations[str(i)] = parsed["correct_explanation"]
            else:
                idx = distractors.index(opt) if opt in distractors else 0
                wrong_explanations = parsed.get("wrong_explanations", [])
                if idx < len(wrong_explanations):
                    option_explanations[str(i)] = wrong_explanations[idx]
                else:
                    option_explanations[str(i)] = parsed.get("wrong_explanation", "틀린 보기입니다.")

        return {
            "question":            parsed["question"],
            "options":             options,
            "answer":              answer,
            "explanation":         parsed["explanation"],
            "option_explanations": option_explanations,
        }

    except Exception as e:
        print(f"  └─ 퀴즈 응답 빌드 실패: {e}")
    return None


class ReviewQuizAgent:
    """토론 내용 단순 복습용 4지선다 퀴즈."""

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
        history_block = build_history_block(history)
        json_block    = build_evidence_block(self.evidence, max_chars=2500)

        system = """당신은 시사 토론 퀴즈 출제자입니다.
아래 토론 기록을 바탕으로 핵심 내용을 복습할 수 있는 4지선다 퀴즈 1개를 만드세요.

퀴즈의 목적은 토론에서 다룬 사실·수치·사건을 잘 이해하고 기억하는지 확인하는 것입니다.
문제와 보기, 해설은 중학생도 이해할 수 있을 만큼 쉽고 친근한 말투로 써주세요.
어려운 전문 용어는 괄호 안에 짧게 풀어서 설명해주세요.

출력 규칙:
- 반드시 아래 JSON 형식만 출력. 설명·마크다운 금지.
- correct      : 정답 텍스트 (1개)
- distractors  : 오답 텍스트 배열 (정확히 3개)
- wrong_explanations: 오답 3개에 대한 설명 배열 (distractors와 순서 동일, 정확히 3개)
- question     : 문제 (토론에서 실제 다룬 사실·수치·사건 기반, 쉬운 말로)
- explanation  : 정답 전체 해설 (2~3문장, 쉬운 말로)
- correct_explanation: 이 보기가 왜 정답인지 (1문장, 쉬운 말로)

JSON 형식:
{
  "question": "문제",
  "correct": "정답",
  "distractors": ["오답1", "오답2", "오답3"],
  "explanation": "정답 해설",
  "correct_explanation": "이 보기가 정답인 이유",
  "wrong_explanations": ["오답1이 틀린 이유", "오답2가 틀린 이유", "오답3이 틀린 이유"]
}"""

        raw = call_ollama(system,
            f"토론 주제: {topic}\n"
            f"유저={self.user_label}, AI={self.ai_label}\n\n"
            f"토론 기록:\n{history_block}\n\n"
            f"기본 증거 (news_data):\n{json_block}"
        )

        parsed = _parse_quiz_json(raw)
        if not parsed:
            return None
        return _build_quiz_response(parsed)


class WeaknessQuizAgent:
    """유저 약점 기반 4지선다 퀴즈."""

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
        history_block = build_history_block(history)
        json_block    = build_evidence_block(self.evidence, max_chars=2500)

        # 1단계: 유저 약점 분석
        weakness = self._analyze_weakness(history_block, topic)
        if not weakness:
            return None

        # 2단계: 약점 기반 퀴즈 생성
        return self._generate_quiz(history_block, json_block, topic, weakness)

    def _analyze_weakness(self, history_block: str, topic: str) -> str:
        """history 전체 분석 → 유저가 헷갈리거나 잘못 이해한 사실·개념 추출."""
        system = f"""당신은 시사 토론 분석가입니다.
유저({self.user_label})의 토론 발언 전체를 분석하여
가장 잘못 이해하거나 근거 없이 말한 사실·개념·수치 1가지를 찾아내세요.

출력 규칙:
- 2~3문장으로 구체적으로 서술
- 어떤 사실·개념·수치를 틀리게 알고 있거나 제대로 설명하지 못했는지 명시
- 토론 스킬(반박 방법 등)이 아닌 내용 자체의 이해 부족에 집중할 것
- 번호·불릿 없이 바로 본문만 출력"""

        raw = call_ollama(system,
            f"토론 주제: {topic}\n\n토론 기록:\n{history_block}"
        )
        return raw.strip()

    def _generate_quiz(
        self,
        history_block: str,
        json_block: str,
        topic: str,
        weakness: str,
    ) -> dict | None:
        """약점 기반 퀴즈 생성."""
        system = """당신은 시사 토론 퀴즈 출제자입니다.
유저가 잘못 이해하거나 헷갈려한 사실·개념을 제대로 알고 있는지 확인하는 4지선다 퀴즈 1개를 만드세요.

퀴즈의 목적은 토론 스킬 평가가 아니라, 토론 주제와 관련된 실제 사실·개념·수치의 이해도 확인입니다.
문제는 반드시 "~은 무엇인가?", "~의 결과는?", "~이 일어난 이유는?" 같이 사실·개념을 묻는 형식으로 만드세요.
"어떻게 반박할까?", "가장 논리적인 반박은?" 같은 토론 전략을 묻는 문제는 절대 만들지 마세요.
문제와 보기, 해설은 중학생도 이해할 수 있을 만큼 쉽고 친근한 말투로 써주세요.
어려운 전문 용어는 괄호 안에 짧게 풀어서 설명해주세요.

출력 규칙:
- 반드시 아래 JSON 형식만 출력. 설명·마크다운 금지.
- correct      : 정답 텍스트 (1개)
- distractors  : 오답 텍스트 배열 (정확히 3개)
- wrong_explanations: 오답 3개에 대한 설명 배열 (distractors와 순서 동일, 정확히 3개)
- question     : 문제 (유저가 헷갈린 사실·개념·수치를 직접 묻는 형식, 쉬운 말로)
- explanation  : 정답 전체 해설 (2~3문장, 쉬운 말로)
- correct_explanation: 이 보기가 왜 정답인지 (1문장, 쉬운 말로)

JSON 형식:
{
  "question": "문제",
  "correct": "정답",
  "distractors": ["오답1", "오답2", "오답3"],
  "explanation": "정답 해설",
  "correct_explanation": "이 보기가 정답인 이유",
  "wrong_explanations": ["오답1이 틀린 이유", "오답2가 틀린 이유", "오답3이 틀린 이유"]
}"""

        raw = call_ollama(system,
            f"토론 주제: {topic}\n\n"
            f"유저가 헷갈린 부분:\n{weakness}\n\n"
            f"토론 기록:\n{history_block}\n\n"
            f"기본 증거 (news_data):\n{json_block}"
        )

        parsed = _parse_quiz_json(raw)
        if not parsed:
            return None
        return _build_quiz_response(parsed)