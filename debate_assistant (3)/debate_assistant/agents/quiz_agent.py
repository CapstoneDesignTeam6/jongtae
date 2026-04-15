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
아래 토론 기록에서 핵심적인 인과관계 또는 사실 하나를 골라 4지선다 퀴즈 1개를 만드세요.

[좋은 문제의 조건]
- 토론에서 실제로 다룬 구체적인 인과관계·근거·수치를 정확히 이해했는지 묻는 문제
- "~이 일어나면 왜 ~가 되는가?", "~의 실제 원인은?", "~의 결과로 나타나는 것은?" 형식 권장
- 정답과 오답이 표면적으로 비슷하게 보여서 제대로 이해한 사람만 고를 수 있어야 함
- 오답은 "완전히 틀린 것"이 아니라 "그럴듯하지만 핵심이 빗나간 것"으로 만들 것
  예) 정답: "에너지 공급 차질 → 물가 상승 → 실질소득 감소"
      오답: "에너지 소비 감소 → 물가 하락" (방향이 반대라 헷갈림)
      오답: "금융시장 불안 → 달러 강세" (관련 있어 보이지만 반대 방향)
      오답: "국방비 증가 → 복지 예산 증가" (국방비와 복지는 연결되지만 방향이 틀림)

[금지 사항]
- "~에 미칠 수 있는 영향으로 알맞은 것은?" 같은 지나치게 넓은 질문 금지
- 보기만 봐도 바로 틀린 게 보이는 뻔한 오답 금지
- 뉴스 자료에만 의존하는 문제 금지 — 토론 내용 중심으로 출제할 것

[언어 규칙]
- 중학생도 이해할 수 있는 쉽고 친근한 말투
- 어려운 용어는 괄호 안에 짧게 풀어서 설명

출력 규칙:
- 반드시 아래 JSON 형식만 출력. 설명·마크다운 금지.
- correct             : 정답 텍스트 (1개)
- distractors         : 오답 텍스트 배열 (정확히 3개, 각각 그럴듯하게 헷갈리는 내용으로)
- wrong_explanations  : 오답 3개에 대한 설명 배열 (distractors와 순서 동일, 정확히 3개, 왜 헷갈렸는지 + 왜 틀렸는지)
- question            : 문제
- explanation         : 정답 전체 해설 (인과관계를 단계별로 설명, 2~3문장)
- correct_explanation : 이 보기가 왜 정답인지 (1문장)

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
            f"참고 자료 (news_data, 보조용):\n{json_block}"
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

        # 1단계: 유저가 제대로 설명하지 못한 개념·사실 특정
        weak_concept = self._analyze_weakness(history_block, topic)
        if not weak_concept:
            return None

        # 2단계: 그 개념·사실 자체를 정면으로 묻는 퀴즈 생성
        return self._generate_quiz(history_block, json_block, topic, weak_concept)

    def _analyze_weakness(self, history_block: str, topic: str) -> str:
        """
        유저 발언에서 근거 없이 주장하거나 인과관계를 잘못 설명한
        구체적인 개념·사실·수치 1가지를 특정한다.
        """
        system = f"""당신은 시사 토론 분석가입니다.
유저({self.user_label})의 발언을 분석해서, 아래 조건에 맞는 약점 1가지를 찾아내세요.

[찾아야 할 약점]
- 유저가 주장은 했지만 실제 배경지식이 부족해 보이는 개념·사실·수치
- 인과관계를 잘못 연결하거나, 수치 없이 막연하게 주장한 부분
- 상대방 주장의 핵심을 제대로 이해하지 못하고 피상적으로 반응한 부분
- 토론 스킬(말하는 방법)이 아닌, 내용(사실·개념·수치) 자체의 이해 부족

[출력 규칙]
- 약점이 되는 개념·사실을 한 단어 또는 짧은 구로 먼저 명시할 것 (예: "이란 경제제재의 실제 영향")
- 왜 그게 약점인지 2문장으로 설명
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
        weak_concept: str,
    ) -> dict | None:
        """약점으로 특정된 개념·사실을 정면으로 묻는 퀴즈 생성."""
        system = """당신은 시사 토론 퀴즈 출제자입니다.
유저가 제대로 이해하지 못한 개념·사실을 직접 묻는 4지선다 퀴즈 1개를 만드세요.

[좋은 문제의 조건]
- "그 개념이 실제로 어떻게 작동하는가?"를 정면으로 묻는 문제
- 복습 퀴즈와 소재·질문 방식이 겹치지 않을 것
- 정답은 그 개념의 핵심 메커니즘(작동 원리·실제 영향)을 담을 것
- 오답은 "완전히 틀린 것"이 아니라 "일부만 맞거나 방향이 반대인 것"으로 만들 것
  → 유저가 실제로 헷갈릴 법한 내용으로 구성할 것

[금지 사항]
- "어떻게 반박할까?", "가장 논리적인 대응은?" 같은 토론 전략 문제 절대 금지
- 보기만 봐도 바로 틀린 게 보이는 뻔한 오답 금지
- 복습 퀴즈와 같은 소재·같은 질문 구조 금지

[언어 규칙]
- 중학생도 이해할 수 있는 쉽고 친근한 말투
- 어려운 용어는 괄호 안에 짧게 풀어서 설명

출력 규칙:
- 반드시 아래 JSON 형식만 출력. 설명·마크다운 금지.
- correct             : 정답 텍스트 (1개)
- distractors         : 오답 텍스트 배열 (정확히 3개, 각각 그럴듯하게 헷갈리는 내용으로)
- wrong_explanations  : 오답 3개에 대한 설명 배열 (distractors와 순서 동일, 정확히 3개, 왜 헷갈렸는지 + 왜 틀렸는지)
- question            : 문제 (유저가 놓친 개념·사실을 정면으로 묻는 형식)
- explanation         : 정답 전체 해설 (개념의 실제 작동 원리를 단계별로, 2~3문장)
- correct_explanation : 이 보기가 왜 정답인지 (1문장)

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
            f"유저가 제대로 이해하지 못한 개념·사실:\n{weak_concept}\n\n"
            f"토론 기록 (참고용):\n{history_block}\n\n"
            f"참고 자료 (news_data, 보조용):\n{json_block}"
        )

        parsed = _parse_quiz_json(raw)
        if not parsed:
            return None
        return _build_quiz_response(parsed)