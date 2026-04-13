"""
debate_assistant.py — 외부 import 통합 진입점
"""

from agents.hint_agent import HintAgent
from agents.summary_agent import SummaryAgent
from agents.quiz_agent import ReviewQuizAgent, WeaknessQuizAgent


class DebateAssistant:
    """
    토론 보조 메인 클래스.

    Parameters
    ----------
    evidence_items : list[dict]
        AI 발언 직후 팀원이 넘겨준 최신 뉴스 데이터.
    user_label : str
        유저 입장 레이블 (예: "이란측", "상승 필요")
    ai_label : str
        AI 입장 레이블 (예: "미국측", "하락 필요")
    """

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        user_label: str = "찬성",
        ai_label: str = "반대",
    ):
        evidence = evidence_items or []
        self._hint          = HintAgent(evidence, user_label, ai_label)
        self._summary       = SummaryAgent(evidence, user_label, ai_label)
        self._review_quiz   = ReviewQuizAgent(evidence, user_label, ai_label)
        self._weakness_quiz = WeaknessQuizAgent(evidence, user_label, ai_label)

    def counter_hint(self, history: list[dict], topic: str) -> dict:
        """재반박 힌트 (AI 반박 직후). 반환: { hint, raw_response }"""
        return self._hint.counter_hint(history, topic)

    def rebuttal_hint(self, history: list[dict], topic: str) -> dict:
        """반박 힌트 (AI 새 주장 직후). 반환: { hint, raw_response }"""
        return self._hint.rebuttal_hint(history, topic)

    def summarize(self, history: list[dict], topic: str) -> dict:
        """토론 종료 후 정리. 라운드 수는 history에서 자동 계산."""
        return self._summary.summarize(history, topic)

    def quiz(self, history: list[dict], topic: str) -> dict:
        """
        퀴즈 생성.
        반환: { review_quiz, weakness_quiz }
        """
        review   = self._review_quiz.generate(history, topic)
        weakness = self._weakness_quiz.generate(history, topic)
        return {
            "review_quiz":   review,
            "weakness_quiz": weakness,
        }