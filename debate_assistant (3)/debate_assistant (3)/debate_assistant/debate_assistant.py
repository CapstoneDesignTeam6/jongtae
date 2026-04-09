"""
debate_assistant.py — 외부 import 통합 진입점

사용 예시:
    from debate_assistant import DebateAssistant
    from data.evidence import load_news_data_json

    evidence = load_news_data_json("news_data.json")
    da = DebateAssistant(evidence_items=evidence, stance=1)

    intro    = da.intro_summary(topic)
    counter  = da.counter_hint(history, topic)
    rebuttal = da.rebuttal_hint(history, topic)
    result   = da.summarize(history, topic, turns=1)
"""

from data.evidence import load_news_data_json
from agents.intro_agent import IntroAgent
from agents.hint_agent import HintAgent
from agents.summary_agent import SummaryAgent


class DebateAssistant:
    """
    토론 보조 메인 클래스.

    Parameters
    ----------
    evidence_items : list[dict]
        load_news_data_json()으로 로드한 news_data.json 아이템 목록.
    stance : int
        1 = 유저 찬성 / -1 = 유저 반대
    """

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        stance: int = 1,
    ):
        evidence = evidence_items or []
        self._intro   = IntroAgent(evidence, stance)
        self._hint    = HintAgent(evidence, stance)
        self._summary = SummaryAgent(evidence, stance)

    def intro_summary(self, topic: str) -> dict:
        """토론 시작 전 주제 배경 요약. 반환: { summary, raw_response }"""
        return self._intro.intro_summary(topic)

    def counter_hint(self, history: list[dict], topic: str) -> dict:
        """재반박 힌트 (4→5 사이). 반환: { hint, raw_response }"""
        return self._hint.counter_hint(history, topic)

    def rebuttal_hint(self, history: list[dict], topic: str) -> dict:
        """반박 힌트 (6→7 사이). 반환: { hint, raw_response }"""
        return self._hint.rebuttal_hint(history, topic)

    def summarize(self, history: list[dict], topic: str, turns: int = 1) -> dict:
        """
        토론 종료 후 정리. 반환:
        { summary, issues, logic_feedback, extra_info, raw_summary, raw_feedback }
        """
        return self._summary.summarize(history, topic, turns)
