"""
agents/hint_agent.py — 재반박 힌트 / 반박 힌트 생성

4→5 사이: AI 반박 후 유저 재반박 전
6→7 사이: AI 새 주장 후 유저 반박 전

데이터 흐름:
    news_data.json → LLM 1회 호출 → 힌트 텍스트 반환
"""

from data.evidence import build_evidence_block, extract_last_ai_claim
from agents.llm import call_ollama


class HintAgent:

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        stance: int = 1,
    ):
        self.evidence   = evidence_items or []
        self.user_label = "찬성" if stance == 1 else "반대"

    def counter_hint(self, history: list[dict], topic: str) -> dict:
        """재반박 힌트 (4→5 사이). 반환: { hint, raw_response }"""
        ai_claim   = extract_last_ai_claim(history)
        json_block = build_evidence_block(self.evidence, max_chars=2500 * 4)

        system = f"""당신은 시사 토론 자료 제공자입니다. 유저는 '{self.user_label}' 입장입니다.
AI가 방금 반박했습니다. 관련 사실만 전달하세요.

작성 규칙:
- 4~5문장의 자연스러운 문장
- 사실·데이터만 전달. 전략 제안 완전 금지
- 실제 사건명·기관명·출처·구체적 수치(금액·인원·연도) 포함
- 모든 수치·통계는 "~에 따르면" 형식으로 출처와 함께 명시
- 총 250자 이내. 제목·번호 없이 바로 본문만 출력
- 확인되지 않은 수치 사용 금지"""

        raw = call_ollama(system,
            f"주제: {topic}\nAI 반박: {ai_claim}\n\n"
            f"━━ 기본 증거 (news_data.json) ━━\n{json_block}"
        )
        return {"hint": raw, "raw_response": raw}

    def rebuttal_hint(self, history: list[dict], topic: str) -> dict:
        """반박 힌트 (6→7 사이). 반환: { hint, raw_response }"""
        ai_claim   = extract_last_ai_claim(history)
        json_block = build_evidence_block(self.evidence, max_chars=2500)

        system = f"""당신은 시사 토론 자료 제공자입니다. 유저는 '{self.user_label}' 입장입니다.
AI가 새로운 주장을 했습니다. 관련 사실만 전달하세요.

작성 규칙:
- 4~5문장의 자연스러운 문장
- 사실·데이터만 전달. 전략 제안 완전 금지
- 실제 사건명·기관명·출처·구체적 수치(금액·기간·인원) 포함
- 모든 수치·통계는 "~에 따르면" 형식으로 출처와 함께 명시
- 총 250자 이내. 제목·번호 없이 바로 본문만 출력
- 확인되지 않은 수치 사용 금지"""

        raw = call_ollama(system,
            f"주제: {topic}\nAI 주장: {ai_claim}\n\n"
            f"━━ 기본 증거 (news_data.json) ━━\n{json_block}"
        )
        return {"hint": raw, "raw_response": raw}
