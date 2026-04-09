"""
agents/intro_agent.py — 토론 시작 전 주제 요약 생성

토론 전 유저에게 주제 배경·양측 논거·핵심 사실을 중립적으로 안내.
news_data.json 기본 증거 활용.
"""

from data.evidence import build_evidence_block
from agents.llm import call_ollama


class IntroAgent:
    """
    토론 시작 전 주제 요약 에이전트.

    사용:
        from agents.intro_agent import IntroAgent

        agent = IntroAgent(evidence_items=evidence, stance=1)
        result = agent.intro_summary(topic)
        print(result["summary"])
    """

    def __init__(
        self,
        evidence_items: list[dict] | None = None,
        stance: int = 1,
    ):
        self.evidence   = evidence_items or []
        self.user_label = "찬성" if stance == 1 else "반대"
        self.ai_label   = "반대" if stance == 1 else "찬성"

    def intro_summary(self, topic: str) -> dict:
        """
        토론 시작 전 주제 배경 요약.

        반환: { "summary", "raw_response" }
        """
        # 증거를 최대한 많이 넘겨 LLM이 풍부한 사례를 뽑을 수 있도록 함
        json_block = build_evidence_block(self.evidence, max_chars=10000)

        system = f"""당신은 시사 토론 진행자입니다. 토론 전 유저에게 주제를 중립적으로 안내합니다.
유저='{self.user_label}' 입장, AI='{self.ai_label}' 입장으로 토론 예정.

작성 규칙:
- 자연스러운 문장 (번호·불릿 금지). 4개 문단, 각 2~4문장, 문단 사이 빈 줄
- 편향 없이 양측 시각 균형 있게 소개
- 증거자료의 구체적 수치·사건명·기관명 반드시 포함
- 모든 수치·통계·사건·자료는 반드시 신뢰 가능한 출처를 함께 명시하고,
  문장 내에 "~에 따르면" 형식으로 자연스럽게 포함하라
- 총 800자 이내. 제목·번호·구분선 없이 바로 본문만 출력

1문단: 주제의 역사적·정치적 배경과 현재 상황 (언제, 어디서, 왜 이 충돌이 발생했는지)
2문단: 찬반 양측이 내세우는 핵심 논거 각 2가지씩, 구체적 근거(수치·기관명) 포함
3문단: 증거자료에서 뽑은 핵심 팩트 3~4가지 — 출처와 함께, 토론에서 바로 쓸 수 있는 수준으로
4문단: 이 토론을 이해하는 데 꼭 필요한 배경 개념을 쉽게 풀어서 설명.
       단순 용어 나열이 아니라 "왜 이게 중요한지"를 포함해 2~3가지 개념을 자연스러운 문장으로 설명.
       예시: 호르무즈 해협이 글로벌 에너지 공급에서 차지하는 비중,
             IRGC가 단순 군대가 아닌 이유, 헬륨이 반도체·AI에 왜 필수인지 등
       증거자료에 있는 내용 기반으로 작성하되, 유저가 토론 중 바로 활용할 수 있도록 구체적으로."""

        raw = call_ollama(system,
            f"토론 주제: {topic}\n\n"
            f"━━ 기본 증거 (news_data.json) ━━\n{json_block}"
        )
        return {"summary": raw, "raw_response": raw}
