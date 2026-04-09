"""
main.py — 실행 진입점

직접 실행: python main.py
"""

from data.evidence import load_news_data_json
from debate_assistant import DebateAssistant


def main():
    topic  = "이란 vs 미국 전쟁, 누가 더 손해인가?"
    stance = 1
    turns  = 1

    evidence = load_news_data_json("news_data.json")
    da = DebateAssistant(evidence_items=evidence, stance=stance)

    history = []

    print("\n" + "=" * 62)
    print("  ★ 초반 주제 요약 (0단계)")
    # print("=" * 62)
    # intro = da.intro_summary(topic)
    # print(intro["summary"])

    history.append({"role": "ai", "content": (
        "이 주제는 군사·경제·외교 등 다양한 측면에서 분석이 필요합니다. "
        "단순한 비용 비교를 넘어 여러 요소를 종합적으로 고려해야 합니다."
    )})
    history.append({"role": "user", "content": (
        "미국은 이라크 전쟁에서 약 2조 달러, 아프가니스탄에서도 막대한 재정 손실을 입었다. "
        "이란과 전쟁이 발생하면 장기전으로 이어질 가능성이 높아 미국이 더 큰 손해를 볼 것이다."
    )})
    history.append({"role": "ai", "content": (
        "이라크 사례를 그대로 적용하기는 무리가 있다. "
        "미국은 F-35 Lightning II 같은 스텔스 전투기와 정밀 유도 무기로 주요 시설을 빠르게 타격할 수 있다. "
        "GDP도 이란보다 훨씬 크기 때문에 전쟁 비용을 감당할 능력도 훨씬 크다."
    )})

    print("\n" + "=" * 62)
    print("  ★ 재반박 힌트 (4→5 사이)")
    print("=" * 62)
    # counter = da.counter_hint(history, topic)
    # print(counter["hint"])

    history.append({"role": "user", "content": (
        "이란은 혁명수비대(IRGC)를 포함한 비정규 전력을 활용할 수 있다. "
        "이라크 전쟁 이후 미군은 IED 공격으로 큰 피해를 입었고, 게릴라전 대응에 막대한 비용이 들었다."
    )})
    history.append({"role": "ai", "content": (
        "미국은 지상군 투입 없이 '공습 + 사이버전 + 제재' 중심 전략을 사용할 가능성이 크다. "
        "Operation Desert Storm에서 미국은 압도적인 공군력으로 단기간에 전쟁을 끝냈다."
    )})

    print("\n" + "=" * 62)
    print("  ★ 반박 힌트 (6→7 사이)")
    print("=" * 62)
    # rebuttal = da.rebuttal_hint(history, topic)
    # print(rebuttal["hint"])

    history.append({"role": "user", "content": (
        "Iraq War에서도 초기 군사 작전은 빠르게 성공했지만 점령 단계에서 막대한 비용이 발생했다. "
        "이란은 이라크보다 훨씬 큰 영토와 인구를 가져 전쟁이 더 복잡해질 가능성이 높다."
    )})
    history.append({"role": "ai", "content": (
        "이란이 지리적으로 방어에 유리한 요소도 있지만, "
        "미국은 위성 감시와 드론 기술로 이를 상당 부분 극복할 수 있다. "
        "동맹국들과 협력해 군사적 부담도 분산할 수 있다."
    )})

    print("\n" + "=" * 62)
    print("  ★ 토론 정리 + 피드백 (10단계)")
    print("=" * 62)
    result = da.summarize(history, topic, turns=turns)

    print("── 토론 요약 ──")
    print(result["summary"])
    if result["issues"]:
        print("\n── 쟁점 분석 ──")
        print(result["issues"])
    if result["logic_feedback"]:
        print("\n── 논리 피드백 ──")
        print(result["logic_feedback"])
    if result["extra_info"]:
        print("\n── 추가 사례 (피드백 기반 검색) ──")
        print(result["extra_info"])


if __name__ == "__main__":
    main()
