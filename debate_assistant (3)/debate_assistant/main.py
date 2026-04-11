"""
main.py — 로컬 테스트 실행 진입점
"""

from data.evidence import load_news_data_json
from debate_assistant import DebateAssistant


def main():
    topic      = "이란 vs 미국 전쟁, 누가 더 손해인가?"
    user_label = "이란"   # 유저: 이란이 더 손해
    ai_label   = "미국"   # AI:   미국이 더 손해
    turns      = 1

    evidence = load_news_data_json("news_data.json")
    history  = []

    da = DebateAssistant(
        evidence_items=evidence, user_label=user_label, ai_label=ai_label
    )

    # ── 1단계: AI 첫 주장 (미국측: 미국이 더 손해) ───────────────
    history.append({"role": "ai", "content": (
        "이란과 미국의 전쟁에서 더 큰 손해를 보는 쪽은 미국입니다. "
        "전쟁이 발생하면 글로벌 금융 시장이 불안정해지고, 미국 증시와 달러 가치가 하락할 가능성이 큽니다. "
        "또한 중동 지역 불안정으로 에너지 가격이 급등하면 미국 내 물가 상승과 소비 위축이 발생해 "
        "실질적으로 미국 국민 전체가 경제적 부담을 떠안게 됩니다. "
        "이는 단순한 군사 비용을 넘어 사회 전반에 걸친 장기적 손실로 이어집니다."
    )})
    #
    # # # ── ★ 반박 힌트 (AI 첫 주장 직후) ───────────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 반박 힌트 (AI 첫 주장 직후)")
    # print("=" * 62)
    # rebuttal1 = da.rebuttal_hint(history, topic)
    # print(rebuttal1["hint"])

    # ── 유저 반박 (이란측: 이란이 더 손해) ───────────────────────
    history.append({"role": "user", "content": (
        "하지만 이란이 입는 사회적·인권적 피해가 훨씬 심각합니다. "
        "전쟁이 발생하면 이란 내 민간인 피해와 난민 문제가 급증하고, "
        "기반 시설 붕괴로 의료·교육 시스템이 마비될 가능성이 큽니다. "
        "이미 제재로 인해 생필품 부족을 겪고 있는 상황에서 전쟁까지 겹치면 "
        "이란 국민들의 삶은 회복 불가능한 수준으로 악화될 것입니다."
    )})

    # ── AI 재반박 (미국측) ────────────────────────────────────────
    history.append({"role": "ai", "content": (
        "이란 내부의 인도적 위기는 심각하지만, 미국 역시 사회적 비용을 무시할 수 없습니다. "
        "전쟁이 장기화될 경우 미국 내 반전 여론이 확산되고 사회적 갈등이 심화될 수 있습니다. "
        "또한 군 복무 확대와 국방비 증가로 인해 복지 예산이 축소되면서 "
        "저소득층과 중산층의 삶의 질이 하락할 가능성이 큽니다. "
        "결국 전쟁은 미국 사회 내부의 불평등과 갈등을 더욱 심화시키는 결과를 초래합니다."
    )})

    # # # ── ★ 재반박 힌트 (AI 재반박 직후) ──────────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 재반박 힌트 (AI 재반박 직후)")
    # print("=" * 62)
    # counter = da.counter_hint(history, topic)
    # print(counter["hint"])

    # ── 유저 재반박 (이란측) ─────────────────────────────────────
    history.append({"role": "user", "content": (
        "그래도 미국은 제도적으로 복구할 수 있는 능력이 있지만, "
        "이란은 그렇지 않습니다. 전쟁으로 인해 정치적 억압이 더 강화되고 "
        "표현의 자유와 기본권이 더욱 제한될 가능성이 큽니다. "
        "즉, 이란은 경제뿐 아니라 인권 측면에서도 구조적인 붕괴를 겪게 됩니다."
    )})

    # ── AI 새 주장 (미국측) ───────────────────────────────────────
    history.append({"role": "ai", "content": (
        "이란의 인권 문제는 분명 심각하지만, 미국 역시 국제적 신뢰를 잃는다는 점에서 큰 손해를 봅니다. "
        "전쟁이 장기화될 경우 미국은 국제사회에서 군사 개입 국가라는 이미지가 강화되고 "
        "동맹국과의 관계에서도 균열이 발생할 수 있습니다. "
        "이는 단순한 군사 충돌을 넘어 외교적 영향력 감소와 글로벌 리더십 약화로 이어지며 "
        "장기적으로 미국의 국제적 지위를 약화시키는 결과를 초래합니다."
    )})

    # # # ── ★ 반박 힌트 (AI 새 주장 직후) ───────────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 반박 힌트 (AI 새 주장 직후)")
    # print("=" * 62)
    # rebuttal2 = da.rebuttal_hint(history, topic)
    # print(rebuttal2["hint"])

    # ── 토론 정리 ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ★ 토론 정리 + 피드백")
    print("=" * 62)
    result = da.summarize(history, topic, turns=turns)

    print("── 토론 요약 ──")
    print(result["summary"])
    if result["logic_feedback"]:
        print("\n── 논리 피드백 ──")
        print(result["logic_feedback"])
    if result["extra_info"]:
        print("\n── 추가 사례·정보 ──")
        print(result["extra_info"])

    # # ── 퀴즈 ─────────────────────────────────────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 퀴즈")
    # print("=" * 62)
    # quiz_result = da.quiz(history, topic)
    #
    # # 복습 퀴즈
    # rq = quiz_result.get("review_quiz")
    # if rq:
    #     print("\n── 복습 퀴즈 ──")
    #     print(f"Q. {rq['question']}\n")
    #     for i, opt in enumerate(rq["options"], 1):
    #         print(f"  {i}. {opt}")
    #     print(f"\n정답: {rq['answer']}번")
    #     print(f"해설: {rq['explanation']}")
    #     print("\n── 보기별 설명 ──")
    #     for i, opt in enumerate(rq["options"], 1):
    #         label = "✅ 정답" if i == rq["answer"] else "❌ 오답"
    #         print(f"  {i}. [{label}] {opt}")
    #         print(f"     → {rq['option_explanations'][str(i)]}")
    # else:
    #     print("\n복습 퀴즈 생성 실패")
    #
    # # 약점 퀴즈
    # wq = quiz_result.get("weakness_quiz")
    # if wq:
    #     print("\n── 약점 퀴즈 ──")
    #     print(f"Q. {wq['question']}\n")
    #     for i, opt in enumerate(wq["options"], 1):
    #         print(f"  {i}. {opt}")
    #     print(f"\n정답: {wq['answer']}번")
    #     print(f"해설: {wq['explanation']}")
    #     print("\n── 보기별 설명 ──")
    #     for i, opt in enumerate(wq["options"], 1):
    #         label = "✅ 정답" if i == wq["answer"] else "❌ 오답"
    #         print(f"  {i}. [{label}] {opt}")
    #         print(f"     → {wq['option_explanations'][str(i)]}")
    # else:
    #     print("\n약점 퀴즈 생성 실패")


if __name__ == "__main__":
    main()