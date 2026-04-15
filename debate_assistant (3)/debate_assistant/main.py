# """
# main.py — 로컬 테스트 실행 진입점
# """
#
# from data.evidence import load_news_data_json
# from debate_assistant import DebateAssistant
#
#
# def main():
#     topic      = "이란과 미국 전쟁,누가 더 손해인가?"
#     user_label = "이란"   # 유저: 이란이 더 손해
#     ai_label   = "미국"   # AI:   미국이 더 손해
#
#     # evidence = load_news_data_json("news_data.json")
#     evidence=[]
#     history  = []
#
#     da = DebateAssistant(
#         evidence_items=evidence, user_label=user_label, ai_label=ai_label
#     )
#
#     # ── 1단계: AI 첫 주장 (미국측: 미국이 더 손해) ───────────────
#     history.append({"role": "ai", "content": (
#         "이란과 미국의 전쟁에서 더 큰 손해를 보는 쪽은 미국입니다. "
#         "전쟁이 발생하면 글로벌 금융 시장이 불안정해지고, 미국 증시와 달러 가치가 하락할 가능성이 큽니다. "
#         "또한 중동 지역 불안정으로 에너지 가격이 급등하면 미국 내 물가 상승과 소비 위축이 발생해 "
#         "실질적으로 미국 국민 전체가 경제적 부담을 떠안게 됩니다. "
#         "이는 단순한 군사 비용을 넘어 사회 전반에 걸친 장기적 손실로 이어집니다."
#     )})
#     #
#     # # ── ★ 반박 힌트 (AI 첫 주장 직후) ───────────────────────────
#     # print("\n" + "=" * 62)
#     # print("  ★ 반박 힌트 (AI 첫 주장 직후)")
#     # print("=" * 62)
#     # rebuttal1 = da.rebuttal_hint(history, topic)
#     # print(rebuttal1["hint"])
#
#     # ── 유저 반박 (이란측: 이란이 더 손해) ───────────────────────
#     history.append({"role": "user", "content": (
#         "하지만 이란이 입는 사회적·인권적 피해가 훨씬 심각합니다. "
#         "전쟁이 발생하면 이란 내 민간인 피해와 난민 문제가 급증하고, "
#         "기반 시설 붕괴로 의료·교육 시스템이 마비될 가능성이 큽니다. "
#         "이미 제재로 인해 생필품 부족을 겪고 있는 상황에서 전쟁까지 겹치면 "
#         "이란 국민들의 삶은 회복 불가능한 수준으로 악화될 것입니다."
#     )})
#
#     # ── AI 재반박 (미국측) ────────────────────────────────────────
#     history.append({"role": "ai", "content": (
#         "이란 내부의 인도적 위기는 심각하지만, 미국 역시 사회적 비용을 무시할 수 없습니다. "
#         "전쟁이 장기화될 경우 미국 내 반전 여론이 확산되고 사회적 갈등이 심화될 수 있습니다. "
#         "또한 군 복무 확대와 국방비 증가로 인해 복지 예산이 축소되면서 "
#         "저소득층과 중산층의 삶의 질이 하락할 가능성이 큽니다. "
#         "결국 전쟁은 미국 사회 내부의 불평등과 갈등을 더욱 심화시키는 결과를 초래합니다."
#     )})
#
#     # # ── ★ 재반박 힌트 (AI 재반박 직후) ──────────────────────────
#     # print("\n" + "=" * 62)
#     # print("  ★ 재반박 힌트 (AI 재반박 직후)")
#     # print("=" * 62)
#     # counter = da.counter_hint(history, topic)
#     # print(counter["hint"])
#
#     # ── 유저 재반박 (이란측) ─────────────────────────────────────
#     history.append({"role": "user", "content": (
#         "뭔소리야"
#     )})
#
#     # ── AI 새 주장 (미국측) ───────────────────────────────────────
#     history.append({"role": "ai", "content": (
#         "이란의 인권 문제는 분명 심각하지만, 미국 역시 국제적 신뢰를 잃는다는 점에서 큰 손해를 봅니다. "
#         "전쟁이 장기화될 경우 미국은 국제사회에서 군사 개입 국가라는 이미지가 강화되고 "
#         "동맹국과의 관계에서도 균열이 발생할 수 있습니다. "
#         "이는 단순한 군사 충돌을 넘어 외교적 영향력 감소와 글로벌 리더십 약화로 이어지며 "
#         "장기적으로 미국의 국제적 지위를 약화시키는 결과를 초래합니다."
#     )})
#
#     # # # # ── ★ 반박 힌트 (AI 새 주장 직후) ───────────────────────────
#     # print("\n" + "=" * 62)
#     # print("  ★ 반박 힌트 (AI 새 주장 직후)")
#     # print("=" * 62)
#     # rebuttal2 = da.rebuttal_hint(history, topic)
#     # print(rebuttal2["hint"])
#     #
#     # # ── 토론 정리 ────────────────────────────────────────────────
#     print("\n" + "=" * 62)
#     print("  ★ 토론 정리 + 피드백")
#     print("=" * 62)
#     result = da.summarize(history, topic)
#
#     print("── 토론 요약 ──")
#     print(result["summary"])
#     if result["logic_feedback"]:
#         print("\n── 논리 피드백 ──")
#         print(result["logic_feedback"])
#     if result["extra_info"]:
#         print("\n── 추가 사례·정보 ──")
#         print(result["extra_info"])
#     #
#     # # ── 퀴즈 ─────────────────────────────────────────────────────
#     print("\n" + "=" * 62)
#     print("  ★ 퀴즈")
#     print("=" * 62)
#     quiz_result = da.quiz(history, topic)
#
#     # 복습 퀴즈
#     rq = quiz_result.get("review_quiz")
#     if rq:
#         print("\n── 복습 퀴즈 ──")
#         print(f"Q. {rq['question']}\n")
#         for i, opt in enumerate(rq["options"], 1):
#             print(f"  {i}. {opt}")
#         print(f"\n정답: {rq['answer']}번")
#         print(f"해설: {rq['explanation']}")
#         print("\n── 보기별 설명 ──")
#         for i, opt in enumerate(rq["options"], 1):
#             label = "✅ 정답" if i == rq["answer"] else "❌ 오답"
#             print(f"  {i}. [{label}] {opt}")
#             print(f"     → {rq['option_explanations'][str(i)]}")
#     else:
#         print("\n복습 퀴즈 생성 실패")
#
#     # 약점 퀴즈
#     wq = quiz_result.get("weakness_quiz")
#     if wq:
#         print("\n── 약점 퀴즈 ──")
#         print(f"Q. {wq['question']}\n")
#         for i, opt in enumerate(wq["options"], 1):
#             print(f"  {i}. {opt}")
#         print(f"\n정답: {wq['answer']}번")
#         print(f"해설: {wq['explanation']}")
#         print("\n── 보기별 설명 ──")
#         for i, opt in enumerate(wq["options"], 1):
#             label = "✅ 정답" if i == wq["answer"] else "❌ 오답"
#             print(f"  {i}. [{label}] {opt}")
#             print(f"     → {wq['option_explanations'][str(i)]}")
#     else:
#         print("\n약점 퀴즈 생성 실패")
#
#
# if __name__ == "__main__":
#     main()

"""
main.py — 로컬 테스트 실행 진입점
"""

from data.evidence import load_news_data_json
from debate_assistant import DebateAssistant


def main():
    topic      = "원격근무와 사무실 출근 중 어떤게 더 생산적인가?"
    user_label = "원격근무"
    ai_label   = "사무실 출근"

    evidence = []
    history  = []

    da = DebateAssistant(
        evidence_items=evidence, user_label=user_label, ai_label=ai_label
    )

    # ── 1단계: AI 첫 주장 (사무실측: 사무실 출근이 더 생산적) ───────────────
    history.append({"role": "ai", "content": (
        "사무실 출근이 원격근무보다 훨씬 더 생산적입니다. "
        "사무실 환경에서는 동료와의 즉각적인 소통이 가능해 의사결정 속도가 빨라지고, "
        "협업 과정에서 발생하는 창의적 아이디어의 교환이 활발하게 이루어집니다. "
        "실제로 Stanford 연구에 따르면 대면 환경에서 팀 프로젝트의 완성도와 속도가 "
        "원격 환경 대비 평균 20% 이상 높게 나타났습니다. "
        "또한 업무와 생활 공간을 물리적으로 분리함으로써 집중력이 유지되고 "
        "번아웃 없이 지속 가능한 업무 리듬을 형성할 수 있습니다."
    )})

    # # ── ★ 반박 힌트 1회차 (AI 첫 주장 직후) ───────────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 반박 힌트 1회차 (AI 첫 주장 직후)")
    # print("=" * 62)
    # rebuttal1 = da.rebuttal_hint(history, topic)
    # print(rebuttal1["hint"])

    # ── 유저 반박 (원격근무측) ───────────────────────────────────────────
    history.append({"role": "user", "content": (
        "하지만 원격근무는 출퇴근 시간을 절약해 실질적인 업무 시간이 늘어납니다. "
        "서울 기준 평균 왕복 출퇴근 시간은 약 1시간 30분으로, "
        "이 시간을 업무나 자기계발에 활용하면 생산성이 훨씬 높아집니다. "
        "또한 개인 맞춤형 업무 환경을 구성할 수 있어 "
        "집중이 필요한 심화 작업에서 원격근무자가 더 뛰어난 성과를 보이는 사례가 많습니다."
    )})

    # ── AI 재반박 (사무실측) ────────────────────────────────────────
    history.append({"role": "ai", "content": (
        "출퇴근 시간 절약은 분명한 장점이지만, 원격근무에는 보이지 않는 비용이 존재합니다. "
        "가정 내 소음, 가족 구성원의 방해, 불충분한 업무 장비 등 "
        "통제하기 어려운 변수들이 집중력을 분산시킵니다. "
        "Microsoft의 2023년 업무 트렌드 보고서에 따르면, "
        "원격근무자의 약 42%가 '디지털 피로'를 경험하며 "
        "회의와 메신저 응답에 쓰는 시간이 오히려 증가했다고 답했습니다. "
        "진정한 생산성이란 단순히 시간의 양이 아니라 결과물의 질로 판단해야 합니다."
    )})
    # #
    # # ── ★ 재반박 힌트 1회차 (AI 재반박 직후) ──────────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 재반박 힌트 1회차 (AI 재반박 직후)")
    # print("=" * 62)
    # counter1 = da.counter_hint(history, topic)
    # print(counter1["hint"])

    # ── 유저 재반박 (원격근무측) ─────────────────────────────────────
    history.append({"role": "user", "content": (
        "디지털 피로는 사무실에서도 발생합니다. "
        "오픈 오피스 환경의 소음과 잦은 대면 인터럽션이 "
        "집중 업무를 방해한다는 연구도 많습니다. "
        "중요한 건 환경이 아니라 개인의 업무 방식과 자율성이며, "
        "원격근무는 그 자율성을 극대화해 높은 몰입도를 이끌어냅니다."
    )})

    # ── AI 새 주장 (사무실측) ───────────────────────────────────────
    history.append({"role": "ai", "content": (
        "자율성은 중요하지만, 특히 주니어 직원이나 신입사원에게는 "
        "사무실 출근이 절대적으로 유리합니다. "
        "현장에서 선배의 업무 방식을 직접 관찰하고, "
        "즉각적인 피드백을 받으며 성장하는 속도는 원격 환경에서 재현하기 어렵습니다. "
        "실제로 글로벌 컨설팅 기업들이 2024년 이후 사무실 복귀 정책을 강화한 것은 "
        "조직 학습 속도의 저하를 막기 위한 조치입니다. "
        "장기적으로는 사무실 기반 문화가 조직 전체의 역량 축적에 더 효과적입니다."
    )})

    # ── ★ 반박 힌트 2회차 (AI 새 주장 직후) ───────────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 반박 힌트 2회차 (AI 새 주장 직후)")
    # print("=" * 62)
    # rebuttal2 = da.rebuttal_hint(history, topic)
    # print(rebuttal2["hint"])

    # ── 유저 반박 2회차 (원격근무측) ────────────────────────────────────
    history.append({"role": "user", "content": (
        "신입사원 육성은 중요하지만, 원격 환경에서도 충분히 가능합니다. "
        "체계적인 온보딩 프로그램과 화상 멘토링, 협업 툴을 활용하면 "
        "오히려 기록이 남아 더 체계적인 학습이 이루어집니다. "
        "실제로 GitLab, Automattic 같은 완전 원격 기업들은 "
        "수천 명 규모에서도 높은 조직 역량을 유지하고 있습니다."
    )})

    # ── AI 재반박 2회차 (사무실측) ───────────────────────────────────
    history.append({"role": "ai", "content": (
        "GitLab 같은 사례는 특수한 IT 업종에 국한된 이야기입니다. "
        "제조업, 의료, 서비스업 등 대다수 산업군에서는 "
        "물리적 현장이 필수적이며 원격근무 자체가 불가능합니다. "
        "또한 완전 원격 기업들도 연 1~2회 오프사이트 미팅에 막대한 비용을 투자하는데, "
        "이는 결국 대면 상호작용의 필요성을 스스로 인정하는 것입니다. "
        "보편적 근무 방식으로서 사무실 출근의 효용은 여전히 압도적입니다."
    )})

    # ── ★ 재반박 힌트 2회차 (AI 재반박 2회차 직후) ──────────────────────
    # print("\n" + "=" * 62)
    # print("  ★ 재반박 힌트 2회차 (AI 재반박 2회차 직후)")
    # print("=" * 62)
    # counter2 = da.counter_hint(history, topic)
    # print(counter2["hint"])

    # ── 토론 정리 ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ★ 토론 정리 + 피드백")
    print("=" * 62)
    result = da.summarize(history, topic)

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
    #

if __name__ == "__main__":
    main()