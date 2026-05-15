"""
main.py — 토론 전체 플로우 테스트
순서: 배경요약 → 사전퀴즈(OX+주관식+평가) → 토론(턴별 점수) → 요약/피드백 → 복습퀴즈(OX) → 약점퀴즈(주관식+평가)
"""

from agents.intro_agent      import IntroAgent
from agents.intro_quiz_agent import IntroQuizAgent
from agents.scoring_agent    import ScoringAgent
from agents.summary_agent    import SummaryAgent
from agents.quiz_agent       import ReviewQuizAgent, WeaknessQuizAgent
from agents.hint_agent       import HintAgent

# ── 설정 ─────────────────────────────────────────────────────────

TOPIC      = "이란과 미국 전쟁, 누가 더 손해인가?"
USER_LABEL = "이란"
AI_LABEL   = "미국"

# 미리 작성된 토론 대화 (AI 선공)
HISTORY = [
    {"role": "ai", "content": (
        "이란과 미국의 전쟁에서 더 큰 손해를 보는 쪽은 미국입니다. "
        "전쟁이 발생하면 글로벌 금융 시장이 불안정해지고, 미국 증시와 달러 가치가 하락할 가능성이 큽니다. "
        "또한 중동 지역 불안정으로 에너지 가격이 급등하면 미국 내 물가 상승과 소비 위축이 발생해 "
        "실질적으로 미국 국민 전체가 경제적 부담을 떠안게 됩니다."
    )},
    {"role": "user", "content": (
        "하지만 이란이 입는 사회적·인권적 피해가 훨씬 심각합니다. "
        "전쟁이 발생하면 이란 내 민간인 피해와 난민 문제가 급증하고, "
        "기반 시설 붕괴로 의료·교육 시스템이 마비될 가능성이 큽니다. "
        "이미 제재로 인해 생필품 부족을 겪고 있는 상황에서 전쟁까지 겹치면 "
        "이란 국민들의 삶은 회복 불가능한 수준으로 악화될 것입니다."
    )},
    {"role": "ai", "content": (
        "이란 내부의 인도적 위기는 심각하지만, 미국 역시 사회적 비용을 무시할 수 없습니다. "
        "전쟁이 장기화될 경우 미국 내 반전 여론이 확산되고 사회적 갈등이 심화될 수 있습니다. "
        "또한 군 복무 확대와 국방비 증가로 인해 복지 예산이 축소되면서 저소득층과 중산층의 삶의 질이 하락합니다."
    )},
    {"role": "user", "content": (
        "UN난민기구(UNHCR) 2023년 보고서에 따르면 이란은 이미 아프가니스탄 난민 380만 명을 수용 중입니다. "
        "전쟁이 발발하면 이 수용 능력이 완전히 붕괴돼 주변국 전체의 인도적 위기로 번질 수 있어요. "
        "반면 미국은 지리적으로 분쟁 현장에서 멀리 떨어져 있어 직접적인 민간 피해 규모가 "
        "이란과는 비교할 수 없을 정도로 작을 수밖에 없습니다."
    )},
]

# 사전 주관식 샘플 답변
PRE_ANSWERS = [
    "이란이 더 손해라고 생각해요. 경제 제재로 이미 많이 힘든 상황에서 전쟁까지 나면 "
    "일반 국민들이 버틸 수가 없을 것 같아요. 미국은 본토가 공격받을 가능성이 낮으니까요.",
    "미국의 군사력이 압도적으로 강하지만 그렇다고 손해가 없는 건 아니에요. "
    "다만 이란은 이미 경제적으로 약한 상태라 전쟁 한 번에 회복 불가능한 타격을 받을 것 같아요.",
]

# 약점 주관식 샘플 답변
WEAKNESS_ANSWERS = [
    "경제 제재가 오래 지속되면 이란의 산업 기반 자체가 무너질 수 있어서 전쟁보다 더 무서울 수도 있어요.",
    "미국이 제재를 강화할수록 이란은 러시아나 중국에 의존하게 되고 결국 미국 입장에서도 외교적 손해가 생겨요.",
]


# ── 출력 헬퍼 ─────────────────────────────────────────────────────

SEP  = "=" * 62
SEP2 = "─" * 62

def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title: str):
    print(f"\n  [{title}]")

def bar(score: int, max_score: int = 5) -> str:
    filled = round(score * 5 / max_score)
    return "█" * filled + "░" * (5 - filled)


# ── 1. 배경 요약 ──────────────────────────────────────────────────

def step_intro(topic: str) -> str:
    section("STEP 1 · 배경 요약  (IntroAgent)")
    agent  = IntroAgent()
    result = agent.run(topic=topic)
    summary = result.get("summary", "")
    print(f"\n{summary}\n")
    return summary


# ── 2. 사전 퀴즈 생성 + 평가 ──────────────────────────────────────

def step_pre_quiz(topic: str, summary: str) -> tuple[list[dict], int]:
    section("STEP 2 · 사전 퀴즈  (IntroQuizAgent)")
    agent  = IntroQuizAgent()
    result = agent.run(topic=topic, summary=summary)

    ox_list  = [q for q in result["quizzes"] if q["type"] == "ox"]
    sub_list = [q for q in result["quizzes"] if q["type"] == "subjective"]

    sub("OX 퀴즈")
    for i, q in enumerate(ox_list, 1):
        ans = "O (참)" if q["answer"] else "X (거짓)"
        print(f"  Q{i}. {q['question']}")
        print(f"       정답: {ans}")
        print(f"       해설: {q['explanation']}")

    sub("주관식 퀴즈")
    for i, q in enumerate(sub_list, 1):
        print(f"  Q{i}. {q['question']}")
        print(f"       안내: {q['context']}")

    # 주관식 평가
    sub("주관식 답변 평가")
    qa_pairs = [
        {"question": sub_list[i]["question"], "answer": PRE_ANSWERS[i]}
        for i in range(min(len(sub_list), len(PRE_ANSWERS)))
    ]
    eval_result = agent.evaluate_subjective(topic=topic, summary=summary, qa_pairs=qa_pairs)

    for r in eval_result["results"]:
        print(f"\n  Q. {r['question']}")
        print(f"  A. {r['answer'][:80]}...")
        print(f"     [{bar(r['score'])}] {r['score']}/5  {r['reason']}")

    pre_total = eval_result["total_score"]
    print(f"\n  사전 주관식 총점: {pre_total}/10")
    return sub_list, pre_total


# ── 3. 힌트 (첫 AI 발언 직후) ────────────────────────────────────

def step_hint(history: list[dict], topic: str):
    section("STEP 3 · 반박 힌트  (HintAgent)")
    agent  = HintAgent(user_label=USER_LABEL, ai_label=AI_LABEL)

    sub("반박 힌트 (1턴 AI 발언 직후)")
    h1 = agent.rebuttal_hint(history[:1], topic)
    print(f"\n{h1.get('hint', '생성 실패')}")

    sub("재반박 힌트 (2턴 AI 발언 직후)")
    h2 = agent.counter_hint(history[:3], topic)
    print(f"\n{h2.get('hint', '생성 실패')}")


# ── 4. 턴별 점수 ─────────────────────────────────────────────────

def step_scoring(history: list[dict], topic: str) -> float:
    section("STEP 4 · 턴별 발언 점수  (ScoringAgent)")

    METRIC_LABELS = {
        "specificity": "발언 구체성",
        "causality":   "인과 연결  ",
        "domain":      "도메인 다양",
        "initiative":  "정보 주도성",
        "bias":        "편향도     ",
    }

    agent = ScoringAgent(user_label=USER_LABEL, ai_label=AI_LABEL)

    # 유저 발언이 등장하는 히스토리 slice 구성
    user_turn_indices = [i for i, h in enumerate(history) if h["role"] == "user"]

    for turn_num, user_idx in enumerate(user_turn_indices, 1):
        partial_history = history[: user_idx + 1]
        result = agent.score_turn(history=partial_history, topic=topic)

        scores = result.get("scores", {})
        total  = result.get("total", 0)
        max_t  = len(METRIC_LABELS) * 5

        print(f"\n  {SEP2}")
        print(f"  {turn_num}턴  |  총점: {total}/{max_t}")
        print(f"  {SEP2}")

        for key, label in METRIC_LABELS.items():
            b      = scores.get(key, {})
            sc     = b.get("score", 0)
            reason = b.get("reason", "")
            print(f"  {label}  [{bar(sc)}] {sc}/5  {reason}")

            if key == "domain" and b.get("domains"):
                print(f"              도메인: {', '.join(b['domains'])}")

            if key == "bias" and b.get("details"):
                d = b["details"]
                print(f"              통계편향: {d.get('stat_bias', '')}")
                print(f"              반례수용: {d.get('counterarg', '')}")
                print(f"              감정선동: {d.get('emotional_bias', '')}")

    final = agent.score_final()
    summary_data = final.get("summary", {})
    total_avg    = final.get("total_avg", 0)

    print(f"\n  {SEP2}")
    print(f"  최종 통계  |  전체 평균: {total_avg}/{len(METRIC_LABELS) * 5}")
    print(f"  {SEP2}")

    for key, label in METRIC_LABELS.items():
        m        = summary_data.get(key, {})
        avg      = m.get("avg", 0)
        trend    = m.get("trend", "유지")
        per_turn = m.get("scores_per_turn", [])
        symbol   = {"상승": "↑", "하락": "↓", "유지": "→"}.get(trend, "?")
        print(f"  {label}  [{bar(avg)}] avg {avg:.1f}/5  {symbol} {trend}  턴별: {per_turn}")
        if key == "domain" and m.get("all_domains"):
            print(f"              누적 도메인: {', '.join(m['all_domains'])}")

    return total_avg


# ── 5. 토론 요약 + 피드백 ─────────────────────────────────────────

def step_summary(history: list[dict], topic: str):
    section("STEP 5 · 토론 요약 + 피드백  (SummaryAgent)")
    agent  = SummaryAgent(user_label=USER_LABEL, ai_label=AI_LABEL)
    result = agent.summarize(history=history, topic=topic)

    sub("토론 요약")
    print(f"\n{result.get('summary', '생성 실패')}")

    if result.get("logic_feedback"):
        sub("논리 피드백")
        print(f"\n{result['logic_feedback']}")

    if result.get("extra_info"):
        sub("추가 사례·정보")
        print(f"\n{result['extra_info']}")


# ── 6. 복습 퀴즈 (OX) ────────────────────────────────────────────

def step_review_quiz(history: list[dict], topic: str):
    section("STEP 6 · 복습 퀴즈  (ReviewQuizAgent — OX 3개)")
    agent  = ReviewQuizAgent(user_label=USER_LABEL, ai_label=AI_LABEL)
    result = agent.generate(history=history, topic=topic)

    if not result:
        print("  생성 실패")
        return

    for i, q in enumerate(result["quizzes"], 1):
        ans = "O (참)" if q["answer"] else "X (거짓)"
        print(f"\n  Q{i}. {q['question']}")
        print(f"       정답: {ans}")
        print(f"       해설: {q['explanation']}")


# ── 7. 약점 퀴즈 (주관식) + 평가 ─────────────────────────────────

def step_weakness_quiz(history: list[dict], topic: str):
    section("STEP 7 · 약점 퀴즈  (WeaknessQuizAgent — 주관식 2개 + 평가)")
    agent  = WeaknessQuizAgent(user_label=USER_LABEL, ai_label=AI_LABEL)
    result = agent.generate(history=history, topic=topic)

    if not result:
        print("  생성 실패")
        return

    sub_list = result["quizzes"]

    sub("주관식 퀴즈")
    for i, q in enumerate(sub_list, 1):
        print(f"\n  Q{i}. {q['question']}")
        print(f"       안내: {q['context']}")

    sub("주관식 답변 평가")
    qa_pairs = [
        {"question": sub_list[i]["question"], "answer": WEAKNESS_ANSWERS[i]}
        for i in range(min(len(sub_list), len(WEAKNESS_ANSWERS)))
    ]
    eval_result = agent.evaluate(topic=topic, history=history, qa_pairs=qa_pairs)

    for r in eval_result["results"]:
        print(f"\n  Q. {r['question']}")
        print(f"  A. {r['answer'][:80]}...")
        print(f"     [{bar(r['score'])}] {r['score']}/5  {r['reason']}")

    print(f"\n  약점 주관식 총점: {eval_result['total_score']}/10")


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    print(f"\n{'★' * 31}")
    print(f"  토론 주제: {TOPIC}")
    print(f"  유저: {USER_LABEL}  |  AI: {AI_LABEL}")
    print(f"{'★' * 31}")

    summary              = step_intro(TOPIC)
    _sub_list, pre_total = step_pre_quiz(TOPIC, summary)
    step_hint(HISTORY, TOPIC)
    total_avg            = step_scoring(HISTORY, TOPIC)
    step_summary(HISTORY, TOPIC)
    step_review_quiz(HISTORY, TOPIC)
    step_weakness_quiz(HISTORY, TOPIC)

    section("완료 · 이해도 변화 요약")
    print(f"  사전 주관식 점수: {pre_total}/10")
    print(f"  토론 중 평균 점수: {total_avg}/{5 * 5}")
    print(f"\n  모든 에이전트 실행 완료 ✅")


if __name__ == "__main__":
    main()