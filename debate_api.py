"""
debate_api.py — 토론 보조 AI 서버 API 호출 함수 모음

설치:
    pip install requests

※ evaluate 함수 없음 — 퀴즈 생성 시 정답·해설·선지별 이유가 모두 포함되어 있으므로
   클라이언트에서 직접 채점하면 됩니다.

채점 예시:
    result = get_intro_quiz(topic, summary)
    for quiz, user_answer in zip(result["quizzes"], user_answers):
        is_correct = user_answer == quiz["correct_index"]
        reason     = quiz["choice_reasons"][user_answer]   # 유저가 고른 선지의 이유
"""

import requests

BASE_URL = "https://undemolished-evelyne-jurisdictionally.ngrok-free.dev"


def check_health() -> dict:
    """서버 상태 확인. 응답: {"status": "ok"}"""
    return requests.get(f"{BASE_URL}/health").json()


# ── 인트로 ────────────────────────────────────────────────────────

def get_intro(topic: str, news_data: list | None = None) -> dict:
    """
    토론 시작 전 주제 배경 정보 요약.
    news_data가 있으면 서버가 우선 사용, 없으면 Tavily로 자체 검색.

    응답:
        {
            "summary":      "배경 요약 텍스트",
            "search_block": "사용된 뉴스/검색 결과 원문 (quiz 호출 시 넘겨줄 것)"
        }
    """
    payload: dict = {"topic": topic}
    if news_data:
        payload["news_data"] = news_data
    return requests.post(f"{BASE_URL}/intro", json=payload).json()


def get_intro_quiz(topic: str, summary: str) -> dict:
    """
    사전 퀴즈 생성. 정답·해설·선지별 이유 포함.

    응답:
        {
            "quizzes": [
                {
                    "quiz_type":      "missing_variable" | "overgeneralization" | "counterargument",
                    "type":           "reasoning",
                    "question":       "...",
                    "choices":        ["...다.", "...다.", "...다.", "...다."],
                    "correct_index":  0~3,
                    "explanation":    "전체 해설 텍스트",
                    "choice_reasons": ["①이유", "②이유", "③이유", "④이유"]
                },
                ...
            ],
            "selected_types": [...]
        }

    클라이언트 채점:
        is_correct     = user_answer == quiz["correct_index"]
        selected_reason = quiz["choice_reasons"][user_answer]
    """
    return requests.post(f"{BASE_URL}/intro/quiz", json={
        "topic": topic, "summary": summary,
    }).json()


# ── 힌트 ──────────────────────────────────────────────────────────

def get_counter_hint(
    topic: str, user_label: str, ai_label: str,
    history: list, news_data: list,
) -> dict:
    """재반박 힌트 (AI 반박 직후). 응답: {"hint": "..."}"""
    return requests.post(f"{BASE_URL}/hint/counter", json={
        "topic": topic, "user_label": user_label, "ai_label": ai_label,
        "history": history, "news_data": news_data,
    }).json()


def get_rebuttal_hint(
    topic: str, user_label: str, ai_label: str,
    history: list, news_data: list,
) -> dict:
    """반박 힌트 (AI 새 주장 직후). 응답: {"hint": "..."}"""
    return requests.post(f"{BASE_URL}/hint/rebuttal", json={
        "topic": topic, "user_label": user_label, "ai_label": ai_label,
        "history": history, "news_data": news_data,
    }).json()


# ── 요약 ──────────────────────────────────────────────────────────

def get_summary(
    topic: str, user_label: str, ai_label: str,
    history: list, news_data: list,
) -> dict:
    """
    토론 종료 후 전체 정리 + 피드백.

    응답:
        {
            "summary":        "토론 요약",
            "logic_feedback": "논리 피드백"
        }
    """
    return requests.post(f"{BASE_URL}/summarize", json={
        "topic": topic, "user_label": user_label, "ai_label": ai_label,
        "history": history, "news_data": news_data,
    }).json()


# ── 복습 퀴즈 ─────────────────────────────────────────────────────

def get_quiz(
    topic: str, user_label: str, ai_label: str,
    history: list, news_data: list,
    search_block: str = "",
) -> dict:
    """
    토론 후 복습 퀴즈 생성. 정답·해설·선지별 이유 포함.
    news_inference 유형은 context_summary(추가 정보)도 함께 전송.

    Args:
        search_block: get_intro()["search_block"] 값 (선택)

    응답:
        {
            "quizzes": [
                {
                    "quiz_type":      "argument_core" | "argument_flaw" | "news_inference",
                    "type":           "reasoning",
                    "question":       "...",
                    "choices":        ["...다.", "...다.", "...다.", "...다."],
                    "correct_index":  0~3,
                    "explanation":    "전체 해설 텍스트",
                    "choice_reasons": ["①이유", "②이유", "③이유", "④이유"],

                    // news_inference 유형만 추가:
                    "context_summary": "추가 정보 요약 (화면에 표시)",
                    "gap_point":       "찾은 미검증 부분"
                },
                ...
            ],
            "selected_types": [...]
        }

    클라이언트 채점:
        is_correct      = user_answer == quiz["correct_index"]
        selected_reason = quiz["choice_reasons"][user_answer]
        # news_inference면 quiz["context_summary"]를 문제 아래에 표시
    """
    return requests.post(f"{BASE_URL}/quiz", json={
        "topic": topic, "user_label": user_label, "ai_label": ai_label,
        "history": history, "news_data": news_data,
        "search_block": search_block,
    }).json()


# ── 평가 ──────────────────────────────────────────────────────────

def get_score_turn(
    topic: str, user_label: str, ai_label: str,
    session_key: str, history: list,
) -> dict:
    """
    턴 종료 직후 호출. 반드시 순서대로.

    응답:
        {
            "turn": 1,
            "scores": {
                "specificity": {"score": 1~5, "reason": "...", "evidence": "..."},
                "causality":   {"score": 1~5, "reason": "...", "evidence": "..."},
                "domain":      {"score": 1~5, "reason": "...", "evidence": "...", "domains": [...]},
                "initiative":  {"score": 1~5, "reason": "...", "evidence": "..."}
            },
            "total": 4~20
        }
    """
    return requests.post(f"{BASE_URL}/score/turn", json={
        "topic": topic, "user_label": user_label, "ai_label": ai_label,
        "session_key": session_key, "history": history,
    }).json()


def get_score_final(session_key: str) -> dict:
    """전체 턴 통계 집계. 호출 후 세션 자동 삭제."""
    return requests.post(f"{BASE_URL}/score/final", json={"session_key": session_key}).json()


def reset_score(session_key: str) -> dict:
    """새 토론 시작 시 기존 평가 기록 초기화."""
    return requests.post(f"{BASE_URL}/score/reset", json={"session_key": session_key}).json()
