"""
debate_api.py — 토론 보조 AI 서버 API 호출 함수 모음

설치:
    pip install requests

사용법:
    from debate_api import get_intro, get_intro_quiz
    from debate_api import get_counter_hint, get_rebuttal_hint, get_summary
    from debate_api import get_score_turn, get_score_final, reset_score

주의:
    - history의 role은 "ai" 또는 "user" 소문자만 가능
    - hint/counter, hint/rebuttal 호출 전 history 마지막은 반드시 role: "ai"
    - news_data는 AI 발언 직후 새로 검색한 누적 뉴스 배열 (필수)
    - user_label, ai_label은 주제에 맞는 문자열 (예: "이란측"/"미국측")
    - get_intro의 news_data는 선택. 외부 서버 뉴스가 있으면 전달, 없으면 생략.
    - get_score_turn은 턴 종료 직후 순서대로 호출. turn_number는 서버가 자동 계산.
    - get_score_final 호출 후 세션이 자동 삭제됨
"""

import requests

BASE_URL = "https://undemolished-evelyne-jurisdictionally.ngrok-free.dev"


def check_health() -> dict:
    """
    서버 상태 확인.
    응답: {"status": "ok"}
    """
    res = requests.get(f"{BASE_URL}/health")
    return res.json()


# ── 인트로 ────────────────────────────────────────────────────────

def get_intro(
    topic: str,
    news_data: list | None = None,
) -> dict:
    """
    토론 시작 전 주제 배경 정보 요약.
    news_data가 있으면 서버가 그것을 우선 사용하고, 없으면 Tavily로 자체 검색.

    Args:
        topic:     토론 주제
        news_data: 외부 서버에서 받은 뉴스 배열 (선택).
                   각 항목 권장 형태: {"title": "...", "content": "...", "url": "..."}

    응답:
        {
            "summary":      "배경 요약 텍스트",
            "search_block": "사용된 뉴스/검색 결과 원문 (디버그용)"
        }
    """
    payload: dict = {"topic": topic}
    if news_data:
        payload["news_data"] = news_data

    res = requests.post(f"{BASE_URL}/intro", json=payload)
    return res.json()


def get_intro_quiz(
    topic: str,
    summary: str,
) -> dict:
    """
    배경 요약 기반 퀴즈 생성 (OX 3개 + 객관식 2개).

    Args:
        topic:   토론 주제
        summary: get_intro()["summary"] 값

    응답:
        {
            "quizzes": [
                {"type": "ox",       "question": "...", "answer": true,  "explanation": "..."},
                {"type": "multiple", "question": "...", "options": [...], "answer": 0, "explanation": "..."},
                ...
            ]
        }
    """
    res = requests.post(f"{BASE_URL}/intro/quiz", json={
        "topic":   topic,
        "summary": summary,
    })
    return res.json()


# ── 힌트 ──────────────────────────────────────────────────────────

def get_counter_hint(
    topic: str,
    user_label: str,
    ai_label: str,
    history: list,
    news_data: list,
) -> dict:
    """
    재반박 힌트 생성 (AI가 반박한 직후 호출).
    history 마지막 항목은 반드시 role: "ai"

    응답: {"hint": "재반박 힌트 텍스트"}
    """
    res = requests.post(f"{BASE_URL}/hint/counter", json={
        "topic":      topic,
        "user_label": user_label,
        "ai_label":   ai_label,
        "history":    history,
        "news_data":  news_data,
    })
    return res.json()


def get_rebuttal_hint(
    topic: str,
    user_label: str,
    ai_label: str,
    history: list,
    news_data: list,
) -> dict:
    """
    반박 힌트 생성 (AI가 새 주장한 직후 호출).
    history 마지막 항목은 반드시 role: "ai"

    응답: {"hint": "반박 힌트 텍스트"}
    """
    res = requests.post(f"{BASE_URL}/hint/rebuttal", json={
        "topic":      topic,
        "user_label": user_label,
        "ai_label":   ai_label,
        "history":    history,
        "news_data":  news_data,
    })
    return res.json()


# ── 요약 ──────────────────────────────────────────────────────────

def get_summary(
    topic: str,
    user_label: str,
    ai_label: str,
    history: list,
    news_data: list,
) -> dict:
    """
    토론 종료 후 전체 정리 + 피드백.

    응답:
        {
            "summary":        "토론 요약",
            "logic_feedback": "논리 피드백 + 보완 정보",
            "extra_info":     "추가 사례"
        }
    """
    res = requests.post(f"{BASE_URL}/summarize", json={
        "topic":      topic,
        "user_label": user_label,
        "ai_label":   ai_label,
        "history":    history,
        "news_data":  news_data,
    })
    return res.json()


# ── 퀴즈 ──────────────────────────────────────────────────────────

def get_quiz(
    topic: str,
    user_label: str,
    ai_label: str,
    history: list,
    news_data: list,
) -> dict:
    """
    퀴즈 생성 (토론 종료 후 호출).

    응답:
        {
            "review_quiz":   { "question", "options", "answer", "explanation", "option_explanations" },
            "weakness_quiz": { "question", "options", "answer", "explanation", "option_explanations" }
        }
    """
    res = requests.post(f"{BASE_URL}/quiz", json={
        "topic":      topic,
        "user_label": user_label,
        "ai_label":   ai_label,
        "history":    history,
        "news_data":  news_data,
    })
    return res.json()


# ── 평가 ──────────────────────────────────────────────────────────

def get_score_turn(
    topic: str,
    user_label: str,
    ai_label: str,
    session_key: str,
    history: list,
) -> dict:
    """
    턴 종료 직후 호출. turn_number는 서버가 자동 계산.
    반드시 턴 순서대로 호출해야 함.

    Args:
        topic       : 토론 주제
        user_label  : 유저 입장 레이블 (예: "이란측")
        ai_label    : AI 입장 레이블   (예: "미국측")
        session_key : 같은 토론 세션 식별용 고유 문자열
        history     : 현재 턴까지의 전체 대화 기록

    응답:
        {
            "turn": 1,
            "scores": {
                "specificity": { "score": 1~5, "reason": "...", "evidence": "..." },
                "causality":   { "score": 1~5, "reason": "...", "evidence": "..." },
                "domain":      { "score": 1~5, "reason": "...", "evidence": "...", "domains": [...] },
                "initiative":  { "score": 1~5, "reason": "...", "evidence": "..." }
            },
            "total": 4~20
        }
    """
    res = requests.post(f"{BASE_URL}/score/turn", json={
        "topic":       topic,
        "user_label":  user_label,
        "ai_label":    ai_label,
        "session_key": session_key,
        "history":     history,
    })
    return res.json()


def get_score_final(session_key: str) -> dict:
    """
    토론 종료 후 전체 턴 통계 집계.
    호출 후 서버에서 세션 자동 삭제.

    응답:
        {
            "turns": [ 각 턴 점수, ... ],
            "summary": {
                "specificity": { "avg": float, "trend": "상승"|"하락"|"유지", "scores_per_turn": [...] },
                "causality":   { ... },
                "domain":      { ..., "all_domains": [...] },
                "initiative":  { ... }
            },
            "total_avg": float
        }
    """
    res = requests.post(f"{BASE_URL}/score/final", json={
        "session_key": session_key,
    })
    return res.json()


def reset_score(session_key: str) -> dict:
    """
    새 토론 시작 시 기존 평가 기록 초기화.

    응답: {"status": "ok", "message": "..."}
    """
    res = requests.post(f"{BASE_URL}/score/reset", json={
        "session_key": session_key,
    })
    return res.json()


def evaluate_intro_subjective(
    topic: str,
    summary: str,
    qa_pairs: list[dict],
) -> dict:
    """
    토론 전 주관식 답변 평가.
    나중에 토론 후 퀴즈 점수와 비교해 이해도 증진 측정용.

    Args:
        topic:    토론 주제
        summary:  get_intro()["summary"] 값
        qa_pairs: [{"question": "질문", "answer": "유저 답변"}, ...]

    응답:
        {
            "results": [
                {"question": "...", "answer": "...", "score": 1~5, "reason": "..."},
                ...
            ],
            "total_score": 2~10   ← 토론 후 퀴즈 점수와 비교용
        }
    """
    res = requests.post(f"{BASE_URL}/intro/quiz/evaluate", json={
        "topic":    topic,
        "summary":  summary,
        "qa_pairs": qa_pairs,
    })
    return res.json()
