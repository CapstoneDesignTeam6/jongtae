"""
debate_api.py — 토론 보조 AI 서버 API 호출 함수 모음

설치:
    pip install requests

사용법:
    from debate_api import get_counter_hint, get_rebuttal_hint, get_summary

주의:
    - history의 role은 "ai" 또는 "user" 소문자만 가능
    - hint/counter, hint/rebuttal 호출 전 history 마지막은 반드시 role: "ai"
    - news_data는 AI 발언 직후 새로 검색한 누적 뉴스 배열 (필수)
    - user_label, ai_label은 주제에 맞는 문자열 (예: "이란측"/"미국측")
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

    Args:
        topic      : 토론 주제
        user_label : 유저 입장 레이블 (예: "이란측")
        ai_label   : AI 입장 레이블   (예: "미국측")
        history    : 그 시점까지 대화 전부
        news_data  : AI 반박 직후 새로 검색한 누적 뉴스 배열

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

    Args:
        topic      : 토론 주제
        user_label : 유저 입장 레이블 (예: "이란측")
        ai_label   : AI 입장 레이블   (예: "미국측")
        history    : 그 시점까지 대화 전부
        news_data  : AI 새 주장 직후 새로 검색한 누적 뉴스 배열

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


def get_summary(
    topic: str,
    user_label: str,
    ai_label: str,
    history: list,
    news_data: list,
    turns: int = 1,
) -> dict:
    """
    토론 종료 후 전체 정리 + 피드백.

    Args:
        topic      : 토론 주제
        user_label : 유저 입장 레이블 (예: "이란측")
        ai_label   : AI 입장 레이블   (예: "미국측")
        history    : 전체 토론 기록
        news_data  : 최종 누적 뉴스 배열
        turns      : 진행된 라운드 수 (기본값 1)

    응답:
        {
            "summary":        "토론 요약 (유저/AI 주장+근거 각각)",
            "logic_feedback": "논리 피드백 + 보완 정보",
            "extra_info":     "추가 사례"
        }
    """
    res = requests.post(f"{BASE_URL}/summarize", json={
        "topic":      topic,
        "user_label": user_label,
        "ai_label":   ai_label,
        "turns":      turns,
        "history":    history,
        "news_data":  news_data,
    })
    return res.json()


def get_quiz(
    topic: str,
    user_label: str,
    ai_label: str,
    history: list,
    news_data: list,
) -> dict:
    """
    퀴즈 생성 (토론 종료 후 호출).

    Args:
        topic      : 토론 주제
        user_label : 유저 입장 레이블 (예: "이란측")
        ai_label   : AI 입장 레이블   (예: "미국측")
        history    : 전체 토론 기록
        news_data  : 최종 누적 뉴스 배열

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
