"""
debate_api.py — 토론 보조 AI 서버 API 호출 함수 모음

설치:
    pip install requests

사용법:
    from debate_api import get_intro, get_counter_hint, get_rebuttal_hint, get_summary

주의:
    - history의 role은 "ai" 또는 "user" 소문자만 가능
    - hint/counter, hint/rebuttal 호출 전 history 마지막은 반드시 role: "ai"
    - news_data는 주제별로 수집한 뉴스 배열 (없으면 서버 기본 데이터 사용)
"""

import requests

# AI 서버 주소 (ngrok 주소 바뀌면 여기만 변경)
BASE_URL = "https://undemolished-evelyne-jurisdictionally.ngrok-free.dev"


def check_health() -> dict:
    """
    서버 상태 확인.
    응답: {"status": "ok", "evidence_count": 50}
    """
    res = requests.get(f"{BASE_URL}/health")
    return res.json()


def get_intro(topic: str, stance: int, news_data: list) -> dict:
    """
    토론 시작 전 주제 배경 요약.

    Args:
        topic     : 토론 주제 (문자열)
        stance    : 유저 입장 (1=찬성, -1=반대)
        news_data : 주제 관련 뉴스 배열

    응답: {"summary": "주제 요약 텍스트"}
    """
    res = requests.post(f"{BASE_URL}/intro", json={
        "topic":     topic,
        "stance":    stance,
        "news_data": news_data,
    })
    return res.json()


def get_counter_hint(topic: str, stance: int, history: list, news_data: list) -> dict:
    """
    재반박 힌트 생성 (AI가 반박한 직후 호출).
    history 마지막 항목은 반드시 role: "ai"

    Args:
        topic     : 토론 주제
        stance    : 유저 입장 (1=찬성, -1=반대)
        history   : 그 시점까지 대화 전부
        news_data : 주제 관련 뉴스 배열

    응답: {"hint": "재반박 힌트 텍스트"}
    """
    res = requests.post(f"{BASE_URL}/hint/counter", json={
        "topic":     topic,
        "stance":    stance,
        "history":   history,
        "news_data": news_data,
    })
    return res.json()


def get_rebuttal_hint(topic: str, stance: int, history: list, news_data: list) -> dict:
    """
    반박 힌트 생성 (AI가 새 주장한 직후 호출).
    history 마지막 항목은 반드시 role: "ai"

    Args:
        topic     : 토론 주제
        stance    : 유저 입장 (1=찬성, -1=반대)
        history   : 그 시점까지 대화 전부
        news_data : 주제 관련 뉴스 배열

    응답: {"hint": "반박 힌트 텍스트"}
    """
    res = requests.post(f"{BASE_URL}/hint/rebuttal", json={
        "topic":     topic,
        "stance":    stance,
        "history":   history,
        "news_data": news_data,
    })
    return res.json()


def get_summary(topic: str, stance: int, history: list, news_data: list, turns: int = 1) -> dict:
    """
    토론 종료 후 전체 정리 + 피드백.

    Args:
        topic     : 토론 주제
        stance    : 유저 입장 (1=찬성, -1=반대)
        history   : 전체 토론 기록
        news_data : 주제 관련 뉴스 배열
        turns     : 진행된 라운드 수 (기본값 1)

    응답:
        {
            "summary":        "토론 요약",
            "issues":         "쟁점 분석",
            "logic_feedback": "논리 피드백 + 보완 정보",
            "extra_info":     "추가 사례"
        }
    """
    res = requests.post(f"{BASE_URL}/summarize", json={
        "topic":     topic,
        "stance":    stance,
        "turns":     turns,
        "history":   history,
        "news_data": news_data,
    })
    return res.json()
