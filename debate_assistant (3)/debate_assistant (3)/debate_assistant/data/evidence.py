"""
data/evidence.py — news_data.json 로드 + 프롬프트용 블록 빌드

LLM 호출 없음. 순수 데이터 가공만 담당.
"""

import json
import os
import re


def _sanitize(text: str) -> str:
    """
    오염된 문자 제거.
    한국어·영어·숫자·기본 특수문자·공백 외의 문자(키릴 문자 등)를 제거.
    허용 범위: ASCII + 한글(가-힣) + 한글 자모
    """
    return re.sub(r"[^\x00-\x7F\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]", "", text)


def load_news_data_json(path: str) -> list[dict]:
    """
    news_data.json 파일을 로드해 기본 증거 아이템 목록 반환.
    파일이 없거나 오류 시 빈 리스트 반환.

    사용:
        from data import load_news_data_json
        evidence = load_news_data_json("news_data.json")
    """
    if not path or not os.path.exists(path):
        print(f"  [news_data] 파일 없음: {path!r}")
        return []
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else [raw]
        print(f"  [news_data] {len(items)}개 아이템 로드 ← {path}")
        return items
    except Exception as e:
        print(f"  [news_data] 로드 실패: {e}")
        return []


def build_evidence_block(items: list[dict], max_chars: int = 3000) -> str:
    """
    news_data.json 아이템 목록 → LLM 프롬프트용 텍스트.
    full_content 우선, 없으면 content 사용.
    max_chars 초과 시 잘라냄.
    """
    if not items:
        return "(증거 없음)"

    parts, total = [], 0
    for i, item in enumerate(items, 1):
        title   = _sanitize(item.get("title", "제목 없음"))
        url     = item.get("url", "")
        content = _sanitize(item.get("full_content") or item.get("content", ""))
        score   = item.get("score", "")

        header = f"[증거 {i}] {title}"
        if url:
            header += f"  (출처: {url})"
        if score:
            header += f"  [관련도: {score}]"
        body = f"{header}\n{content}"

        if total + len(body) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(body[:remaining] + "\n...(생략)")
            break
        parts.append(body)
        total += len(body)

    return "\n\n".join(parts)


def build_history_block(history: list[dict]) -> str:
    """
    토론 히스토리 리스트 → LLM 프롬프트용 텍스트.
    history 형식: [{"role": "ai"|"user", "content": "..."}]
    """
    lines = []
    for h in history:
        role = "AI" if h["role"] == "ai" else "유저"
        lines.append(f"[{role}] {h['content']}")
    return "\n".join(lines)


def extract_last_ai_claim(history: list[dict]) -> str:
    """히스토리에서 마지막 AI 발언 반환. 없으면 빈 문자열."""
    for h in reversed(history):
        if h["role"] == "ai":
            return h["content"]
    return ""
