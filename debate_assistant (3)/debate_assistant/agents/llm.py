"""
agents/llm.py — Ollama LLM 호출 공통 래퍼

모든 agents가 LLM을 호출할 때 이 함수만 사용.
모델명·옵션은 config.py에서 가져옴.
"""

import re

import ollama

from config import MODEL_NAME, OLLAMA_OPTIONS


def clean_llm_output(text: str) -> str:
    """<think> 블록 제거."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(system_prompt: str, user_content: str) -> str:
    """
    Ollama LLM 호출.
    반환: LLM 응답 문자열 (<think> 블록 제거 후)
    오류 발생 시 "[ERROR] ..." 문자열 반환.
    """
    try:
        res = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            options=OLLAMA_OPTIONS,
            think=False,
        )
        return clean_llm_output(res.message.content)
    except Exception as e:
        return f"[ERROR] {e}"
