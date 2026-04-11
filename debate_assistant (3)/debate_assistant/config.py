"""
config.py — 전역 설정
"""

# ── Ollama ────────────────────────────────────────────────────────
MODEL_NAME     = "gemma4:26b"
OLLAMA_OPTIONS = {"num_gpu": -1, "num_ctx": 4096 * 8}
