"""
server.py — Flask API 서버

실행:
    cd debate_assistant/
    python server.py

엔드포인트:
    GET  /health                서버 상태 확인
    POST /intro                 토론 주제 배경 정보 요약
    POST /intro/quiz            배경 요약 기반 사전 퀴즈 생성 (정답·해설 포함)
    POST /hint/counter          재반박 힌트 (AI 반박 직후)
    POST /hint/rebuttal         반박 힌트   (AI 새 주장 직후)
    POST /summarize             토론 정리 + 피드백
    POST /quiz                  토론 후 복습 퀴즈 생성 (정답·해설 포함)
    POST /score/turn            턴별 유저 발언 평가
    POST /score/final           전체 턴 통계 집계
    POST /score/reset           세션 초기화

※ evaluate 엔드포인트 없음 — 정답·해설을 퀴즈 생성 시 함께 전송하므로
   클라이언트가 직접 채점합니다.
"""

import socket
from flask import Flask, request, jsonify
from flask_cors import CORS

from debate_assistant import DebateAssistant
from agents.intro_agent import IntroAgent
from agents.intro_quiz_agent import IntroQuizAgent
from agents.quiz_agent import ReviewQuizAgent
from agents.scoring_agent import ScoringAgent

app = Flask(__name__)
CORS(app)

_intro_agent      = IntroAgent()
_intro_quiz_agent = IntroQuizAgent()
_scoring_agents: dict[str, ScoringAgent] = {}


# ── 공통 헬퍼 ─────────────────────────────────────────────────────

def _get_da(user_label, ai_label, news_data):
    return DebateAssistant(evidence_items=news_data, user_label=user_label, ai_label=ai_label)

def _get_scoring_agent(user_label, ai_label, session_key):
    if session_key not in _scoring_agents:
        _scoring_agents[session_key] = ScoringAgent(user_label=user_label, ai_label=ai_label)
    return _scoring_agents[session_key]

def _parse_history(raw):
    return [{"role": h["role"], "content": h["content"]}
            for h in raw if h.get("role") in ("user", "ai") and h.get("content")]

def _require(data, *keys):
    for key in keys:
        if not data.get(key):
            return f"{key} 필드 필요"
    return None

def _format_quiz(q: dict) -> dict:
    """
    퀴즈 하나를 클라이언트 전송용으로 정규화.
    explanation을 파싱해 각 선지별 reason 배열로 분리.
    """
    explanation = q.get("explanation", "")
    choices     = q.get("choices", [])

    # (1)~(4) 패턴으로 선지별 해설 분리
    import re
    parts = re.split(r"\(([1-4])\)", explanation)
    reasons = [""] * 4
    for i in range(1, len(parts), 2):
        idx = int(parts[i]) - 1
        if 0 <= idx <= 3:
            reasons[idx] = parts[i + 1].strip() if i + 1 < len(parts) else ""

    result = {
        "quiz_type":     q.get("quiz_type"),
        "type":          q.get("type", "reasoning"),
        "question":      q.get("question"),
        "choices":       choices,
        "correct_index": q.get("correct_index"),
        "explanation":   explanation,
        "choice_reasons": reasons,   # 선지별 이유 [0]~[3]
    }

    # news_inference 전용 추가 정보
    if q.get("context_summary"):
        result["context_summary"] = q["context_summary"]
    if q.get("gap_point"):
        result["gap_point"] = q["gap_point"]

    return result


# ── 헬스체크 ──────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── 인트로 ────────────────────────────────────────────────────────

@app.route("/intro", methods=["POST"])
def intro():
    """
    요청: { "topic": "...", "news_data": [...] }
    응답: { "summary": "...", "search_block": "..." }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "topic"):
        return jsonify({"error": err}), 400
    try:
        result = _intro_agent.run(topic=data["topic"], news_data=data.get("news_data") or [])
        return jsonify({"summary": result["summary"], "search_block": result["search_block"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/intro/quiz", methods=["POST"])
def intro_quiz():
    """
    사전 퀴즈 생성. 정답·해설·선지별 이유 포함.

    요청: { "topic": "...", "summary": "..." }
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
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "topic", "summary"):
        return jsonify({"error": err}), 400
    try:
        result = _intro_quiz_agent.run(topic=data["topic"], summary=data["summary"])
        result["quizzes"] = [_format_quiz(q) for q in result["quizzes"]]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 힌트 ──────────────────────────────────────────────────────────

def _hint_common(mode):
    data = request.get_json()
    if not data:
        return None, (jsonify({"error": "JSON body 필요"}), 400)
    if err := _require(data, "topic", "user_label", "ai_label"):
        return None, (jsonify({"error": err}), 400)
    history   = _parse_history(data.get("history", []))
    news_data = data.get("news_data", [])
    if not history:
        return None, (jsonify({"error": "history 필드 필요"}), 400)
    if not news_data:
        return None, (jsonify({"error": "news_data 필드 필요"}), 400)
    da = _get_da(data["user_label"], data["ai_label"], news_data)
    fn = da.counter_hint if mode == "counter" else da.rebuttal_hint
    return fn, (history, data["topic"])

@app.route("/hint/counter", methods=["POST"])
def hint_counter():
    fn, args = _hint_common("counter")
    if fn is None:
        return args
    try:
        history, topic = args
        return jsonify({"hint": fn(history, topic)["hint"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/hint/rebuttal", methods=["POST"])
def hint_rebuttal():
    fn, args = _hint_common("rebuttal")
    if fn is None:
        return args
    try:
        history, topic = args
        return jsonify({"hint": fn(history, topic)["hint"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 요약 ──────────────────────────────────────────────────────────

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "topic", "user_label", "ai_label"):
        return jsonify({"error": err}), 400
    history   = _parse_history(data.get("history", []))
    news_data = data.get("news_data", [])
    if not history:
        return jsonify({"error": "history 필드 필요"}), 400
    if not news_data:
        return jsonify({"error": "news_data 필드 필요"}), 400
    try:
        result = _get_da(data["user_label"], data["ai_label"], news_data).summarize(history, data["topic"])
        return jsonify({
            "summary":        result["summary"],
            "logic_feedback": result["logic_feedback"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 복습 퀴즈 ─────────────────────────────────────────────────────

@app.route("/quiz", methods=["POST"])
def quiz():
    """
    토론 후 복습 퀴즈 생성. 정답·해설·선지별 이유 포함.
    news_inference 유형은 context_summary(추가 정보)도 함께 전송.

    요청:
    {
        "topic":        "...",
        "user_label":   "...",
        "ai_label":     "...",
        "history":      [...],
        "news_data":    [...],
        "search_block": "..."  ← 선택
    }
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
                // news_inference만 추가:
                "context_summary": "추가 정보 요약",
                "gap_point":       "찾은 미검증 부분"
            },
            ...
        ],
        "selected_types": [...]
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "topic", "user_label", "ai_label"):
        return jsonify({"error": err}), 400
    history   = _parse_history(data.get("history", []))
    news_data = data.get("news_data", [])
    if not history:
        return jsonify({"error": "history 필드 필요"}), 400
    if not news_data:
        return jsonify({"error": "news_data 필드 필요"}), 400
    try:
        agent = ReviewQuizAgent(
            evidence_items=news_data,
            user_label=data["user_label"],
            ai_label=data["ai_label"],
        )
        result = agent.generate(
            history=history,
            topic=data["topic"],
            search_block=data.get("search_block", ""),
        )
        if result is None:
            return jsonify({"error": "퀴즈 생성 실패"}), 500
        result["quizzes"] = [_format_quiz(q) for q in result["quizzes"]]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 평가 ──────────────────────────────────────────────────────────

@app.route("/score/turn", methods=["POST"])
def score_turn():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "topic", "user_label", "ai_label"):
        return jsonify({"error": err}), 400
    history = _parse_history(data.get("history", []))
    if not history:
        return jsonify({"error": "history 필드 필요"}), 400
    session_key = data.get("session_key") or f"{data['topic']}_{data['user_label']}"
    try:
        agent  = _get_scoring_agent(data["user_label"], data["ai_label"], session_key)
        result = agent.score_turn(history=history, topic=data["topic"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/score/final", methods=["POST"])
def score_final():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "session_key"):
        return jsonify({"error": err}), 400
    session_key = data["session_key"]
    if session_key not in _scoring_agents:
        return jsonify({"error": f"'{session_key}' 평가 기록 없음"}), 404
    try:
        result = _scoring_agents.pop(session_key).score_final()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/score/reset", methods=["POST"])
def score_reset():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "session_key"):
        return jsonify({"error": err}), 400
    _scoring_agents.pop(data["session_key"], None)
    return jsonify({"status": "ok", "message": f"'{data['session_key']}' 초기화 완료"})


# ── 서버 시작 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'=' * 50}")
    print(f"  Debate Assistant API 서버 시작")
    print(f"  로컬:          http://127.0.0.1:5000")
    print(f"  같은 네트워크: http://{local_ip}:5000")
    print(f"{'=' * 50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
