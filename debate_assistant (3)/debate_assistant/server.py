"""
server.py — Flask API 서버

실행:
    cd debate_assistant/
    python server.py

엔드포인트:
    GET  /health            서버 상태 확인
    POST /intro             토론 주제 배경 정보 요약
    POST /intro/quiz        배경 요약 기반 퀴즈 생성
    POST /hint/counter      재반박 힌트 (AI 반박 직후)
    POST /hint/rebuttal     반박 힌트   (AI 새 주장 직후)
    POST /summarize         토론 정리 + 피드백
    POST /quiz              퀴즈 생성
    POST /score/turn        턴별 유저 발언 평가
    POST /score/final       전체 턴 통계 집계
    POST /score/reset       세션 초기화
"""

import socket

from flask import Flask, request, jsonify
from flask_cors import CORS

from debate_assistant import DebateAssistant
from agents.intro_agent import IntroAgent
from agents.intro_quiz_agent import IntroQuizAgent
from agents.scoring_agent import ScoringAgent

app = Flask(__name__)
CORS(app)

# ── 싱글턴 에이전트 ───────────────────────────────────────────────
_intro_agent      = IntroAgent()
_intro_quiz_agent = IntroQuizAgent()
_scoring_agents: dict[str, ScoringAgent] = {}


# ── 공통 헬퍼 ────────────────────────────────────────────────────

def _get_da(user_label: str, ai_label: str, news_data: list) -> DebateAssistant:
    return DebateAssistant(evidence_items=news_data, user_label=user_label, ai_label=ai_label)


def _get_scoring_agent(user_label: str, ai_label: str, session_key: str) -> ScoringAgent:
    if session_key not in _scoring_agents:
        _scoring_agents[session_key] = ScoringAgent(user_label=user_label, ai_label=ai_label)
    return _scoring_agents[session_key]


def _parse_history(raw: list) -> list[dict]:
    result = []
    for h in raw:
        role    = h.get("role", "")
        content = h.get("content", "")
        if role in ("user", "ai") and content:
            result.append({"role": role, "content": content})
    return result


def _require(data: dict, *keys) -> str | None:
    """필수 필드 누락 시 에러 메시지 반환. 없으면 None."""
    for key in keys:
        if not data.get(key):
            return f"{key} 필드 필요"
    return None

# server.py에 추가할 엔드포인트

@app.route("/intro/quiz/evaluate", methods=["POST"])
def intro_quiz_evaluate():
    """
    주관식 답변 평가.

    요청:
    {
        "topic":    "토론 주제",
        "summary":  "IntroAgent가 생성한 배경 요약",
        "qa_pairs": [
            {"question": "질문1", "answer": "유저 답변1"},
            {"question": "질문2", "answer": "유저 답변2"}
        ]
    }
    응답:
    {
        "results": [
            {"question": "...", "answer": "...", "score": 1~5, "reason": "..."},
            ...
        ],
        "total_score": 2~10
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "topic", "summary"):
        return jsonify({"error": err}), 400

    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        return jsonify({"error": "qa_pairs 필드 필요"}), 400

    try:
        result = _intro_quiz_agent.evaluate_subjective(
            topic    = data["topic"],
            summary  = data["summary"],
            qa_pairs = qa_pairs,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 헬스체크 ─────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── 인트로 ───────────────────────────────────────────────────────

@app.route("/intro", methods=["POST"])
def intro():
    """
    요청: { "topic": "...", "news_data": [...] }
    응답: { "summary": "...", "search_block": "..." }
    news_data 있으면 Tavily 검색 스킵.
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
    요청: { "topic": "...", "summary": "..." }
    응답: { "quizzes": [ {OX 3개 + 객관식 2개} ] }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "topic", "summary"):
        return jsonify({"error": err}), 400

    try:
        result = _intro_quiz_agent.run(topic=data["topic"], summary=data["summary"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 힌트 ─────────────────────────────────────────────────────────

def _hint_common(mode: str) -> tuple:
    """counter / rebuttal 공통 처리."""
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
    """재반박 힌트 (AI 반박 직후). 응답: { "hint": "..." }"""
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
    """반박 힌트 (AI 새 주장 직후). 응답: { "hint": "..." }"""
    fn, args = _hint_common("rebuttal")
    if fn is None:
        return args
    try:
        history, topic = args
        return jsonify({"hint": fn(history, topic)["hint"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 요약 ─────────────────────────────────────────────────────────

@app.route("/summarize", methods=["POST"])
def summarize():
    """
    요청: { "topic", "user_label", "ai_label", "history", "news_data" }
    응답: { "summary", "logic_feedback", "extra_info" }
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
        result = _get_da(data["user_label"], data["ai_label"], news_data).summarize(history, data["topic"])
        return jsonify({
            "summary":        result["summary"],
            "logic_feedback": result["logic_feedback"],
            "extra_info":     result["extra_info"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 퀴즈 ─────────────────────────────────────────────────────────

@app.route("/quiz", methods=["POST"])
def quiz():
    """
    요청: { "topic", "user_label", "ai_label", "history", "news_data" }
    응답: { "review_quiz": {...}, "weakness_quiz": {...} }
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
        result = _get_da(data["user_label"], data["ai_label"], news_data).quiz(history, data["topic"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 평가 ─────────────────────────────────────────────────────────
# server.py — /score/turn 엔드포인트만 수정 (나머지 동일)

@app.route("/score/turn", methods=["POST"])
def score_turn():
    """
    턴 종료 직후 호출. turn_number는 서버가 자동 계산.

    요청: { "topic", "user_label", "ai_label", "session_key", "history" }
    응답: {
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
        result = agent.score_turn(history=history, topic=data["topic"])  # turn_number 제거
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/score/final", methods=["POST"])
def score_final():
    """
    토론 종료 후 호출. 전체 턴 통계 집계. /score/turn을 먼저 호출해야 함.

    요청: { "session_key": "..." }
    응답: {
        "turns": [...],
        "summary": {
            "specificity": { "avg": float, "trend": "상승"|"하락"|"유지", "scores_per_turn": [...] },
            "causality":   { ... },
            "domain":      { ..., "all_domains": [...] },
            "initiative":  { ... }
        },
        "total_avg": float,
        "overall_comment": "..."
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "session_key"):
        return jsonify({"error": err}), 400

    session_key = data["session_key"]
    if session_key not in _scoring_agents:
        return jsonify({"error": f"'{session_key}' 평가 기록 없음. /score/turn을 먼저 호출하세요."}), 404

    try:
        result = _scoring_agents.pop(session_key).score_final()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/score/reset", methods=["POST"])
def score_reset():
    """
    새 토론 시작 시 기존 세션 초기화.

    요청: { "session_key": "..." }
    응답: { "status": "ok", "message": "..." }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400
    if err := _require(data, "session_key"):
        return jsonify({"error": err}), 400

    session_key = data["session_key"]
    _scoring_agents.pop(session_key, None)
    return jsonify({"status": "ok", "message": f"'{session_key}' 초기화 완료"})


# ── 서버 시작 ────────────────────────────────────────────────────

if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'=' * 50}")
    print(f"  Debate Assistant API 서버 시작")
    print(f"  로컬:          http://127.0.0.1:5000")
    print(f"  같은 네트워크: http://{local_ip}:5000")
    print(f"{'=' * 50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)