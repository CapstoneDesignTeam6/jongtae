"""
server.py — Flask API 서버

실행:
    cd debate_assistant/
    python server.py

엔드포인트:
    GET  /health            서버 상태 확인
    POST /hint/counter      재반박 힌트 (AI 반박 직후)
    POST /hint/rebuttal     반박 힌트   (AI 새 주장 직후)
    POST /summarize         토론 정리 + 피드백
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from debate_assistant import DebateAssistant

app = Flask(__name__)
CORS(app)


def _get_da(user_label: str, ai_label: str, news_data: list) -> DebateAssistant:
    return DebateAssistant(evidence_items=news_data, user_label=user_label, ai_label=ai_label)


def _parse_history(raw: list) -> list[dict]:
    result = []
    for h in raw:
        role    = h.get("role", "")
        content = h.get("content", "")
        if role in ("user", "ai") and content:
            result.append({"role": role, "content": content})
    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/hint/counter", methods=["POST"])
def hint_counter():
    """
    요청:
    {
        "topic":      "토론 주제",
        "user_label": "이란측",
        "ai_label":   "미국측",
        "history":    [ {"role": "ai"|"user", "content": "..."} ],
        "news_data":  [ ... ]
    }
    응답:
    {
        "hint": "재반박 힌트 텍스트"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400

    topic      = data.get("topic")
    user_label = data.get("user_label")
    ai_label   = data.get("ai_label")
    history    = _parse_history(data.get("history", []))
    news_data  = data.get("news_data", [])

    if not topic:
        return jsonify({"error": "topic 필드 필요"}), 400
    if not user_label or not ai_label:
        return jsonify({"error": "user_label, ai_label 필드 필요"}), 400
    if not history:
        return jsonify({"error": "history 필드 필요"}), 400
    if not news_data:
        return jsonify({"error": "news_data 필드 필요"}), 400

    try:
        result = _get_da(user_label, ai_label, news_data).counter_hint(history, topic)
        return jsonify({"hint": result["hint"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/hint/rebuttal", methods=["POST"])
def hint_rebuttal():
    """
    요청:
    {
        "topic":      "토론 주제",
        "user_label": "이란측",
        "ai_label":   "미국측",
        "history":    [ {"role": "ai"|"user", "content": "..."} ],
        "news_data":  [ ... ]
    }
    응답:
    {
        "hint": "반박 힌트 텍스트"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400

    topic      = data.get("topic")
    user_label = data.get("user_label")
    ai_label   = data.get("ai_label")
    history    = _parse_history(data.get("history", []))
    news_data  = data.get("news_data", [])

    if not topic:
        return jsonify({"error": "topic 필드 필요"}), 400
    if not user_label or not ai_label:
        return jsonify({"error": "user_label, ai_label 필드 필요"}), 400
    if not history:
        return jsonify({"error": "history 필드 필요"}), 400
    if not news_data:
        return jsonify({"error": "news_data 필드 필요"}), 400

    try:
        result = _get_da(user_label, ai_label, news_data).rebuttal_hint(history, topic)
        return jsonify({"hint": result["hint"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    요청:
    {
        "topic":      "토론 주제",
        "user_label": "이란측",
        "ai_label":   "미국측",
        "turns":      3,
        "history":    [ {"role": "ai"|"user", "content": "..."} ],
        "news_data":  [ ... ]
    }
    응답:
    {
        "summary":        "토론 요약",
        "issues":         "쟁점 분석",
        "logic_feedback": "논리 피드백 + 보완 정보",
        "extra_info":     "추가 사례"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400

    topic      = data.get("topic")
    user_label = data.get("user_label")
    ai_label   = data.get("ai_label")
    turns      = data.get("turns", 1)
    history    = _parse_history(data.get("history", []))
    news_data  = data.get("news_data", [])

    if not topic:
        return jsonify({"error": "topic 필드 필요"}), 400
    if not user_label or not ai_label:
        return jsonify({"error": "user_label, ai_label 필드 필요"}), 400
    if not history:
        return jsonify({"error": "history 필드 필요"}), 400
    if not news_data:
        return jsonify({"error": "news_data 필드 필요"}), 400

    try:
        result = _get_da(user_label, ai_label, news_data).summarize(history, topic, turns=turns)
        return jsonify({
            "summary":        result["summary"],
            "logic_feedback": result["logic_feedback"],
            "extra_info":     result["extra_info"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/quiz", methods=["POST"])
def quiz():
    """
    요청:
    {
        "topic":      "토론 주제",
        "user_label": "이란측",
        "ai_label":   "미국측",
        "history":    [ {"role": "ai"|"user", "content": "..."} ],
        "news_data":  [ ... ]
    }
    응답:
    {
        "review_quiz": {
            "question": "...",
            "options":  ["1번", "2번", "3번", "4번"],
            "answer":   2,
            "explanation": "...",
            "option_explanations": {"1": "...", "2": "...", "3": "...", "4": "..."}
        },
        "weakness_quiz": { ... }
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body 필요"}), 400

    topic      = data.get("topic")
    user_label = data.get("user_label")
    ai_label   = data.get("ai_label")
    history    = _parse_history(data.get("history", []))
    news_data  = data.get("news_data", [])

    if not topic:
        return jsonify({"error": "topic 필드 필요"}), 500
    if not user_label or not ai_label:
        return jsonify({"error": "user_label, ai_label 필드 필요"}), 600
    if not history:
        return jsonify({"error": "history 필드 필요"}), 700
    if not news_data:
        return jsonify({"error": "news_data 필드 필요"}), 800

    try:
        result = _get_da(user_label, ai_label, news_data).quiz(history, topic)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  Debate Assistant API 서버 시작")
    print(f"  로컬:          http://127.0.0.1:5000")
    print(f"  같은 네트워크: http://{local_ip}:5000")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)