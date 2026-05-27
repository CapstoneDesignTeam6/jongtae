"""
agents/intro_quiz_agent.py

사전 퀴즈 생성 에이전트 — Ollama 단일 백엔드
topic + summary만 넣으면 어떤 주제든 동작 (범용)
"""

import json
import re

GENERATE_TOKENS = 2000
MAX_RETRIES = 2


def _call_llm(prompt: str, max_tokens: int = GENERATE_TOKENS) -> str:
    try:
        from agents.llm import call_ollama
        return call_ollama(prompt, "", num_predict=max_tokens)
    except Exception as e:
        return f"[ERROR] Ollama: {e}"


# ──────────────────────────────────────────────────────────────────
# 퀴즈 유형 정의 (범용)
# ──────────────────────────────────────────────────────────────────

QUIZ_TYPES: dict[str, dict] = {
    "missing_variable": {
        "name": "누락 변수 감지",
        "measure": "부분 사실 하나로 전체를 단정할 때 빠진 핵심 변수를 찾는 능력",
        "question_template": (
            "'{partial_fact}'라는 사실만으로 '{conclusion}'을 결론 내릴 때, "
            "논증이 성립하기 위해 반드시 비교·고려해야 하지만 빠진 핵심 변수는?"
        ),
        "correct_criteria": (
            "이 결론이 성립하려면 반드시 함께 비교해야 할 변수. "
            "이걸 빠뜨리면 비교 자체가 무의미해진다."
        ),
        "distractor_criteria": (
            "각 오답은 다음 중 하나여야 한다:\n"
            "- 결론을 강화하지만 '빠진 변수'가 아닌 것\n"
            "- 관련 있어 보이지만 이 논증과 다른 차원\n"
            "- 주제 내 실제 변수지만 이 특정 비교에서는 부차적\n"
            "모든 오답은 실제 전문가나 해당 분야 종사자가 진지하게 고민할 만해야 한다."
        ),
    },
    "overgeneralization": {
        "name": "일반화 범위 판단",
        "measure": "하나의 사례에서 끌어낼 수 있는 결론의 적절한 범위를 판단하는 능력",
        "question_template": (
            "다음 사례를 보고 가장 적절한 결론을 고르시오: '{case}'"
        ),
        "correct_criteria": (
            "사례 하나가 직접 말해주는 것만 담은 결론. "
            "범위를 벗어나지 않으면서 너무 좁지도 넓지도 않다."
        ),
        "distractor_criteria": (
            "각 오답은 다음 중 하나여야 한다:\n"
            "- 사례를 훨씬 넓은 범주로 과도하게 일반화\n"
            "- 사례의 의미를 너무 좁게 해석하거나 전면 부정\n"
            "- 그럴듯하게 들리지만 사례와 논리 연결이 없는 비약\n"
            "모든 오답은 표면적으로 '맞는 말 같아' 보여야 한다."
        ),
    },
    "counterargument": {
        "name": "반론 구성력",
        "measure": "주장의 숨은 전제를 직접 흔드는 반론을 찾는 능력",
        "correct_criteria": (
            "주장이 성립하려면 반드시 참이어야 하는 숨은 전제를 직접 부정하거나 "
            "성립하지 않을 수 있음을 보여주는 반론. "
            "이 반론이 받아들여지면 주장의 핵심 논리가 무너진다."
        ),
        "distractor_criteria": (
            "각 오답은 반드시 서로 다른 논리 차원이어야 한다:\n"
            "- 주장의 결론 방향을 강화하거나 보완하는 것 (반론이 아닌 지지)\n"
            "- 주장의 전제는 그대로 두고 부수적 수치나 사실만 다투는 것\n"
            "- 주제와 관련 있지만 이 주장의 핵심 논점과 다른 차원을 건드리는 것\n"
            "★ 4개 선지 모두 서로 다른 논리 공격 방향이어야 한다. 같은 말을 다르게 표현한 선지는 금지.\n"
            "★ 선지는 짧고 명확하게: 한 문장 안에 핵심 논리만 담을 것. 조건절을 중첩하거나 길게 늘이지 말 것."
        ),
    },
}

DEFAULT_TYPES = list(QUIZ_TYPES.keys())
MAX_RETRIES = 2
GENERATE_TOKENS = 2000


# ──────────────────────────────────────────────────────────────────
# 메인 에이전트
# ──────────────────────────────────────────────────────────────────

class IntroQuizAgent:

    def run(self, topic: str, summary: str, types: list[str] | None = None) -> dict:
        selected = types if types is not None else list(DEFAULT_TYPES)
        _validate_types(selected)

        print(f"\n[IntroQuizAgent] 주제: {topic}")
        print(f"  유형: {[QUIZ_TYPES[t]['name'] for t in selected]}")

        quizzes = []
        for qtype in selected:
            quiz = self._make_one(topic, summary, qtype)
            if quiz:
                quizzes.append(quiz)
                print(f"  [{QUIZ_TYPES[qtype]['name']}] 완료")
            else:
                print(f"  [{QUIZ_TYPES[qtype]['name']}] 실패 — 건너뜀")

        print(f"  최종 퀴즈: {len(quizzes)}개")
        return {"quizzes": quizzes, "selected_types": selected}

    def evaluate(self, quizzes: list[dict], user_answers: list[int]) -> dict:
        results = []
        for quiz, user_idx in zip(quizzes, user_answers):
            correct_idx = quiz.get("correct_index", -1)
            is_correct = user_idx == correct_idx
            results.append({
                "quiz_type": quiz.get("quiz_type"),
                "question": quiz.get("question"),
                "user_index": user_idx,
                "correct_index": correct_idx,
                "correct": is_correct,
                "explanation": quiz.get("explanation"),
            })
            mark = "✓" if is_correct else "✗"
            name = QUIZ_TYPES.get(quiz.get("quiz_type", ""), {}).get("name", "?")
            print(f"  [{name}] {mark}  유저:{user_idx} / 정답:{correct_idx}")

        return {
            "results": results,
            "total_score": sum(1 for r in results if r["correct"]),
            "detail": {r["quiz_type"]: r["correct"] for r in results},
        }

    # ── 문항 1개 생성 (단일 LLM 호출) ──────────────────────────────

    def _make_one(self, topic: str, summary: str, qtype: str) -> dict | None:
        meta = QUIZ_TYPES[qtype]

        for attempt in range(1, MAX_RETRIES + 2):
            prompt = _build_prompt(topic, summary, qtype, meta)
            raw = _call_llm(prompt, GENERATE_TOKENS)

            if not raw or "[ERROR]" in raw:
                print(f"    [{meta['name']}] 시도 {attempt} LLM 오류")
                continue

            quiz = _parse_single_quiz(raw, qtype)
            if quiz:
                return quiz

            print(f"    [{meta['name']}] 시도 {attempt} 파싱 실패, 재시도")

        return None


# ──────────────────────────────────────────────────────────────────
# 프롬프트 빌더 (핵심: 단일 호출로 고품질 출력)
# ──────────────────────────────────────────────────────────────────

def _build_prompt(topic: str, summary: str, qtype: str, meta: dict) -> str:
    counterargument_extra = (
        "[반론 구성력 유형 추가 규칙]\n"
        "- 선지는 한 문장, 간결하게. 조건절 중첩 금지\n"
        "- 현실 팩트 단정 선지 금지 — 논리 구조 공격이 목적\n"
        "- 4개 선지가 각각 다른 논리 차원을 공격해야 함\n"
        "- 선지 시작 표현 금지: '만약 ~', 'If ~' 로 시작하는 선지 절대 금지\n"
        "- 선지 끝 표현 금지: '~인가?', '~무엇인가?' 로 끝나는 선지 절대 금지\n"
        "- 선지는 반드시 '...다.' 또는 '...된다.' 로 끝나는 평서형\n"
    ) if qtype == "counterargument" else ""

    lines = [
        "당신은 비판적 사고 평가 전문가입니다.",
        "아래 토론 주제와 배경 요약을 바탕으로, 지정된 유형의 4지선다 퀴즈를 하나 만드세요.",
        "",
        "토론 주제: " + topic,
        "배경 요약: " + summary,
        "퀴즈 유형: " + meta["name"] + " - " + meta["measure"],
        "",
        "[질문]",
        "- 주제의 핵심 대상(국가명, 인물명, 개념 등)을 반드시 포함할 것",
        "- 단순 암기가 아닌 추론을 요구해야 함",
        "- 한 문장, 명확하게",
        "",
        "[정답 기준]",
        meta["correct_criteria"],
        "",
        "[오답 설계 - 가장 중요]",
        meta["distractor_criteria"],
        "",
        "[핵심 요구사항]",
        "- 4개 선지 중 어느 것도 '이건 아닌데?' 싶으면 안 된다",
        "- 처음 읽을 때 4개 모두 그럴듯해야 한다",
        "- 정답은 추론 끝에 '아, 이게 맞구나' 하고 납득되어야 한다",
        "- 오답은 '이것도 틀린 건 아닌데...' 싶지만 결정적으로 부족한 것",
    ]
    if counterargument_extra:
        lines += ["", counterargument_extra]

    choice_format = "- 선지: ...다. 또는 ...된다. 로 끝나는 평서형. 한 문장, 간결하게."
    lines += [
        "",
        "[형식]",
        choice_format,
        "- correct_index: choices 배열 0부터 시작 (정답 위치를 매번 다르게 섞을 것)",
        "- explanation: 정확히 4문장. 1234 기호 대신 (1)(2)(3)(4) 사용.",
        "  정답 문장 예시: (1)은 ~~이기 때문에 정답이다.",
        "  오답 문장 예시: (2)는 ~~처럼 보이지만, ~~점에서 정답보다 덜 핵심적이라 오답이다.",
        "",
        "JSON만 출력. 백틱/마크다운 금지. 다른 말 금지.",
        "",
        '{',
        '  "quiz_type": "' + qtype + '",',
        '  "type": "reasoning",',
        '  "question": "질문 텍스트?",',
        '  "choices": ["선지A다.", "선지B다.", "선지C다.", "선지D다."],',
        '  "correct_index": 0,',
        '  "explanation": "(1)은 ... (2)는 ... (3)은 ... (4)는 ..."',
        '}',
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# 파싱 & 정규화
# ──────────────────────────────────────────────────────────────────

def _validate_types(types: list[str]) -> None:
    invalid = [t for t in types if t not in QUIZ_TYPES]
    if invalid:
        raise ValueError(f"알 수 없는 유형: {invalid}. 가능: {list(QUIZ_TYPES)}")


def _parse_single_quiz(raw: str, qtype: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    obj = _find_balanced_object(cleaned) or _find_balanced_object(raw)
    if not obj:
        return None

    try:
        item = json.loads(obj)
    except Exception:
        return None

    if not isinstance(item, dict):
        return None

    choices = None
    for key in ("choices", "options", "answers"):
        val = item.get(key)
        if isinstance(val, list) and len(val) == 4:
            choices = [str(c).strip() for c in val]
            break
    if not choices:
        return None

    question = item.get("question", "").strip()
    explanation = item.get("explanation", "").strip()
    if not question or not explanation:
        return None

    correct_idx = _parse_correct_index(item, choices)
    if correct_idx is None:
        return None

    correct_idx = _sync_correct_index(correct_idx, explanation)
    choices = _normalize_choices(choices)

    return {
        "quiz_type": qtype,
        "type": "reasoning",
        "question": question,
        "choices": choices,
        "correct_index": correct_idx,
        "explanation": explanation,
    }


def _parse_correct_index(item: dict, choices: list[str]) -> int | None:
    raw = item.get("correct_index", item.get("answer_index"))
    try:
        idx = int(raw)
        if 0 <= idx <= 3:
            return idx
    except (TypeError, ValueError):
        pass
    ans_text = item.get("correct_answer") or item.get("answer")
    if isinstance(ans_text, str):
        for i, c in enumerate(choices):
            if c.strip() == ans_text.strip():
                return i
    return None


def _sync_correct_index(correct_idx: int, explanation: str) -> int:
    # 원형 기호 방식: ①②③④
    label_map = {"①": 0, "②": 1, "③": 2, "④": 3}
    hits = re.findall(r"([①②③④])[^。.]*?정답|정답[^。.]*?([①②③④])", explanation)
    indices = set()
    for a, b in hits:
        label = a or b
        if label in label_map:
            indices.add(label_map[label])
    # 괄호 숫자 방식: (1)(2)(3)(4)
    if not indices:
        paren_map = {"(1)": 0, "(2)": 1, "(3)": 2, "(4)": 3}
        hits2 = re.findall(r"(\(\d\))[^.]*?정답|정답[^.]*?(\(\d\))", explanation)
        for a, b in hits2:
            label = a or b
            if label in paren_map:
                indices.add(paren_map[label])
    if len(indices) == 1:
        synced = next(iter(indices))
        if synced != correct_idx:
            print(f"    [보정] correct_index {correct_idx} -> {synced}")
        return synced
    return correct_idx


def _normalize_choices(choices: list[str]) -> list[str]:
    """선지를 올바른 종결형으로 정규화한다.

    - 의문문(? 로 끝나는 것)은 그대로 유지: "~인가?" -> "~인가?"
    - 평서형은 "...다." 로 통일
    - "~인가?다." 같은 이중 종결 제거
    """
    result = []
    for c in choices:
        c = c.strip()
        # 이중 종결 제거: "~인가?다." / "~된다.다." 등
        c = c.rstrip(".")
        while c.endswith("다") and (c.endswith("인가?다") or c.endswith("된다다") or c.endswith("다다")):
            c = c[:-1]
        # 의문문은 그대로
        if c.endswith("?"):
            result.append(c)
        # 이미 "다"로 끝나면 마침표만 추가
        elif c.endswith("다"):
            result.append(c + ".")
        # 그 외는 "다." 추가
        else:
            result.append(c + "다.")
    return result


def _find_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = in_str = escape = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            escape = (ch == "\\") if not escape else 0
            if not escape and ch == '"':
                in_str = 0
        elif ch == '"':
            in_str = 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# 하위 호환
QUIZ_TYPE_META = QUIZ_TYPES
ALL_TYPES = list(QUIZ_TYPES.keys())
