import re
from difflib import SequenceMatcher


FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[0].get("content", "")
    return ""


def extract_answer(text):
    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def format_reward_func(completions, **kwargs):
    return [1.0 if FORMAT_RE.match(_text(item)) else 0.0 for item in completions]


def answer_reward_func(completions, solution, **kwargs):
    """Case-insensitive answer-only similarity; reasoning text cannot inflate it."""
    scores = []
    for completion, target in zip(completions, solution):
        prediction = extract_answer(_text(completion)).casefold()
        reference = extract_answer(str(target)).casefold()
        scores.append(SequenceMatcher(None, prediction, reference).ratio())
    return scores


def clinical_consistency_reward_func(completions, solution, **kwargs):
    """Penalize high-risk negation and laterality contradictions."""
    neg = {"no", "not", "none", "without", "absent", "negative"}
    sides = ({"left", "左"}, {"right", "右"})
    scores = []
    for completion, target in zip(completions, solution):
        pred = set(re.findall(r"[\w]+", extract_answer(_text(completion)).casefold()))
        ref = set(re.findall(r"[\w]+", extract_answer(str(target)).casefold()))
        contradiction = bool(pred & neg) != bool(ref & neg)
        contradiction |= any(bool(pred & a) and bool(ref & b) or bool(pred & b) and bool(ref & a) for a, b in [sides])
        scores.append(-1.0 if contradiction else 0.0)
    return scores


# Kept for compatibility with old launch scripts.
levenshtein_reward_func = answer_reward_func


def make_overlong_reward(max_completion_length, soft_ratio=0.8):
    """DAPO-style soft overlong punishment based on generated token counts."""
    soft_limit = int(max_completion_length * soft_ratio)
    window = max(max_completion_length - soft_limit, 1)

    def overlong_reward_func(completions, completion_ids=None, **kwargs):
        if completion_ids is None:
            # Older TRL releases do not expose token ids to reward functions.
            lengths = [len(_text(item).split()) for item in completions]
        else:
            lengths = [len(ids) for ids in completion_ids]
        return [0.0 if n <= soft_limit else -min((n - soft_limit) / window, 1.0) for n in lengths]

    overlong_reward_func.__name__ = "overlong_reward_func"
    return overlong_reward_func
