from medvlm_grpo.rewards import answer_reward_func, clinical_consistency_reward_func, extract_answer, format_reward_func, make_overlong_reward


def test_format_and_answer_rewards():
    completions = [[{"content": "<think>visible finding</think><answer>Yes</answer>"}]]
    format_scores = format_reward_func(completions)
    answer_scores = answer_reward_func(completions, ["yes"])
    print("\n=== format and answer rewards ===")
    print(f"completion={completions[0][0]['content']}")
    print(f"target=yes | format_reward={format_scores[0]} | answer_reward={answer_scores[0]}")
    assert format_scores == [1.0]
    assert answer_scores == [1.0]
    assert extract_answer(completions[0][0]["content"]) == "Yes"
    invalid_score = format_reward_func(["Yes"])
    print(f"completion=Yes | format_reward={invalid_score[0]} (invalid format)")
    assert invalid_score == [0.0]


def test_overlong_soft_penalty():
    reward = make_overlong_reward(10, soft_ratio=0.8)
    scores = reward(["x", "x", "x"], completion_ids=[[1] * 8, [1] * 9, [1] * 10])
    print("\n=== overlong reward ===")
    print(f"max_tokens=10 | soft_limit=8 | lengths=[8, 9, 10] | rewards={scores}")
    assert scores == [0.0, -0.5, -1.0]


def test_clinical_contradictions():
    c = [[{"content": "<think>x</think><answer>left</answer>"}]]
    contradictory = clinical_consistency_reward_func(c, ["right"])
    consistent = clinical_consistency_reward_func(c, ["left lung"])
    negation = clinical_consistency_reward_func(["<answer>no</answer>"], ["yes"])
    chinese = clinical_consistency_reward_func(["<answer>左</answer>"], ["右"])
    print("\n=== clinical consistency reward ===")
    print(f"prediction=left | target=right | reward={contradictory[0]}")
    print(f"prediction=left | target=left lung | reward={consistent[0]}")
    print(f"prediction=no | target=yes | reward={negation[0]}")
    print(f"prediction=Chinese-left | target=Chinese-right | reward={chinese[0]}")
    assert contradictory == [-1.0]
    assert consistent == [0.0]
    assert negation == [-1.0]
    assert chinese == [-1.0]
