from medvlm_grpo.smoke_rewards import proxy_combined_reward, tag_presence


def test_proxy_reward_formula_and_format():
    output = "<think>The image supports the finding.</think><answer>yes</answer>"
    result = proxy_combined_reward(output, "yes")
    expected = 0.5 * result["semantic_proxy"] + 0.4 * result["fluency_proxy"] + 0.1
    assert result["tag_presence"] == 1.0
    assert result["combined_reward"] == expected
    assert 0.0 <= result["combined_reward"] <= 1.0


def test_missing_tags_get_zero_format_score():
    assert tag_presence("yes") == 0.0
