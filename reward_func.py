import re
import torch
from Levenshtein import ratio as levenshtein_ratio
def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # print(completions) #debug
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matches = [re.match(pattern, content[0]['content'], re.DOTALL) for content in completions]
    # for content in completions:
    #     print('prediction=='+content[0]['content']+'\n\n')
    return torch.tensor([1.0 if match else 0.0 for match in matches], requires_grad=False)
    
def levenshtein_reward_func(completions, solution, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    for completion, sol in zip(completions, solution):
        completion = completion[0]['content']
        if '</think>' in completion:
            t = completion.split('</think>')[-1]    # calculate result distance
            res.append(levenshtein_ratio(t, sol))
        else:
            res.append(0.0)
    # print(res)
    print('\n\n')
    return torch.tensor(res, requires_grad=False)