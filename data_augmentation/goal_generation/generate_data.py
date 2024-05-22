import json
import random
from tqdm import tqdm
from utils.parallel_gpt_generator import parallel_gpt_generate
random.seed(42)


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def goal_generation():
    with open('../state_construction/output/states_sampled.json', encoding='utf-8') as f:
        bs = json.load(f)
    with open('../templates/template_goal.txt', encoding='utf-8') as f:
        template_goal = f.read()
    # random.shuffle(bs)
    inputs = []
    for state in bs:
        tmp = []
        # goal generation
        prompt_goal = template_goal.format(str(state["belief_state"]))
        inputs.append({
            "gpt_input": prompt_goal,
            "align_data": state,
        })
    results = parallel_gpt_generate(inputs)
    with open('output/output.json', 'w', encoding='utf-8') as file:
        lst = []
        for align_data, gpt_out in results:
            lst.append([align_data, gpt_out])
        json.dump(lst, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    goal_generation()
