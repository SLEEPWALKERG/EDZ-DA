import json
import random
from tqdm import tqdm
from utils.parallel_gpt_generator import parallel_gpt_generate
random.seed(42)


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


dialogue_history_template = '''## Dialogue History:
{}
'''


turn_utterance_template = '''## Current Turn Utterances:
{}

'''


inferable_information_template = '''## Co-reference:
{}

'''


turn_state_template = '''## Turn State:
{}

'''


instruction = '''## Instruction:
Your task is to modify the current turn utterances to express the co-reference information implicitly rather than explicitly stating the full name of the value. Note that the rest of the values in the current turn state should still be expressed explicitly. The modified utterance should be the paraphrase of the original utterance in the current turn for both the system and the user. Only to modify the information for these co-reference value.

Remember to maintain the meaning of the modified utterance with the original one. 

If the expression of the co-reference value in the original utterance is already in a implicit way, please do not modify the utterances and just copy it.

## Output Format:
{
“description”: <description of the co-reference information in natural language>,
“system”: <modified system utterance of the current turn>,
“user”: <modified user utterance of the current turn>
}
'''


def infer_generation():
    with open('../dialogue_gen_based_on_flow/output/fine-grained.json', encoding='utf-8') as f:
        data = json.load(f)
    inputs = []
    idx_list = []
    for idx, each in enumerate(data):
        tmp = {}
        for k, v in each['turn_label'].items():
            if k in each['infer'].keys():
                tmp[k] = each['infer'][k]
        if len(tmp) == 0:
            continue
        # infer generation
        history = dialogue_history_template.format(construct_history(each['history']))
        turn_utterance = turn_utterance_template.format('System: ' + each['system'] + '\n' + 'User: ' + each['user'])
        co_ref_information = inferable_information_template.format(str(tmp))
        turn_state = turn_state_template.format(str(each['turn_label']))
        prompt = history + turn_utterance + turn_state + co_ref_information + instruction
        # print(prompt)
        idx_list.append(idx)
        inputs.append({
            "gpt_input": prompt,
            "align_data": idx,
        })
    with open('./output/modified_idx.json', 'w', encoding='utf-8') as f:
        json.dump(idx_list, f, ensure_ascii=False, indent=2)
    print(len(inputs))
    results = parallel_gpt_generate(inputs)
    with open('./output/output.json', 'w', encoding='utf-8') as file:
        lst = []
        for result in results:
            if result is None:
                lst.append([{}, ""])
            align_data, gpt_out = result
            lst.append([align_data, gpt_out])
        json.dump(lst, file, ensure_ascii=False, indent=2)


def construct_history(history):
    ans = 'System: How can I assist you today ?\n'
    for idx, each in enumerate(history):
        if idx == 0:
            continue
        if idx % 2 == 0:
            ans += 'System: '
        else:
            ans += 'User: '
        ans += each
        ans += '\n'
    return ans


if __name__ == '__main__':
    infer_generation()
