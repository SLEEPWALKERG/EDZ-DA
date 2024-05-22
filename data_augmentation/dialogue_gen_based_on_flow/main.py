import json
import random
from utils.parallel_gpt_generator import parallel_gpt_generate
random.seed(42)


instruction = '''## Instruction:
Based on the dialogue flow, please generate a dialogue between a agent and a user. You should include information provided in the description and the turn state while construct a dialogue turn. The synthetic reference number should be a string consists of 8 random character. Remember no turn in the dialogue flow can be omitted and no additional dialogue turn can be added. The generated dialogue turns must correspond one-to-one with the provided dialogue flow of, neither more nor less.

Remember no turns in the dialogue flow can be omitted and no additional dialogue turn can be added.

## Output Format:
[
{
"role": "agent",
"description": <a copy of the corresponding description>,
"content": <the agent utterance according to the description and the turn state>
},
{
"role": "user",
"description": <a copy of the corresponding description>,
"content": <the user utterance according to the description and the turn state>
}
]
'''


def func():
    with open('../dialogue_flow_generation/output/parsed_dialogue_flow.json', encoding='utf-8') as f:
        data = json.load(f)
    inputs = []
    for each in data:
        information = "## Information for the Agent: \n\n"
        if 'meta_information' in each:
            information += str(each['meta_information']) + '\n\n'
        else:
            information += '[]\n'
        flow = each['gpt']
        flow[0]['description'] = "The agent greets the user and asks what they can assist them with today."
        dialogue_flow = '## Dialogue Flow: \n' + str(flow) + '\n\n'
        # print(information + dialogue_flow + instruction)
        inputs.append({
            "gpt_input": information + dialogue_flow + instruction,
            "align_data": each,
        })
    results = parallel_gpt_generate(inputs)
    with open('output/output.json', 'w', encoding='utf-8') as file:
        lst = []
        for each in results:
            if each is None:
                continue
            align_data, gpt_out = each
            lst.append([align_data, gpt_out])
        json.dump(lst, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    func()
