import json
import random
from utils.parallel_gpt_generator import parallel_gpt_generate
random.seed(42)


instruction = '''## Instruction:
The dialogue state summarize the dialogue between the user and the agent. Your task is to generate a dialogue flow. Each element in the dialogue flow determine the information of a dialogue turn. Each element should contain a description for the dialogue turn and the corresponding turn state. The description is used to guide the subsequent turn utterance generation. The state is a subset of the dialogue state that describe the information the user mentioned or confirmed in the dialogue turn. Note that the dialogue is started with the agent and ended with the user and must follow an alternating pattern between the agent and the user. After a successful booking service, the model should provide a synthetic reference number, which consists of 8 random characters. Sometimes there are a lot of restriction for a domain, so you should construct more dialogue turns to make sure that the user express no more than six new restrictions of domain-slots in one dialogue turn. The value 'dontcare' means that the user does not care about the restriction of that domain-slot and this information must be expressed by the user.

Remember that all the states in the dialogue state should be used and no additional information can be added.

Remember that the agent does not have prior knowledge of the user's goals.

Ensure that the user express no more than six new restriction of domain-slots in one dialogue turn. Be meticulous to confirm that the user does not introduce more than six new restrictions of domain-slots within a single turn. To comply with this, divide constraints across multiple turns when necessary.

## Output Format:
[
{
“description”: < natural language description for dialogue turn1>
“turn state”: {“domain-slot1”: “value”, “domain-slot2”: “value”, ...}
},
{
“description”: < natural language description for dialogue turn2>
“turn state”: {“domain-slot1”: “value”, “domain-slot2”: “value”, ...}
},
....
{
“description”: < natural language description for dialogue turnN>
“turn state”: {“domain-slot1”: “value”, “domain-slot2”: “value”, ...}
}
]
'''


def func():
    with open('../goal_generation/output/parsed_goals.json', encoding='utf-8') as f:
        data = json.load(f)
    # random.shuffle(data)
    inputs = []
    for i in range(len(data)):
        sample = data[i]
        ds = "## Dialogue State:\n" + str(sample['belief_state']['belief_state']) + '\n'
        goal = "## User Goal:\n" + sample['goal'] + '\n'
        information = "## Information for the Agent: \n"
        if 'meta_information' in sample['belief_state']:
            information += str(sample['belief_state']['meta_information']) + '\n'
        else:
            information += '[]\n'
        inputs.append({
            "gpt_input": ds + goal + information + instruction,
            "align_data": sample,
        })
    results = parallel_gpt_generate(inputs)
    with open('output/output.json', 'w', encoding='utf-8') as file:
        lst = []
        for align_data, gpt_out in results:
            lst.append([align_data, gpt_out])
        json.dump(lst, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    func()
