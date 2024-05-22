import json
from pprint import pprint


def watch():
    with open('./output/output.json', encoding='utf-8') as f:
        data = json.load(f)
    for each in data:
        dial = eval(each[1])
        if len(dial) % 2 == 0:
            continue
            # pprint(dial)
        else:
            pprint(dial)
        print('-' * 200)


def post_process():
    with open('./output/output.json', encoding='utf-8') as f:
        data = json.load(f)
    lst = []
    for each in data:
        dic = {}
        dic['gpt'] = eval(each[1])
        dic['goal'] = each[0]['goal']
        dic['infer'] = each[0]['belief_state']['infer']
        dic['state'] = each[0]['belief_state']['belief_state']
        dic['meta_information'] = each[0]['belief_state']['meta_information']
        lst.append(dic)
    with open('./output/parsed_dialogue_flow.json', 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # watch()
    post_process()
