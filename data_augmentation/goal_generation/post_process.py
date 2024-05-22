import json


def func():
    with open('./output/output.json', encoding='utf-8') as f:
        data = json.load(f)
    lst = []
    for each in data:
        gpt = each[1]
        align_data = each[0]
        lst.append({
            'belief_state': align_data,
            'goal': gpt,
        })
    with open('./output/parsed_goals.json', 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    func()
