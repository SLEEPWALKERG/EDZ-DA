import json
from pprint import pprint
from utils.mwz_create_data import normalize


def watch():
    with open('./output/output.json', encoding='utf-8') as f:
        data = json.load(f)
    for each in data:
        modified = eval(each[1])
        origin = each[0]
        pprint(origin['infer'])
        print("Original Utterances:")
        print(origin['system'])
        print(origin['user'])
        pprint(modified)
        print('-' * 200)


def post_process():
    with open('../dialogue_gen_based_on_flow/output/fine-grained.json', encoding='utf-8') as f:
        origin = json.load(f)
    with open('./output/output.json', encoding='utf-8') as f:
        modified = json.load(f)
    dic = {x[0]: x[1] for x in modified}
    cnt = 0
    for idx, each in enumerate(origin):
        if idx in dic:
            m = eval(dic[idx])
            if len(m) == 0:
                continue
            cnt += 1
            each['system'] = normalize(replace_quotes(m['system']), False)
            each['user'] = normalize(replace_quotes(m['user']), False)
    print(cnt)
    with open('./output/add_infer.json', 'w', encoding='utf-8') as f:
        json.dump(origin, f, ensure_ascii=False, indent=2)


def replace_quotes(input_str):
    result_str = ''
    for i in range(len(input_str)):
        if input_str[i] == "'":
            if ((i > 0 and input_str[i - 1] == ' ') and (i < len(input_str) - 1 and input_str[i + 1] != ' ')) or ((i > 0 and input_str[i - 1] != ' ') and (i < len(input_str) - 1 and input_str[i + 1] == ' ')):
                result_str += ''
            else:
                result_str += input_str[i]
        else:
            result_str += input_str[i]
    return result_str


if __name__ == '__main__':
    # watch()
    post_process()
