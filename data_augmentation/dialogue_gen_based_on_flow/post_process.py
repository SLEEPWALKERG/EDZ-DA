import json
from pprint import pprint
from utils.fix_label import fix_general_label_error
from utils.mwz_create_data import normalize


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def get_turn_state(flow, slots):
    s = set()
    lst = []
    for i in range(0, len(flow) - 1, 2):
        turn_state = {}
        sys_state = flow[i]['turn state']
        user_state = flow[i + 1]['turn state']
        for k, v in sys_state.items():
            if k in slots:
                if k + '==' + v not in s:
                    turn_state[k] = v
                    s.add(k + '==' + v)
            else:
                print(k)
        for k, v in user_state.items():
            if k in slots:
                if k + '==' + v not in s:
                    turn_state[k] = v
                    s.add(k + '==' + v)
            else:
                print(k)
        lst.append(turn_state)
    return lst


def watch():
    with open("../../svag/data/MWZProcessor/data/mwz2_1/ontology.json", encoding="utf-8") as f:
        ontology = json.load(f)
    slots = get_slot_information(ontology)
    with open('./output/output.json', encoding='utf-8') as f:
        data = json.load(f)
    for each in data:
        dial = eval(each[1])
        align_data = each[0]
        turn_states = get_turn_state(align_data['gpt'], slots)
        idx = 0
        for i in range(0, len(dial) - 1, 2):
            pprint(dial[i])
            pprint(dial[i + 1])
            pprint(turn_states[idx])
            idx += 1
        print('-' * 100)


def post_process():
    with open("../../svag/MWZProcessor/data/mwz2_1/ontology.json", encoding="utf-8") as f:
        ontology = json.load(f)
    slots = get_slot_information(ontology)
    with open('output/output.json', encoding='utf-8') as f:
        data = json.load(f)
    ans = []
    for each in data:
        history = []
        align_data = each[0]
        dial = eval(each[1])
        if len(dial) != len(align_data['gpt']):
            print('长度不匹配')
            continue
        dial, flow = fix_dial(eval(each[1].lower()), align_data['gpt'])
        if dial is None:
            continue
        turn_states = get_turn_state(flow, slots)
        idx = 0
        for i in range(0, len(dial) - 1, 2):
            turn_label = fix_general_label_error([{"slots": [[k, v]]} for k, v in turn_states[idx].items()], False, slots)
            fix_turn_label(turn_label)
            if idx == 0:
                system = ''
            else:
                system = normalize(replace_quotes(dial[i]['content']), False)
            user = normalize(replace_quotes(dial[i + 1]['content']), False)
            ans.append({
                'dialogue_idx': 'd_id',
                'turn_idx': idx,
                "system": system,
                'user': user,
                "turn_label": turn_label,
                "history": history[:],
                "infer": align_data['infer'],
                "state": align_data['state']
            })
            idx += 1
            history.append(system)
            history.append(user)
    with open('output/parsed.json', 'w', encoding='utf-8') as f:
        json.dump(ans, f, ensure_ascii=False, indent=2)


def fix_dial(dial, turn_labels):
    for each in dial:
        lst = each['description'].lower().split(' ')
        if lst[0] == 'the':
            r = lst[1]
            if r == 'agent' and each['role'] == 'user':
                each['role'] = 'agent'
        else:
            print('-' * 100)
            print(lst)
    flag = 'agent'
    lst = []
    flow = []
    for x, each in enumerate(dial):
        if each['role'] == flag:
            lst.append(each)
            flow.append(turn_labels[x])
            if flag == 'agent':
                flag = 'user'
            else:
                flag = 'agent'
        else:
            if flag == 'user':
                del lst[-1]
                del flow[-1]
                lst.append(each)
                flow.append(turn_labels[x])
            else:
                print('error')
                return None
    return lst, flow


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


def fix_turn_label(turn_label):
    # to_be_deleted = []
    # for k, v in turn_label.items():
    #     if k not in slots:
    #         print(k)
    #         print(v)
    #         to_be_deleted.append(k)
    #     if k + '==' + v in bs:
    #         print(k + '----' + v)
    #         to_be_deleted.append(k)
    # for k in to_be_deleted:
    #     del turn_label[k]
    for k in turn_label:
        if k == 'hotel-type' and turn_label[k] == 'guest house':
            turn_label[k] = 'guesthouse'
        if k == 'hotel-parking' and turn_label[k] == 'free':
            turn_label[k] = 'yes'
        if k == 'hotel-internet' and turn_label[k] == 'wifi':
            turn_label[k] = 'yes'
        turn_label[k] = normalize((turn_label[k]))


def catch_name_error():
    with open('output/parsed.json', encoding='utf-8') as f:
        data = json.load(f)
    for idx, each in enumerate(data):
        domains = set()
        for k, v in each['turn_label'].items():
            domain, slot = k.split('-')
            if slot != 'name':
                domains.add(domain)
        to_be_deleted = []
        for k, v in each['turn_label'].items():
            domain, slot = k.split('-')
            if slot == 'name' and domain in domains:
                tmp = each['system'] + each['user']
                tmp = tmp.replace(' ', '')
                if tmp.find(v.replace(' ', '')) == -1:
                    tmp_next = data[idx + 1]['system'] + data[idx + 1]['user']
                    tmp_next = tmp_next.replace(' ', '')
                    if tmp_next.find(v.replace(' ', '')) != -1:
                        pprint(each)
                        print('-' * 100)
                        pprint(data[idx + 1])
                        check = 1
                        if check == 1:
                            to_be_deleted.append(k)
                            data[idx + 1]['turn_label'][k] = v
                        else:
                            continue
        for deleted in to_be_deleted:
            del each['turn_label'][deleted]
    with open('./output/modified.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def catch_turn_label_error():
    with open('output/modified.json', encoding='utf-8') as f:
        data = json.load(f)
    cnt = 0
    for idx, each in enumerate(data):
        turn_idx = each['turn_idx']
        if turn_idx > 0:
            # former_turn_utterance = data[idx - 1]['system'].replace(' ', '') + data[idx - 1]['user'].replace(' ', '')
            former_turn_utterance = data[idx - 1]['user'].replace(' ', '')
            former_turn_state = data[idx - 1]['turn_label']
        else:
            former_turn_utterance = ''
            former_turn_state = {}
        turn_utterance = each['system'].replace(' ', '') + each['user'].replace(' ', '')
        to_be_deleted = []
        for k, v in each['turn_label'].items():
            domain, slot = k.split('-')
            if 'internet' == slot or 'parking' == slot:
                continue
            if former_turn_utterance.find(v) != -1 and v not in former_turn_state.values():
                flag = True
                if slot in {'book stay', 'book people'}:
                    flag = check_num(v, data[idx - 1]['user'])
                if flag:
                    cnt += 1
                    print(k, v)
                    print('-' * 100)
                    print('current turn:')
                    print(each['system'])
                    print(each['user'])
                    print(each['turn_label'])
                    print('-' * 100)
                    print('former turn:')
                    print(data[idx - 1]['system'])
                    print(data[idx - 1]['user'])
                    print(former_turn_state)
                    print('+' * 200)
                    to_be_deleted.append(k)
                    data[idx - 1]['turn_label'][k] = v
        for ds in to_be_deleted:
            del each['turn_label'][ds]
    # print(cnt)
    # print(len(data))
    # print(cnt / len(data))
    with open('output/fine-grained.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_num(x, s):
    t = s.find(x)
    if t == 0 and s[t + 1] != ' ':
        return False
    if t == len(s) - 1 and s[t - 1] != ' ':
        return False
    if s[t - 1] != ' ' or s[t + 1] != ' ':
        return False
    return True


if __name__ == '__main__':
    # watch()
    post_process()
    catch_name_error()
    catch_turn_label_error()
