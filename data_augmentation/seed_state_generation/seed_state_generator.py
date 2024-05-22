import json
from utils.gpt_generator import GPTGenerator
from tqdm import tqdm
from copy import deepcopy
from itertools import permutations, combinations
from utils.code_extractor import extract_code_from_markdown


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def judgement():
    generator = GPTGenerator()
    template = '''There are five domains that the AI agent supported, and their slots are listed in the following:
1. Hotel: {{area, type, internet, parking, name, book day, price range, star, book stay, book people}}
2. Restaurant: {{area, book day, book people, book time, food, name, price range}}
3. Attraction: {{area, name, type}}
4. Taxi: {{arrive by, departure, destination, leave at}}
5. Train: {{book people, day, departure, destination, leave at}}

Categorical slots and their possible values:
1. Area: centre, east, south, west, north
2. Internet: yes, no
3. Parking: yes, no
4. Price range: cheap, moderate, expensive
5. Star: 1, 2, 3, 4, 5
6. Day: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
7. Hotel-type: hotel, guest house
8. Book people: 1, 2, 3, 4, 5, 6, 7, 8
9. Book stay: 1, 2, 3, 4, 5, 6, 7, 8
10. Arrive by & leave at: time in forms of “xx:xx” such as “13:00”

I will give you some of these domains. Your task is to determine whether these domains can be fused together in a dialogue reasonably. The order of these domains does not matter, and you can determine the order of these domains yourself. Please output it in a json format like {{'is_reasonable': <1 for reasonable and 0 the opposite>, 'explanation': <give some explanation about your judgement>}}.

# domains:
{}.'''
    lst = []
    for i in range(2, 3):
        for each in tqdm(permutations(EXPERIMENT_DOMAINS, i)):
            prompt = template.format(', '.join(each))
            is_success, out = generator.generate([{'role': 'user', 'content': prompt}], 0)
            if is_success:
                lst.append({
                    'domains': list(each),
                    'out': out,
                })
            else:
                lst.append({
                    'domains': list(each),
                    'out': out,
                })
    with open('./output/judgement.json', 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def seed_state_generation():
    generator = GPTGenerator()
    with open('../templates/template_seed_state.txt', encoding='utf-8') as f:
        template_seed_state = f.read()
    with open('./output/judgement.json', encoding='utf-8') as f:
        data = json.load(f)
    lst = []
    for each in data:
        judge = extract_code_from_markdown(each['out'])[0]
        # judge = eval(judge)
        # print(judge)
        try:
            judge = eval(judge)
        except:
            judge = {'is_reasonable': 0}
        if judge['is_reasonable'] == 1:
            message = [{'role': 'assistant', 'content': each['out']}, {'role': 'user', 'content': template_seed_state}]
            is_success, out = generator.generate(message, 0)
            if is_success:
                lst.append({
                    'domains': each['domains'],
                    'out': out,
                })
            else:
                lst.append({
                    'domains': each['domains'],
                    'out': '',
                })
    with open('./output/seed_states.json', 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def ds_reformator(ds, dic):
    d, s = ds.split('-')
    s = s.replace('_', ' ')
    if s == 'arrive by':
        slot = 'arriveby'
    elif s == 'leave at':
        slot = 'leaveat'
    elif s == 'price' or s == 'price range':
        slot = 'pricerange'
    elif s == 'star':
        slot = 'stars'
    elif s == 'number of people':
        slot = 'book people'
    else:
        slot = s
    if d + '-' + slot not in dic[d]:
        if d + '-' + 'book ' + s in dic[d]:
            slot = 'book ' + s
        else:
            print(d)
            print(slot)
            return d, None
    return d, slot


def is_legal(inferred):
    if 'taxi-destination' in inferred and inferred['taxi-destination'] == 'train-destination':
        return False
    if 'taxi-departure' in inferred and inferred['taxi-departure'] == 'train-departure':
        return False
    if 'taxi-leaveat' in inferred and inferred['taxi-leaveat'] == 'train-leaveat':
        return False
    if 'taxi-arriveby' in inferred and inferred['taxi-arriveby'] == 'train-arriveby':
        return False
    if 'train-leaveat' in inferred and inferred['train-leaveat'] == 'taxi-leaveat':
        return False
    if 'train-arriveby' in inferred and inferred['train-arriveby'] == 'taxi-arriveby':
        return False

    if 'taxi-destination' in inferred and inferred['taxi-destination'] == 'train-departure':
        if 'taxi-leaveat' in inferred and inferred['taxi-leaveat'] == 'train-arriveby':
            return False
    if 'taxi-arriveby' in inferred and inferred['taxi-arriveby'] == 'restaurant-book time':
        if 'taxi-departure' in inferred and inferred['taxi-departure'] == 'restaurant-name':
            return False
    return True


def post_process():
    ans = {
        1: [],
        2: [],
        3: [],
        4: [],
    }
    with open('../../svag/data/MWZProcessor/data/mwz2_1/ontology_modified.json', encoding='utf-8') as f:
        ontology = json.load(f)
    dic = {}
    # single domain data
    for ds, v in ontology.items():
        d, s = ds.split('-')
        if d not in dic:
            dic[d] = [ds]
        else:
            dic[d].append(ds)
    for d, dss in dic.items():
        tmp = {}
        for ds in dss:
            tmp[ds] = ''
        ans[1].append(deepcopy(tmp))
    # multi domain data
    with open('./output/seed_states.json', encoding='utf-8') as f:
        seed_data = json.load(f)
    for each in seed_data:
        try:
            parsed = eval(extract_code_from_markdown(each['out'].lower())[0])
        except:
            print(each)
            continue
        occurred = []
        f = False
        for x in parsed:
            tmp = {}
            for ds, v in x.items():
                domain, slot = ds_reformator(ds, dic)
                if slot is None:
                    continue
                if '-' in v:
                    if slot == 'name' or slot == 'book time':
                        continue
                    inferred_from_domain, inferred_from_slot = ds_reformator(v, dic)
                    if inferred_from_slot is None or inferred_from_domain not in each['domains']:
                        continue
                    if slot != inferred_from_slot:
                        if slot in {'departure', 'destination'}:
                            if domain == 'train':
                                value = ''
                            else:
                                if inferred_from_slot == 'name':
                                    value = inferred_from_domain + '-' + inferred_from_slot
                                elif inferred_from_slot == 'area':
                                    assert inferred_from_domain + '-' + 'name' in dic[inferred_from_domain]
                                    value = inferred_from_domain + '-' + 'name'
                                else:
                                    value = inferred_from_domain + '-' + inferred_from_slot
                        else:
                            if slot == 'area' or inferred_from_slot == 'area':
                                value = ''
                            elif slot in {'arriveby', 'leaveat', 'book time'} and inferred_from_slot in {'day', 'book day'}:
                                value = ''
                            elif slot == 'leaveat' and inferred_from_slot == 'book time':
                                value = ''
                            elif inferred_from_slot in {'arriveby', 'leaveat', 'book time'} and slot in {'day', 'book day'}:
                                value = ''
                            else:
                                value = inferred_from_domain + '-' + inferred_from_slot
                    else:
                        if domain != inferred_from_domain:
                            value = inferred_from_domain + '-' + inferred_from_slot
                        else:
                            value = ''
                    if value != '':
                        tmp[domain + '-' + slot] = value
                else:
                    value = ''
            if len(tmp) == 0 and not f:
                continue
                for dd in each['domains']:
                    for ds in dic[dd]:
                        if ds not in tmp:
                            tmp[ds] = ''
                ans[len(each['domains'])].append(deepcopy(tmp))
                f = True
            else:
                if not is_legal(tmp):
                    continue
                else:
                    flag = True
                    t = set([a + '==' + b for a, b in tmp.items()])
                    for occur in occurred:
                        if t == occur:
                            flag = False
                            break
                    if flag:
                        for dd in each['domains']:
                            for ds in dic[dd]:
                                if ds not in tmp:
                                    tmp[ds] = ''
                        ans[len(each['domains'])].append(deepcopy(tmp))
                        occurred.append(t)
    with open('./output/seed_state_parsed.json', 'w', encoding='utf-8') as f:
        json.dump(ans, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    judgement()
    seed_state_generation()
    post_process()
