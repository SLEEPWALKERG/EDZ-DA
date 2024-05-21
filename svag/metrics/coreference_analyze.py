import json
import re
from copy import deepcopy
from utils.fix_label import fix_general_label_error
import argparse


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def get_turn_label(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dic = {}
    for each in data:
        if each["flag"] not in dic:
            dic[each["flag"]] = {}
        dic[each["flag"]][each["generate"]] = each["value"]
    return dic


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def compute_jga(dic, mwz_ver, remove_blank):
    with open("./{}/ontology.json".format(mwz_ver), encoding="utf-8") as f:
        ontology = json.load(f)
    slots = get_slot_information(ontology)
    with open("./{}/test_dials.json".format(mwz_ver), encoding="utf-8") as f:
        data = json.load(f)
    acc, cnt = 0, 0
    for each in data:
        dialogue_idx = each["dialogue_idx"]
        for turn in each["dialogue"]:
            if 'coreference' not in turn:
                continue
            k = dialogue_idx + '-' + str(turn["turn_idx"])
            if k in dic:
                turn_label = dic[k]
            else:
                turn_label = {}
            turn_label = fix_general_label_error([{"slots": [[k, fix_time_label(v)]]} for k, v in turn_label.items()],
                                                        False, slots)
            for s, v in turn['coreference'].items():
                if remove_blank == 0:
                    if s in turn_label and turn_label[s] == v:
                        acc += 1
                else:
                    if s in turn_label and turn_label[s].replace(' ', '') == v.replace(' ', ''):
                        acc += 1
                cnt += 1
    print(acc)
    print(cnt)
    print("Co-reference Acc: {:.4f} %".format(acc / cnt * 100))


# def update_state(state, turn_label):
#     for s, v in turn_label.items():
#         slot, value = s, fix_time_label(v)
#         if value == "none":
#             try:
#                 del state[slot]
#             except:
#                 continue
#         else:
#             if value == "parking" or value == "wifi":
#                 value = "yes"
#             state[slot] = value
#     return state


def fix_time_label(value):
    x = re.search(r"\d\d\D\d\d", value)
    if x is not None:
        x = x.group()
        biaodian = re.search(r"\D", x).group()
        if biaodian != ":":
            return value.replace(biaodian, ':')
    return additional_normalize(value)


def additional_normalize(value):
    # Some of the mismatch of predicted values and the ground truth ones is because of the normalization
    # inconsistency, such as ("saint john s college", "saint johns college"), ("pizza hut fen ditton", "pizza hut
    # fenditton"), and ("guesthouse", "guest house") ... We do additional normalize to the predicted values to
    # mitigate the influence of such inconsistency.
    if ' s ' in value:
        return value.replace(' s ', 's ')
    if 'guesthouse' in value:
        return value.replace('guesthouse', 'guest house')
    if 'fen ditton' in value:
        return value.replace('fen ditton', 'fenditton')
    if 'pizzahut' in value:
        return value.replace('pizzahut', 'pizza hut')
    if 'steak house' in value:
        return value.replace('steak house', 'steakhouse')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default=r'', type=str)
    parser.add_argument('--remove_blank', default=1, type=int)
    args = parser.parse_args()
    compute_jga(get_turn_label(args.result_path), "mwz2_3", args.remove_blank)
