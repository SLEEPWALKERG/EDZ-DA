import argparse
import json
import os
import re
from copy import deepcopy
from utils.fix_label import fix_general_label_error

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def analyze(path):
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    dic = {}
    acc = 0
    samples = 0
    for each in result:
        samples += 1
        k = False
        if each["generate"] == each["slot"]:
            k = True
        if k:
            acc += 1
            if each["flag"] not in dic:
                dic[each["flag"]] = 1
        else:
            dic[each["flag"]] = 0
    turns = 0
    turn_acc = 0
    for k, v in dic.items():
        turns += 1
        if v == 1:
            turn_acc += 1
    print("Single Level Acc: {:.2f} %".format(acc / samples * 100))
    print("Turn Level Acc: {:.2f} %".format(turn_acc / turns * 100))


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


def compute_jga(dic):
    with open("./mwz2_4/ontology.json", encoding="utf-8") as f:
        ontology = json.load(f)
    slots = get_slot_information(ontology)
    with open("./mwz2_4/test_dials.json", encoding="utf-8") as f:
        data = json.load(f)
    j_acc = 0
    turns = 0
    err_lst = []
    for each in data:
        dialogue_idx = each["dialogue_idx"]
        last_belief_state = {}
        for turn in each["dialogue"]:
            turns += 1
            ground_truth = set(sv["slots"][0][0] + '==' + sv["slots"][0][1] for sv in turn["belief_state"])
            k = dialogue_idx + '-' + str(turn["turn_idx"])
            if k in dic:
                turn_label = dic[k]
                last_belief_state = update_state(last_belief_state, turn_label)
            last_belief_state = fix_general_label_error([{"slots": [[k, v]]} for k, v in last_belief_state.items()],
                                                        False, slots)
            predict = set(s + '==' + v for s, v in last_belief_state.items())
            if predict == ground_truth:
                j_acc += 1
            else:
                err_lst.append({
                    "flag": k,
                    "predict": deepcopy(last_belief_state),
                    "ground_truth": turn["belief_state"],
                    "system": turn["system_transcript"],
                    "user": turn["transcript"],
                })
    print("Joint Goal Acc: {:.4f} %".format(j_acc / turns * 100))
    # with open("./result/err.json", 'w', encoding="utf-8") as f:
    #     json.dump(err_lst, f, ensure_ascii=False, indent=2)


def update_state(state, turn_label):
    for s, v in turn_label.items():
        slot, value = s, fix_time_label(v)
        if value == "none":
            try:
                del state[slot]
            except:
                continue
        else:
            if value == "parking" or value == "wifi":
                value = "yes"
            state[slot] = value
    return state


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
    parser.add_argument('--result_path', default=r'E:\Temp\svag-new\T5DST\output\result-seed-20-ratio-1-augmented-fine-infer-permute.json', type=str)
    args = parser.parse_args()
    # analyze(args.result_path)
    # compute_jga(get_turn_label(args.result_path))
    for file_name in os.listdir(r'../T5DST/output'):
        print(file_name)
        compute_jga(get_turn_label(r'../T5DST/output/{}'.format(file_name)))
