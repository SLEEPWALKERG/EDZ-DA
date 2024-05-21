import json
import argparse
import os
from transformers import T5Tokenizer
from tqdm import tqdm
import itertools
import random
random.seed(42)


prompt_value = [
    "belief states: slot={}, value=",
    "belief states: {}=",
    "{} is the slot of ",
    "what is the value of {}?"
]


prompt_slot = [
    "belief states: value={}, slot=",
    "belief states: {}=",
    "{} is the value of ",
    "what is the slot type of {}?"
]


prefix = "answer the question:"


class DSTProcess:
    def __init__(self, args):
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        self.prefix_encoded = self.tokenizer.encode(prefix)
        self.cutoff = args.cutoff

    def process(self, data):
        lst = []
        for sample in tqdm(data):
            history = ';'.join(sample["history"]) + ';' + sample["system"] + ';' + sample["user"]
            history_tokenized = self.tokenizer.encode(history)[-self.cutoff:]
            for s, v in sample["turn_label"].items():
                slot, value = s, v
                if "internet" in slot:
                    if value == "yes":
                        value = "wifi"
                    # elif value == 'no':
                    #     value = "no internet need"
                if "parking" in slot:
                    if value == "yes":
                        value = "parking"
                    # elif value == 'no':
                    #     value = "no parking need"
                prompt = self.tokenizer.encode(prompt_slot[3].format(value))
                prompt_inverse = self.tokenizer.encode(prompt_value[3].format(slot))
                lst.append({
                    "flag": sample["dialogue_idx"] + '-' + str(sample["turn_idx"]),
                    "prompt": self.prefix_encoded + history_tokenized + prompt,
                    "inverse_prompt": self.prefix_encoded + history_tokenized + prompt_inverse,
                    "value_encoded": self.tokenizer.encode(value),
                    "slot_encoded": self.tokenizer.encode(slot),
                    "value": value,
                    "slot": slot,
                })
        return lst


def func_dst(args):
    pre_process = DSTProcess(args)
    with open('../../data_augmentation/{}.json'.format(args.aug_name), encoding='utf-8') as f:
        data = json.load(f)
    lst = pre_process.process(data)
    print(f'dst_sample: {len(lst)}')
    with open('./data/mwz2_1/dst_augmented_{}.json'.format(args.aug_name), 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


class ValProcess:
    def __init__(self, args):
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        self.is_permute = args.is_permute
        self.cutoff = args.val_cutoff
        self.history_flag = "<extra_id_0>"
        self.turn_flag = "<extra_id_1>"
        prefix_get_intend = "get the requests that the user confirmed or mentioned in this turn"
        self.history_flag_id = self.tokenizer.encode(self.history_flag)[0]
        self.prefix_encoded = self.tokenizer.encode(prefix_get_intend)

    def process(self, data):
        lst = []
        cnt = 0
        for sample in tqdm(data):
            values = []
            for s, v in sample["turn_label"].items():
                slot, value = s, v
                if "internet" in slot:
                    if value == "yes":
                        value = "wifi"
                    # elif value == 'no':
                    #     value = "no internet need"
                if "parking" in slot:
                    if value == "yes":
                        value = "parking"
                    # elif value == 'no':
                    #     value = "no parking need"
                values.append(value)
            history = self.history_flag + ';'.join(sample["history"]) + self.turn_flag + sample["system"] + ';' + sample["user"]
            history_tokenized = self.tokenizer.encode(history)
            if len(history_tokenized) > self.cutoff:
                cnt += 1
            history_tokenized = history_tokenized[-self.cutoff:]
            history_tokenized[0] = self.history_flag_id
            if self.is_permute == 1:
                tmp = list(itertools.permutations(values, len(values)))
            else:
                tmp = [values]
            if len(tmp) > 120:
                tmp = random.sample(tmp, 120)
            for each in tmp:
                label = self.tokenizer.encode(" | ".join(each))
                dic = {
                    "flag": sample["dialogue_idx"] + '-' + str(sample["turn_idx"]),
                    "input_id": self.prefix_encoded + history_tokenized,
                    "values": values,
                    "label": label,
                }
                lst.append(dic)
        print(f'val_length_shift: {cnt / len(data)}')
        return lst


def func_val(args):
    pre_process = ValProcess(args)
    with open('../../data_augmentation/{}.json'.format(args.aug_name), encoding='utf-8') as f:
        data = json.load(f)
    lst = pre_process.process(data)
    print(f'val_sample: {len(lst)}')
    if args.is_permute == 1:
        with open('./data/mwz2_1/val_augmented_{}_permuted.json'.format(args.aug_name), 'w', encoding='utf-8') as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
    else:
        with open('./data/mwz2_1/val_augmented_{}.json'.format(args.aug_name), 'w', encoding='utf-8') as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=r't5-small', type=str)
    parser.add_argument("--cutoff", default=480, type=int)
    parser.add_argument('--val_cutoff', default=384, type=int)
    parser.add_argument('--is_permute', default=0, type=int)
    parser.add_argument("--aug_name", default="fine-infer", type=str)
    args = parser.parse_args()
    func_dst(args)
    func_val(args)
