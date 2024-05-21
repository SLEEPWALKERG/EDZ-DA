import json
import argparse
import os
from transformers import T5Tokenizer
from tqdm import tqdm
import itertools
import random
random.seed(42)


class PreProcess:
    def __init__(self, args):
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        self.is_permute = args.is_permute
        self.cutoff = args.cutoff
        self.history_flag = "<extra_id_0>"
        self.turn_flag = "<extra_id_1>"
        prefix_get_intend = "get the requests that the user confirmed or mentioned in this turn"
        self.history_flag_id = self.tokenizer.encode(self.history_flag)[0]
        self.prefix_encoded = self.tokenizer.encode(prefix_get_intend)

    def process(self, stage, data):
        lst = []
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
            history_tokenized = self.tokenizer.encode(history)[-self.cutoff:]
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
        return lst


def func(args):
    pre_process = PreProcess(args)
    seed_list = [10, 20, 48]
    data_ratio_list = [1, 5, 10]
    # if 'flan' not in args.model_path.lower():
    #     lst = pre_process.process("dev", json.load(open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/dev_raw.json')))
    #     with open("./data/dev.json", 'w', encoding='utf-8') as f:
    #         json.dump(lst, f, ensure_ascii=False, indent=2)
    #     lst = pre_process.process("test", json.load(open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')))
    #     with open("./data/test.json", 'w', encoding='utf-8') as f:
    #         json.dump(lst, f, ensure_ascii=False, indent=2)
    # else:
    #     lst = pre_process.process("dev", json.load(
    #         open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/dev_raw.json')))
    #     with open("./data/flan_dev.json", 'w', encoding='utf-8') as f:
    #         json.dump(lst, f, ensure_ascii=False, indent=2)
    #     lst = pre_process.process("test", json.load(
    #         open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')))
    #     with open("./data/flan_test.json", 'w', encoding='utf-8') as f:
    #         json.dump(lst, f, ensure_ascii=False, indent=2)
    for seed in seed_list:
        for data_ratio in data_ratio_list:
            # dir_path = f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}'
            # if os.path.exists(dir_path):
            #     print(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}' + ' already exists.')
            # else:
            #     os.mkdir(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}')
            with open(
                    f'../MWZProcessor/data_processed/mwz2_1/seed-{seed}-ratio-{data_ratio}/train_raw.json') as f:
                data = json.load(f)
            lst = pre_process.process("train", data)
            if 'flan' not in args.model_path.lower():
                if args.is_permute == 1:
                    with open(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/train_permuted.json', 'w', encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)
                else:
                    with open(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/train.json', 'w', encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)
            else:
                if args.is_permute == 1:
                    with open(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/flan_train_permuted.json', 'w', encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)
                else:
                    with open(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/flan_train.json', 'w', encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=r't5-small', type=str)
    parser.add_argument("--is_permute", default=0, type=int)
    parser.add_argument("--cutoff", default=384, type=int)
    args = parser.parse_args()
    func(args)
