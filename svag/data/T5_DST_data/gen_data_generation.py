import json
import argparse
import os
from transformers import T5Tokenizer
from tqdm import tqdm


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


class PreProcess:
    def __init__(self, args):
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        self.prefix_encoded = self.tokenizer.encode(prefix)
        self.cutoff = args.cutoff

    def process(self, stage, data, values):
        lst = []
        for sample in tqdm(data):
            flag = sample["dialogue_idx"] + '-' + str(sample["turn_idx"])
            if flag in values:
                for v in values[flag]:
                    history = ';'.join(sample["history"]) + ';' + sample["system"] + ';' + sample["user"]
                    history_tokenized = self.tokenizer.encode(history)[-self.cutoff:]
                    prompt = self.tokenizer.encode(prompt_slot[3].format(v))
                    lst.append({
                        "flag": flag,
                        "prompt": self.prefix_encoded + history_tokenized + prompt,
                        "inverse_prompt": [0],
                        "value_encoded": [0],
                        "slot_encoded": [0],
                        "value": v,
                        "slot": "",
                    })
        return lst


def func(args):
    pre_process = PreProcess(args)
    dic = {}
    with open(args.value_path, encoding='utf-8') as f:
        val = json.load(f)
    for each in val:
        dic[each["flag"]] = list(set(each["predict"].split(' | ')) - {' ', ''})
    lst = pre_process.process("test", json.load(open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')), values=dic)
    with open(f"./data/mwz2_1/{args.output_name}.json", 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def func_batch(args):
    pre_process = PreProcess(args)
    seed_list = [10, 20, 48]
    data_ratio_list = [1, 5, 10]
    for seed in seed_list:
        for data_ratio in data_ratio_list:
            # if not (seed == 20 and data_ratio == 10):
            #     continue
            dic = {}
            if args.is_permute == 0:
                if 'flan' not in args.model_path.lower():
                    with open(f'../../T5_val_gen/output/seed-{seed}-ratio-{data_ratio}.json', encoding='utf-8') as f:
                        val = json.load(f)
                    for each in val:
                        dic[each["flag"]] = list(set(each["predict"].split(' | ')) - {' ', ''})
                    lst = pre_process.process("test", json.load(
                        open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')), values=dic)
                    with open(f"./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/gen_data.json", 'w',
                              encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)
                else:
                    with open(f'../../T5_val_gen/output/seed-{seed}-ratio-{data_ratio}-flan.json',
                              encoding='utf-8') as f:
                        val = json.load(f)
                    for each in val:
                        dic[each["flag"]] = list(set(each["predict"].split(' | ')) - {' ', ''})
                    lst = pre_process.process("test", json.load(
                        open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')), values=dic)
                    with open(f"./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/gen_data_flan.json", 'w',
                              encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)
            else:
                if 'flan' not in args.model_path.lower():
                    with open(f'../../T5_val_gen/output/seed-{seed}-ratio-{data_ratio}-permuted.json',
                              encoding='utf-8') as f:
                        val = json.load(f)
                    for each in val:
                        dic[each["flag"]] = list(set(each["predict"].split(' | ')) - {' ', ''})
                    lst = pre_process.process("test", json.load(
                        open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')), values=dic)
                    with open(f"./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/gen_data_permute.json", 'w',
                              encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)
                else:
                    with open(f'../../T5_val_gen/output/seed-{seed}-ratio-{data_ratio}-flan-permuted.json',
                              encoding='utf-8') as f:
                        val = json.load(f)
                    for each in val:
                        dic[each["flag"]] = list(set(each["predict"].split(' | ')) - {' ', ''})
                    lst = pre_process.process("test", json.load(
                        open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')), values=dic)
                    with open(f"./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/gen_data_permute_flan.json", 'w',
                              encoding='utf-8') as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=r't5-large', type=str)
    parser.add_argument("--cutoff", default=480, type=int)
    parser.add_argument('--is_permute', default=0, type=int)
    parser.add_argument('--value_path', default='', type=str)
    parser.add_argument('--output_name', default='', type=str)
    args = parser.parse_args()
    if args.value_path == '':
        func_batch(args)
    else:
        func(args)

