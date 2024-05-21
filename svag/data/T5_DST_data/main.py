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

    def process(self, stage, data):
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


def func(args):
    pre_process = PreProcess(args)
    seed_list = [10, 20, 48]
    data_ratio_list = [1, 5, 10]
    if 'flan' not in args.model_path.lower():
        lst = pre_process.process("dev", json.load(open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/dev_raw.json')))
        with open("./data/dev.json", 'w', encoding='utf-8') as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
        lst = pre_process.process("test", json.load(open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')))
        with open("./data/test.json", 'w', encoding='utf-8') as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
    else:
        lst = pre_process.process("dev", json.load(
            open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/dev_raw.json')))
        with open("./data/flan_dev.json", 'w', encoding='utf-8') as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
        lst = pre_process.process("test", json.load(
            open(f'../MWZProcessor/data_processed/mwz2_1/seed-10-ratio-1/test_raw.json')))
        with open("./data/flan_test.json", 'w', encoding='utf-8') as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
    for seed in seed_list:
        for data_ratio in data_ratio_list:
            dir_path = f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}'
            if os.path.exists(dir_path):
                print(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}' + ' already exists.')
            else:
                os.mkdir(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}')
            with open(
                    f'../MWZProcessor/data_processed/mwz2_1/seed-{seed}-ratio-{data_ratio}/train_raw.json') as f:
                data = json.load(f)
            lst = pre_process.process("train", data)
            if 'flan' not in args.model_path.lower():
                with open(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/train.json', 'w', encoding='utf-8') as f:
                    json.dump(lst, f, ensure_ascii=False, indent=2)
            else:
                with open(f'./data/mwz2_1/seed-{seed}-ratio-{data_ratio}/flan_train.json', 'w', encoding='utf-8') as f:
                    json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=r't5-small', type=str)
    parser.add_argument("--cutoff", default=480, type=int)
    args = parser.parse_args()
    func(args)
