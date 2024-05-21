import pytorch_lightning as pl
import json
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    input_id = [torch.tensor(each["input_id"], dtype=torch.long) for each in batch]
    label = [torch.tensor(each["label"], dtype=torch.long) for each in batch]
    flag = [each["flag"] for each in batch]
    values = [each["values"] for each in batch]
    input_ids = pad_sequence(input_id, batch_first=True)
    labels = pad_sequence(label, batch_first=True, padding_value=-100)
    attn_mask = torch.ne(input_ids, 0).long()
    ret = {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "labels": labels,
        "flag": flag,
        "values": values,
    }
    return ret


class MWZDataSet(pl.LightningDataModule):
    def __init__(self, args):
        super(MWZDataSet, self).__init__()
        self.args = args
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.test_batch_size = args.test_batch_size
        self.n_workers = args.n_workers
        self.train = []
        self.dev = []
        self.test = []

    def setup(self, stage):
        if stage == "fit" or stage is None:
            with open("../data/T5_val_gen_data/data/{}/{}.json".format(self.args.mwz_ver, self.args.train_dir), encoding="utf-8") as f:
                self.train = json.load(f)
            if self.args.augmented_data_dir != '':
                with open("../data/augmented_data/data/{}/{}.json".format(self.args.mwz_ver, self.args.augmented_data_dir), encoding="utf-8") as f:
                    self.train.extend(json.load(f))
            with open("../data/T5_val_gen_data/data/{}/{}.json".format(self.args.mwz_ver, self.args.dev_name), encoding="utf-8") as f:
                self.dev = json.load(f)
        if stage == 'test' or stage is None:
            with open("../data/T5_val_gen_data/data/{}/{}.json".format(self.args.mwz_ver, self.args.test_name), encoding="utf-8") as f:
                self.test = json.load(f)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        print(len(self.train))
        return DataLoader(self.train, shuffle=True, batch_size=self.train_batch_size, num_workers=self.n_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev, shuffle=False, batch_size=self.dev_batch_size, num_workers=self.n_workers,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=self.test_batch_size, num_workers=self.n_workers,
                          collate_fn=collate_fn)