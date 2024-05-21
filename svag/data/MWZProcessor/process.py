import subprocess
import os


def func():
    seed_list = [10, 20, 48]
    data_ratio_list = [1, 5, 10]
    stages = ["train", "dev", "test"]
    for seed in seed_list:
        for data_ratio in data_ratio_list:
            dir_path = f'./data_processed/mwz2_1/seed-{seed}-ratio-{data_ratio}'
            if os.path.exists(dir_path):
                print(f'./data_processed/mwz2_1/seed-{seed}-ratio-{data_ratio}' + ' already exists.')
            else:
                os.mkdir(f'./data_processed/mwz2_1/seed-{seed}-ratio-{data_ratio}')
            for stage in stages:
                os.system(f'python main.py --seed {seed} --data_ratio {data_ratio} --stage {stage}')


if __name__ == '__main__':
    func()
