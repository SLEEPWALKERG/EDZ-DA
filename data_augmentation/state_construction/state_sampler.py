import json
import random
random.seed(42)


def func():
    with open('./output/states.json', encoding='utf-8') as f:
        data = json.load(f)
    x = 550
    num_1 = int(x / 3.5)
    num_2 = int(num_1 * 2)
    num_3 = int(num_1 * 0.5)
    print(num_1, num_2, num_3)
    lst = []
    for i in range(1, 4):
        tmp = data[str(i)]
        print(len(tmp))
        random.shuffle(tmp)
        eval('lst.extend(tmp[:num_{}])'.format(i))
    print(len(lst))
    with open('./output/states_sampled.json', 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    func()
