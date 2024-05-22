import json
import random
from copy import deepcopy
from utils.get_domains import count_domains
from utils.determine_fill_order import determine_fill_order
random.seed(42)


class MWZDatabase:
    def __init__(self):
        # hotel
        with open('../db/hotel_db.json', encoding='utf-8') as f:
            self.hotel = json.load(f)
        # attraction
        with open('../db/attraction_db.json', encoding='utf-8') as f:
            self.attraction = json.load(f)
        # restaurant
        with open('../db/restaurant_db.json', encoding='utf-8') as f:
            self.restaurant = json.load(f)
        # train
        with open('../db/train_db.json', encoding='utf-8') as f:
            self.train = json.load(f)
        self.hotel_is_used = [0] * len(self.hotel)
        self.attraction_is_used = [0] * len(self.attraction)
        self.restaurant_is_used = [0] * len(self.restaurant)

    def get_restaurant(self, restriction=None):
        if len(restriction) == 0:
            i = 0
            while i < len(self.restaurant_is_used):
                if self.restaurant_is_used[i] == 0:
                    break
                i += 1
            if i == len(self.restaurant_is_used):
                return random.choice(self.restaurant)
            else:
                self.restaurant_is_used[i] = 1
                return self.restaurant[i]
        else:
            res = set(k + '==' + v for k, v in restriction.items())
            flag = False
            i = 0
            lst = []
            while i < len(self.restaurant_is_used):
                tmp = set(k + '==' + str(v) for k, v in self.restaurant[i].items())
                if len(res - tmp) == 0:
                    if self.restaurant_is_used[i] == 0:
                        flag = True
                        break
                    else:
                        lst.append(self.restaurant[i])
                i += 1
            if flag:
                self.restaurant_is_used[i] = 1
                return self.restaurant[i]
            else:
                if len(lst) == 0:
                    print('err')
                    print(restriction)
                return random.choice(lst)

    def get_attraction(self, restriction=None):
        if len(restriction) == 0:
            i = 0
            while i < len(self.attraction_is_used):
                if self.attraction_is_used[i] == 0:
                    break
                i += 1
            if i == len(self.attraction_is_used):
                return random.choice(self.attraction)
            else:
                self.attraction_is_used[i] = 1
                return self.attraction[i]
        else:
            res = set(k + '==' + v for k, v in restriction.items())
            flag = False
            i = 0
            lst = []
            while i < len(self.attraction_is_used):
                tmp = set(k + '==' + str(v) for k, v in self.attraction[i].items())
                if len(res - tmp) == 0:
                    if self.attraction_is_used[i] == 0:
                        flag = True
                        break
                    else:
                        lst.append(self.attraction[i])
                i += 1
            if flag:
                self.attraction_is_used[i] = 1
                return self.attraction[i]
            else:
                if len(lst) == 0:
                    print('err')
                    print(restriction)
                return random.choice(lst)

    def get_hotel(self, restriction=None):
        if len(restriction) == 0:
            i = 0
            while i < len(self.hotel_is_used):
                if self.hotel_is_used[i] == 0:
                    break
                i += 1
            if i == len(self.hotel_is_used):
                return random.choice(self.hotel)
            else:
                self.hotel_is_used[i] = 1
                return self.hotel[i]
        else:
            res = set(k + '==' + v for k, v in restriction.items())
            flag = False
            i = 0
            lst = []
            while i < len(self.hotel_is_used):
                tmp = set(k + '==' + str(v) for k, v in self.hotel[i].items())
                if len(res - tmp) == 0:
                    if self.hotel_is_used[i] == 0:
                        flag = True
                        break
                    else:
                        lst.append(self.hotel[i])
                i += 1
            if flag:
                self.hotel_is_used[i] = 1
                return self.hotel[i]
            else:
                if len(lst) == 0:
                    print('err')
                    print(restriction)
                return random.choice(lst)


class TimeLabeler:
    def __init__(self):
        pass


class ValueMarker:
    def __init__(self):
        with open('../../svag/data/MWZProcessor/data/mwz2_1/ontology_modified.json', encoding='utf-8') as f:
            self.ontology = json.load(f)
        self.dic = {}
        self.dic_idx = {}
        for ds, v in self.ontology.items():
            tmp = []
            for each in v:
                if '|' in each:
                    continue
                    for vv in each.split('|'):
                        tmp.append(vv)
                else:
                    tmp.append(each)
            tmp = list(set(tmp) - {'dontcare'})
            random.shuffle(tmp)
            self.dic[ds] = tmp[:]
            self.dic_idx[ds] = 0
        names = set(self.dic['attraction-name'] + self.dic['hotel-name'] + self.dic['restaurant-name'])
        for ds in {'train-destination', "train-departure", "taxi-destination", "taxi-departure"}:
            tmp = set(self.dic[ds][:])
            self.dic[ds] = list(tmp - names)

    def get(self, ds):
        ret = self.dic[ds][self.dic_idx[ds] % len(self.dic[ds])]
        self.dic_idx[ds] = (self.dic_idx[ds] + 1)
        return ret

    def current_status(self):
        for ds, v in self.dic.items():
            print(f'{ds}: {self.dic_idx[ds] / len(v)}')

    def get_schema(self):
        dic = {}
        for each in self.ontology.keys():
            domain, slot = each.split('-')
            if domain not in dic:
                dic[domain] = set()
            dic[domain].add(slot)
        return dic


def func():
    value_marker = ValueMarker()
    schema = value_marker.get_schema()
    database = MWZDatabase()
    with open('../seed_state_generation/output/seed_state_parsed.json', encoding='utf-8') as f:
        seed_states = json.load(f)
    ans_dic = {
        '1': [],
        '2': [],
        '3': [],
    }
    num2aug = {'1': 50, '2': 20, '3': 20}
    for num_domains in ans_dic.keys():
        for seed_state in seed_states[num_domains]:
            bs, inferred = co_ref_detection(seed_state)
            if len(inferred) == 0:
                fill_order = count_domains(bs)
            else:
                fill_order = determine_fill_order(inferred)
            if fill_order is None:
                continue
            for _ in range(num2aug[num_domains]):
                state = {}
                meta_information = []
                for domain in fill_order:
                    restrict = {}
                    for k, v in inferred.items():
                        d, s = k.split('-')
                        if domain == d and 'book' not in s:
                            restrict[s] = state[v]
                    if domain == 'restaurant':
                        tmp = database.get_restaurant(restrict)
                    elif domain == 'attraction':
                        tmp = database.get_attraction(restrict)
                    elif domain == 'hotel':
                        tmp = database.get_hotel(restrict)
                    else:
                        tmp = {}
                    if len(tmp) > 0:
                        meta_information.append(tmp)
                    for slot in schema[domain]:
                        ds = domain + '-' + slot
                        if ds in inferred:
                            state[ds] = state[inferred[ds]]
                        elif slot in tmp:
                            state[ds] = tmp[slot]
                        else:
                            state[ds] = value_marker.get(ds)
                            if ds in inferred.values() and state[ds] == 'dontcare':
                                state[ds] = value_marker.get(ds)
                            if ds == 'taxi-departure' and 'taxi-destination' in state and state['taxi-destination'] == state[ds]:
                                state[ds] = value_marker.get(ds)
                            if ds == 'taxi-destination' and 'taxi-departure' in state and state['taxi-departure'] == state[ds]:
                                state[ds] = value_marker.get(ds)
                            if ds == 'train-departure' and 'train-destination' in state and state['train-destination'] == state[ds]:
                                state[ds] = value_marker.get(ds)
                            if ds == 'train-destination' and 'train-departure' in state and state['train-departure'] == state[ds]:
                                state[ds] = value_marker.get(ds)
                fix_time(state, inferred)
                slot_compression(state, inferred)
                ans_dic[num_domains].append({
                    'belief_state': state,
                    'meta_information': meta_information[:],
                    'infer': inferred,
                })
    with open('output/states.json', 'w', encoding='utf-8') as f:
        json.dump(ans_dic, f, ensure_ascii=False, indent=2)


def co_ref_detection(state):
    inferred = {}
    ret = {}
    for ds, v in state.items():
        if v in state:
            inferred[ds] = v
        ret[ds] = v
    return ret, inferred


def compare_times(time1, time2):
    # Split the time strings into hours and minutes
    hours1, minutes1 = map(int, time1.split(':'))
    hours2, minutes2 = map(int, time2.split(':'))

    # Compare hours
    if hours1 < hours2:
        return -1
    elif hours1 > hours2:
        return 1
    else:
        # If hours are equal, compare minutes
        if minutes1 < minutes2:
            return -1
        elif minutes1 > minutes2:
            return 1
        else:
            # If both hours and minutes are equal, return 0 (equal)
            return 0


def fix_time(bs, inferred):
    if 'taxi-leaveat' in bs and 'taxi-arriveby' in bs:
        if 'taxi-leaveat' in inferred:
            del bs['taxi-arriveby']
        elif 'taxi-arrveby' in inferred:
            del bs['taxi-leaveat']
        else:
            if random.random() > 0.5:
                del bs['taxi-arriveby']
            else:
                del bs['taxi-leaveat']
    if 'train-leaveat' in bs and 'train-arriveby' in bs:
        if compare_times(bs['train-leaveat'], bs['train-arriveby']) == 1:
            bs['train-leaveat'], bs['train-arriveby'] = bs['train-arriveby'], bs['train-leaveat']
    if 'taxi-arriveby' in bs and 'restaurant-name' in bs and bs['taxi-destination'] == bs['restaurant-name']:
        if 'restaurant-book time' in bs:
            if compare_times(bs['taxi-arriveby'], bs['restaurant-book time']) == 1:
                bs['taxi-arriveby'], bs['restaurant-book time'] = bs['restaurant-book time'], bs['taxi-arriveby']
    if 'taxi-leaveat' in bs and 'restaurant-name' in bs and bs['taxi-destination'] == bs['restaurant-name']:
        if 'restaurant-book time' in bs:
            if compare_times(bs['taxi-leaveat'], bs['restaurant-book time']) == 1:
                bs['taxi-leaveat'], bs['restaurant-book time'] = bs['restaurant-book time'], bs['taxi-leaveat']


def slot_compression(bs, inferred):
    if 'hotel-parking' in bs and 'hotel-internet' in bs:
        if bs['hotel-parking'] == 'no' and bs['hotel-internet'] == 'no':
            if random.random() > 0.5:
                del bs['hotel-parking']
                bs['hotel-internet'] = 'dontcare'
            else:
                del bs['hotel-internet']
                if random.random() > 0.5:
                    bs['hotel-parking'] = 'dontcare'
                else:
                    del bs['hotel-parking']
            return
    if 'hotel-parking' in bs:
        if bs['hotel-parking'] == 'no':
            if random.random() > 0.5:
                bs['hotel-parking'] = 'dontcare'
            else:
                del bs['hotel-parking']
    if 'hotel-internet' in bs:
        if bs['hotel-internet'] == 'no':
            bs['hotel-parking'] = 'dontcare'
    # if 'hotel-stars' in bs and bs['hotel-stars'] == '0':
    #     del bs['hotel-stars']
    # can_be_deleted = []
    # for k, v in bs.items():
    #     domain, slot = k.split('-')
    #     if domain in {'taxi', 'attraction', 'train'}:
    #         continue
    #     if slot in {'area', 'name'}:
    #         continue
    #     if k in inferred.keys() or k in inferred.values():
    #         continue
    #     if random.random() > 0.8:
    #         can_be_deleted.append(k)
    # for deleted in can_be_deleted:
    #     del bs[deleted]
    # cnt = 0
    # for val in bs.values():
    #     if val == 'dontcare':
    #         cnt += 1
    # for k in bs:
    #     domain, slot = k.split('-')
    #     if domain not in {'taxi', 'attraction', 'train'}:
    #         if slot in {'internet', 'parking', 'area', 'food'} and slot not in inferred.keys() and slot not in inferred.values():
    #             if random.random() > 0.5 and cnt < 2:
    #                 bs[k] = 'dontcare'
    #                 cnt += 1


if __name__ == '__main__':
    func()
