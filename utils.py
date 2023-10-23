import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while user not in user_train.keys() or len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def test_load(dataset):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    test_user = []
    test_candidates = []
    for u in range(1, usernum+1):
        if u not in train.keys() or len(train[u]) < 1 or len(test[u]) < 1: continue
        rated = set(train[u])
        rated.add(0)
        rated.add(valid[u][0])
        item_idx = [test[u][0]]
        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        test_user.append(u)
        test_candidates.append(item_idx)
    return test_user, test_candidates

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args, s_emb, test_user, test_candidates):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    HT_5 = 0.0
    NDCG_5 = 0.0
    HT_10 = 0.0
    NDCG_10 = 0.0
    HT_20 = 0.0
    NDCG_20 = 0.0

    test_num = 0.0

    for k in range(len(test_user)):
        u = test_user[k]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = test_candidates[k]


        if args.model == 'SASRec':
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        else:
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, s_emb[u]]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        test_num += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if test_num % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return HT_5 / test_num, NDCG_5 / test_num, HT_10 / test_num, NDCG_10 / test_num, HT_20 / test_num, NDCG_20 / test_num

def valid_load(dataset):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    valid_user = []
    valid_candidates = []
    for u in range(1, usernum+1):
        if u not in train.keys() or len(train[u]) < 1 or len(valid[u]) < 1: continue
        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        valid_user.append(u)
        valid_candidates.append(item_idx)

    return valid_user, valid_candidates
# evaluate on val set
def evaluate_valid(model, dataset, args, s_emb, valid_user, valid_candidates):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    HT_5 = 0.0
    NDCG_5 = 0.0
    HT_10 = 0.0
    NDCG_10 = 0.0
    HT_20 = 0.0
    NDCG_20 = 0.0
    valid_num = 0.0

    for k in range(len(valid_user)):
        u = valid_user[k]


        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = valid_candidates[k]

        if args.model == 'SASRec':
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        else:
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, s_emb[u]]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_num += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if valid_num % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return HT_5 / valid_num, NDCG_5 / valid_num, HT_10 / valid_num, NDCG_10 / valid_num, HT_20 / valid_num, NDCG_20 / valid_num