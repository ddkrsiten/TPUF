import os
import time
import torch
import argparse
import datetime
from model import *
from utils import *
import pickle
import numpy as np
import random

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Amazon_book')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=32, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default="./Amazon_book_TPUF_AdamW/\
TPUF.dataset=Amazon_book.epoch=400.lr=0.001.hidden=32.maxlen=50.l2=0.5.loss_w0.8.combine_w0.5_HR_0.2113.pth",
                    type=str)
parser.add_argument('--model', default='TPUF', type=str)  # TPUF, SASRec
parser.add_argument('--s_model', default='mlp', type=str)  # mlp,mf,sas
parser.add_argument('--combine_weight', default=0.5, type=float)
parser.add_argument('--loss_weight', default=0.8, type=float)
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.model + '_AdamW'):
    os.makedirs(args.dataset + '_' + args.model + '_AdamW')
# with open(os.path.join(args.dataset + '_' + args.train_dir + '_AdamW', 'args.txt'), 'w') as f:
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
if not os.path.isdir(args.dataset + '_' + args.model + '_AdamW' + '/results'):
    os.makedirs(args.dataset + '_' + args.model + '_AdamW' + '/results')

if __name__ == '__main__':
    # global dataset


    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)






    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    valid_user, valid_candidates = valid_load(dataset)
    test_user, test_candidates = test_load(dataset)

    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    now_time = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now_time, '%m-%d.%H.%M.%S')
    print(now_time)
    if args.model == 'TPUF' or args.model == 'TPUF_wo_TFM':
        f = open(os.path.join(args.dataset + '_' + args.model + '_AdamW' + '/results',
                              now_time + '_' + 'four_layer_' + args.s_model + '_l2_' + str(args.l2_emb) + '_' + str(args.loss_weight) + '_' \
                              + str(args.combine_weight) + '_' + str(args.hidden_units) + '.txt'), 'w')
    elif args.model == 'SASRec':
        f = open(os.path.join(args.dataset + '_' + args.model + '_AdamW' + '/results',
                              now_time + '_l2_' + str(args.l2_emb) + str(args.dataset) \
                              + '_hidden_units_' + str(args.hidden_units) + '.txt'), 'w')

    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.write('\n')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)
    if args.model == 'SASRec':
        model = SASRec(usernum, itemnum, args).to(args.device)
    elif args.model == 'TPUF':
        model = TPUF(usernum, itemnum, args).to(args.device)
    elif args.model == 'TPUF_wo_TFM':
        model = TPUF_wo_TFM(usernum, itemnum, args).to(args.device)

    if args.dataset == 'douban_book' and (args.model == 'TPUF' or args.model == 'TPUF_wo_TFM'):
        if args.s_model == 'mf':
            s_model = MF(1599, 13712, args.hidden_units)
            s_model = s_model.cuda()
            s_model.load_state_dict(torch.load('./s_model/douban_movie_mf_32_0.005_epoch49_hits0.6471_ndcgs0.3417.model'))
            s_emb = s_model.get_embedding(torch.tensor(np.arange(usernum + 1)).cuda()).detach().cpu().numpy()
        elif args.s_model == 'mlp':
            s_model = MLP_rec(1599, 13712, args.hidden_units)
            s_model = s_model.cuda()
            s_model.load_state_dict(torch.load('./s_model/douban_movie_mlp_32_0.005_epoch49_hits0.6133_ndcgs0.3414.model'))
            s_emb = s_model.MLP_Embedding_User(torch.tensor(np.arange(usernum + 1)).cuda()).detach().cpu().numpy()
        elif args.s_model == 'sas':
            data_dir = "s_model/SAS_douban_movie_0.6520650813516896.pickle"

            with open(data_dir, "rb") as fid:
                s_emb = pickle.load(fid)
            s_emb = np.array(s_emb).astype(np.float32)
    elif args.dataset == 'Amazon_book' and (args.model == 'TPUF' or args.model == 'TPUF_wo_TFM'):
        if args.s_model == 'mf':
            s_model = MF(4250, 20542, args.hidden_units)
            s_model = s_model.cuda()
            s_model.load_state_dict(torch.load('./s_model/Amazon_movie_mf_32_0.005_epoch99_hits0.4846_ndcgs0.2895.model'))
            s_emb = s_model.get_embedding(torch.tensor(np.arange(usernum + 1)).cuda()).detach().cpu().numpy()
        elif args.s_model == 'mlp':
            s_model = MLP_rec(4250, 20542, args.hidden_units)
            s_model = s_model.cuda()
            s_model.load_state_dict(torch.load('./s_model/Amazon_movie_mlp_32_0.005_epoch99_hits0.4573_ndcgs0.2486.model'))
            s_emb = s_model.MLP_Embedding_User(torch.tensor(np.arange(usernum + 1)).cuda()).detach().cpu().numpy()
        elif args.s_model == 'sas':
            data_dir = "s_model/SAS_Amazon_movie_0.4801129677571193.pickle"
            with open(data_dir, "rb") as fid:
                s_emb = pickle.load(fid)
            s_emb = np.array(s_emb).astype(np.float32)
    elif args.dataset == 'Amazon_cloth' and (args.model == 'TPUF' or args.model == 'TPUF_wo_TFM'):
        if args.s_model == 'mf':
            s_model = MF(3332, 10200, args.hidden_units)
            s_model = s_model.cuda()
            s_model.load_state_dict(torch.load('./s_model/Amazon_sport_mf_32_0.005_epoch99_hits0.3209_ndcgs0.1817.model'))
            # s_model.load_state_dict(torch.load('./s_model/Amazon_sport_mf_32_0.005_epoch99_hits0.2984_ndcgs0.1661.model'))
            s_emb = s_model.get_embedding(torch.tensor(np.arange(usernum + 1)).cuda()).detach().cpu().numpy()
        elif args.s_model == 'mlp':
            s_model = MLP_rec(3332, 10200, args.hidden_units)
            s_model = s_model.cuda()
            s_model.load_state_dict(torch.load('./s_model/Amazon_sport_mlp_32_0.005_epoch99_hits0.2924_ndcgs0.1523.model'))
            #s_model.load_state_dict(torch.load('./s_model/Amazon_sport_mlp_32_0.005_epoch99_hits0.2566_ndcgs0.1400.model'))
            s_emb = s_model.MLP_Embedding_User(torch.tensor(np.arange(usernum + 1)).cuda()).detach().cpu().numpy()
        elif args.s_model == 'sas':
            data_dir = "s_model/SAS_Amazon_sport_0.23776643650555387.pickle"
            with open(data_dir, "rb") as fid:
                s_emb = pickle.load(fid)
            s_emb = np.array(s_emb).astype(np.float32)

    else:
        s_emb = []

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.inference_only and args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args, s_emb, test_user, test_candidates)
        print('')
        print('test (HR@10: %.4f, NDCG@10: %.4f)'
              % (t_test[2], t_test[3]))
        print('test (HR@5: %.4f, NDCG@5: %.4f), test (HR@20: %.4f, NDCG@20: %.4f)'
              % (t_test[0], t_test[1], t_test[4], t_test[5]))


    bce_criterion = torch.nn.BCELoss()
    adam_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            if args.model == 'SASRec':
                pos_logits, neg_logits = model(u, seq, pos, neg)

            elif args.model == 'TPUF' or args.model == 'TPUF_wo_TFM':
                if args.s_model == 'mf':
                    s_embs = s_model.get_embedding(torch.tensor(u).cuda())
                elif args.s_model == 'mlp':
                    s_embs = s_model.MLP_Embedding_User(torch.tensor(u).cuda())
                elif args.s_model == 'sas':
                    s_embs = torch.tensor(s_emb[u], dtype=torch.float32).cuda()
                pos_logits, neg_logits, pos_embs, neg_embs = model(u, seq, pos, neg, s_embs)


            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            pos_domains, neg_domains = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                     device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            if args.model == 'TPUF' or args.model == 'TPUF_wo_TFM':
                classify_loss = bce_criterion(pos_embs, pos_domains)
                classify_loss += bce_criterion(neg_embs, neg_domains)
                loss = loss + classify_loss * args.loss_weight
            # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                             loss.item()))  # expected 0.4~0.6 after init few epochs






        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            # t_test = evaluate(model, dataset, args, s_emb, valid_dataset)
            t_test = evaluate(model, dataset, args, s_emb, test_user, test_candidates)
            # t_valid = evaluate_valid(model, dataset, args, s_emb, test_dataset)
            t_valid = evaluate_valid(model, dataset, args, s_emb, valid_user, valid_candidates)
            print('epoch:%d, time: %f(s), valid (HR@10: %.4f, NDCG@10: %.4f), test (HR@10: %.4f, NDCG@10: %.4f)'
                  % (epoch, T, t_valid[2], t_valid[3], t_test[2], t_test[3]))
            print('epoch:%d, time: %f(s), test (HR@5: %.4f, NDCG@5: %.4f), test (HR@20: %.4f, NDCG@20: %.4f)'
                  % (epoch, T, t_test[0], t_test[1], t_test[4], t_test[5]))

            f.write("epoch=" + str(epoch) + ", " + str(t_valid[2:4]) + ' ' + str(t_test[2:4]) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
            # folder = args.dataset + '_' + args.model + '_AdamW'
            # fname = args.model + '.dataset={}.epoch={}.lr={}.hidden={}.maxlen={}.l2={}.loss_w{}.combine_w{}_HR_{:.4f}.{}.pth'
            # fname = fname.format(args.dataset, epoch, args.lr, args.hidden_units, args.maxlen, args.l2_emb,
            #                      args.loss_weight, args.combine_weight, t_test[2], now_time)
            # torch.save(model.state_dict(), os.path.join(folder, fname))
            # print(fname)


    if args.model == 'SASRec':

        seq_u = []
        for u in range(0, usernum + 1):
            if u in user_train.keys():
                seq = user_train[u][-args.maxlen:]
            else:
                seq = []
            if len(seq) < args.maxlen:
                seq = [0] * (args.maxlen - len(seq)) + seq

            seq_u.append(seq)
        seq_u = np.array(seq_u)

        feature = model.get_user_feature(seq_u)
        feature = feature.detach().cpu().numpy()

        outfile = './s_model/SAS_' + args.dataset + '_' + str(t_test[2]) + '.pickle'
        print(outfile)
        with open(outfile, "wb") as fid:
            pickle.dump(feature, fid, -1)

    f.close()
    sampler.close()
    print("Done")
