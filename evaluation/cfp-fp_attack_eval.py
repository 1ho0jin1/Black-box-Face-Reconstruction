import os
import sys
sys.path.append('./../')
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from encoder.fetch import fetch_encoder
from encoder.blackbox_encoder import BlackboxEncoder
from utils import cosine_similarity, roc_curve, get_best_threshold



def evaluate_attack_F(args, encoder, flip=True, metric='cosine'):
    if metric == 'cosine':
        pass
    else:
        raise NotImplementedError(f'metric "{metric}" is not implemented!')

    # prepare image transforms
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # prepare pair dict
    with open(args.txt_dir+'/Pair_list_F.txt','r') as fp:
        Flines = fp.readlines()
    with open(args.txt_dir+'/Pair_list_P.txt','r') as fp:
        Plines = fp.readlines()

    Fdict = {}
    for line in Flines:
        num, path = line.strip().split()
        plist = path.split('/')
        plist[2] = f'{args.align}_aligned'
        path = '/'.join(plist)
        Fdict[num] = args.txt_dir+'/'+path
    Pdict = {}
    for line in Plines:
        num, path = line.strip().split()
        plist = path.split('/')
        plist[2] = f'{args.align}_aligned'
        path = '/'.join(plist)
        Pdict[num] = args.txt_dir+'/'+path

    # compute similarity scores
    real_feat_dict = {}  # save features for effective computation
    attack_feat_dict = {}
    pos_scores = []
    neg_scores = []
    att_scores_type1 = []
    att_scores_type2 = []

    # iterate through all 10 folds
    for i in range(1,11):
        print(f'split {i}')
        split_dir = args.txt_dir+'/Split/FP/'+str(i).zfill(2)
        # positive pair
        with open(split_dir+'/same.txt','r') as fp:
            lines = fp.readlines()
        for line in tqdm(lines):
            F, P = line.strip().split(',')
            if F not in attack_feat_dict.keys():
                real_feat_dict['F'+F] = None
                imf = Fdict[F]
                ima = os.path.join(args.attack_img_dir, F+'.jpg')  # Type-1 attack
                imf = Image.open(imf)
                ima = Image.open(ima)
                imf = trf(imf).unsqueeze(0).to(args.device)
                ima = trf(ima).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featf = encoder(imf, flip=flip)
                    feata = encoder(ima, flip=flip)
                    real_feat_dict['F'+F] = featf.cpu()
                    attack_feat_dict['F'+F] = feata.cpu()
            if P not in real_feat_dict.keys():
                real_feat_dict['P' + P] = None
                imp = Pdict[P]
                imp = Image.open(imp)
                imp = trf(imp).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featp = encoder(imp, flip=flip)
                    real_feat_dict['P'+P] = featp.cpu()
            pos_scores.append(cosine_similarity(real_feat_dict['F'+F], real_feat_dict['P'+P]).item())
            att_scores_type1.append(cosine_similarity(attack_feat_dict['F'+F], real_feat_dict['F'+F]).item())  # Type-1 attack scores
            att_scores_type2.append(cosine_similarity(attack_feat_dict['F'+F], real_feat_dict['P'+P]).item())  # Type-2 attack scores

        # negative pair
        with open(split_dir + '/diff.txt', 'r') as fp:
            lines = fp.readlines()
        for line in tqdm(lines):
            F, P = line.strip().split(',')
            if F not in real_feat_dict.keys():
                real_feat_dict['F' + F] = None
                imf = Fdict[F]
                imf = Image.open(imf)
                imf = trf(imf).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featf = encoder(imf, flip=flip)
                    real_feat_dict['F'+F] = featf.cpu()
            if P not in real_feat_dict.keys():
                real_feat_dict['P' + P] = None
                imp = Pdict[P]
                imp = Image.open(imp)
                imp = trf(imp).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featp = encoder(imp, flip=flip)
                    real_feat_dict['P'+P] = featp.cpu()
            neg_scores.append(cosine_similarity(real_feat_dict['F' + F], real_feat_dict['P' + P]).item())
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    att_scores_type1 = np.array(att_scores_type1)
    att_scores_type2 = np.array(att_scores_type2)

    # total 3,863,149 negative pairs
    Flist = sorted([int(f[1:]) for f in real_feat_dict.keys() if 'F' in f])  # Frontal images list: len=3940
    Plist = sorted([int(p[1:]) for p in real_feat_dict.keys() if 'P' in p])  # Profile images list: len=1961

    Fset1, Pset1 = torch.FloatTensor([]), torch.FloatTensor([])  # set1: ID   1~250
    Fset2, Pset2 = torch.FloatTensor([]), torch.FloatTensor([])  # set2: ID 251~500
    for F in Flist:
        if F <= 2500:
            Fset1 = torch.cat((Fset1, real_feat_dict['F'+str(F)]))
        else:
            Fset2 = torch.cat((Fset2, real_feat_dict['F'+str(F)]))
    for P in Plist:
        if P <= 1000:
            Pset1 = torch.cat((Pset1, real_feat_dict['P'+str(P)]))
        else:
            Pset2 = torch.cat((Pset2, real_feat_dict['P'+str(P)]))
    neg_scores_temp = cosine_similarity(Fset1, Pset2).view(-1)
    neg_scores_extend = cosine_similarity(Fset2, Pset1).view(-1)
    neg_scores_extend = torch.cat((neg_scores_extend, neg_scores_temp))
    neg_scores_extend = neg_scores_extend.view(-1).numpy()

    return pos_scores, neg_scores, neg_scores_extend, att_scores_type1, att_scores_type2



def evaluate_attack_P(args, encoder, flip=True, metric='cosine'):
    """
    Here, Profile face images are set as targets.
    For matching, Frontal face images are used.
    """
    if metric == 'cosine':
        pass
    else:
        raise NotImplementedError(f'metric "{metric}" is not implemented!')

    # prepare image transforms
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # prepare pair dict
    with open(args.txt_dir+'/Pair_list_F.txt','r') as fp:
        Flines = fp.readlines()
    with open(args.txt_dir+'/Pair_list_P.txt','r') as fp:
        Plines = fp.readlines()

    Fdict = {}
    for line in Flines:
        num, path = line.strip().split()
        plist = path.split('/')
        plist[2] = f'{args.align}_aligned'
        path = '/'.join(plist)
        Fdict[num] = args.txt_dir+'/'+path
    Pdict = {}
    for line in Plines:
        num, path = line.strip().split()
        plist = path.split('/')
        plist[2] = f'{args.align}_aligned'
        path = '/'.join(plist)
        Pdict[num] = args.txt_dir+'/'+path

    # compute similarity scores
    real_feat_dict = {}  # save features for effective computation
    attack_feat_dict = {}
    pos_scores = []
    neg_scores = []
    att_scores_type1 = []
    att_scores_type2 = []

    # iterate through all 10 folds
    for i in range(1,11):
        print(f'split {i}')
        split_dir = args.txt_dir+'/Split/FP/'+str(i).zfill(2)
        # positive pair
        with open(split_dir+'/same.txt','r') as fp:
            lines = fp.readlines()
        for line in tqdm(lines):
            F, P = line.strip().split(',')
            if P not in attack_feat_dict.keys():
                real_feat_dict['P'+P] = None
                imp = Pdict[P]
                ima = os.path.join(args.attack_img_dir, P+'.jpg')  # Type-1 attack
                imp = Image.open(imp)
                ima = Image.open(ima)
                imp = trf(imp).unsqueeze(0).to(args.device)
                ima = trf(ima).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featp = encoder(imp, flip=flip)
                    feata = encoder(ima, flip=flip)
                    real_feat_dict['P'+P] = featp.cpu()
                    attack_feat_dict['P'+P] = feata.cpu()
            if F not in real_feat_dict.keys():
                real_feat_dict['F' + F] = None
                imp = Fdict[F]
                imp = Image.open(imp)
                imp = trf(imp).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featf = encoder(imp, flip=flip)
                    real_feat_dict['F'+F] = featf.cpu()
            pos_scores.append(cosine_similarity(real_feat_dict['F'+F], real_feat_dict['P'+P]).item())          # Positive scores
            att_scores_type1.append(cosine_similarity(attack_feat_dict['P'+P], real_feat_dict['P'+P]).item())  # Type-1 attack scores
            att_scores_type2.append(cosine_similarity(attack_feat_dict['P'+P], real_feat_dict['F'+F]).item())  # Type-2 attack scores

        # negative pair
        with open(split_dir + '/diff.txt', 'r') as fp:
            lines = fp.readlines()
        for line in tqdm(lines):
            F, P = line.strip().split(',')
            if F not in real_feat_dict.keys():
                real_feat_dict['F' + F] = None
                imf = Fdict[F]
                imf = Image.open(imf)
                imf = trf(imf).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featf = encoder(imf, flip=flip)
                    real_feat_dict['F'+F] = featf.cpu()
            if P not in real_feat_dict.keys():
                real_feat_dict['P' + P] = None
                imp = Pdict[P]
                imp = Image.open(imp)
                imp = trf(imp).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    featp = encoder(imp, flip=flip)
                    real_feat_dict['P'+P] = featp.cpu()
            neg_scores.append(cosine_similarity(real_feat_dict['F' + F], real_feat_dict['P' + P]).item())
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    att_scores_type1 = np.array(att_scores_type1)
    att_scores_type2 = np.array(att_scores_type2)

    # total 3,863,149 negative pairs
    Flist = sorted([int(f[1:]) for f in real_feat_dict.keys() if 'F' in f])  # Frontal images list: len=3940
    Plist = sorted([int(p[1:]) for p in real_feat_dict.keys() if 'P' in p])  # Profile images list: len=1961

    Fset1, Pset1 = torch.FloatTensor([]), torch.FloatTensor([])  # set1: ID   1~250
    Fset2, Pset2 = torch.FloatTensor([]), torch.FloatTensor([])  # set2: ID 251~500
    for F in Flist:
        if F <= 2500:
            Fset1 = torch.cat((Fset1, real_feat_dict['F'+str(F)]))
        else:
            Fset2 = torch.cat((Fset2, real_feat_dict['F'+str(F)]))
    for P in Plist:
        if P <= 1000:
            Pset1 = torch.cat((Pset1, real_feat_dict['P'+str(P)]))
        else:
            Pset2 = torch.cat((Pset2, real_feat_dict['P'+str(P)]))
    neg_scores_temp = cosine_similarity(Fset1, Pset2).view(-1)
    neg_scores_extend = cosine_similarity(Fset2, Pset1).view(-1)
    neg_scores_extend = torch.cat((neg_scores_extend, neg_scores_temp))
    neg_scores_extend = neg_scores_extend.view(-1).numpy()

    return pos_scores, neg_scores, neg_scores_extend, att_scores_type1, att_scores_type2




def main(args):
    encoder = fetch_encoder(args.target_encoder)
    encoder = BlackboxEncoder(encoder, img_size=encoder.img_size).to(args.device)

    if args.mode == 'Frontal':
        pos_scores, neg_scores, neg_scores_extend, att_scores_type1, att_scores_type2 = evaluate_attack_F(args, encoder, flip=True)
    elif args.mode == 'Profile':
        pos_scores, neg_scores, neg_scores_extend, att_scores_type1, att_scores_type2 = evaluate_attack_P(args, encoder, flip=True)
    else:
        raise NotImplementedError(f"Invalid evaluation mode: {args.mode}. Choose from ['Frontal', 'Profile']")


    # show histogram
    plt.hist(pos_scores, bins=20, alpha=0.6, density=True)
    plt.hist(neg_scores_extend, bins=100, alpha=0.6, density=True)
    plt.hist(att_scores_type1, bins=20, alpha=0.6, density=True)
    plt.hist(att_scores_type2, bins=20, alpha=0.6, density=True)
    plt.legend(['Positive', 'Negative', 'Type-1', 'Type-2'], fontsize=10)
    plt.xlabel('cosine similarity', fontsize=16)
    plt.savefig(args.attack_img_dir + f'/../score_hist_{args.target_encoder}.jpg', bbox_inches='tight')

    # for saving TAR@FAR and val. accuracy
    res_array = np.zeros((4, 4))

    # compute validation accuracy using 10-fold CV
    sub_mask = np.array([False] * 3500, dtype=np.bool)
    threshold, pos_acc, att_type1_acc, att_type2_acc = [], [], [], []
    for i in range(10):
        sub_mask[350 * i:350 * (i + 1)] = True
        pos_test = pos_scores[sub_mask]
        neg_test = neg_scores[sub_mask]
        att_type1_test = att_scores_type1[sub_mask]
        att_type2_test = att_scores_type2[sub_mask]
        pos_train = pos_scores[~sub_mask]
        neg_train = neg_scores[~sub_mask]
        sub_mask[350 * i:350 * (i + 1)] = False

        best_threshold = get_best_threshold(pos_train, neg_train)
        true_pos_pairs = (pos_test >= best_threshold).sum()
        succ_type1_pairs = (att_type1_test >= best_threshold).sum()
        succ_type2_pairs = (att_type2_test >= best_threshold).sum()
        true_neg_pairs = (neg_test < best_threshold).sum()

        threshold.append(best_threshold)
        pos_acc.append((true_pos_pairs + true_neg_pairs) / 700)
        att_type1_acc.append((succ_type1_pairs + true_neg_pairs) / 700)
        att_type2_acc.append((succ_type2_pairs + true_neg_pairs) / 700)
    threshold = np.array(threshold).mean()
    pos_acc = np.array(pos_acc).mean()
    att_type1_acc = np.array(att_type1_acc).mean()
    att_type2_acc = np.array(att_type2_acc).mean()
    res_array[0, 3] = "{:.4f}".format(threshold)
    res_array[1, 3] = "{:.4f}".format(pos_acc)
    res_array[2, 3] = "{:.4f}".format(att_type1_acc)
    res_array[3, 3] = "{:.4f}".format(att_type2_acc)
    print('match acc:{:.4f}\ntype-1 attack acc:{:.4f}\ntype-2 attack acc:{:.4f}'.format(pos_acc, att_type1_acc,
                                                                                        att_type2_acc))

    # compute TAR(SAR) @ FAR
    far, tar, thresholds = roc_curve(pos_scores, neg_scores_extend)
    far_list = [0.0001, 0.001, 0.01]
    for i, far_tgt in enumerate(far_list):
        abs_diff = np.abs(far - far_tgt)
        idx = np.argmin(abs_diff)
        thr = thresholds[idx]
        sar_type1 = (att_scores_type1 > thr).sum() / att_scores_type1.shape[0]
        sar_type2 = (att_scores_type2 > thr).sum() / att_scores_type2.shape[0]
        res_array[0, i] = "{:.4f}".format(thr)
        res_array[1, i] = "{:.4f}".format(tar[idx])
        res_array[2, i] = "{:.4f}".format(sar_type1.item())
        res_array[3, i] = "{:.4f}".format(sar_type2.item())
        print('FAR={:.4f}: TAR={:.4f}, SAR-Type1={:.4f}, SAR-Type2={:.4f}'.format(far_tgt, tar[idx],
                                                                                  sar_type1.item(), sar_type2.item()))
    # save as xls
    columns = ['0.0001', '0.0010', '0.0100', 'Accuracy']
    rows = ['Threshold', 'TAR', 'Type-1', 'Type-2']
    df = pd.DataFrame(res_array, rows, columns)
    df.to_excel(args.attack_img_dir + f'/../eval_{args.target_encoder}.xls')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_img_dir', type=str,
                        default='/data/New_Projects/ZOIA/results/cfp-fp-F/FaceNet/BB_Wplus_top10_iter200/attack_images')
    parser.add_argument('--mode', type=str, default='Frontal', help='["Frontal", "Profile"]')
    parser.add_argument('--align', type=str, default='mtcnn', help='["mtcnn", "FXZoo"]')
    parser.add_argument('--target_encoder', type=str, default='FaceNet')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    args.device = f'cuda:{args.device_id}'
    args.txt_dir = '/home/yoon/datasets/face/cfp/Protocol'  # directory for cfp Pair_list txt
    main(args)