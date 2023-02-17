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



def evaluate_attack(args, img_dir, attack_img_dir, txt_dir, encoder, metric='cosine', flip=True):
    if metric == 'cosine':
        pass
    else:
        raise NotImplementedError(f'metric "{metric}" is not implemented!')

    # set encoder to eval mode
    encoder.eval()

    # prepare image transforms
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


    # compute similarity scores
    real_feat_dict = {}  # save features for effective computation
    attack_feat_dict = {}
    pos_scores = []
    neg_scores = []
    att_scores_type1 = []
    att_scores_type2 = []

    with open(txt_dir, 'r') as fp:
        lines = fp.readlines()
    for i, line in tqdm(enumerate(lines)):
        if i == 0: continue
        elem = line[:-1].split('\t')
        if len(elem) == 3:  # positive pair
            n1 = elem[0] + '_{:04d}.jpg'.format(int(elem[1]))
            n2 = elem[0] + '_{:04d}.jpg'.format(int(elem[2]))
            if n1 not in attack_feat_dict.keys():
                im1 = os.path.join(img_dir, elem[0], n1)
                ima = os.path.join(attack_img_dir, n1)  # Type-1 attack
                im1 = Image.open(im1)
                ima = Image.open(ima)
                im1 = trf(im1).unsqueeze(0).to(args.device)
                ima = trf(ima).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    feat1 = encoder(im1, flip=flip)
                    feata = encoder(ima, flip=flip)
                    real_feat_dict[n1] = feat1.cpu()
                    attack_feat_dict[n1] = feata.cpu()

            if n2 not in real_feat_dict.keys():
                im2 = os.path.join(img_dir, elem[0], n2)
                im2 = Image.open(im2)
                im2 = trf(im2).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    feat2 = encoder(im2, flip=flip)
                    real_feat_dict[n2] = feat2.cpu()
            pos_scores.append(cosine_similarity(real_feat_dict[n1], real_feat_dict[n2]).item())
            att_scores_type1.append(cosine_similarity(real_feat_dict[n1], attack_feat_dict[n1]).item())  # Type-1 attack scores
            att_scores_type2.append(cosine_similarity(real_feat_dict[n2], attack_feat_dict[n1]).item())  # Type-2 attack scores

        elif len(elem) == 4:  # negative pair
            n1 = elem[0] + '_{:04d}.jpg'.format(int(elem[1]))
            n2 = elem[2] + '_{:04d}.jpg'.format(int(elem[3]))
            if n1 not in real_feat_dict.keys():
                im1 = os.path.join(img_dir, elem[0], n1)
                im1 = Image.open(im1)
                im1 = trf(im1).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    feat1 = encoder(im1, flip=flip)
                    real_feat_dict[n1] = feat1.cpu()

            if n2 not in real_feat_dict.keys():
                im2 = os.path.join(img_dir, elem[2], n2)
                im2 = Image.open(im2)
                im2 = trf(im2).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    feat2 = encoder(im2, flip=flip)
                    real_feat_dict[n2] = feat2.cpu()
            neg_scores.append(cosine_similarity(real_feat_dict[n1], real_feat_dict[n2]).item())

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    att_scores_type1 = np.array(att_scores_type1)
    att_scores_type2 = np.array(att_scores_type2)


    # to accurately compute FAR=0.0001, wecompute 14,826,350 negative pairs
    nlist = sorted(list(real_feat_dict.keys()))
    n1list = nlist[:3851]  # until Jose_Serra_0009.jpg
    n2list = nlist[3851:]  # from Jose_Theodore_0001.jpg
    n1feat = torch.FloatTensor([])
    n2feat = torch.FloatTensor([])
    for n in n1list:
        n1feat = torch.cat((n1feat, real_feat_dict[n]))
    for n in n2list:
        n2feat = torch.cat((n2feat, real_feat_dict[n]))
    neg_scores_extend = cosine_similarity(n1feat, n2feat)
    neg_scores_extend = neg_scores_extend.view(-1).numpy()

    return pos_scores, neg_scores, neg_scores_extend, att_scores_type1, att_scores_type2


def main(args):
    encoder = fetch_encoder(args.target_encoder)
    encoder = BlackboxEncoder(encoder, img_size=encoder.img_size).to(args.device)

    pos_scores, neg_scores, neg_scores_extend, att_scores_type1, att_scores_type2 = evaluate_attack(args, args.img_dir, args.attack_img_dir, args.txt_dir,
                                                                                                    encoder, flip=True)
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
    sub_mask = np.array([False] * 3000, dtype=np.bool)
    threshold, pos_acc, att_type1_acc, att_type2_acc = [], [], [], []
    for i in range(10):
        sub_mask[300 * i:300 * (i + 1)] = True
        pos_test = pos_scores[sub_mask]
        neg_test = neg_scores[sub_mask]
        att_type1_test = att_scores_type1[sub_mask]
        att_type2_test = att_scores_type2[sub_mask]
        pos_train = pos_scores[~sub_mask]
        neg_train = neg_scores[~sub_mask]
        sub_mask[300 * i:300 * (i + 1)] = False

        best_threshold = get_best_threshold(pos_train, neg_train)
        true_pos_pairs = (pos_test >= best_threshold).sum()
        succ_type1_pairs = (att_type1_test >= best_threshold).sum()
        succ_type2_pairs = (att_type2_test >= best_threshold).sum()
        true_neg_pairs = (neg_test < best_threshold).sum()

        threshold.append(best_threshold)
        pos_acc.append((true_pos_pairs + true_neg_pairs) / 600)
        att_type1_acc.append((succ_type1_pairs + true_neg_pairs) / 600)
        att_type2_acc.append((succ_type2_pairs + true_neg_pairs) / 600)
    threshold = np.array(threshold).mean()
    pos_acc = np.array(pos_acc).mean()
    att_type1_acc = np.array(att_type1_acc).mean()
    att_type2_acc = np.array(att_type2_acc).mean()
    res_array[0,3] = "{:.4f}".format(threshold)
    res_array[1,3] = "{:.4f}".format(pos_acc)
    res_array[2,3] = "{:.4f}".format(att_type1_acc)
    res_array[3,3] = "{:.4f}".format(att_type2_acc)
    print('match acc:{:.4f}\ntype-1 attack acc:{:.4f}\ntype-2 attack acc:{:.4f}'.format(pos_acc, att_type1_acc,
                                                                                        att_type2_acc))

    # compute TAR(SAR) @ FAR
    far_list = [0.0001, 0.001, 0.01]
    far, tar, thresholds = roc_curve(pos_scores, neg_scores_extend)
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
                        default='/data/New_Projects/ZOIA/results/lfw/FaceNet/BB_Wplus_top10_iter200/attack_images')
    parser.add_argument('--align', type=str, default='mtcnn', help='["mtcnn", "FXZoo"]')
    parser.add_argument('--target_encoder', type=str, default='FaceNet')
    parser.add_argument('--device_id', type=int, default=1)
    args = parser.parse_args()
    args.device = f'cuda:{args.device_id}'
    args.img_dir = f'/home/yoon/datasets/face/lfw/{args.align}_aligned'  # directory for lfw images
    args.txt_dir = '/home/yoon/datasets/face/lfw/pairs.txt'  # directory for lfw pairs.txt
    main(args)