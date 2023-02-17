import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import glob
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from encoder.fetch import fetch_encoder
from encoder.blackbox_encoder import BlackboxEncoder
from utils import cosine_similarity, threshold_at_far, plot_scores, roc_curve




class lfw_dataset(Dataset):
    def __init__(self, img_dirs, trf):
        self.img_dirs = img_dirs
        self.trf = trf

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        dir = self.img_dirs[idx]
        imname = dir.split('/')[-1]
        img = Image.open(dir)
        img = self.trf(img)
        return img, imname


class lfw_evaluator():
    def __init__(self, args, img_dir, targets_txt, encoder, metric='cosine', flip=True):
        if metric == 'cosine':
            pass
        else:
            raise NotImplementedError(f'metric "{metric}" is not implemented!')
        self.trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.device = args.device

        with open(targets_txt, 'r') as fp:
            lines = fp.readlines()
        pos_ids = [l.strip()[:-9] for l in lines]
        all_ids = os.listdir(img_dir)
        neg_ids = list(set(all_ids) - set(pos_ids))
        pos_dirs = []
        for name in pos_ids:
            pos_dirs += glob.glob(os.path.join(img_dir, name, '*'))
        neg_dirs = []
        for name in neg_ids:
            neg_dirs += glob.glob(os.path.join(img_dir, name, '*'))
        pos_set = lfw_dataset(pos_dirs, self.trf)
        neg_set = lfw_dataset(neg_dirs, self.trf)
        self.pos_loader = DataLoader(pos_set, batch_size=60, shuffle=False, num_workers=4)
        self.neg_loader = DataLoader(neg_set, batch_size=60, shuffle=False, num_workers=4)

        # attack images
        # self.att_dirs = [os.path.join(args.attack_img_dir, target) for target in os.listdir(args.attack_img_dir)]
        self.att_dirs = []
        for target in os.listdir(args.attack_img_dir):
            if target[:-9] in pos_ids:
                self.att_dirs.append(os.path.join(args.attack_img_dir, target))

        print("initializing lfw_evaluator...")
        self.att_features, self.att_imnames = self.compute_features(encoder, 'attack', flip)
        self.pos_features, self.pos_imnames = self.compute_features(encoder, 'positive', flip)
        self.neg_features, _                = self.compute_features(encoder, 'negative', flip)


    @torch.no_grad()
    def compute_features(self, encoder, type, flip=True):
        encoder.eval()
        if type == 'positive':
            features = torch.FloatTensor([])
            imnames = []
            for i, (img, name) in tqdm(enumerate(self.pos_loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
                imnames += list(name)
        elif type == 'negative':
            features = torch.FloatTensor([])
            imnames = None
            for i, (img, name) in tqdm(enumerate(self.neg_loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
        elif type == 'attack':
            features = torch.FloatTensor([])
            imnames = []
            for dir in self.att_dirs:
                imname = dir.split('/')[-1]
                imnames.append(imname)
                img = self.trf(Image.open(dir))
                img = img.unsqueeze(0).to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
        else:
            raise ValueError(f'{type} is not a valid type; ["positive","negative","attack"]')

        return features, imnames

    @torch.no_grad()
    def positive_scores(self):
        pos_name = []
        for imname in self.pos_imnames:
            pos_name.append('_'.join(imname.split('_')[:-1]))
        pos_name = np.array([pos_name])  # 1 x N array
        pos_idx = (pos_name.T == pos_name)  # N x N binary mask
        pos_idx = torch.BoolTensor(pos_idx)

        # compute cosine similarity
        scores = cosine_similarity(self.pos_features, self.pos_features)

        pos_scores = torch.FloatTensor([])
        for i in range(scores.size(0)):
            mask = pos_idx[i].clone()  # ignore negative components
            mask[i] = False            # ignore diagonal components (itself)
            pos_scores = torch.cat((pos_scores, scores[i][mask]), dim=0)

        return pos_scores.numpy()

    @torch.no_grad()
    def negative_scores(self):
        # compute cosine similarity
        scores = cosine_similarity(self.pos_features, self.neg_features)
        neg_scores = scores.view(-1)

        return neg_scores.numpy()

    @torch.no_grad()
    def attack_scores(self):
        """
        defined by paper: https://arxiv.org/abs/1703.00832
        type1 attack: same identity, same image
        type2 attack: same identity, different image (e.g., George_Bush_0001 vs George_Bush_0002)
        """
        pos_names = ['_'.join(n.split('_')[:-1]) for n in self.pos_imnames]
        att_names = ['_'.join(n.split('_')[:-1]) for n in self.att_imnames]
        pos_names = np.array([pos_names])
        att_names = np.array([att_names])
        pos_imnames = np.array([self.pos_imnames])  # [1, 3166]
        att_imnames = np.array([self.att_imnames])  # [1, 200]
        mask_type1 = (att_imnames.T == pos_imnames) # [200, 3166]
        mask_type2 = (att_names.T == pos_names)     # [200, 3166]
        mask_type2 &= ~mask_type1                   # exclude type1 attacks

        # compute cosine similarity
        scores = cosine_similarity(self.att_features, self.pos_features)

        scores_type1 = torch.FloatTensor([])
        scores_type2 = torch.FloatTensor([])
        for i in range(len(self.att_imnames)):
            mask1, mask2 = mask_type1[i], mask_type2[i]
            scores_type1 = torch.cat((scores_type1, scores[i][mask1]), dim=0)
            scores_type2 = torch.cat((scores_type2, scores[i][mask2]), dim=0)

        return scores_type1.numpy(), scores_type2.numpy()



def main(args):
    encoder = fetch_encoder(args.target_encoder).to(args.device)
    encoder = BlackboxEncoder(encoder, img_size=encoder.img_size)

    project_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    with open(f'{project_dir}/dataset/dataset_conf.yaml') as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)
        conf = conf['lfw-200']
    img_dir = conf['image_dir']
    img_dir = img_dir + f'/{args.align}_aligned'
    targets_txt = conf['targets_txt']

    evaluator = lfw_evaluator(args, img_dir, targets_txt, encoder)

    # pos: 369,924 / neg: 31,872,122 / Type-1: 200, Type-2: 2,966
    pos_scores = evaluator.positive_scores()
    neg_scores = evaluator.negative_scores()
    att_scores_type1, att_scores_type2 = evaluator.attack_scores()

    # successful attack rate at different far
    far_arr, tar_arr, thr_arr = roc_curve(pos_scores, neg_scores)
    idx = np.argmax(tar_arr - far_arr)
    best_threshold = thr_arr[idx]
    neg_acc = (neg_scores < best_threshold).sum() / len(neg_scores)
    pos_acc = (pos_scores >= best_threshold).sum() / len(pos_scores)
    type1_acc = (att_scores_type1 >= best_threshold).sum() / len(att_scores_type1)
    type2_acc = (att_scores_type2 >= best_threshold).sum() / len(att_scores_type2)

    type1_arr, type2_arr = [], []
    for thr in thr_arr:
        sar_type1 = (att_scores_type1 >= thr).sum() / len(att_scores_type1)
        sar_type2 = (att_scores_type2 >= thr).sum() / len(att_scores_type2)
        type1_arr.append(sar_type1)
        type2_arr.append(sar_type2)

    res_array = np.zeros((4,4))
    for i, far_tgt in enumerate([0.0001, 0.001, 0.01]):
        threshold = threshold_at_far(thr_arr, far_arr, far_tgt)
        res_array[0, i] = threshold
        res_array[1, i] = (pos_scores >= threshold).sum() / len(pos_scores)
        res_array[2, i] = (att_scores_type1 >= threshold).sum() / len(att_scores_type1)
        res_array[3, i] = (att_scores_type2 >= threshold).sum() / len(att_scores_type2)
    res_array[0,3] = best_threshold
    res_array[1,3] = 0.5 * (pos_acc + neg_acc)
    res_array[2,3] = 0.5 * (type1_acc + neg_acc)
    res_array[3,3] = 0.5 * (type2_acc + neg_acc)

    for i in range(4):
        for j in range(4):
            if i != 0:
                res_array[i, j] *= 100  # use % except for threshold
            res_array[i, j] = "{:.2f}".format(res_array[i, j])

    # save as xls
    columns = ['0.0001', '0.0010', '0.0100', 'Acc']
    rows = ['Threshold','TAR','Type-1', 'Type-2']
    df = pd.DataFrame(res_array, rows, columns)
    df.to_excel(f'{args.attack_img_dir}/../eval_{args.target_encoder}.xlsx')

    # plot score distributions
    fig,ax = plot_scores(pos_scores, neg_scores, att_scores_type1, att_scores_type2)
    fig.savefig(f'{args.attack_img_dir}/../score_hist_{args.target_encoder}.png', bbox_inches='tight')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, type=int, help='which gpu to use')
    parser.add_argument("--target_encoder", default="SwinTransformer", type=str,
                        help="target encoder architecture")
    parser.add_argument('--align', default='FXZoo', type=str)
    parser.add_argument('--attack_img_dir', type=str, help='directory of attack images',
                        default='/data/New_Projects/icip2023/results/lfw-200/SwinTransformer/BB_Wplus_top1_iter200/attack_images')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device_id}')

    main(args)