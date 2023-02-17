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




class cfp_fp_dataset(Dataset):
    def __init__(self, img_dirs, trf):
        self.img_dirs = img_dirs
        self.trf = trf

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        dir = self.img_dirs[idx]
        id = int(dir.split('/')[-3]) - 1
        img = Image.open(dir)
        img = self.trf(img)
        return img, id, dir


class cfp_fp_evaluator():
    def __init__(self, args, targets_txt, encoder, metric='cosine', flip=True):
        if metric == 'cosine':
            pass
        else:
            raise NotImplementedError(f'metric "{metric}" is not implemented!')

        self.trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.device = args.device

        # prepare pair dict
        with open(args.txt_dir + '/Pair_list_F.txt', 'r') as fp:
            Flines = fp.readlines()
        with open(args.txt_dir + '/Pair_list_P.txt', 'r') as fp:
            Plines = fp.readlines()

        Fdict = {}
        for line in Flines:
            num, path = line.strip().split()
            num = int(num) - 1
            plist = path.split('/')
            plist[2] = f'{args.align}_aligned'
            path = '/'.join(plist)
            Fdict[num] = args.txt_dir + '/' + path
        Pdict = {}
        for line in Plines:
            num, path = line.strip().split()
            num = int(num)-1
            plist = path.split('/')
            plist[2] = f'{args.align}_aligned'
            path = '/'.join(plist)
            Pdict[num] = args.txt_dir + '/' + path

        # interchanges w.r.t. mode (F or P)
        if args.mode == 'F':
            self.kp, self.kn = 10, 4
            pdict, ndict = Fdict, Pdict
        elif args.mode == 'P':
            self.kp, self.kn = 4, 10
            pdict, ndict = Pdict, Fdict
        else:
            raise NotImplementedError(f"{args.mode} is not a valid mode!")

        with open(targets_txt, 'r') as fp:
            lines = fp.readlines()

        # in cfp-fp, there are 10 frontal and 4 profile images per identity
        pos_ids = [(int(l.strip())-1)//self.kp for l in lines]
        all_ids = [i for i in range(500)]
        neg_ids = list(set(all_ids) - set(pos_ids))
        pos_p_dirs, pos_n_dirs = [], []
        neg_p_dirs, neg_n_dirs = [], []
        for pid in pos_ids:
            for k in range(self.kn*pid, self.kn*(pid+1)):
                pos_n_dirs.append(ndict[k])
            # skip image #1, since it is set as the target
            for k in range(self.kp*pid+1, self.kp*(pid+1)):
                pos_p_dirs.append(pdict[k])
        for nid in neg_ids:
            for k in range(self.kn*nid, self.kn*(nid+1)):
                neg_n_dirs.append(ndict[k])
            for k in range(self.kp*nid, self.kp*(nid+1)):
                neg_p_dirs.append(pdict[k])

        pos_p_set = cfp_fp_dataset(pos_p_dirs, self.trf)
        pos_n_set = cfp_fp_dataset(pos_n_dirs, self.trf)
        neg_p_set = cfp_fp_dataset(neg_p_dirs, self.trf)
        neg_n_set = cfp_fp_dataset(neg_n_dirs, self.trf)
        self.pos_p_loader = DataLoader(pos_p_set, batch_size=self.kp, shuffle=False, num_workers=4)
        self.pos_n_loader = DataLoader(pos_n_set, batch_size=self.kn, shuffle=False, num_workers=4)
        self.neg_p_loader = DataLoader(neg_p_set, batch_size=self.kp, shuffle=False, num_workers=4)
        self.neg_n_loader = DataLoader(neg_n_set, batch_size=self.kn, shuffle=False, num_workers=4)

        # attack images
        self.att_dirs = [os.path.join(args.attack_img_dir, target) for target in os.listdir(args.attack_img_dir)]

        print("initializing lfw_evaluator...")
        self.att_features, self.att_ids = self.compute_features(encoder, 'attack', flip)
        self.pos_p_features, self.pos_p_ids = self.compute_features(encoder, 'pos-p', flip)
        self.pos_n_features, self.pos_n_ids = self.compute_features(encoder, 'pos-n', flip)
        self.neg_p_features, _              = self.compute_features(encoder, 'neg-p', flip)
        self.neg_n_features, _              = self.compute_features(encoder, 'neg-n', flip)

    @torch.no_grad()
    def compute_features(self, encoder, type, flip=True):
        encoder.eval()
        features = torch.FloatTensor([])
        ids = []
        if 'pos' in type:
            loader = self.pos_p_loader if type=='pos-p' else self.pos_n_loader
            for i, (img, id, dir) in tqdm(enumerate(loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
                ids += id.tolist()
        elif 'neg' in type:
            loader = self.neg_p_loader if type == 'neg-p' else self.neg_n_loader
            for i, (img, id, dir) in tqdm(enumerate(loader)):
                img = img.to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
        elif type == 'attack':
            for dir in self.att_dirs:
                id = (int(dir.split('/')[-1][:-4])-1) // self.kp
                ids.append(id)
                img = self.trf(Image.open(dir))
                img = img.unsqueeze(0).to(self.device)
                feat = encoder(img, flip=flip).cpu()
                features = torch.cat((features, feat), dim=0)
        else:
            raise ValueError(f'{type} is not a valid type')
        return features, ids

    @torch.no_grad()
    def positive_scores(self):
        lenP, lenN = len(self.pos_p_ids), len(self.pos_n_ids)
        pos_p_ids = torch.LongTensor(self.pos_p_ids).unsqueeze(1)
        pos_n_ids = torch.LongTensor(self.pos_n_ids).unsqueeze(1)

        # F-F, P-P, F-P masks
        mask_pp = pos_p_ids.eq(pos_p_ids.T).float()
        mask_nn = pos_n_ids.eq(pos_n_ids.T).float()
        mask_pn = pos_p_ids.eq(pos_n_ids.T)
        mask_pp -= torch.eye(lenP)  # remove diagonal (itself)
        mask_nn -= torch.eye(lenN)  # remove diagonal (itself)

        scores_pp = cosine_similarity(self.pos_p_features, self.pos_p_features)
        scores_nn = cosine_similarity(self.pos_n_features, self.pos_n_features)
        scores_pn = cosine_similarity(self.pos_p_features, self.pos_n_features)
        scores_pp = scores_pp[mask_pp.bool()]
        scores_nn = scores_nn[mask_nn.bool()]
        scores_pn = scores_pn[mask_pn.bool()]
        scores = torch.cat((scores_pp, scores_pn, scores_nn))

        return scores.numpy()

    @torch.no_grad()
    def negative_scores(self):
        # compute cosine similarity
        scores_pp = cosine_similarity(self.pos_p_features, self.neg_p_features)
        scores_nn = cosine_similarity(self.pos_n_features, self.neg_n_features)
        scores_pn = cosine_similarity(self.pos_p_features, self.neg_n_features)
        scores_np = cosine_similarity(self.pos_n_features, self.neg_p_features)
        scores = torch.cat((scores_pp.view(-1), scores_nn.view(-1),
                            scores_pn.view(-1),scores_np.view(-1)))

        return scores.numpy()

    @torch.no_grad()
    def attack_scores(self):
        """
        defined by paper: https://arxiv.org/abs/1703.00832
        type1 attack: same identity, same image
        type2 attack: same identity, different image (e.g., George_Bush_0001 vs George_Bush_0002)
        NOTE: for cfp-fp, we are matching Profile-Frontal faces, thus there is no type-1 attack
        """
        pos_p_ids = torch.LongTensor(self.pos_p_ids).unsqueeze(1)
        pos_n_ids = torch.LongTensor(self.pos_n_ids).unsqueeze(1)
        att_ids = torch.LongTensor(self.att_ids).unsqueeze(1)
        mask_p = att_ids.eq(pos_p_ids.T)  # [200, 1800]
        mask_n = att_ids.eq(pos_n_ids.T)  # [200, 800]

        # compute cosine similarity
        scores_p = cosine_similarity(self.att_features, self.pos_p_features)
        scores_n = cosine_similarity(self.att_features, self.pos_n_features)
        scores_p = scores_p[mask_p]
        scores_n = scores_n[mask_n]
        scores = torch.cat((scores_p, scores_n))

        return scores.numpy()



def main(args):
    encoder = fetch_encoder(args.target_encoder).to(args.device)
    encoder = BlackboxEncoder(encoder, img_size=encoder.img_size)

    project_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    with open(f'{project_dir}/dataset/dataset_conf.yaml') as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)
        conf = conf[f'cfp-fp-200-{args.mode}']
    img_dir = conf['image_dir']
    img_dir = img_dir + f'/{args.align}_aligned'
    targets_txt = conf['targets_txt']

    evaluator = cfp_fp_evaluator(args, targets_txt, encoder)

    # pos: 369,924 / neg: 31,872,122 / Type-1: 200, Type-2: 2,966
    pos_scores = evaluator.positive_scores()
    neg_scores = evaluator.negative_scores()
    att_scores_type2 = evaluator.attack_scores()

    # successful attack rate at different far
    far_arr, tar_arr, thr_arr = roc_curve(pos_scores, neg_scores)
    idx = np.argmax(tar_arr - far_arr)
    best_threshold = thr_arr[idx]
    neg_acc = (neg_scores < best_threshold).sum() / len(neg_scores)
    pos_acc = (pos_scores >= best_threshold).sum() / len(pos_scores)
    type2_acc = (att_scores_type2 >= best_threshold).sum() / len(att_scores_type2)

    type1_arr, type2_arr = [], []
    for thr in thr_arr:
        sar_type2 = (att_scores_type2 >= thr).sum() / len(att_scores_type2)
        type2_arr.append(sar_type2)

    res_array = np.zeros((3,4))
    for i, far_tgt in enumerate([0.0001, 0.001, 0.01]):
        threshold = threshold_at_far(thr_arr, far_arr, far_tgt)
        res_array[0, i] = threshold
        res_array[1, i] = (pos_scores >= threshold).sum() / len(pos_scores)
        res_array[2, i] = (att_scores_type2 >= threshold).sum() / len(att_scores_type2)
    res_array[0,3] = best_threshold
    res_array[1,3] = 0.5 * (pos_acc + neg_acc)
    res_array[2,3] = 0.5 * (type2_acc + neg_acc)

    for i in range(3):
        for j in range(4):
            if i != 0:
                res_array[i, j] *= 100  # use % except for threshold
            res_array[i, j] = "{:.2f}".format(res_array[i, j])

    # save as xls
    columns = ['0.0001', '0.0010', '0.0100', 'Acc']
    rows = ['Threshold','TAR', 'Type-2']
    df = pd.DataFrame(res_array, rows, columns)
    df.to_excel(f'{args.attack_img_dir}/../eval_{args.target_encoder}.xlsx')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=1, type=int, help='which gpu to use')
    parser.add_argument('--mode', default='F', type=str,
                        help='["F", "P"]; if F, the attack images were generated from Frontal images and vice versa')
    parser.add_argument("--target_encoder", default="VGGNet19", type=str,
                        help="target encoder architecture")
    parser.add_argument('--align', default='mtcnn', type=str)
    parser.add_argument('--attack_img_dir', type=str, help='directory of attack images',
                        default='/data/New_Projects/NBNet/attack/FaceNet/cfp-fp-200-F/attack_images')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device_id}')
    args.txt_dir = '/home/yoon/datasets/face/cfp/Protocol'

    main(args)