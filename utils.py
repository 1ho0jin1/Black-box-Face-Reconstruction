import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F



def cosine_similarity(x,w):
    x_norm = F.normalize(x,dim=1)
    w_norm = F.normalize(w,dim=1)
    cosine = torch.mm(x_norm, w_norm.T).clamp(-1, 1)
    return cosine


def postprocess(images):  # for visualization during debug mode
    """
    images: `torch.Tensor` with range -1~1
    return: `numpy.ndarray` with range 0~1
    """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - images.min()) / (images.max() - images.min())  # min-max norm
    return images


def stylegan_postprocess(img, crop_size=192, v_offset=10):
    '''
    postprocessing for StyleGAN-FFHQ-256
    crops and normalizes the generated image
    '''
    _, _, cy, cx = img.shape
    assert len(img.shape) == 4 and img.shape[1] == 3, 'img must be a Bx3xHxW numpy array'
    assert cy >= crop_size and cx >= crop_size, 'crop size must be smaller than the given image'
    cy = cy // 2 + v_offset  # vertical offset
    cx = cx // 2
    w = crop_size // 2
    img = img[:, :, cy-w:cy+w, cx-w:cx+w]
    img = 2 * (img - img.min()) / (img.max() - img.min()) - 1  # normalize -1~1
    return img


def roc_curve(pos_scores, neg_scores, num_intervals=1000):
    # pos_scores & neg_scores are np.ndarray of shape (N,)
    tar = np.zeros(num_intervals)
    far = np.zeros(num_intervals)
    min_ = min(pos_scores.min(), neg_scores.min())
    max_ = neg_scores.max()
    thr = np.linspace(min_, max_, num_intervals)
    for i, th in enumerate(thr):
        tar[i] = (pos_scores > th).sum() / pos_scores.shape[0]
        far[i] = (neg_scores > th).sum() / neg_scores.shape[0]
    return far, tar, thr


def get_best_threshold(pos_scores, neg_scores):
    fpr, tpr, thresholds = roc_curve(pos_scores, neg_scores)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def threshold_at_far(thr_arr, far_arr, far_tgt):
    """
    thr_arr: thresholds array
    far_arr: false accept rate array
    far_tgt: target far value, e.g. far=0.001 (0.1%)
    **NOTE : there can be multiple thresholds that meets the given FAR (e.g., FAR=1.000)
             if so, we use the average threshold
    """
    abs_diff = np.abs(far_arr - far_tgt)
    minval = abs_diff.min()
    mask = abs_diff == minval
    candidates = thr_arr[mask]
    return candidates.mean()


def plot_scores(pos_scores, neg_scores, att_scores_type1, att_scores_type2):
    # neg_scores are too long to plot; random sample
    neg_scores_sampled = np.random.permutation(neg_scores)
    neg_scores_sampled = neg_scores_sampled[:len(pos_scores)]

    sns.set()
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    sns.distplot(pos_scores, ax=ax[0])
    sns.distplot(neg_scores_sampled, ax=ax[0])
    sns.distplot(att_scores_type1, ax=ax[0])
    sns.distplot(pos_scores, ax=ax[1])
    sns.distplot(neg_scores_sampled, ax=ax[1])
    sns.distplot(att_scores_type2, ax=ax[1])
    ax[0].set_title("Type-1", fontsize=15)
    ax[0].set_xlabel("Cosine Similarity", fontsize=15)
    ax[0].set_ylabel("Density", fontsize=15)
    ax[0].legend(['Positive', 'Negative', 'Attack'], fontsize=12)
    ax[1].set_title("Type-2", fontsize=15)
    ax[1].set_xlabel("Cosine Similarity", fontsize=15)
    ax[1].set_ylabel("")
    ax[1].legend(['Positive', 'Negative', 'Attack'], fontsize=12)
    return fig, ax



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')



def plot_cosine_hist(log_base):
    logs = os.listdir(log_base)
    cos_hist = []
    for log in logs:
        with open(log_base + '/' + log, 'r') as fp:
            lines = fp.readlines()
        hist = []
        for l in lines:
            hist.append(float(l.strip()))
        cos_hist.append(hist)
    cos_hist = np.array(cos_hist)
    N, d = cos_hist.shape
    y = cos_hist.reshape(-1)
    x = np.arange(N * d) % d
    sns.lineplot(x, y)
    plt.show()



if __name__ == '__main__':
    print('hi')