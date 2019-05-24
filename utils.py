import numpy as np
from torch.nn import PairwiseDistance

l2_dist = PairwiseDistance(2)


def rank1(embeddings_anc, embeddings_pos):
    n = len(embeddings_anc)
    n_good = 0

    A = conjagate_matrix(embeddings_anc, embeddings_pos)
    for i, anc_base_dists in enumerate(A):
        j = np.argmin(anc_base_dists)
        if i == j:
            n_good += 1
        # print(i,j)
    return n_good / n


def roc_curve(embeddings_anc, embeddings_pos):
    A = conjagate_matrix(embeddings_anc, embeddings_pos)
    A = (A - A.min()) / (A.max() - A.min())
    trshs = []
    tprs = [0]
    fprs = [0]
    for th in np.sort(np.unique(A.ravel())):
        tpr, fpr = tpr_fpr(A, th)
        trshs.append(th)
        tprs.append(tpr)
        fprs.append(fpr)
    fprs.append(1)
    tprs.append(1)

    return trshs, fprs, tprs


def tpr_fpr(Conj, th):
    n, _ = Conj.shape
    B = Conj < th
    tpr = np.trace(B)

    fpr = np.sum(B) - tpr

    tpr = tpr / n
    fpr = fpr / n ** 2

    return tpr, fpr


def conjagate_matrix(embeddings_anc, embeddings_pos):
    n = embeddings_anc.shape[0]
    A = np.zeros((n, n), dtype=np.float)
    for i, pos in enumerate(embeddings_pos):
        A[:, i] = l2_dist(embeddings_anc, pos.unsqueeze(0).expand_as(embeddings_anc))
    return A
