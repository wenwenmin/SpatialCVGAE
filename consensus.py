import os
import sys
import glob
import itertools
import seaborn as sns
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import anndata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA,NMF
from numpy.linalg import  *
import random
from typing import List
from scipy.spatial import distance
from scipy.cluster import hierarchy


def labels_connectivity_mat(labels: np.ndarray):
    _labels = labels - np.min(labels)
    n_classes = np.unique(_labels)
    mat = np.zeros([labels.size, labels.size])
    for i in n_classes:
        indices = np.squeeze(np.where(_labels == i))  #将属于各个类的标签提取出来
        row_indices, col_indices = zip(*itertools.product(indices, indices))
        mat[row_indices, col_indices] = 1
    return mat


def consensus_matrix(labels_list: List[np.ndarray]):
    mat = 0
    for labels in labels_list:
        mat += labels_connectivity_mat(labels)
    return mat / float(len(labels_list))


def plot_consensus_map(cmat, method="average", return_linkage=True, **kwargs):
    row_linkage = hierarchy.linkage(distance.pdist(cmat), method=method)
    col_linkage = hierarchy.linkage(distance.pdist(cmat.T), method=method)
    figure = sns.clustermap(cmat, row_linkage=row_linkage, col_linkage=col_linkage, **kwargs)
    if return_linkage:
        return row_linkage, col_linkage, figure
    else:
        return figure


def get_consensus_matrix(section_id='151507'):
    save_dir = os.path.join(section_id)
    name = section_id + "_pred.npy"

    sys.setrecursionlimit(100000)
    label_files = glob.glob(save_dir + f"/version_*/{name}")
    labels_list = [np.load(file) for file in label_files]
    df = pd.DataFrame(labels_list).T

    return df



def consensus_summary(section_id, n_clusters, methods='Hierarchical-based', maxiter_em=100, eps_em=0.001, seed=2023):
    '''
    :param section_id: Slice number in the DLPFC dataset
    :param save_dir: Consensus clustering heatmaps and predicted label preservation paths
    :param df: a data frame representing the clustering results, the rows represent samples, and the columns represent different situations
    :param methods='Average-based' or methods='Onehot-based' or methods='NMF-based' or methods='wNMF-based'
    :param n_clusters: number of principal components for NMF
    :param maxiter_em: the maximum number of iterations, only useful when methods='NMF-based' or 'wNMF-based'
    :param eps_em: threshold, only useful when methods='NMF-based' or 'wNMF-based'
    :param seed: random number seed
    '''

    save_dir = os.path.join(section_id)

    df = get_consensus_matrix(section_id)

    # 计算连接矩阵并加权求和（权重为1/T）
    n = len(df)
    matrix_sum = np.zeros(shape=(n, n))
    matrix_list = []
    for i in df.columns:
        labels = df[i].tolist()
        temp = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n)[i + 1:]:
                if labels[i] == labels[j]:
                    temp[i, j] = temp[j, i] = 1
        matrix_sum += temp
        matrix_list.append(temp)
    matrix_mean = matrix_sum / df.shape[1]

    if methods == 'Hierarchical-based':
        row_linkage, _, figure = plot_consensus_map(matrix_mean, return_linkage=True)
        figure.savefig(os.path.join(save_dir, "consensus_clustering.png"), dpi=300)  # 保存图片
        labels_consensus = hierarchy.cut_tree(row_linkage, n_clusters).squeeze()
        np.save(os.path.join(save_dir, "consensus_labels"), labels_consensus)
    if methods == 'Kmeans-based':
        # 对matrix_mean进行kmeans聚类
        kmeans = KMeans(n_clusters, random_state=seed, n_init=10)
        kmeans.fit(matrix_mean)
        labels_consensus = kmeans.labels_
    elif methods == 'Onehot-based':
        # 对df每列的聚类标签进行one-hot编码得到k*n的矩阵，并将所有的矩阵vstack得到matrix_all
        k = n_clusters
        matrix_all = np.zeros(shape=(1, n))
        for i in df.columns:
            temp = np.zeros(shape=(k, n))
            for j in range(n):
                labels = df[i].tolist()
                if labels[j] >= k:
                    labels[j] = k - 1
                temp[labels[j], j] = 1
            matrix_all = np.vstack((matrix_all, temp))
        matrix_all = matrix_all[1:, :]

        # 对matrix_all进行kmeans聚类
        kmeans = KMeans(n_clusters, random_state=seed, n_init=10)
        kmeans.fit(matrix_all.T)
        labels_consensus = kmeans.labels_

    elif methods == 'NMF-based':
        # 对matrix_mean进行NMF
        model = NMF(n_components=n_clusters, init='random', solver='mu', random_state=seed, max_iter=1000)
        W = model.fit_transform(matrix_mean)
        H = model.components_

        H_init = (W + H.T) / 2
        H_hat = np.dot(H_init, (np.dot(H_init.T, H_init)) ** (-1 / 2))
        D = np.diag(np.diagonal(np.dot(H_init.T, H_init)))
        U0 = np.dot(np.dot(H_hat, D), H_hat.T)

        # 通过NMF分解的乘法更新过程求解H_hat(n*n_clusters)和D
        diff = []
        diff.append(np.linalg.norm(matrix_mean - U0))
        for i in range(maxiter_em):
            # print(f"**********This is {i}th iteration**********")
            if len(diff) > 1 and ((abs(diff[-1] - diff[-2]) / abs(diff[-2])) < eps_em):
                break
            else:
                H_update = np.multiply(H_hat, np.sqrt(
                    np.dot(np.dot(matrix_mean, H_hat), D) / np.dot(np.dot(np.dot(H_hat, H_hat.T), matrix_mean),
                                                                   np.dot(H_hat, D))))
                D_update = np.multiply(D, np.sqrt(
                    np.dot(np.dot(H_hat.T, matrix_mean), H_hat) / np.dot(np.dot(np.dot(H_hat.T, H_hat), D),
                                                                         np.dot(H_hat.T, H_hat))))
                U_update = np.dot(np.dot(H_update, D_update), H_update.T)
                diff.append(np.linalg.norm(matrix_mean - U_update))
                H_hat = H_update
                D = D_update
        # H_hat每一行最大值对应的索引即为labels
        labels_nmf = []
        for i in range(len(H_hat)):
            labels_nmf.append(np.argmax(H_hat[i]))
        labels_consensus = labels_nmf

    elif methods == 'wNMF-based':
        model = NMF(n_components=n_clusters, init='random', solver='mu', random_state=seed, max_iter=1000)
        W = model.fit_transform(matrix_mean)
        H = model.components_

        H_init = (W + H.T) / 2
        H_hat = np.dot(H_init, (np.dot(H_init.T, H_init)) ** (-1 / 2))
        D = np.diag(np.diagonal(np.dot(H_init.T, H_init)))
        U0 = np.dot(np.dot(H_hat, D), H_hat.T)

        # step1:固定w(w=1/T)求解H_hat
        diff = []
        diff.append(np.linalg.norm(matrix_mean - U0))
        for i in range(maxiter_em):
            # print(f"**********This is {i}th iteration**********")
            if len(diff) > 1 and ((abs(diff[-1] - diff[-2]) / abs(diff[-2])) < eps_em):
                break
            else:
                H_update = np.multiply(H_hat, np.sqrt(
                    np.dot(np.dot(matrix_mean, H_hat), D) / np.dot(np.dot(np.dot(H_hat, H_hat.T), matrix_mean),
                                                                   np.dot(H_hat, D))))
                D_update = D
                U_update = np.dot(np.dot(H_update, D_update), H_update.T)
                diff.append(np.linalg.norm(matrix_mean - U_update))
                H_hat = H_update

        # 固定H_hat求解w(具有T个变量的线性约束的二次函数优化问题)
        p = np.ones((len(matrix_list), len(matrix_list)))
        for i in range(len(matrix_list)):
            for j in range(len(matrix_list)):
                p[i, j] = p[j, i] = np.multiply(matrix_list[i], matrix_list[j]).sum()
        q = list()
        for i in range(df.shape[1]):
            q.append((-2 * (np.dot(np.dot(H_hat.T, matrix_list[i]), H_hat))).sum())
        G = -np.identity(len(matrix_list))
        h = np.zeros((len(matrix_list), 1))
        A = np.ones((1, len(matrix_list)))
        b = matrix([1.0])
        result = solvers.qp(matrix(p), matrix(q), matrix(G), matrix(h), matrix(A), b)  # result['x']即为权重

        # 根据得到的权重对T个连接矩阵进行加权，得到最终的矩阵matrix_weigh
        matrix_weigh = np.zeros(shape=(n, n))
        for i in range(len(matrix_list)):
            matrix_weigh += result['x'][i] * matrix_list[i]

        # 对matrix_weigh进行kmeans聚类，得到最终的labels
        kmeans = KMeans(n_clusters, random_state=seed, n_init=10)
        kmeans.fit(matrix_weigh.T)
        labels_weigh = kmeans.labels_
        labels_consensus = labels_weigh

    return labels_consensus
