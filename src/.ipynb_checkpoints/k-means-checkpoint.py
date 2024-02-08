from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from dataset import MyDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
import random
import os
import json
from scipy.sparse import csr_matrix
import re
import sys
from clusters import ClusteringEnsemble

sys.path.append('../aenets')
from net import AE, AE_test



def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def make_datasets(network, ngenes, root_dir, is_X=False):
    datasets = []
    for c, ngene in enumerate(ngenes):
        labels = []
        file_names = []
        # print('ndim = ' + str(ndim))
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        start_time = time.time()
        Q_concat = []
        for cell, label in network:
            labels.append(label)
            file_name = cell + '_chr' + c + '.npy'
            file_names.append(file_name)
            Q_concat.append(np.load(file_name))
        # print(labels[112], file_names[112], Q_concat.__len__(), Q_concat[0].__len__())
        dataset = MyDataset(root_dir=root_dir, Q_concat=Q_concat, labels=labels, file_names=file_names
                            , chr_num=c, is_mask=True, random_mask=True, mask_rate=0.1, update_mask=False,
                            is_train=True, is_shuffle=True)
        end_time = time.time()
        print('Load and make dataset for chromosome', c, 'take', end_time - start_time, 'seconds')
        datasets.append(dataset)
    return datasets


def run_on_model(model_dir, train_epochs, network, ngenes, nc, ndim=20, is_X=False, prct=20, cluster='k-means'):
    assert cluster in ['k-means', 'dbscan', 'agg', 'spec', 'ensemble'], print('no such cluster!')
    matrix = []
    ndims = []
    for c, ngene in enumerate(ngenes):
        file_names = []
        # print('ndim = ' + str(ndim))
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        model_path = os.path.join(model_dir, 'chr' + c + '_' + str(train_epochs) + 'epochs.pth')

        with open(os.path.join(model_dir, 'chr' + c + '_datasize.json'), 'r') as json_file:
            d = json.load(json_file)
            ipt_size = d['ipt']
            opt_size = d['opt']

        # 创建模型实例
        model = AE(ipt_size, opt_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)

        Q_concat = []
        with torch.no_grad():
            for cell in network:
                file_name = cell + '_chr' + c + '.npy'
                file_names.append(file_name)

                test_data = np.load(file_name)
                test_data = torch.from_numpy(test_data).to(device)
                embedding = model.encoder(test_data)
                reconstructed_datas = model.decoder(embedding)

                # Q_concat.append(test_data)
                # Q_concat.append(embedding.cpu().numpy())
                Q_concat.append(reconstructed_datas.cpu().numpy())

        Q_concat = np.array(Q_concat)

        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])

        ndim = int(min(Q_concat.shape) * 0.2) - 1
        ndims.append(ndim)
        print(Q_concat.shape)
        # pca = PCA(n_components=ndim)
        # R_reduce = pca.fit_transform(Q_concat)
        # matrix.append(R_reduce)
        matrix.append(Q_concat)
    matrix = np.concatenate(matrix, axis=1)
    print(matrix.shape)
    pca = PCA(n_components=min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    # ndim = 30
    ndim = min(ndims)
    print('ndim = ' + str(ndim))
    # 下载到PC端可视化
    np.save('matrix_reduced.npy', matrix_reduce)
    
    if cluster == 'k-means':
        # k-means
        kmeans = KMeans(n_clusters=nc, n_init=500).fit(matrix_reduce[:, :ndim])
        labels = kmeans.labels_

    elif cluster == 'dbscan':
        # DBSCAN
        dbscan = DBSCAN(eps=15, min_samples=30)
        labels = dbscan.fit_predict(matrix_reduce[:, :ndim])

    elif cluster == 'agg':
        # 层次聚类（Agglomerative Clustering）
        agg_clustering = AgglomerativeClustering(n_clusters=nc)
        labels = agg_clustering.fit_predict(matrix_reduce[:, :ndim])

    elif cluster == 'spec':
        # 谱聚类（Spectral Clustering）
        spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
        labels = spectral_clustering.fit_predict(matrix_reduce[:, :ndim])

    elif cluster == 'ensemble':
        kmeans = KMeans(n_clusters=nc, n_init=500)
        agg_clustering = AgglomerativeClustering(n_clusters=nc)
        spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
        ensemble_model = ClusteringEnsemble(models=[kmeans, agg_clustering, spectral_clustering])
        ensemble_model.fit(matrix_reduce[:, :ndim])
        labels = ensemble_model.predict(matrix_reduce[:, :ndim])

    else:
        print('报错也能出问题？再改改！')
        exit(1)

    return labels


def run_original_data(network, ngenes, nc, ndim=20, is_X=False, prct=20):
    matrix = []
    for c, ngene in enumerate(ngenes):
        labels = []
        file_names = []
        # print('ndim = ' + str(ndim))
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        start_time = time.time()
        Q_concat = []
        for cell in network:
            file_name = cell + '_chr' + c + '.npy'
            file_names.append(file_name)
            Q_concat.append(np.load(file_name))

        Q_concat = np.array(Q_concat)

        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])

        ndim = int(min(Q_concat.shape) * 0.2) - 1
        print(Q_concat.shape)
        # U, S, V = torch.svd(Q_concat, some=True)
        # R_reduce = torch.mm(U[:, :ndim], torch.diag(S[:ndim])).cuda().numpy()
        pca = PCA(n_components=ndim)
        R_reduce = pca.fit_transform(Q_concat)
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    pca = PCA(n_components=min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    # ndim = 30
    print('ndim = ' + str(ndim))
    kmeans = KMeans(n_clusters=nc, n_init=500, init='k-means++').fit(matrix_reduce[:, :ndim])
    return kmeans.labels_


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)

    # *******************************调参部分*****************************************

    dataset = 'Ramani'
    sdir = 'diag8'
    extra = 'm20_o6'
    train_epochs = 500
    prct = 20
    # k-means, dbscan, agg, spec, ensemble
    cluster = 'k-means'

    # ********************************************************************************

    # 分类数
    if dataset == 'Lee':
        nc = 14
    elif dataset == '4DN':
        nc = 5
    elif dataset == 'Ramani':
        nc = 4
    else:
        assert 0, print('check dataset name!')

    ndim = 20
    # 含X染色体总数
    chr_num = 23
    is_X = False if dataset == 'Lee' else True

    # 加载数据位置
    root_dir = '../../Datas/vectors/{}/{}'.format(dataset, sdir)
    data_info_path = os.path.join(root_dir, 'data_info.json')

    # 模型保存文件
    model_dir = '../../models/{}/{}{}'.format(dataset, sdir, extra)
    print('model_dir=', model_dir)

    # 加载ngenes
    with open(data_info_path, 'r') as f:
        ngenes = json.load(f)['chr_lens']

    label_dirs = get_subdirectories(root_dir)
    if 'masks' in label_dirs:
        label_dirs.remove('masks')
    y = []
    network = []

    str2dig = {}
    x = []

    for i, label_name in enumerate(label_dirs):
        str2dig[label_name] = i

    print(str2dig)

    for label_dir in label_dirs:
        sub_path = os.path.join(root_dir, label_dir)
        files = os.listdir(sub_path)
        file_num = 0
        cell_numbers = []
        for file in files:
            file_num += 1
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).npy', file)
            cell_number = int(match.group(1))
            if cell_number not in cell_numbers:
                cell_numbers.append(cell_number)
        cell_num = int(file_num / chr_num)
        # for i in range(1, cell_num + 1):
        #     cell_path = os.path.join(sub_path, 'cell_' + str(i))
        #     network.append(cell_path)
        #     y.append(str2dig[label_dir])
        for i in cell_numbers:
            cell_path = os.path.join(sub_path, 'cell_' + str(i))
            network.append(cell_path)
            y.append(str2dig[label_dir])

    combined = list(zip(network, y))
    random.shuffle(combined)
    network[:], y[:] = zip(*combined)

    if train_epochs <= 0:
        cluster_labels = run_original_data(network, ngenes, nc, ndim, is_X, prct)
    else:
        cluster_labels = run_on_model(model_dir, train_epochs, network, ngenes, nc, ndim, is_X, prct, cluster=cluster)

    y = np.array(y)
    
    np.save('labels.npy', y)
    np.save('predicts.npy', cluster_labels)
    print(list(cluster_labels))

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score

    # 计算调整兰德指数和归一化互信息
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    hm = homogeneity_score(y, cluster_labels)
    fm = completeness_score(y, cluster_labels)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    print("Homogeneity (HM):", hm)
    print("Completeness (FM):", fm)

    print('root_dir={}\nmodel_dir={}\ncluster={}\nnc={}\nprct={}\ntrain_epochs={}'.format(root_dir, model_dir, cluster, nc, prct,
                                                                              train_epochs))
