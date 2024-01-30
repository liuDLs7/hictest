import torch
import numpy as np
import time
import os
from scipy.sparse import csr_matrix
import re


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def get_avg(network, ngenes, is_X=False):
    Q_avgs = []
    for c, ngene in enumerate(ngenes):
        start_time = time.time()
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        Q_concat = np.zeros((ngene, ngene))
        cell_nums = 0.0
        for cell in network:
            D = np.loadtxt(cell + '_chr' + c + '.txt')
            A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
            Q_concat += A
            cell_nums += 1.0
        Q_concat = Q_concat.astype(float)
        Q_concat /= cell_nums
        Q_avgs.append(Q_concat)
        end_time = time.time()
        print('Load and avg chromosome', c, 'take', end_time - start_time, 'seconds')
    return Q_avgs


def upperDiagCsr(matrix, ndiags):
    max_len = len(np.diag(matrix))
    upper_matrix = []
    for i in range(ndiags):
        arr = np.diag(matrix, i)
        upper_matrix.append(np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=0))
    upper_matrix = np.stack(upper_matrix, axis=0)

    return upper_matrix


def SCCbydiag(m1, m2, ndiags: int):
    m1D = upperDiagCsr(m1, ndiags)
    m2D = upperDiagCsr(m2, ndiags)
    # nSamplesD--Nk
    nSamplesD = np.count_nonzero(m1D + m2D, axis=1)
    rowSumM1D = m1D.sum(axis=1)
    rowSumM2D = m2D.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        # r1k * Nk
        r1k = (np.multiply(m1D, m2D).sum(axis=1) - rowSumM1D * rowSumM2D / nSamplesD) / nSamplesD
        # rhoD为rou1k
        r2k = np.sqrt(
            (np.square(m1D).sum(axis=1) / nSamplesD - np.square(rowSumM1D / nSamplesD)) *
            (np.square(m2D).sum(axis=1) / nSamplesD - np.square(rowSumM2D / nSamplesD)))

        wk = np.multiply(nSamplesD, r2k) / np.multiply(nSamplesD, r2k).sum()

    return wk


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)

    # *******************************调参部分*****************************************

    dataset = 'Ramani'
    cal_avg = False
    cal_weight = True

    # ********************************************************************************

    chr_num = 23
    is_X = False if dataset == 'Lee' else True

    # 加载数据位置
    root_dir = '../../../Downloads/CTPredictor/Data_filter/{}'.format(dataset)

    # 加载ngenes
    if dataset == 'Ramani':
        chr_lens = [250, 244, 198, 192, 181, 171, 160, 147, 142, 136, 135, 134, 116, 108, 103, 91, 82, 79, 60, 63, 49,
                    52, 155]
    elif dataset == '4DN':
        chr_lens = [250, 244, 198, 192, 181, 172, 160, 147, 142, 136, 135, 134, 116, 108, 103, 91, 82, 79, 60, 63, 49,
                    52, 156]
    elif dataset == 'Lee':
        chr_lens = [251, 245, 200, 193, 182, 173, 161, 148, 143, 137, 137, 135, 116, 108, 103, 92, 83, 80, 61, 65, 49,
                    52, 157]
    else:
        assert 0, print('no such dataset!')

    ngenes = chr_lens

    t1 = time.time()

    if cal_avg:
        label_dirs = get_subdirectories(root_dir)
        if 'avgs' in label_dirs:
            label_dirs.remove('avgs')
        if 'weights' in label_dirs:
            label_dirs.remove('weights')
        network = []

        target_sub_dir = os.path.join(root_dir, 'avgs')

        for label_dir in label_dirs:
            sub_path = os.path.join(root_dir, label_dir)
            file_names = os.listdir(sub_path)
            file_num = 0
            cell_numbers = []
            for file_name in file_names:
                file_num += 1
                match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
                cell_number = int(match.group(1))
                if cell_number not in cell_numbers:
                    cell_numbers.append(cell_number)
            cell_num = int(file_num / chr_num)
            for i in cell_numbers:
                cell_path = os.path.join(sub_path, 'cell_' + str(i))
                network.append(cell_path)

        Q_avgs = get_avg(network, ngenes, is_X)

        os.makedirs(target_sub_dir, exist_ok=True)
        for c, Q_avg in enumerate(Q_avgs):
            c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
            save_path = os.path.join(target_sub_dir, 'chr' + c + '_avg.npy')
            np.save(save_path, Q_avg)

        print('avgs have been saved in:', target_sub_dir)

    if cal_weight:
        t2 = time.time()
        avgs_dir = os.path.join(root_dir, 'avgs')
        t_dir = os.path.join(root_dir, 'weights')
        os.makedirs(t_dir, exist_ok=True)
        avg_files = os.listdir(avgs_dir)
        for c in range(len(avg_files)):
            c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
            avg_file = os.path.join(avgs_dir, 'chr' + c + '_avg.npy')
            Q_avg = np.load(avg_file)
            weight = SCCbydiag(Q_avg, Q_avg, 10)
            # print('chr' + c + '\'s weight:', np.log2(weight / min(weight)) + 1)
            print('chr' + c + '\'s weight:', weight / min(weight))
            save_path = os.path.join(t_dir, 'chr' + c + '_weight.npy')
            np.save(save_path, weight)
        print('weights have been saved in:', t_dir)
        print('calculate weights use:', str(time.time()-t2))

    print('total use time:', str(time.time()-t1))


