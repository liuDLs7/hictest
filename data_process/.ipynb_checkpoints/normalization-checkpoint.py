import datetime
import json
import re
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def get_max_chr_len(root_dir, chr_num):
    # 获取每条染色体的最大长度用于对齐
    max_len = [0] * chr_num
    subdirectories = get_subdirectories(root_dir)

    for subdirectory in subdirectories:
        # 文件夹路径
        folder_path = os.path.join(root_dir, subdirectory)
        # 获取文件夹下的所有文件
        files = os.listdir(folder_path)

        for file_name in files:
            # 根据文件名获取染色体信息
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
            cell_number = int(match.group(1))
            chromosome_number = int(match.group(2)) if match.group(2) != 'X' else chr_num

            file_path = os.path.join(root_dir, subdirectory, file_name)
            # 打开文件
            with open(file_path, 'r') as file:
                # 读取第一行
                first_line = file.readline().strip()
                # 转换为整数
                ngene = int(float(first_line)) + 1
                max_len[chromosome_number - 1] = max(ngene, max_len[chromosome_number - 1])

    return max_len


def neighbor_ave_gpu(A, pad):
    if pad == 0:
        return torch.from_numpy(A).float().cuda()
    ll = pad * 2 + 1
    conv_filter = torch.ones(1, 1, ll, ll).cuda()
    B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().cuda(), conv_filter, padding=pad * 2)
    return B[0, 0, pad:-pad, pad:-pad] / float(ll * ll)


def random_walk_gpu(A, rp):
    ngene, _ = A.shape
    A = A - torch.diag(torch.diag(A))
    A = A + torch.diag(torch.sum(A, 0) == 0).float()

    P = torch.div(A, torch.sum(A, 0))
    Q = torch.eye(ngene).cuda()
    I = torch.eye(ngene).cuda()
    for i in range(30):
        Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
        delta = torch.norm(Q - Q_new, 2)
        Q = Q_new
        if delta < 1e-6:
            break
    return Q


read_file_time = 0.0


def impute_gpu(ngene, pad, rp, file_path, is_weight: bool, weights):
    global read_file_time
    t = time.time()
    D = np.loadtxt(file_path)
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
    read_file_time += time.time() - t
    if is_weight:
        diags = []
        for i in range(len(weights)):
            diag = np.diag(np.diag(A, i) * (weights[i] - 1), i)
            diags.append(diag)

    diag = np.diag(np.diag(A))
    A = A + A.T - diag + 1

    if is_weight:
        for diag in diags:
            A = A + diag

    A = np.log2(A)

    A = neighbor_ave_gpu(A, pad)
    if rp == -1:
        Q = A[:]
    else:
        Q = random_walk_gpu(A, rp)
    return Q


def normalize_by_chr(ngene, pad, rp, file_path, is_weight: bool, weights, mode='None'):
    # not to conv when pad == 0, not to random_walk when rp == -1
    Q = impute_gpu(ngene, pad, rp, file_path, is_weight, weights)
    assert mode in ['None', 'chr_max', 'chr_sum', 'hicembed'], \
        print('normalize_mode should in [\'None\', \'chr_max\', \'chr_sum\']')
    if mode == 'None':
        param = 1.0
    elif mode == 'chr_max':
        param = torch.max(Q)
    elif mode == 'chr_sum':
        param = torch.sum(Q)
    elif mode == 'hicembed':
        # 对矩阵的每行进行求和
        row_sum = torch.sum(Q, dim=1)
        # 构造对角阵
        D = torch.diag_embed(row_sum)
        # 计算 D 的伪逆矩阵 D_pinv
        D_pinv = torch.pinverse(D)
        # 计算 D^(-1/2)AD^(-1/2)
        result = torch.sqrt(D_pinv) @ Q @ torch.sqrt(D_pinv)
        # print(list(result))
        # showw = result.cpu().numpy()
        # np.save('tmp.npy', showw)
        return result

    else:
        assert 0, exit(2)
    return Q / param


def myflatten(A, process_pattern: str = 'row', m: int = -1):
    if process_pattern == 'row':
        # 按行取
        # 获取上三角矩阵的索引
        if m != -1:
            A = A[:m, :]
        # 拉伸为一维向量
        indices = np.triu_indices_from(A)

        B = A[indices].flatten()

    elif process_pattern == 'diag':
        # 按对角线取，只取靠近主对角线的m条（含主对角线)
        if m != -1:
            upper_diags = [np.diagonal(A, offset=i) for i in range(0, m)]
        else:
            upper_diags = [np.diagonal(A, offset=i) for i in range(0, A.shape[0])]

        B = np.concatenate(upper_diags)

    else:
        assert 0, print('error!')

    return B


def main():
    # **********************************调参部分*******************************************
    dataset = 'Ramani'
    pad = 0
    rp = -1
    mode = 'chr_max'
    process_pattern = 'diag'
    m = 8
    
    extra = ''

    is_weight = True
    weights = [2]

    notes = 'None'
    # ************************************************************************************

    chr_num = 23 if dataset in ['Ramani', '4DN', 'Lee'] else 20

    root_dir = '../../CTPredictor/Data_filter/{}'.format(dataset)
    target_dir = '../../Datas/vectors/{0}/{1}{2}{3}'.format(
        dataset, process_pattern, 'all' if m == -1 else str(m), extra)
    processed_dir = '../../Datas/{0}/{0}_processed'.format(dataset)

    sub_dirs = get_subdirectories(root_dir)

    # chr_lens = get_max_chr_len(processed_dir, chr_num=chr_num)
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
        chr_lens = get_max_chr_len(processed_dir, chr_num=chr_num)
        # assert 0, print('check dataset name!')
    print(chr_lens)

    start = time.time()

    tt_normalize_by_chr_time = 0.0
    tt_myflatten_time = 0.0
    tt_read_file_time = 0.0

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        target_sub_dir = os.path.join(target_dir, sub_dir)
        file_names = os.listdir(sub_path)

        normalize_by_chr_time = 0.0
        myflatten_time = 0.0
        global read_file_time
        read_file_time = 0.0

        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
            cell_number = int(match.group(1))
            chromosome_number = int(match.group(2)) if match.group(2) != 'X' else chr_num
            ngene = chr_lens[chromosome_number - 1]
            t1 = time.time()
            M = normalize_by_chr(ngene=ngene, pad=pad, rp=rp, file_path=file_path, mode=mode, is_weight=is_weight, weights=weights)
            t2 = time.time()
            M = M.cpu().numpy()
            # np.save('tmp2.npy', M)
            # exit(0)
            M_vector = myflatten(M, process_pattern=process_pattern, m=m)
            t3 = time.time()
            # M_vector = M.flatten()

            normalize_by_chr_time += t2 - t1
            myflatten_time += t3 - t2

            # 将处理后的数据写入文件
            # 可以先用这个创建文件夹
            os.makedirs(target_sub_dir, exist_ok=True)
            target_file = os.path.join(target_sub_dir, file_name[:-4] + '.npy')
            np.save(target_file, M_vector)

        print(sub_dir + ' has been processed!')
        print('normalize use time: ' + str(normalize_by_chr_time) + ' seconds')
        print('flatten use time: ' + str(myflatten_time) + ' seconds')
        print('rd_fl use time: ' + str(read_file_time) + ' seconds')

        tt_normalize_by_chr_time += normalize_by_chr_time
        tt_myflatten_time += myflatten_time
        tt_read_file_time += read_file_time

    print('use time: ' + str(time.time() - start))
    print('total normalize use time: ' + str(tt_normalize_by_chr_time) + ' seconds')
    print('total flatten use time: ' + str(tt_myflatten_time) + ' seconds')
    print('total rd_fl use time: ' + str(tt_read_file_time) + ' seconds')

    data_info = {
        'root_dir': root_dir,
        'processed_dir': processed_dir,
        'target_dir': target_dir,
        'chr_num': chr_num,
        'pad': pad,
        'rp': rp,
        'mode': mode,
        'process_pattern': process_pattern,
        'm': m,
        'chr_lens': chr_lens,
        'is_weight': is_weight,
        'weights': weights,
        'last_update': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'notes': notes
    }

    data_info_file_path = os.path.join(target_dir, 'data_info.json')

    with open(data_info_file_path, 'w') as json_file:
        json.dump(data_info, json_file, indent=3)

    print(target_dir)


if __name__ == '__main__':
    main()
