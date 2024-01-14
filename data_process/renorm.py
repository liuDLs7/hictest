import datetime
import json
import re
import os
import time
import numpy as np
import torch


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def normalize_by_chr(A, mode='None'):
    assert mode in ['None', 'chr_max', 'chr_sum'], \
        print('normalize_mode should in [\'None\', \'chr_max\', \'chr_sum\']')
    if mode == 'None':
        param = 1.0
    elif mode == 'chr_max':
        param = np.max(A)
    elif mode == 'chr_sum':
        param = np.sum(A)
    else:
        assert 0, exit(2)
    return A / param


if __name__ == '__main__':
    # **********************************调参部分*******************************************
    dataset = 'Ramani'
    process_pattern = 'diag'
    m = 8
    extra1 = '_sm'
    extra2 = ''
    mode = 'chr_max'

    notes = 'None'
    # ************************************************************************************

    root_dir = '../../Datas/vectors/{0}/{1}{2}{3}'.format(
        dataset, process_pattern, 'all' if m == -1 else str(m), extra1)

    target_dir = '{}{}'.format(root_dir, extra2)

    # chr_lens = get_max_chr_len(processed_dir, chr_num=chr_num)
    if dataset not in ['Ramani', '4DN', 'Lee']:
        assert 0, print('check dataset name!')

    sub_dirs = get_subdirectories(root_dir)

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        target_sub_dir = os.path.join(target_dir, sub_dir)
        file_names = os.listdir(sub_path)

        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            A = np.load(file_path)
            A = normalize_by_chr(A, mode)

            # 将处理后的数据写入文件
            # 可以先用这个创建文件夹
            os.makedirs(target_sub_dir, exist_ok=True)
            target_file = os.path.join(target_sub_dir, file_name)
            np.save(target_file, A)

        print(sub_dir + ' has been processed!')

    data_info_s = os.path.join(root_dir, 'data_info.json')
    data_info_t = os.path.join(target_dir, 'data_info.json')

    with open(data_info_s, 'r') as sf:
        data_info = json.load(sf)
    data_info['mode'] = mode
    with open(data_info_t, 'w') as tf:
        json.dump(data_info, tf)

    print(target_dir)
