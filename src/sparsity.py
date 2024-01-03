import numpy as np
import os
import statistics


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def get_non_zero(root_path, rate):
    sub_dirs = get_subdirectories(root_path)

    sum_length = 0.0
    sum_non_zero = 0.0
    unzero_rate = []

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_path, sub_dir)
        file_names = os.listdir(sub_path)
        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            vec = np.load(file_path)
            ll = int(rate * len(vec))
            c = np.count_nonzero(vec[:ll])
            sum_length += ll
            sum_non_zero += c
            unzero_rate.append(float(ll) / float(c))

    return sum_length, sum_non_zero, unzero_rate


def get_contact(root_path, rate):
    sub_dirs = get_subdirectories(root_path)

    sum_contact = 0.0
    pre_contact = 0.0
    contact_rate = []

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_path, sub_dir)
        file_names = os.listdir(sub_path)
        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            vec = np.load(file_path)
            pre = int(rate * len(vec))
            s = np.sum(vec)
            p = np.count_nonzero(vec[:pre])
            sum_contact += s
            pre_contact += p
            contact_rate.append(float(s) / float(p))

    return sum_contact, pre_contact, contact_rate


if __name__ == '__main__':
    root_dir = '../../Datas/vectors/Lee/diagall'
    r1 = 1
    r2 = 0.5

    sl, snz, ur = get_non_zero(root_dir, r1)
    sc, pc, cr = get_contact(root_dir, r2)

    # 非0值分布特征
    print('non_zero : {} / {} = '.format(snz, sl), snz / sl)

    # 平均值
    mean = statistics.mean(ur)
    print("Mean:", mean)

    # 中位数
    median = statistics.median(ur)
    print("Median:", median)

    # 方差
    variance = statistics.variance(ur)
    print("Variance:", variance)

    # 标准差
    std_dev = statistics.stdev(ur)
    print("Standard Deviation:", std_dev)

    # contact分布特征
    print('contact : {} / {} = '.format(pc, sc), pc / sc)

    # 平均值
    mean = statistics.mean(cr)
    print("Mean:", mean)

    # 中位数
    median = statistics.median(cr)
    print("Median:", median)

    # 方差
    variance = statistics.variance(cr)
    print("Variance:", variance)

    # 标准差
    std_dev = statistics.stdev(cr)
    print("Standard Deviation:", std_dev)
