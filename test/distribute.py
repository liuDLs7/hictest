import scipy.stats as stats
import numpy as np
import os
from tqdm import tqdm


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories

root_dir = '../../Datas/vectors/Ramani/diag8'
sub_dirs = get_subdirectories(root_dir)
normal = []
expon = []
gamma = []
cells = 0.0

for sub_dir in sub_dirs:
    sub_path = os.path.join(root_dir, sub_dir)
    file_names = os.listdir(sub_path)
    for file_name in tqdm(file_names, desc=sub_dir):
        file_path = os.path.join(sub_path, file_name)
        data = np.load(file_path)

        # 拟合正态分布
        normal_params = stats.norm.fit(data)
        normal_dist = stats.norm(*normal_params)
        normal_pvalue = stats.kstest(data, normal_dist.cdf).pvalue
        normal.append(normal_pvalue)

        # 拟合指数分布
        expon_params = stats.expon.fit(data)
        expon_dist = stats.expon(*expon_params)
        expon_pvalue = stats.kstest(data, expon_dist.cdf).pvalue
        expon.append(expon_pvalue)

        # 拟合伽马分布
        gamma_params = stats.gamma.fit(data)
        gamma_dist = stats.gamma(*gamma_params)
        gamma_pvalue = stats.kstest(data, gamma_dist.cdf).pvalue
        gamma.append(gamma_pvalue)

        cells += 1

# 输出结果
print("正态分布平均概率 = ", sum(normal) / cells)
print("指数分布平均概率 = ", sum(expon) / cells)
print("伽马分布平均概率 = ", sum(gamma) / cells)

