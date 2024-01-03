from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dataset import MyDataset
import torch
import numpy as np
import time
import os
import json
import re
import sys
sys.path.append('../aenets')
from net1 import AutoEncoder
from net2 import AutoEncoder2


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def run_on_model(model_dir, train_epochs, network, network2, ngenes, is_X=False, prct=20):
    matrix = []
    for c, ngene in enumerate(ngenes):
        # print('ndim = ' + str(ndim))
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        model_path = os.path.join(model_dir, 'chr' + c + '_' + str(train_epochs) + 'epochs.pth')

        with open(os.path.join(model_dir, 'chr' + c + '_datasize.json'), 'r') as json_file:
            d = json.load(json_file)
            ipt_size = d['ipt']
            opt_size = d['opt']

        # 创建模型实例
        model = AutoEncoder2(ipt_size, opt_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)

        Q_concat = []
        with torch.no_grad():
            for cell in network:
                file_name = cell + '_chr' + c + '.npy'

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
        
        print('saving chr' + c)

        for j, cell2 in enumerate(network2):
            file_name2 = cell2 + '_chr' + c + '.npy'
            np.save(file_name2, Q_concat[j])

        print('chr' + c + ' saved!')

    return


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)

    # *******************************调参部分*****************************************

    dataset = 'Lee'
    train_epochs = 350
    dir_name = 'diagall2'

    # 模型保存文件
    model_dir = '../../models/{}/diagall2'.format(dataset)
    os.makedirs(model_dir, exist_ok=True)

    # 含X染色体总数
    chr_num = 23
    is_X = False
    prct = -1

    # ********************************************************************************

    # 加载数据位置
    root_dir = '../../Datas/vectors/{}/{}'.format(dataset, dir_name)
    t_dir = '../../Datas/vectors/{}/ipt_data/{}_{}epochs'.format(dataset, dir_name, str(train_epochs))
    data_info_path = os.path.join(root_dir, 'data_info.json')

    # 加载ngenes
    with open(data_info_path, 'r') as f:
        ngenes = json.load(f)['chr_lens']

    label_dirs = get_subdirectories(root_dir)
    network = []
    network2 = []

    for label_dir in label_dirs:
        sub_path = os.path.join(root_dir, label_dir)
        tsub_path = os.path.join(t_dir, label_dir)
        os.makedirs(tsub_path, exist_ok=True)
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
            cell_path2 = os.path.join(tsub_path, 'cell_' + str(i))
            network.append(cell_path)
            network2.append(cell_path2)

    run_on_model(model_dir, train_epochs, network, network2, ngenes, is_X, prct)

    print(t_dir)
