import json
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MyDataset
import os
import time
import numpy as np
import sys
from tqdm import tqdm as progress_bar
from tqdm import trange
import re

sys.path.append('../aenets')
from net import AE, AE_test


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def make_datasets(network, ngenes, root_dir, is_X, update_mask, mask_rate):
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
                            , chr_num=c, is_mask=True, random_mask=True, mask_rate=mask_rate, update_mask=update_mask,
                            is_train=True, is_shuffle=True)
        end_time = time.time()
        print('Load and make dataset for chromosome', c, 'take', end_time - start_time, 'seconds')
        datasets.append(dataset)
    return datasets


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # *******************************调参部分*****************************************

    ds = 'Ramani'
    sd = 'diag8'
    extra = 'm20_o6'

    # 含X染色体总数
    chr_num = 23 if ds in['Ramani', '4DN', 'Lee'] else 20
    is_X = False if ds == 'Lee' else True

    # 是否使用训练过的模型继续训练
    is_pretrained = False
    if is_pretrained:
        load_epochs = 500
    else:
        load_epochs = 'None'
    save_epochs = 500

    batch_size = 256
    lr = 1e-3
    update_mask = True
    mask_rate = 0.20
    # 用来调整embedding层大小
    opt_rate = 1.0 / 6.0

    # ********************************************************************************

    # 加载数据位置
    root_dir = '../../Datas/vectors/{}/{}'.format(ds, sd)
    data_info_path = os.path.join(root_dir, 'data_info.json')

    # 模型保存文件
    model_dir = '../../models/{}/{}{}'.format(ds, sd, extra)
    os.makedirs(model_dir, exist_ok=True)

    # 加载ngenes
    with open(data_info_path, 'r') as f:
        ngenes = json.load(f)['chr_lens']

    label_dirs = get_subdirectories(root_dir)
    if 'masks' in label_dirs:
        label_dirs.remove('masks')
    y = []
    network = []

    for label_dir in label_dirs:
        sub_path = os.path.join(root_dir, label_dir)
        files = os.listdir(sub_path)
        file_num = 0
        cell_numbers = []
        for filen in files:
            file_num += 1
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).npy', filen)
            cell_number = int(match.group(1))
            if cell_number not in cell_numbers:
                cell_numbers.append(cell_number)
        cell_num = int(file_num / chr_num)
        for i in cell_numbers:
            cell_path = os.path.join(sub_path, 'cell_' + str(i))
            network.append(cell_path)
            y.append(label_dir)

    network_zip = list(zip(network, y))

    train_datasets = make_datasets(network=network_zip, ngenes=ngenes, root_dir=root_dir, is_X=is_X,
                                   update_mask=update_mask, mask_rate=mask_rate)

    start_time = time.time()

    for c, train_dataset in enumerate(train_datasets):
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        load_model_path = os.path.join(model_dir, 'chr' + c + '_' + str(load_epochs) + 'epochs.pth')
        save_model_path = os.path.join(model_dir, 'chr' + c + '_' + str(save_epochs) + 'epochs.pth')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        ipt_size = train_dataset.datasize
        opt_size = int(min(len(train_dataset), ipt_size) * opt_rate) - 1

        size_data = {
            'ipt': ipt_size,
            'opt': opt_size
        }

        with open(os.path.join(model_dir, 'chr' + c + '_datasize.json'), 'w') as f:
            json.dump(size_data, f)

        # 创建模型实例并将其移动到GPU上
        model = AE(ipt_size, opt_size)
        if is_pretrained:
            model.load_state_dict(torch.load(load_model_path, map_location=device))
        model.to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        num_epochs = save_epochs - load_epochs if is_pretrained else save_epochs
        start = time.time()
        for epoch in trange(num_epochs, desc='chr' + c):
            # train_dataset.gen_mask_time = 0.0
            # train_dataset.read_dic_time = 0.0
            # print(f"start Epoch [{epoch + 1}/{num_epochs}]")
            running_loss = 0.0  # 用于累积整个训练集上的损失值
            for train_data in train_loader:
                datas, _ = train_data
                if isinstance(datas, list):
                    # 此时datas是由[original_datas,masked_datas]组成
                    original_datas = datas[0]
                    masked_datas = datas[1]
                else:
                    original_datas = datas
                    masked_datas = datas

                original_datas = original_datas.view(original_datas.size(0), -1).to(device)
                masked_datas = masked_datas.view(masked_datas.size(0), -1).to(device)

                reconstructed_datas = model(masked_datas)
                loss = criterion(reconstructed_datas, original_datas)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 手动释放内存
                # del original_datas
                # del masked_datas
                # torch.cuda.empty_cache()

                running_loss += loss.item() * original_datas.size(0)  # 累积损失值

            epoch_loss = running_loss / len(train_loader.dataset)  # 计算整个训练集上的平均损失值
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss * 1e4:.4f}")

        print('chr' + c + ' complete train!')
        print('chr' + c + ' use time: ' + str(time.time() - start))
        # print('read_dic_time: ' + str(train_dataset.read_dic_time))
        # print('gen_mask_time: ' + str(train_dataset.gen_mask_time))
        # 保存模型
        print('saving model...')
        torch.save(model.state_dict(), save_model_path)
        print('model saved!')
        # time.sleep(5)

    print('total use time: ' + str(time.time() - start_time) + 'seconds')

    print(
        'root_dir={}\nmodel_dir={}\nload_epochs={}\nsave_epochs={}\nbatch_size={}\nlr={}\nupdate_mask={}\nmask_rate={}\nopt_rate={}'.format(
            root_dir, model_dir, load_epochs, save_epochs, batch_size, lr, update_mask, mask_rate, opt_rate))
