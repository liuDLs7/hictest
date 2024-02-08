import numpy as np
import os


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


if __name__ == '__main__':

    # **********************************调参部分*******************************************
    dataset = 'Ramani'
    process_pattern = 'diag'
    m = 8
    chr_num = 23
    extra1 = ''
    extra2 = '_v6_2'

    # ************************************************************************************

    root_dir1 = '../../Datas/vectors/{0}/{1}{2}{3}'.format(
        dataset, process_pattern, 'all' if m == -1 else str(m), extra1)
    root_dir2 = '../../Datas/vectors/{0}/{1}{2}{3}'.format(
        dataset, process_pattern, 'all' if m == -1 else str(m), extra2)

    sub_dirs = get_subdirectories(root_dir1)
    if 'masks' in sub_dirs:
        sub_dirs.remove('masks')

    flag = True
    dif_sum = 0
    same_files = []

    for sub_dir in sub_dirs:
        sub_path1 = os.path.join(root_dir1, sub_dir)
        sub_path2 = os.path.join(root_dir2, sub_dir)
        file_names = os.listdir(sub_path1)
        for file_name in file_names:
            file_path1 = os.path.join(sub_path1, file_name)
            file_path2 = os.path.join(sub_path2, file_name)
            # 读取两个 .npy 文件的数据
            data1 = np.load(file_path1)
            data2 = np.load(file_path2)

            # 比较两个数组是否相等
            if not np.array_equal(data1, data2):
                if flag:
                    print('{} and {} are different'.format(file_path1, file_path2))
                dif_sum += 1
                flag = False
            else:
                same_files.append((file_path1, file_path2))

    if flag:
        print('{} and {} are the same!'.format(root_dir1, root_dir2))
    else:
        print('%d files are different' % dif_sum)
        if input('show the same ones? (y/n)') == 'y':
            print(same_files)
