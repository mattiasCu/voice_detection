import os
import random
import pandas as pd

def get_file_paths_and_labels(root_dir):
    data = []
    for label, subdir in enumerate(['fake', 'true']):
        subdir_path = os.path.join(root_dir, subdir)
        for file_name in os.listdir(subdir_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(subdir_path, file_name)
                data.append((file_path, label))
    return data

def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data

def save_to_csv(data, file_name):
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    df.to_csv(file_name, index=False)
    print(f"保存到 {file_name}")

def main():
    root_dir = '/home/cu/workplace/python/fake_sound_detection/data'  # 替换为你的数据集根目录
    train_ratio = 0.8  # 训练集比例

    # 获取数据
    data = get_file_paths_and_labels(root_dir)

    # 打乱并划分数据集
    train_data, val_data = split_data(data, train_ratio)

    # 保存到CSV
    save_to_csv(train_data, 'train.csv')
    save_to_csv(val_data, 'val.csv')

    print(f"总样本数: {len(data)}")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")

if __name__ == "__main__":
    main()
