import numpy as np
import os
import pickle
import tarfile
import urllib.request


def _download(url, file_name):
    """下载CIFAR-10数据集"""
    if os.path.exists(file_name):
        return

    print(f"正在下载 {file_name} ...")
    urllib.request.urlretrieve(url, file_name)
    print("下载完成")


def _extract(file_name):
    """解压tar文件"""
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall("./dataset")


def _load_batch(file_path):
    """加载单个batch文件"""
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.array(batch[b"labels"])
    return data, labels


def load_cifar10(normalize=True):
    """加载CIFAR-10数据集

    Parameters
    ----------
    normalize : 是否将像素值标准化到0.0~1.0区间

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = "cifar-10-python.tar.gz"
    file_path = os.path.join(dataset_dir, file_name)

    if not os.path.exists(file_path):
        _download(url, file_path)
        _extract(file_path)

    dataset_dir = os.path.join(dataset_dir, "cifar-10-batches-py")

    # 加载训练数据
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(dataset_dir, f"data_batch_{i}")
        data, labels = _load_batch(batch_file)
        train_data.append(data)
        train_labels.append(labels)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # 加载测试数据
    test_file = os.path.join(dataset_dir, "test_batch")
    test_data, test_labels = _load_batch(test_file)

    if normalize:
        train_data = train_data.astype(np.float32) / 255.0
        test_data = test_data.astype(np.float32) / 255.0

    return (train_data, train_labels), (test_data, test_labels)


if __name__ == "__main__":
    # 测试代码
    (x_train, t_train), (x_test, t_test) = load_cifar10()

    print("Train data shape:", x_train.shape)  # (50000, 3, 32, 32)
    print("Train labels shape:", t_train.shape)  # (50000,)
    print("Test data shape:", x_test.shape)  # (10000, 3, 32, 32)
    print("Test labels shape:", t_test.shape)  # (10000,)
