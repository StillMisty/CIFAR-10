import numpy as np
import matplotlib.pyplot as plt
from dataset.cifar10 import load_cifar10
from self_deep_convent import SelfDeepConvNet


def plot_cifar10_samples(x_train, t_train, num_samples=5):
    """可视化CIFAR-10数据集样本"""
    plt.figure(figsize=(15, 3))
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_train[i].transpose(1, 2, 0))  # 转换通道顺序以正确显示
        plt.title(f"{classes[t_train[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_filters(network, layer_name: str):
    """可视化卷积核"""
    filters = network.params[layer_name]

    plt.figure(figsize=(12, 3))
    for i in range(16):  # 展示前16个过滤器
        plt.subplot(2, 8, i + 1)
        plt.imshow(filters[i, 0], cmap="gray")  # 只显示第一个通道
        plt.axis("off")
    plt.suptitle("First Layer Conv Filters")
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(network, image):
    """可视化特征图"""
    # 获取第一个卷积层的特征图
    x = image[np.newaxis, :]
    conv1_out = None
    for key, layer in network.layers.items():
        x = layer.forward(x)
        if key == "Conv1":
            conv1_out = x
            break

    # 显示特征图
    plt.figure(figsize=(15, 8))
    for i in range(16):  # 显示前16个特征图
        plt.subplot(2, 8, i + 1)
        plt.imshow(conv1_out[0, i], cmap="viridis")
        plt.axis("off")
    plt.suptitle("Conv1 Feature Maps")
    plt.tight_layout()
    plt.show()


def analyze_model(network_path="results/cifar10_params.pkl"):
    """整体可视化分析"""
    # 加载数据和模型
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    network = SelfDeepConvNet(input_dim=(3, 32, 32))
    network.read_params(network_path)

    # 1. 显示原始数据样本
    print("显示CIFAR-10数据集样本:")
    plot_cifar10_samples(x_train, t_train)

    # 2. 显示卷积核
    print("\n显示第一层卷积核:")
    visualize_filters(network, "W1")
    print("\n显示第二层卷积核:")
    visualize_filters(network, "W2")
    print("\n显示第三层卷积核:")
    visualize_filters(network, "W3")

    # 3. 显示特征图
    print("\n显示第一个样本的特征图:")
    visualize_feature_maps(network, x_train[0])


if __name__ == "__main__":
    pkl_path = "results/cifar10_params-L2.pkl"
    analyze_model(pkl_path)
