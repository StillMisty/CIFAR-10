import numpy as np
import matplotlib.pyplot as plt
from common.layers import *
from common.trainer import Trainer
from dataset.cifar10 import load_cifar10
from self_deep_convent import SelfDeepConvNet


def plot_training_curves(trainer):
    """绘制训练过程的损失和准确率曲线"""
    plt.figure(figsize=(15, 5))

    # 绘制损失函数
    plt.subplot(121)
    x = np.arange(trainer.max_iter)
    plt.plot(x, trainer.train_loss_list, marker="o", label="train", markevery=100)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend(loc="upper right")

    # 绘制准确率
    plt.subplot(122)
    plt.plot(
        np.arange(len(trainer.train_acc_list)),
        trainer.train_acc_list,
        marker="o",
        label="train",
        markevery=2,
    )
    plt.plot(
        np.arange(len(trainer.test_acc_list)),
        trainer.test_acc_list,
        marker="s",
        label="test",
        markevery=2,
    )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.title("Accuracy")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def preprocess_cifar10(x_train, x_test):
    # 标准化(Z-score标准化)
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    return x_train, x_test


def train_cifar10():
    # 加载CIFAR-10数据集
    (x_train, t_train), (x_test, t_test) = load_cifar10()

    x_train, x_test = preprocess_cifar10(x_train, x_test)

    # 创建网络(输入为3通道32x32图像)
    # network = SelfDeepConvNet(input_dim=(3, 32, 32))
    network = SelfDeepConvNet(input_dim=(3, 32, 32))

    # 配置训练器
    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=30,
        mini_batch_size=100,
        optimizer="Adam",
        optimizer_param={"lr": 0.001},
        evaluate_sample_num_per_epoch=1000,
    )

    # 训练网络
    trainer.train()

    # 计算最终测试准确率
    test_acc = network.accuracy(x_test, t_test)
    print("=============== Final Test Accuracy ===============")
    print(f"Test accuracy: {test_acc:.4f}")

    # 保存训练好的模型参数
    network.save_params("results/cifar10_params.pkl")
    print("别摸了，跑完了！")

    # 绘制训练过程曲线
    plot_training_curves(trainer)


if __name__ == "__main__":
    train_cifar10()
