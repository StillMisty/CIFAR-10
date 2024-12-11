from common.trainer import Trainer
from dataset.cifar10 import load_cifar10
from simple_convnet import SimpleConvNet

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(trainer: Trainer):
    # 绘制训练过程的损失和准确率曲线

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


def train_cifar10():
    # 加载CIFAR-10数据
    (x_train, t_train), (x_test, t_test) = load_cifar10()

    # 归一化处理
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    network = SimpleConvNet(input_dim=(3, 32, 32))

    # 跑20轮，使用Adam优化器，学习率为0.001
    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=20,
        mini_batch_size=100,
        optimizer="Adam",
        optimizer_param={"lr": 0.001},
        evaluate_sample_num_per_epoch=1000,
    )

    # 训练网络
    trainer.train()

    # 计算并打印最终的测试准确率
    test_acc = network.accuracy(x_test, t_test)
    print("=============== Final Test Accuracy ===============")
    print(f"Test accuracy: {test_acc:.4f}")

    # 保存模型参数
    network.save_params("result/cifar10_params-self_simple_convent.pkl")
    print("别摸了，跑完了！")

    # 绘制训练过程的损失和准确率曲线
    plot_training_curves(trainer)


# 使用CIFAR-10数据集训练简单卷积神经网络
if __name__ == "__main__":
    train_cifar10()
