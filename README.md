# CUFAR-10数据集训练

## 环境管理

本项目使用UV进行环境管理，具体使用方法如下：

```shell
# 安装UV（Windows环境）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 同步环境
uv sync
```

## 训练代码

train_simple_convent.py是一个简单的卷积神经网络训练代码

train_self_deep_convent.py是一个更深的卷积神经网络训练代码,具体结构如下：

- 3个卷积层,每层后接ReLU激活和池化层
- 2个全连接层,中间有ReLU激活
- 使用Softmax进行最终分类
- 权重使用He初始化方法
- Z-score标准化

train_self_deep_convent_L2.py相对于train_self_deep_convent.py使用了L2正则化，以及添加了Dropout层

训练过程中会自动检测数据集是否存在，如果不存在会自动下载，耐心等待即可

## 训练结果

所有最终的模型文件存储在result文件夹下
