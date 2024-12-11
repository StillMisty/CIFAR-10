import pickle
import numpy as np
from collections import OrderedDict

from common.layers import (
    Affine,
    Convolution,
    Dropout,
    Pooling,
    Relu,
    SoftmaxWithLoss,
)


class SelfDeepConvNetL2:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param_1={"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_2={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_3={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
        hidden_size=50,
        output_size=10,
        weight_decay_lambda=0.1,
        dropout_ration=0.5,
    ):
        """_summary_

        Args:
            weight_decay_lambda (float, optional): L2正则化系数. Defaults to 0.1.
        """
        # 初始化权重
        pre_node_nums = np.array(
            [1 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 64 * 7 * 7, hidden_size]
        )
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0]
        # 初始化第一个卷积层
        self.params["W1"] = weight_init_scales[0] * np.random.randn(
            conv_param_1["filter_num"],
            pre_channel_num,
            conv_param_1["filter_size"],
            conv_param_1["filter_size"],
        )
        self.params["b1"] = np.zeros(conv_param_1["filter_num"])

        # 初始化第二个卷积层
        self.params["W2"] = weight_init_scales[1] * np.random.randn(
            conv_param_2["filter_num"],
            conv_param_1["filter_num"],
            conv_param_2["filter_size"],
            conv_param_2["filter_size"],
        )
        self.params["b2"] = np.zeros(conv_param_2["filter_num"])

        # 初始化第三个卷积层
        self.params["W3"] = weight_init_scales[2] * np.random.randn(
            conv_param_3["filter_num"],
            conv_param_2["filter_num"],
            conv_param_3["filter_size"],
            conv_param_3["filter_size"],
        )
        self.params["b3"] = np.zeros(conv_param_3["filter_num"])

        # 初始化全连接层
        self.params["W4"] = weight_init_scales[3] * np.random.randn(
            64 * 4 * 4, hidden_size
        )
        self.params["b4"] = np.zeros(hidden_size)
        self.params["W5"] = weight_init_scales[4] * np.random.randn(
            hidden_size, output_size
        )
        self.params["b5"] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param_1["stride"],
            conv_param_1["pad"],
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Conv2"] = Convolution(
            self.params["W2"],
            self.params["b2"],
            conv_param_2["stride"],
            conv_param_2["pad"],
        )
        self.layers["Relu2"] = Relu()
        self.layers["Pool2"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Conv3"] = Convolution(
            self.params["W3"],
            self.params["b3"],
            conv_param_3["stride"],
            conv_param_3["pad"],
        )
        self.layers["Relu3"] = Relu()
        self.layers["Pool3"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Affine1"] = Affine(self.params["W4"], self.params["b4"])
        self.layers["Relu4"] = Relu()
        # 添加Dropout层
        self.layers["Dropout1"] = Dropout(dropout_ration)
        self.layers["Affine2"] = Affine(self.params["W5"], self.params["b5"])

        self.last_layer = SoftmaxWithLoss()

        self.weight_decay_lambda = weight_decay_lambda

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)

        weight_decay = 0
        for idx in range(1, 6):  # 对所有权重进行遍历
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 保存并返回梯度
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Conv2"].dW, self.layers["Conv2"].db
        grads["W3"], grads["b3"] = self.layers["Conv3"].dW, self.layers["Conv3"].db
        grads["W4"], grads["b4"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W5"], grads["b5"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def read_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params[f"W{i+1}"]
            self.layers[key].b = self.params[f"b{i+1}"]


if __name__ == "__main__":
    network = SelfDeepConvNetL2()
    print(network.params)
