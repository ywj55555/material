import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    # dy=y*(1-y)
    return y


def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.savefig('./res/sigmoid.png')
    plt.show()


import math
from matplotlib import pyplot as plt
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.where(x < 0, 0, x)


def prelu(x):
    return np.where(x < 0, 0.1 * x, x)


'''
def sigmoid(x):
    result = 1 / (1 + math.e ** (-x))
    return result
'''


def plot_softmax():
    x = np.linspace(-10, 10, 200)
    y = softmax(x)
    plt.plot(x, y, label="softmax", linestyle='-', color='blue')
    plt.legend()
    plt.savefig("softmax.png")
    # plt.show()


def plot_sigmoid():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(-10, 10)
    y = sigmoid(x)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.set_yticks([-1, -0.5, 0.5, 1])
    plt.plot(x, y, label="Sigmoid", linestyle='-', color='blue')
    plt.legend()
    plt.savefig("sigmoid.png")
    # plt.show()


def plot_tanh():
    x = np.arange(-10, 10, 0.1)
    y = tanh(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, label="tanh", linestyle='-', color='blue')
    plt.legend()
    plt.xlim([-10.05, 10.05])
    plt.ylim([-1.02, 1.02])
    ax.set_yticks([-1.0, -0.5, 0.5, 1.0])
    ax.set_xticks([-10, -5, 5, 10])
    plt.tight_layout()
    plt.savefig("tanh.png")
    # plt.show()


def plot_relu():
    x = np.arange(-10, 10, 0.1)
    y = relu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y, label="relu", linestyle='-', color='blue')
    plt.legend()
    plt.xlim([-10.05, 10.05])
    plt.ylim([0, 10.02])
    ax.set_yticks([2, 4, 6, 8, 10])
    plt.tight_layout()
    plt.savefig("relu.png")
    # plt.show()


def plot_prelu():
    x = np.arange(-10, 10, 0.1)
    y = prelu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, label="leaky-relu", linestyle='-', color='blue')
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("./res/leaky-relu.png")
    # plt.show()


if __name__ == "__main__":
    # plot_softmax()
    # plot_sigmoid()
    # plot_tanh()
    # plot_relu()
    plot_prelu()
# if __name__ == '__main__':
#     plot_sigmoid()