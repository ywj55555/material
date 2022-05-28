import torch
import torch.nn.functional as F
from utils.focalloss import *
#在使用DICE loss时，对小目标是十分不利的
#尝试 dice + focal loss 或者 dice + weight loss

def dice_loss(pred, target, smooth=1.): #B*C*H*W C为1 单纯的dice_loss,交集比上并集，smooth为Laplace smoothing
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean() #返回的值是0-1之间 loss越小越好

#DICE loss和交叉熵不是一个数量级 对DICE加一个 -log平衡一下
def calc_loss(pred, target, bce_weight=0.5):
    # print('pre:',pred.size())
    bce = F.binary_cross_entropy_with_logits(pred, target)
    # print('bce:',bce)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    # loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = bce * bce_weight - torch.log(1-dice) * (1 - bce_weight)
    return loss

def focal_dice_loss(pred, target,tlabel,foc_weight=0.5,weight_fun = 3,exp=10):
    # print('pre:',pred.size())
    # bce = F.binary_cross_entropy_with_logits(pred, target)
    true_label = torch.argmax(target,1)
    focal = focus_loss(pred.size()[1],pred,true_label,tlabel,cuda=True,weight_fun = weight_fun,exp=exp)
    # print('bce:',bce)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    # loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = focal * foc_weight - torch.log(1-dice) * (1 - foc_weight)
    return loss

def calc_loss_per(pred, target, bce_weight=0.5, percent=1):
    # print('pre:',pred.size())
    bce = F.binary_cross_entropy_with_logits(pred, target) #多标签二分类损失函数，有些问题，本实验是多分类问题，实际上是对每个类做二分类判别，看实际效果
    # print('bce:',bce)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = loss * percent
    # print('per:',percent)
    return loss

#语义分割输出是B*C*H*W 可以转为B*H*W C的二维tensor，展开成每个像素，然后每个像素做多分类任务，focal loss公式就是那个公式
#改成pytorch的代码就好了
#也可以改成one-hot的形式，每个通道做二分类的focal loss,然后相加
# def test_softmax_cross_entropy_with_logits(n_classes, logits, true_label):
#     epsilon = 1.e-8
#     # 得到y_true和y_pred
#     y_true = tf.one_hot(true_label, n_classes)
#     softmax_prob = tf.nn.softmax(logits)
#     y_pred = tf.clip_by_value(softmax_prob, epsilon, 1. - epsilon)
#     # 得到交叉熵，其中的“-”符号可以放在好几个地方，都是等效的，最后取mean是为了兼容batch训练的情况。
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_pred)))
#
# def test_softmax_focal_ce_3(n_classes, gamma, alpha, logits, label):
#     epsilon = 1.e-8
#     # y_true and y_pred
#     y_true = tf.one_hot(label, n_classes)
#     probs = tf.nn.softmax(logits)
#     y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
#
#     # weight term and alpha term【因为y_true是只有1个元素为1其他元素为0的one-hot向量，所以对于每个样本，只有y_true位置为1的对应类别才有weight，其他都是0】这也是为什么网上有的版本会用到tf.gather函数，这个函数的作用就是只把有用的这个数取出来，可以省略一些0相关的运算。
#     weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
#     if alpha != 0.0:  # 我这实现中的alpha只是起到了调节loss倍数的作用（调节倍数对训练没影响，因为loss的梯度才是影响训练的关键），要想起到调节类别不均衡的作用，要替换成数组，数组长度和类别总数相同，每个元素表示对应类别的权重。另外[这篇](https://blog.csdn.net/Umi_you/article/details/80982190)博客也提到了，alpha在多分类Focal loss中没作用，也就是只能调节整体loss倍数，不过如果换成数组形式的话，其实是可以达到缓解类别不均衡问题的目的。
#         alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
#     else:
#         alpha_t = tf.ones_like(y_true)
#
#     # origin x ent，这里计算原始的交叉熵损失
#     xent = tf.multiply(y_true, -tf.log(y_pred))
#
#     # focal x ent，对交叉熵损失进行调节，“-”号放在上一行代码了，所以这里不需要再写“-”了。
#     focal_xent = tf.multiply(alpha_t, tf.multiply(weight, xent))
#
#     # in this situation, reduce_max is equal to reduce_sum，因为经过y_true选择后，每个样本只保留了true label对应的交叉熵损失，所以使用max和使用sum是同等作用的。
#     reduced_fl = tf.reduce_max(focal_xent, axis=1)
#     return tf.reduce_mean(reduced_fl)
# weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
# no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
def L2Loss(model,alpha):
    l2_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        # if 'bias' not in name:
        if 'bias' not in name and "bn" not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma,2)))
    return l2_loss

def L1Loss(model,beta):
    l1_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name and "bn" not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(parma))
    return l1_loss
