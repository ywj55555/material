import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def computer_weights(piexlist):
    sum_list = sum(piexlist) # 先求和
    
    piexlist_w = [((sum_list/x) if x!=0 else 0) for x in piexlist] #类别样本数量越少，权重越大
    sum_w = sum(piexlist_w) # 再求和 权重和
    w_final = [x/sum_w for x in piexlist_w]
    return w_final #返回每个类别的权重列表

def computer_exp_weights(piexlist,exp = np.exp(1)): #权重和数量成指数下降关系，最后再归一化
    sum_list = sum(piexlist)  # 先求和
    '''
    可选的比例方式，第一种应该要好一些
    piexlist_w = [(np.power(10, ((sum_list - x) / sum_list)) if x != 0 else 0) for x in piexlist]
    sum_w = sum(piexlist_w)  # 再求和 权重和,归一化
    w_final = [x / sum_w for x in piexlist_w]
    print(w_final)

    piexlist_w = [((sum_list / x) if x != 0 else 0) for x in piexlist]
    sum_w = sum(piexlist_w)  # 再求和
    w_final = [x / sum_w for x in piexlist_w]
    print(w_final)

    piexlist_w = [(np.power(1.01, (sum_list / x)) if x != 0 else 0) for x in piexlist]
    sum_w = sum(piexlist_w)  # 再求和 权重和,归一化
    w_final = [x / sum_w for x in piexlist_w]
    '''
    #少一个负号
    # piexlist_w = [(np.power(exp,(x/sum_list)) if x != 0 else 0) for x in piexlist]  # 类别样本数量越少，权重越大
    piexlist_w = [(np.power(exp, -(x / sum_list)) if x != 0 else 0) for x in piexlist]
    sum_w = sum(piexlist_w)  # 再求和 权重和,归一化,归一化后值变小了！！！
    w_final = [x / sum_w for x in piexlist_w]
    return w_final  # 返回每个类别的权重列表

def computer_10exp_weights(piexlist,exp = 10): #权重和数量成指数下降关系，最后再归一化
    sum_list = sum(piexlist)  # 先求和
    piexlist_w = [(np.power(exp, ((sum_list - x) / sum_list)) if x != 0 else 0) for x in piexlist]
    sum_w = sum(piexlist_w)  # 再求和 权重和,归一化,归一化后值变小了！！！
    w_final = [x / sum_w for x in piexlist_w]
    return w_final  # 返回每个类别的权重列表

'''在多分类中，我记得有篇文章里面有个动态图，当我们对神经网络的输出图output进行softmax之后，我们需要根据target，也就是打的标签
里面的每一个像素的值来选择output中每个像素某一个维度的值，然后我看网上公开的代码，其实就是类似这样操作的'''
def focus_loss(num_classes, input_data, target, tlabel,cuda=True,weight_fun = 3 ,exp = 10):
    n, h, w = target.shape
    input_data = torch.softmax(input_data, dim=1) # 先对数据进行softmax

    '''接下来就要根据打标签的图像来选择选取output中每个像素的最终输出值了'''
    # classes_mask = torch.zeros_like(input_data) # 这个mask用来跟input相乘，相当于要去对softmax输出的维度进行筛选
    # classes_mask.scatter_(1, target, 1)
    classes_mask = F.one_hot(target,num_classes)
    classes_mask = classes_mask.float()
    classes_mask = classes_mask.permute(0, 3, 1, 2) # 这一行代码是用来验证上一行代码是否正确
    # print(classes_mask) classes_mask就是one-hot编码？验证一下
    # 这个函数，我感觉我这样用是没问题的，将最终值输出进行验证，感觉也没有问题，这个函数就是根据target的数值做索引。
    input_data = torch.sum(input_data * classes_mask, dim=1) # 二者相乘，并且求和，这样就将每个像素对应的预测值选取成功

    gamma = 2

    '''接下来就是对一张输入图中不同标签类型的权重进行求解'''
    num_class_list = []
    #计算类别0出现问题，得传入原始label来计算数量，传入三个参数
    for i in range(num_classes):
        num_class_list.append(torch.sum(tlabel == i).item()) # 将每一种类别所占的像素总和找到并且放到一个list里面，计算类别0出现问题

    if weight_fun==1:
        weights_alpha = computer_weights(num_class_list)
    elif weight_fun==2:
        weights_alpha = computer_exp_weights(num_class_list,exp)
    else :
        weights_alpha = computer_10exp_weights(num_class_list,exp) # 这是固定的求解公式，相当于所有类别的权重之和为1，我们通过一定的公式将各自的权重找到

    weights_alpha = torch.tensor(weights_alpha)
    # print('before : ',weights_alpha.shape,'target : ',target.shape,'view ',target.view(-1).shape)
    weights_alpha = weights_alpha[target.view(-1)].reshape(n, 1, h, w) # 这一行代码有意思，值得注意，这一行代码实现了将target中的每一个像素点进行了权值分配
    # print('after : ',weights_alpha.shape)
    # print(weights_alpha.shape, weights_alpha)
    if cuda:
        weights_alpha = weights_alpha.cuda() # 这里是需要的，否则会因为数据在cpu和在gpu上的差距报错

    # loss = -(torch.pow((1-input_data), gamma))*torch.log(input_data) # 这个是没有α参数的损失函数
    # print(weights_alpha.size())
    # print(torch.pow((1-input_data), gamma).size())
    # print(input_data.size()[1])
    # print(torch.log(input_data).size())
    loss = -(weights_alpha * torch.pow((1-input_data), gamma) * torch.log(input_data)) # 这个是加了α的损失函数
    loss = torch.mean(loss)
    # print(loss.item())
    return loss


if __name__ == '__main__':
    pred = torch.rand((1, 4, 512, 512))
    y = torch.from_numpy(np.random.randint(0, 4, (1, 512, 512))).long()
    focus_loss(4, pred, y)

