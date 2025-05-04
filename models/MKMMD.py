import random
import time

import torch
from torch import nn


class MKMMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MKMMD, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def forward(self, source, target):
        batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
        kernels = self.guassian_kernel(source, target)
        # 根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss  # 因为一般都是n==m，也就是batch_size，所以L矩阵一般不加入计算


    def guassian_kernel(self, source, target):
        '''
        将源域数据和目标域数据转化为核矩阵并求和得到K
        Params:
    	    source: 源域数据（batch_size , d_model)
    	    target: 目标域数据（batch_size , d_model)
    	Return:
    		sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并[2*batch_size, d_model]
        # 将total复制（n+m）份 [1, 2*batch_size, d_model] -> [2*batch_size, 2*batch_size, d_model]
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total1 - total0) ** 2).sum(2)
        # 调整高斯核函数的sigma值
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)  # /len(kernel_val)


class MKMMD2(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MKMMD2, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def forward(self, total, batch_size):
        source, target = total.split(split_size=batch_size, dim=0)
        batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
        kernels = self.guassian_kernel(source, target)
        # 根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss  # 因为一般都是n==m，也就是batch_size，所以L矩阵一般不加入计算


    def guassian_kernel(self, source, target):
        '''
        将源域数据和目标域数据转化为核矩阵并求和得到K
        Params:
    	    source: 源域数据（batch_size , d_model)
    	    target: 目标域数据（batch_size , d_model)
    	Return:
    		sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并[2*batch_size, d_model]
        # 将total复制（n+m）份 [1, 2*batch_size, d_model] -> [2*batch_size, 2*batch_size, d_model]
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total1 - total0) ** 2).sum(2)
        # 调整高斯核函数的sigma值
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)  # /len(kernel_val)

if __name__ == '__main__':
    batch_size , d_model = 2, 3
    source = torch.ones((batch_size, d_model), dtype=torch.float32)
    target = torch.zeros((batch_size, d_model), dtype=torch.float32)
    SAMPLE_SIZE = 500
    buckets = 50

    # 第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
    mu = -0.6
    sigma = 0.15  # 将输出数据限制到0-1之间
    diff1 = []
    for i in range(10):
        diff1.append([random.lognormvariate(mu, sigma) for _ in range(0, SAMPLE_SIZE)])
    # 第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
    alpha = 1
    beta = 10
    diff2 = []
    for i in range(10):
        diff2.append([random.betavariate(alpha, beta) for _ in range(0, SAMPLE_SIZE)])
    X = torch.Tensor(diff1)
    Y = torch.Tensor(diff2)
    mk_mmd = MKMMD()
    loss = mk_mmd(X, Y)
    # print(loss)
    X = torch.randn((8, 256, 80, 80), device="cuda")
    Y = torch.randn((8, 256, 80, 80), device="cuda")
    start_time = time.time()
    loss = mk_mmd(X.flatten(1), Y.flatten(1))
    # loss = mk_mmd(X.flatten(1), Y.flatten(1))
    end_time = time.time()
    print(f"计算耗时: {end_time - start_time}")
    print(loss)
    total = torch.cat([X, Y], dim=0)
    mk_mmd2 = MKMMD2()
    start_time = time.time()
    loss = mk_mmd2(total.flatten(1), 8)
    end_time = time.time()
    print(f"计算耗时: {end_time - start_time}")
    print(loss)

