import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import shutil

import base_utilize
from base_utilize import get_vm_tasks_capacity, get_similar_performance_vm
# from torchsummary import summary
import globals

GAMMA = 0.7  # reward discount，惩罚项
TARGET_REPLACE_ITER = 50  # target update frequency，每过多少轮更新TargetNet


class DQN(object):
    # 每次把一个任务分配给一个虚拟机
    def __init__(self, state_dim, vms):
        self.state_dim = state_dim # 状态维度
        self.vms = vms  # 虚拟机数量
        self.a_dim = self.vms   # 动作空间

        self.lr = 0.003  # learning rate
        self.batch_size = 32  # 128
        self.epsilon = 1.0
        # self.epsilon = 0.95   # epsilon初始值
        self.epsilon_decay = 0.999998  # epsilon退化率
        self.epsilon_min = 0.1      # epsilon最小值
        self.step = 0
        self.target_prob = 1.0      # 学习目标算法的概率，若目标算法为fine grain，则有50%的几率在随机选择动作时以fine grain策略选择动作

        # print("state_dim: ", self.state_dim)
        # print("a_dim: ", self.a_dim)

        self.eval_net = QNet_v1(self.state_dim, self.a_dim)

        # 打印网络结构参数
        # device = torch.device('cpu')
        # net_graph = self.eval_net.to(device)
        # summary(net_graph)

        self.eval_net.apply(self.weights_init)
        self.target_net = QNet_v1(self.state_dim, self.a_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.hard_update(self.target_net, self.eval_net)  # 初始化为相同权重

        self.loss_f = nn.MSELoss()

    # 一个状态传入，给状态选择一个动作
    def choose_action(self, state):
        # print("state: ", state)
        if self.epsilon > self.epsilon_min:  # epsilon最小值
            self.epsilon *= self.epsilon_decay
        # else:
        #     print("epsilon reached epsilon_min!")
        # print("epsilon val: ", self.epsilon)
        if np.random.uniform() > self.epsilon:  # np.random.uniform()输出0到1之间的一个随机数
            self.eval_net.eval()    # 进入evaluate模式，区别于train模式，在eval模式，框架会自动把BatchNormalize和Dropout固定住，不会取平均，而是用训练好的值
            actions_value = self.eval_net(state)
            # print("actions_value:")
            # print(actions_value)
            # 原始方式，直接根据最大值选择动作
            action = torch.max(actions_value, 1)[1].data.numpy()
            # print("action: ", action)
            action = action[0]
            # print("actions_value: ", actions_value)
            # print("action[0]: ", action)
        else:
            # 增加虚拟机分配的合理性，避免任务都集中分配到某一个机器上
            # 选择随机机器策略
            action = np.random.randint(0, self.vms)
        return action

    def learn(self):
        # 更新 Target Net
        if self.step % TARGET_REPLACE_ITER == 0:
            self.hard_update(self.target_net, self.eval_net)

        # 训练Q网络
        self.eval_net.train()
        # Q预测值
        q_eval = self.eval_net(self.bstate).gather(1, self.baction)  # shape (batch, 1), gather表示获取每个维度action为下标的Q值
        # print("q_eval: ", q_eval)

        self.target_net.eval()
        q_next = self.target_net(self.bstate_).detach()  # 设置 Target Net 不需要梯度

        # Q现实值
        # Tensor.view()返回的新tensor与原先的tensor共用一个内存,只是将原tensor中数据按照view(M,N)中，M行N列显示出来
        # Tensor.max(1)表示返回每一行中最大值的那个元素，且返回其索引
        # Tensor.max(1)[0]表示只返回每一行中最大值的那个元素
        # Tensor.unsqueeze(dim)用来扩展维度，在指定位置加上维数为1的维度，dim可以取0,1,...或者负数，这里的维度和pandas的维度是一致的，0代表行扩展，1代表列扩展
        # Tensor.squeeze()则用来对维度进行压缩，去掉所有维数为1（比如1行或1列这种）的维度，不为1的维度不受影响
        # Tensor.expand()将单个维度扩展成更大的维度，返回一个新的tensor

        # 先用Q_eval即最新的神经网络估计Q_next即Q现实中的Q(S',a')中的最大动作值所对应的索引
        self.eval_net.eval()
        q_eval_next = self.eval_net(self.bstate_).detach()
        q_eval_action = q_eval_next.max(1)[1].view(self.batch_size, 1)

        # 然后用这个被Q_eval估计出来的动作来选择Q现实中的Q(s')
        q_target_prime = q_next.gather(1, q_eval_action)
        q_target = self.breward + GAMMA * q_target_prime
        loss = self.loss_f(q_eval, q_target)

        # 将梯度初始化为零
        self.optimizer.zero_grad()
        # 反向传播求梯度
        loss.backward()
        # 更新所有参数
        self.optimizer.step()
        return loss.detach().numpy()

    def store_memory(self, state_all, action_all, reward_all):
        indexs = np.random.choice(len(state_all[:-1]), size=self.batch_size)

        self.bstate = torch.from_numpy(state_all[indexs, :]).float()
        self.bstate_ = torch.from_numpy(state_all[indexs + 1, :]).float()
        self.baction = torch.LongTensor(action_all[indexs, :])
        self.breward = torch.from_numpy(reward_all[indexs, :]).float()  # 奖励值值越大越好

    # 全部更新
    def hard_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    # 初始化网络参数
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):  # 批归一化层初始化
            nn.init.uniform_(m.bias)  # 初始化为U(0,1)
            nn.init.constant_(m.bias, 0)


class QNet_v1(nn.Module):  # 通过 s 预测出 a
    def __init__(self, s_dim, a_dim):
        super(QNet_v1, self).__init__()
        self.state_dim = s_dim
        self.action_dim = a_dim
        # task_num = 200:  task_dim--64--128--a_dim
        self.layer1 = nn.Sequential(
            nn.Linear(s_dim, 32),
            torch.nn.Dropout(0.2),              # Dropout层
            nn.BatchNorm1d(32),                 # 归一化层，参数为维度
            nn.LeakyReLU(),                     # 激活函数
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 64),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, self.action_dim)        # 输出为动作的维度
        )

    def forward(self, x):
        # list转tensor
        x = torch.Tensor(x)
        x_dim = len(x.size())
        if x_dim == 1:
            x = x.reshape(1, self.state_dim)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
