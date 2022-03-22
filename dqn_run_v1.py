import os
import numpy as np
import time
import random
import sys
from util.fileio import FileIo
import torch

from core.task import Task
from dqn.DQN_model_v1 import DQN
from base_utilize import *


# 通过任务和机器获取状态
# 状态定义：[任务利用率，vm1_utilization, vm2_utilization, ...]
def get_state(tasks_list, machines):
    start_time = tasks_list[0].start_time  # 当前批次任务的开始时间
    machines_state = []
    for machine in machines:
        machines_state.append(machine.mips)
        # machines_state.append(max(machine.next_start_time - start_time, 0))  # 等待时间
    tasks_state = []
    for i, task in enumerate(tasks_list):
        task_state = []
        # task_state.append(task.mi)
        task_state.append(task.cpu_utilization)
        # task_state.append(task.mi / machines[0].speed)  # 传输时间
        task_state += machines_state  # 由于是DQN，所以一个任务状态加上多个虚拟机状态
        # if (i == 1): print(task_state)
        tasks_state.append(task_state)
    # 返回值 [[[153.0, 0.79, 0.34, 600, 0, 600, 0, 500, 0, 500, 0, 400, 0, 400, 0, 300, 0, 300, 0, 200, 0, 200, 0]... ]]
    #           任务长度，任务利用率，任务传输时间，vm1_mips, vm1_waitTime, vm2....
    return tasks_state


# 初始化状态
def init_first_state(taskNum):
    initState = []
    for i in range(taskNum):
        initState.append(-1)
    return initState


# 初始化运行时间列表
def init_run_time_list(vmNum):
    vm_run_time_list = []
    for i in range(vmNum):
        vm_run_time_list.append(0)
    return vm_run_time_list


# 计算所有虚拟机的最迟运行时间
def get_task_run_time(state, tasks, machines):
    # 先计算得到每个虚拟机的运行时间
    vm_run_time_list = init_run_time_list(len(machines))
    for i, vmId in enumerate(state):
        # print("vmId:", vmId)
        # print("type of vmId: ", type(vmId))
        if vmId != -1:
            # 两种运行时间计算方式，一种考虑任务cpu利用率，一种不考虑
            # vm_run_time_list[vmId] += tasks[i].mi / machines[vmId].mips
            vm_run_time_list[vmId] += tasks[i].mi / (machines[vmId].mips * tasks[i].cpu_utilization)

    # 再得到所有虚拟机中的最迟完成时间
    biggest_idx = 0
    for i in range(len(vm_run_time_list)-1):
        if (vm_run_time_list[i+1] > vm_run_time_list[biggest_idx]):
            biggest_idx = i+1
    return vm_run_time_list[biggest_idx]


# 计算某个状态的平均任务排队时间
def get_state_avg_task_waiting_time(state, tasks, machines):
    # 先计算得到每个虚拟机的运行时间
    vm_run_time_list = init_run_time_list(len(machines))
    task_waiting_time_list = []
    for i, vmId in enumerate(state):
        # print("vmId:", vmId)
        # print("type of vmId: ", type(vmId))
        if vmId != -1:
            # 当前任务的排队时间即vm_run_time_list[vmId]
            task_waiting_time_list.append(vm_run_time_list[vmId])
            # 两种运行时间计算方式，一种考虑任务cpu利用率，一种不考虑
            # vm_run_time_list[vmId] += tasks[i].mi / machines[vmId].mips
            vm_run_time_list[vmId] += tasks[i].mi / (machines[vmId].mips * tasks[i].cpu_utilization)

    # 取task_waiting_time_list的均值即可
    ret = np.mean(task_waiting_time_list)
    return ret


def is_valid(result_list, dec_rate, step_per_rate, result, search_epoch):
    # print(f"result_list[-1]: {result_list[-1]}, result_list[-1]*1.5: {result_list[-1] * 1.5}, result: {result}")
    if search_epoch > 50 and result < result_list[-1] * 1.2:
        return True
    if search_epoch > 100 and result < result_list[-1] * 1.5:
        return True
    if search_epoch > 150 and result < result_list[-1] * 2.0:
        return True
    if search_epoch > 300 and result < result_list[-1] * 3.0:
        return True
    if result_list[-1] * (dec_rate + step_per_rate * 5) > result > result_list[-1] * (dec_rate - step_per_rate * 5):
        return True
    else:
        return False


def main(cluster, filepath_input, file_str):
    # 虚拟机的数目
    vmNum = len(cluster.machines)
    print("vmsNum: %d" % vmNum)
    # 任务数目
    taskNum = globals.TASK_NUM
    stateDim = taskNum

    # 从输入文件路径读取所有的任务, all_batch_tasks的长度为
    all_batch_tasks = FileIo(filepath_input).readAllBatchLines()
    # print(all_batch_tasks[0])
    print("batchNum: %d" % len(all_batch_tasks))
    print("taskNum: %d" % len(all_batch_tasks))
    print("环境创建成功！")

    tasks_list = []
    for task in all_batch_tasks:
        tasks_list.append(Task(task[0][0], task[0][1], task[0][2], task[0][3]))  # 构建任务
        # tasks_list.append(Task(task[0], task[1], task[2], task[3]))  # 构建任务
    print("len(tasks_list):", len(tasks_list))

    # 初始化初始状态，每个状态定义为每个任务分配给的虚拟机id，例[vm1, vm2, vm5, vm4, ...]
    initState = init_first_state(taskNum)


    # 用同个batch_tasks多次训练，外层再套个循环作为训练的epoch
    epoches = 20
    step = 0

    # 下降速率
    dec_rate = 0.5
    target_rate = 1.0
    step_per_rate = (target_rate - dec_rate) / epoches
    result_list = []

    while step < epoches:
        print("epoch[%d]" % step)
        state_all = []  # 存储所有的状态 [None,2+2*20]
        action_all = []  # 存储所有的动作 [None,1]
        reward_all = []  # 存储所有的奖励 [None,1]

        best_time_results = 0
        best_std = 0
        best_state = []
        dec_rate += step_per_rate
        init_epoch = 10
        search_epoch = 0

        DRL = DQN(stateDim, vmNum)
        print("DQN网络初始化成功！")
        found = False

        while True:
            state = initState
            search_epoch += 1
            print("search_epoch: ", search_epoch)

            # i是任务的索引
            for i in range(taskNum):
                # print("state: ", state)
                state_all.append(state)
                # print("state_all: ", state_all)
                # print("state: ", state)
                # print("np.array(state): ", np.array(state))
                machines_id = DRL.choose_action(state)  # 通过调度算法得到分配 id
                # print("task[%d] --> machine[%d]" % (i, machines_id))
                # machines_id = machines_id.astype(int).tolist()
                action_all.append([machines_id])
                state[i] = machines_id

                # 取最后一次迭代的结果，作为分配策略
                task_run_time = get_task_run_time(state, tasks_list, cluster.machines)
                vm_utilization, vm_utilization_std = get_vm_avg_cpu_utilization(state, tasks_list)
                if i == taskNum - 1:
                    # print(f"state: {state}")
                    if len(result_list) == 0:
                        best_time_results = max(best_time_results, task_run_time)
                        best_std = vm_utilization_std
                        best_state = state
                        init_epoch -= 1
                        if init_epoch == 0:
                            result_list.append(best_time_results)
                            print("best_time_results: ", best_time_results)
                            print("best_state: ", best_state)
                            found = True
                    elif is_valid(result_list, dec_rate, step_per_rate, task_run_time, search_epoch):
                        best_time_results = task_run_time
                        best_std = vm_utilization_std
                        best_state = state
                        result_list.append(best_time_results)
                        print("best_time_results: ", best_time_results)
                        print("best_state: ", best_state)
                        found = True

                """  ---------奖惩函数的设计----------  """
                reward = 1 / math.log(task_run_time)
                reward_all.append([reward])  # 计算奖励

            # 减少存储数据量
            if len(state_all) > 20000:
                # print("Cutting data now!")
                state_all = state_all[-10000:]
                action_all = action_all[-10000:]
                reward_all = reward_all[-10000:]

            # 先学习一些经验，再学习
            if step > 10:
                # 截取最后10000条记录
                new_state = np.array(state_all, dtype=np.float32)[-10000:-1]
                new_action = np.array(action_all, dtype=np.float32)[-10000:-1]
                new_reward = np.array(reward_all, dtype=np.float32)[-10000:-1]

                DRL.store_memory(new_state, new_action, new_reward)
                DRL.step = step
                loss = DRL.learn()
                if i == taskNum - 1:
                    print("step:", step, ", loss:", loss)

            if found:
                break

        # 每次迭代结束记录结果
        best_avg_task_waiting_time = get_state_avg_task_waiting_time(best_state, tasks_list,
                                                                     cluster.machines)
        if not os.path.exists(file_str):
            with open(file_str, 'w') as f:
                f.write("task_num\twait_time\texecute_time\tcpu_std\n")
        with open(file_str, 'a') as f:
            f.write("%d\t%.3f\t%.3f\t%.3f\n" % (
                taskNum, best_avg_task_waiting_time, best_time_results, best_std))

        # 每轮step自增1
        step += 1


if __name__ == '__main__':
    start_time = time.time()
    cluster = creat_cluster_from_file()

    filepath_input = "data/Alibaba/Alibaba-Cluster-trace-" + str(globals.TASK_NUM) + ".txt"
    file_path = "result/double_dqn_result.txt"
    if os.path.exists(file_path):
        os.remove(file_path)

    main(cluster, filepath_input, file_path)

    finish_time = time.time()
    print("Time used: %.2f s" % (finish_time - start_time))
