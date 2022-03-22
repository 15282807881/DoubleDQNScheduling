import math

from core.cluster import Cluster
from core.machine import Machine
from util.fileio import FileIo
import globals

# 创建集群

# 从文件读入虚拟机数据
def creat_cluster_from_file():
    # 共globals.VM_NUM台机器
    cluster = Cluster()
    lines = []
    vm_data = FileIo(r"D:\dev\task-scheduler-based-on-DQN-main\代码\pycloudsim-master\pycloudsim-master\data\create\vm_normal.txt").readAllLines(lines)
    # print("lines[1]:", task_data[0][1])
    for i in range(globals.VM_NUM):
        cluster.add_machine(Machine(mips=vm_data[i][1], speed=vm_data[i][1] / 10, micost=1))
    # print("len[cluster]:", len(cluster.machines))
    return cluster

def creat_cluster():
    cluster = Cluster()
    eachNum = 10
    for i in range(eachNum):
        cluster.add_machine(Machine(mips=600, speed=450, micost=1))  # 构建虚拟机
    for i in range(eachNum):
        cluster.add_machine(Machine(mips=500, speed=450, micost=1))
    for i in range(eachNum):
        cluster.add_machine(Machine(mips=400, speed=450, micost=1))
    for i in range(eachNum):
        cluster.add_machine(Machine(mips=300, speed=450, micost=1))
    for i in range(eachNum):
        cluster.add_machine(Machine(mips=200, speed=450, micost=1))
    return cluster

# 创建大集群
def creat_cluster_large():
    cluster = Cluster()
    for i in range(5):
        cluster.add_machine(Machine(mips=2000, speed=500, micost=1))  # 构建虚拟机
    for i in range(5):
        cluster.add_machine(Machine(mips=1000, speed=500, micost=1))
    for i in range(5):
        cluster.add_machine(Machine(mips=500, speed=500, micost=1))
    for i in range(5):
        cluster.add_machine(Machine(mips=250, speed=500, micost=1))
    return cluster

# 创建大集群
def creat_cluster_large_multiple():
    cluster = Cluster()
    for i in range(4):
        cluster.add_machine(Machine(mips=2000, speed=560, micost=1.71))  # 构建虚拟机
    for i in range(4):
        cluster.add_machine(Machine(mips=1500, speed=529, micost=2.23))
    for i in range(4):
        cluster.add_machine(Machine(mips=1000, speed=511, micost=2.3))
    for i in range(4):
        cluster.add_machine(Machine(mips=500, speed=448, micost=2.19))
    for i in range(4):
        cluster.add_machine(Machine(mips=250, speed=467, micost=2.18))
    return cluster

# 根据机器性能分配任务数
def get_vm_tasks_capacity(cluster):
    # 定义数组用于存储最终结果
    vm_tasks_capacity = []
    # 所有的机器
    machines = cluster.machines
    # 所有机器的总mips
    total_mips = 0
    for machine in machines:
        vm_tasks_capacity.append(0)
        total_mips += machine.mips
    # print("total_mips: ", total_mips)
    # 计算每个机器mips占总mips的比例
    for i, machine in enumerate(machines):
        vm_tasks_capacity[i] = math.ceil(((float)(machine.mips) / total_mips) * globals.TASK_NUM)
    # print("vm_tasks_capacity: ", vm_tasks_capacity)
    return vm_tasks_capacity

# 将所有机器按照性能排序，返回排序好的list
def get_vms_sorted_by_perf(cluster):
    # 所有的机器
    machines = cluster.machines
    # 把所有的机器按照mips降序排序
    # 排序结果存储为虚拟机的id
    sort_results = []
    for i in range(len(machines)):
        sort_results.append(i)
    for i in range(globals.VM_NUM - 1):
        for j in range(i+1, globals.VM_NUM):
            if machines[sort_results[j]].mips > machines[sort_results[i]].mips:
                tmp = sort_results[i]
                sort_results[i] = sort_results[j]
                sort_results[j] = tmp
    # 打印排序结果
    # print("sorted vm: ", sort_results)
    # for id in sort_results:
    #     print("vm[%d].mips: %d \n" % (id, machines[id].mips))
    return sort_results


# 寻找与某个机器性能相近的机器
def get_similar_performance_vm(cluster, machine_id):
    # 获取按照性能排序的机器列表
    sorted_results = get_vms_sorted_by_perf(cluster)
    # 创建返回结果列表
    ret_list = []
    input_idx = -1
    for i, id in enumerate(sorted_results):
        if id == machine_id:
            input_idx = i
    # 用两个指针分别从input_idx位置开始前后遍历
    # print("input_idx: ", input_idx)
    pre_ptr = input_idx - 1
    post_ptr = input_idx + 1
    while (pre_ptr >= 0 and post_ptr < globals.VM_NUM):
        ret_list.append(sorted_results[pre_ptr])
        ret_list.append(sorted_results[post_ptr])
        pre_ptr -= 1
        post_ptr += 1
    while (pre_ptr >= 0):
        ret_list.append(sorted_results[pre_ptr])
        pre_ptr -= 1
    while (post_ptr < globals.VM_NUM):
        ret_list.append(sorted_results[post_ptr])
        post_ptr += 1
    # print("ret_list: ", ret_list)
    # print("current id: ", machine_id)
    return ret_list


# 将所有任务按照任务量排序
def get_tasks_sorted_by_mips(tasks_list):
    # 把所有的任务按照任务量降序排序
    # 排序结果存储为task的id
    sort_results = []
    for i in range(len(tasks_list)):
        sort_results.append(i)
    for i in range(globals.TASK_NUM - 1):
        for j in range(i+1, globals.TASK_NUM):
            if tasks_list[sort_results[j]].mi > tasks_list[sort_results[i]].mi:
                tmp = sort_results[i]
                sort_results[i] = sort_results[j]
                sort_results[j] = tmp
    # 打印排序结果
    # print("sorted tasks: ", sort_results)
    # for id in sort_results:
    #     print("task[%d].mips: %d \n" % (id, tasks_list[id].mi))
    return sort_results


# 先做排序，然后将所有排序好的任务平均分配给排序好的机器
def distribute_machine_for_tasks(cluster, tasks_list):
    # 获取按照性能排序的机器列表
    machine_sorted_results = get_vms_sorted_by_perf(cluster)
    # print("machine_sorted_results: ", machine_sorted_results)
    # 获取按照任务量排序的任务列表
    task_sorted_results = get_tasks_sorted_by_mips(tasks_list)
    # print("task_sorted_results: ", task_sorted_results)

    ret_dict = {}
    task_num = len(task_sorted_results)
    machine_num = len(machine_sorted_results)
    each_num = math.ceil(task_num / machine_num)
    tmp_num = each_num
    machine_idx = 0
    for task_id in task_sorted_results:
        ret_dict[str(task_id)] = machine_sorted_results[machine_idx]
        tmp_num -= 1
        if (tmp_num == 0):
            tmp_num = each_num
            machine_idx += 1
    # print("distribute_results: ", ret_dict)
    return ret_dict


# 按照资源细粒度的方式规划虚拟机性能和任务分配策略，使DQN基于此进行搜索，返回基本fine grain策略为task_id对应的machine_id
def choose_action_by_fine_grain_strategy(task_id, distribute_results):
    ret = distribute_results[str(task_id)]
    # print("action: ", ret)
    # print("choose_action_by_fine_grain_strategy return vm[%d] for task[%d]" % (ret, task_id))
    return ret


# 按照资源细粒度的方式进行任务调度，返回一个调度状态
def get_state_by_fine_grain_strategy(distribute_results):
    ret_state = []
    for i in range(globals.TASK_NUM):
        ret_state.append(distribute_results[str(i)])
    # print("action: ", ret)
    # print("choose_action_by_fine_grain_strategy return vm[%d] for task[%d]" % (ret, task_id))
    return ret_state


# 计算某个状态的所有虚拟机的平均cpu利用率、cpu利用率方差
def get_vm_avg_cpu_utilization(state, tasks_list):
    # 初始化数组
    vm_avg_utilization = []
    vm_tasks_num = []
    vm_total_avg_utilization = 0
    vm_occupied = []     # 占用的虚拟机数，因为某个状态可能并没有占用所有的虚拟机
    for i in range(globals.VM_NUM):
        vm_avg_utilization.append(0)
        vm_tasks_num.append(0)
    # 计算每个虚拟机的平均利用率
    for i, vm_id in enumerate(state):
        vm_avg_utilization[vm_id] += tasks_list[i].cpu_utilization
        vm_tasks_num[vm_id] += 1
        if (vm_id not in vm_occupied):
            vm_occupied.append(vm_id)
    for i in range(globals.VM_NUM):
        if vm_tasks_num[i] > 0:
            vm_avg_utilization[i] = vm_avg_utilization[i] / (float)(vm_tasks_num[i])
            vm_total_avg_utilization += vm_avg_utilization[i]
    # 计算所有虚拟机的平均利用率
    vm_total_avg_utilization /= len(vm_occupied)
    # 计算所有虚拟机的利用率的标准差
    vm_utilization_std = 0
    for i in range(globals.VM_NUM):
        if vm_tasks_num[i] > 0:
            vm_utilization_std += math.pow((vm_avg_utilization[i] - vm_total_avg_utilization), 2)
    vm_utilization_std /= len(vm_occupied)
    vm_utilization_std = math.sqrt(vm_utilization_std)
    return round(vm_total_avg_utilization, 4), round(vm_utilization_std, 4)




