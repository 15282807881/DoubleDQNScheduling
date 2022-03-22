import numpy as np
from util.fileio import FileIo
import globals

# 仅读取文件
def create_tasks_just_from_file(taskNum):
    all_tasks = []  # 保存所有的任务
    lines = []
    task_data = FileIo(r"D:\dev\task-scheduler-based-on-DQN-main\代码\pycloudsim-master\pycloudsim-master\data\create\task.txt").readAllLines(lines)
    submit_time = 1
    for i in range(taskNum):
        all_tasks.append([submit_time, task_data[i][1], task_data[i][3], task_data[i][2]])

    return all_tasks

# 创建任务量相同的任务
def create_same_length_tasks(poisson_lam):
    task_num = 1000000  # 生成100万条任务
    task_mi = 5000  # 固定长度
    task_cpu_utilization = 0.12  # cpu利用率平均数
    task_data_size = 531  # 任务数据大小

    # 可以改动的地方是下面的size参数
    task_last_time = 10000
    num_per_second = np.random.poisson(lam=poisson_lam, size=task_last_time)  # lam单位时间随机时间发生次数的平均值
    task_time = []  # 存储每条任务的提交时间,从1开始
    for i, num in enumerate(num_per_second):
        if num > (poisson_lam + 2):  # 对每秒提交的任务数量进行限制
            num = poisson_lam + 2
        elif num < max(1, poisson_lam - 2):
            num = max(1, poisson_lam - 2)
        for j in range(num):
            task_time.append(i + 1)
    mi = []  # 长度固定
    for i in range(task_num):
        mi.append(task_mi)
    cpu_utilization = np.around(np.random.normal(loc=task_cpu_utilization, scale=0.5, size=task_num),
                                decimals=2).tolist()  # 均值mean,标准差std,数量. 保留两位小数
    data_size = list(map(int, np.random.normal(loc=task_data_size, scale=400, size=task_num)))

    all_tasks = []  # 保存所有的任务
    for t, l, c, d in zip(task_time, mi, cpu_utilization, data_size):   # 任务数据属性：任务提交时间，mips，cpu利用率，数据量大小
        '''
        if c < 0.3:  # 对cpu利用率限制在合理范围
            c = 0.3
        elif c > 0.9:
            c = 0.9
        '''
        if c < 0.1:  # 对cpu利用率限制在合理范围
            c = 0.1
        if d < 100:
            d = 100
        if l < 1000:
            l = 1000
        all_tasks.append([t, l, c, d])

    # 最后加上要分配的任务的数据
    lines = []
    task_data = FileIo(r"D:\dev\task-scheduler-based-on-DQN-main\代码\pycloudsim-master\pycloudsim-master\data\create\task.txt").readAllLines(lines)
    submit_time = task_last_time + 1
    # print("lines[1]:", task_data[0][1])
    for line in task_data:
        all_tasks.append([submit_time, 5000, line[3], line[2]])
    # print(task_data)

    return all_tasks

# 创建任务量不同的任务
def create_tasks(poisson_lam):
    task_num = 1000000  # 生成100万条任务
    task_mi = 5318  # 长度平均值
    task_cpu_utilization = 0.5  # cpu利用率平均数
    task_data_size = 531  # 任务数据大小

    # 可以改动的地方是下面的size参数
    task_last_time = 2000
    num_per_second = np.random.poisson(lam=poisson_lam, size=task_last_time)  # lam单位时间随机时间发生次数的平均值
    task_time = []  # 存储每条任务的提交时间,从1开始
    for i, num in enumerate(num_per_second):
        if num > (poisson_lam + 2):  # 对每秒提交的任务数量进行限制
            num = poisson_lam + 2
        elif num < max(1, poisson_lam - 2):
            num = max(1, poisson_lam - 2)
        for j in range(num):
            task_time.append(i + 1)
    mi = list(map(int, np.random.normal(loc=task_mi, scale=4000, size=task_num)))  # 均值mean,标准差std,数量
    cpu_utilization = np.around(np.random.normal(loc=task_cpu_utilization, scale=0.5, size=task_num),
                                decimals=2).tolist()  # 均值mean,标准差std,数量. 保留两位小数
    data_size = list(map(int, np.random.normal(loc=task_data_size, scale=400, size=task_num)))

    all_tasks = []  # 保存所有的任务
    for t, l, c, d in zip(task_time, mi, cpu_utilization, data_size):   # 任务数据属性：任务提交时间，mips，cpu利用率，数据量大小
        '''
        if c < 0.3:  # 对cpu利用率限制在合理范围
            c = 0.3
        elif c > 0.9:
            c = 0.9
        '''
        if c < 0.1:  # 对cpu利用率限制在合理范围
            c = 0.1
        if c > 1:
            c = 1
        if d < 100:
            d = 100
        if l < 1000:
            l = 1000
        all_tasks.append([t, l, c, d])

    # 最后加上要分配的任务的数据
    lines = []
    task_data = FileIo(r"D:\dev\task-scheduler-based-on-DQN-main\代码\pycloudsim-master\pycloudsim-master\data\create\task.txt").readAllLines(lines)
    submit_time = task_last_time + 1
    # print("lines[1]:", task_data[0][1])
    for line in task_data:
        all_tasks.append([submit_time, line[1], line[3], line[2]])
    # print(task_data)

    return all_tasks


if __name__ == '__main__':
    poisson_lam_li = [200]  # 1, 3, 5, 7, 9
    for i in poisson_lam_li:
        poisson_lam = i  # 表示平均每秒提交任务的个数
        # 任务参数设置
        is_just_from_file = True
        is_same_task_length = False
        taskNum = globals.TASK_NUM
        if (is_just_from_file):
            all_tasks = create_tasks_just_from_file(taskNum)
            print("任务生成成功！")
            result = FileIo("create_tasks_" + str(globals.TASK_NUM) + ".txt").twoListToFile(all_tasks, 'w')
            print("任务存储成功!")
        elif (is_same_task_length):
            # 创建任务量相同的任务
            all_tasks = create_same_length_tasks(poisson_lam)
            print("任务生成成功！")
            result = FileIo("create_tasks_" + str(poisson_lam) + ".txt").twoListToFile(all_tasks, 'w')
            print("任务存储成功!")
        else:
            # 创建任务量不同的任务
            all_tasks = create_tasks(poisson_lam)
            print("任务生成成功！")
            result = FileIo("create_tasks_" + str(poisson_lam) + ".txt").twoListToFile(all_tasks, 'w')
            print("任务存储成功!")