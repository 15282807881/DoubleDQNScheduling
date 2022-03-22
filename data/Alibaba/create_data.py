import pandas as pd


# 创建指定大小的数据集
def create_dataset(filePath):
    df = pd.read_csv(filePath)
    # 修改列名
    df.columns = ['instance_id', 'instance_name', 'task_name', 'job_name', 'task_type', 'status', 'start_time',
                  'end_time', 'machine_id', 'seq_no', 'total_seq_no', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
    df['length'] = (df['end_time'] - df['start_time']) * 1000
    df['cpu_avg'] /= 100
    df['size'] = df['length'] / 10
    df = df.drop(df[df['status'] == 'Running'].index)
    df = df[['length', 'cpu_avg', 'size']]
    df = df.drop(df[(df['length'] == 0) | (df['cpu_avg'] == 0) | (df['cpu_avg'] > 1)].index)

    # print(df)
    idx = 0
    for i in range(100, 1100, 100):
        tmp_df = df[idx : idx+i]
        fileName = 'Alibaba-Cluster-trace-' + str(i) + '.txt'
        tmp_df.to_csv(fileName, header=False)
        idx += i


if __name__ == '__main__':
    # 解决控制台输出省略号的问题
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

    filePath = "Alibaba-Cluster-trace-v2018.csv"
    create_dataset(filePath)