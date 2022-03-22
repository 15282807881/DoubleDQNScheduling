import pandas as pd


if __name__ == '__main__':
    file_path = r"D:\data\batch_instance\batch_instance.csv"

    # 解决控制台输出省略号的问题
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

    with pd.read_csv(file_path, chunksize=10000) as reader:
        print_num = 1
        for chunk in reader:
            # 经观察发现，在1000行左右时会有一段数据存在缺失值，所以这里取一个chunk块，然后截取后2000条数据
            # chunk = chunk[-10000:]
            print("len(chunk): ", len(chunk))
            chunk.to_csv(r"Alibaba-Cluster-trace-v2018.csv")
            # print(chunk)
            # print(chunk.columns.values.tolist())
            print_num -= 1
            if print_num == 0:
                break
