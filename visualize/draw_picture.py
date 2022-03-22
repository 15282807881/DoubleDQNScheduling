#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

# 设置汉字格式
plt.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


def plot_data():
    data = pd.read_csv("../result/double_dqn_result.txt", delimiter='\t')
    # data = data.sort_values(by=['execute_time'], ascending=False)
    show_list = data['execute_time'].tolist()
    print(show_list)
    # 指定画布大小
    plt.figure(figsize=(10, 4))

    # 设置图标标题
    plt.title(u"任务makespan")

    # 设置图表x轴名称和y轴名称
    plt.xlabel(u"迭代数")
    plt.ylabel("makespan")

    # 将数据进行处理，以便于画图（需要将DataFrame转为List）
    # x轴数据，只需一份即可
    linewidth = 2.5
    plt.plot(show_list, label="double dqn", linewidth=linewidth)

    # 设置图例
    plt.legend(loc='best')

    # 保存图片
    plt.savefig(f"../pic/results.png")

    # 展示折线图
    plt.show()


if __name__ == '__main__':
    plot_data()
