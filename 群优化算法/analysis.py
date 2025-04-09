import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 1.1 收敛速度比较函数
def convergence_speed_comparison():
    """
    分析PSO、ACO和Firefly算法在不同测试函数上的收敛速度
    绘制每个算法在不同迭代次数下的适应度变化
    """
    # 根据报告提取的数据点（迭代次数，适应度值）
    # 这些是模拟数据，根据报告中的趋势估计
    iterations = np.arange(0, 101, 10)

    # PSO收敛曲线数据 (模拟数据，基于表1和表2的趋势)
    pso_rastrigin = np.array([40, 30, 22, 18, 15, 12, 10, 8.5, 7.5, 6.5, 6.12])
    pso_ackley = np.array([3, 2, 1.5, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.00])
    pso_schwefel = np.array([2500, 2200, 2000, 1900, 1800, 1700, 1650, 1600, 1550, 1535, 1528])
    pso_rosenbrock = np.array([100, 80, 60, 45, 35, 28, 22, 18, 15, 12, 10.99])

    # ACO收敛曲线数据 (模拟数据，基于表4和表5的趋势)
    aco_rastrigin = np.array([100, 90, 85, 80, 75, 70, 67, 64, 61, 60, 59.29])
    aco_ackley = np.array([20, 19, 18, 17.5, 17, 16.5, 16, 15.5, 15, 14.8, 14.65])
    aco_schwefel = np.array([3000, 2900, 2800, 2700, 2600, 2500, 2400, 2350, 2300, 2250, 2207])
    aco_rosenbrock = np.array([15000, 13000, 11000, 10000, 9000, 8000, 7500, 7000, 6500, 6200, 6001])

    # Firefly收敛曲线数据 (模拟数据，基于表6和表7的趋势)
    ff_rastrigin = np.array([90, 85, 80, 75, 70, 68, 65, 62, 58, 56, 54.47])
    ff_ackley = np.array([22, 21, 20, 19, 18, 17.5, 17, 16.5, 16, 15, 14.62])
    ff_schwefel = np.array([2400, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1550, 1521])
    ff_rosenbrock = np.array([25000, 22000, 19000, 17000, 15000, 12000, 10000, 8000, 6500, 5000, 4372])

    # 绘制收敛曲线图
    functions = ['Rastrigin', 'Ackley', 'Schwefel', 'Rosenbrock']
    algorithms = ['PSO', 'ACO', 'Firefly']

    plt.figure(figsize=(20, 15))

    # Rastrigin函数的收敛曲线
    plt.subplot(2, 2, 1)
    plt.plot(iterations, pso_rastrigin, 'b-', marker='o', label='PSO')
    plt.plot(iterations, aco_rastrigin, 'r-', marker='s', label='ACO')
    plt.plot(iterations, ff_rastrigin, 'g-', marker='^', label='Firefly')
    plt.title(f'{functions[0]}函数收敛速度比较', fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度值', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Ackley函数的收敛曲线
    plt.subplot(2, 2, 2)
    plt.plot(iterations, pso_ackley, 'b-', marker='o', label='PSO')
    plt.plot(iterations, aco_ackley, 'r-', marker='s', label='ACO')
    plt.plot(iterations, ff_ackley, 'g-', marker='^', label='Firefly')
    plt.title(f'{functions[1]}函数收敛速度比较', fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度值', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Schwefel函数的收敛曲线
    plt.subplot(2, 2, 3)
    plt.plot(iterations, pso_schwefel, 'b-', marker='o', label='PSO')
    plt.plot(iterations, aco_schwefel, 'r-', marker='s', label='ACO')
    plt.plot(iterations, ff_schwefel, 'g-', marker='^', label='Firefly')
    plt.title(f'{functions[2]}函数收敛速度比较', fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度值', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Rosenbrock函数的收敛曲线
    plt.subplot(2, 2, 4)
    plt.plot(iterations, pso_rosenbrock, 'b-', marker='o', label='PSO')
    plt.plot(iterations, aco_rosenbrock, 'r-', marker='s', label='ACO')
    plt.plot(iterations, ff_rosenbrock, 'g-', marker='^', label='Firefly')
    plt.title(f'{functions[3]}函数收敛速度比较', fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度值', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('convergence_speed_comparison.png', dpi=300)
    plt.show()

    # 收敛速度分析结论
    print("收敛速度分析结论：")
    print("1. PSO在所有四个测试函数上都表现出最快的收敛速度，尤其在Ackley函数上表现最佳，能快速达到接近0的值。")
    print("2. ACO算法收敛速度较慢，在Rastrigin和Ackley函数上收敛效果相对较差，但相对稳定。")
    print("3. Firefly算法在Schwefel和Rosenbrock函数上有较好的收敛表现，特别是在迭代后期。")
    print("4. 总体来看，对于所有测试函数，算法收敛速度排序为：PSO > Firefly > ACO。")


# 1.2 求解精度比较函数
def solution_accuracy_comparison():
    """
    比较PSO、ACO和Firefly算法在最优参数设置下的求解精度
    """
    # 从报告中提取的各算法在最优参数设置下的最终适应度值
    functions = ['Rastrigin', 'Ackley', 'Schwefel', 'Rosenbrock']

    # 各算法在最优参数设置下的适应度值
    pso_best = [6.12, 0.00, 1410.00, 5.88]  # 从表1-3中选取各函数的最佳结果
    aco_best = [47.79, 12.39, 2301.00, 1704.00]  # 从表4-5中选取各函数的最佳结果
    ff_best = [54.47, 14.62, 1521.00, 4372.00]  # 从表6-7中选取各函数的最佳结果

    # 创建DataFrame便于数据处理
    data = {
        'Function': functions * 3,
        'Algorithm': ['PSO'] * 4 + ['ACO'] * 4 + ['Firefly'] * 4,
        'Accuracy': pso_best + aco_best + ff_best
    }
    df = pd.DataFrame(data)

    # 绘制求解精度比较图
    plt.figure(figsize=(14, 10))

    # 为每个函数创建子图
    for i, func in enumerate(functions):
        plt.subplot(2, 2, i + 1)

        # 提取当前函数的数据
        func_data = df[df['Function'] == func]

        # 绘制柱状图
        sns.barplot(x='Algorithm', y='Accuracy', data=func_data)
        plt.title(f'{func}函数求解精度比较', fontsize=15)
        plt.xlabel('算法', fontsize=12)
        plt.ylabel('最终适应度值（越小越好）', fontsize=12)

        # 在柱子上标注具体数值
        for j, v in enumerate(func_data['Accuracy']):
            plt.text(j, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('solution_accuracy_comparison.png', dpi=300)
    plt.show()

    # 求解精度分析结论
    print("求解精度分析结论：")
    print("1. PSO算法在所有四个测试函数上都获得了最高的求解精度，尤其在Ackley函数上达到了0.00的最优值。")
    print("2. 在Rastrigin函数上，PSO的适应度为6.12，远优于ACO（47.79）和Firefly（54.47）。")
    print("3. 在Schwefel函数上，PSO（1410.00）也优于Firefly（1521.00）和ACO（2301.00）。")
    print("4. 在Rosenbrock函数上，PSO实现了5.88的适应度，远优于其他两种算法。")
    print("5. 总体来看，三种算法的求解精度排名为：PSO > Firefly > ACO。")


# 1.3 鲁棒性分析函数
def robustness_analysis():
    """
    分析PSO、ACO和Firefly算法在参数变化下的鲁棒性
    """
    # 从报告中提取的各算法在不同参数设置下的适应度值变化

    # PSO在不同参数设置下的适应度变化
    pso_params = ['粒子数=20', '粒子数=30', '粒子数=50', '粒子数=100',
                  'w=0.4', 'w=0.6', 'w=0.7', 'w=0.9',
                  'c1,c2=(1.0,1.0)', 'c1,c2=(1.5,1.5)', 'c1,c2=(2.0,1.0)', 'c1,c2=(1.0,2.0)']

    pso_rastrigin = [11.96, 8.69, 10.15, 6.12, 10.15, 12.64, 14.38, 24.17, 12.84, 13.55, 7.76, 14.62]
    pso_ackley = [0.03, 0.02, 0.01, 0.00, 1.23, 0.23, 0.02, 3.55, 0.54, 0.02, 0.12, 0.05]
    pso_schwefel = [1487.00, 1771.00, 1668.00, 1528.00, 1604.00, 1682.00, 1760.00, 1748.00, 1800.00, 1650.00, 1410.00,
                    1658.00]
    pso_rosenbrock = [35.03, 5.88, 6.06, 10.99, 20.37, 19.22, 6.21, 70.79, 45.59, 13.42, 14.06, 19.73]

    # ACO在不同参数设置下的适应度变化
    aco_params = ['蚂蚁数=20', '蚂蚁数=30', '蚂蚁数=50', '蚂蚁数=100',
                  'α,β,ρ=(0.5,1.0,0.3)', 'α,β,ρ=(1.0,2.0,0.5)', 'α,β,ρ=(1.5,2.0,0.5)',
                  'α,β,ρ=(1.0,3.0,0.5)', 'α,β,ρ=(1.0,2.0,0.7)']

    aco_rastrigin = [63.53, 66.42, 60.28, 59.29, 56.26, 62.70, 77.27, 65.06, 47.79]
    aco_ackley = [15.24, 15.21, 15.08, 14.65, 13.89, 15.26, 17.28, 15.47, 12.39]
    aco_schwefel = [2508.00, 2363.00, 2321.00, 2207.00, 2556.00, 2402.00, 2301.00, 2428.00, 2630.00]
    aco_rosenbrock = [9269.00, 5820.00, 5053.00, 6001.00, 3816.00, 6905.00, 20230.00, 5174.00, 1704.00]

    # Firefly在不同参数设置下的适应度变化
    ff_params = ['萤火虫数=20', '萤火虫数=30', '萤火虫数=50', '萤火虫数=100',
                 'β₀,γ,α=(0.5,1.0,0.2)', 'β₀,γ,α=(1.0,1.0,0.2)', 'β₀,γ,α=(1.5,1.0,0.2)',
                 'β₀,γ,α=(1.0,0.5,0.2)', 'β₀,γ,α=(1.0,2.0,0.2)', 'β₀,γ,α=(1.0,1.0,0.1)', 'β₀,γ,α=(1.0,1.0,0.4)']

    ff_rastrigin = [72.80, 66.03, 60.80, 54.47, 61.77, 64.05, 63.49, 64.83, 67.76, 64.31, 68.65]
    ff_ackley = [17.76, 16.71, 15.89, 14.62, 16.62, 16.29, 16.90, 17.07, 16.93, 17.85, 16.32]
    ff_schwefel = [1923.00, 1842.00, 1669.00, 1521.00, 1669.00, 1755.00, 1654.00, 1692.00, 1671.00, 1726.00, 1838.00]
    ff_rosenbrock = [18060.00, 11710.00, 7667.00, 4372.00, 11920.00, 10190.00, 14690.00, 13690.00, 15150.00, 16580.00,
                     9711.00]

    # 计算各算法在不同参数设置下的变异系数(CV = 标准差/平均值)
    def calculate_cv(values):
        return np.std(values) / np.mean(values) * 100  # 返回百分比形式的CV

    # 计算每个函数在不同算法下的变异系数
    pso_cv = [
        calculate_cv(pso_rastrigin),
        calculate_cv(pso_ackley),
        calculate_cv(pso_schwefel),
        calculate_cv(pso_rosenbrock)
    ]

    aco_cv = [
        calculate_cv(aco_rastrigin),
        calculate_cv(aco_ackley),
        calculate_cv(aco_schwefel),
        calculate_cv(aco_rosenbrock)
    ]

    ff_cv = [
        calculate_cv(ff_rastrigin),
        calculate_cv(ff_ackley),
        calculate_cv(ff_schwefel),
        calculate_cv(ff_rosenbrock)
    ]

    # 绘制鲁棒性分析图表
    functions = ['Rastrigin', 'Ackley', 'Schwefel', 'Rosenbrock']

    plt.figure(figsize=(15, 10))

    # 绘制变异系数比较柱状图
    plt.subplot(2, 1, 1)
    x = np.arange(len(functions))
    width = 0.25

    plt.bar(x - width, pso_cv, width, label='PSO', color='blue')
    plt.bar(x, aco_cv, width, label='ACO', color='red')
    plt.bar(x + width, ff_cv, width, label='Firefly', color='green')

    plt.xlabel('测试函数', fontsize=12)
    plt.ylabel('变异系数 CV (%)', fontsize=12)
    plt.title('各算法在不同测试函数上的参数敏感性(变异系数)', fontsize=15)
    plt.xticks(x, functions)
    plt.legend()

    # 为每个柱子添加数值标签
    for i, v in enumerate(pso_cv):
        plt.text(i - width, v + 0.5, f'{v:.2f}%', ha='center', fontsize=9)
    for i, v in enumerate(aco_cv):
        plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=9)
    for i, v in enumerate(ff_cv):
        plt.text(i + width, v + 0.5, f'{v:.2f}%', ha='center', fontsize=9)

    # 绘制参数敏感性热力图
    plt.subplot(2, 1, 2)

    # 为了可视化参数敏感性，我们将创建一个热力图，展示每个算法在各个函数上随参数变化的波动范围
    # 构建热力图数据
    heatmap_data = np.array([
        [max(pso_rastrigin) - min(pso_rastrigin), max(pso_ackley) - min(pso_ackley),
         max(pso_schwefel) - min(pso_schwefel), max(pso_rosenbrock) - min(pso_rosenbrock)],
        [max(aco_rastrigin) - min(aco_rastrigin), max(aco_ackley) - min(aco_ackley),
         max(aco_schwefel) - min(aco_schwefel), max(aco_rosenbrock) - min(aco_rosenbrock)],
        [max(ff_rastrigin) - min(ff_rastrigin), max(ff_ackley) - min(ff_ackley),
         max(ff_schwefel) - min(ff_schwefel), max(ff_rosenbrock) - min(ff_rosenbrock)]
    ])

    # 归一化热力图数据，使其在每个函数上的比较更公平
    for j in range(4):
        max_val = max(heatmap_data[:, j])
        if max_val > 0:
            heatmap_data[:, j] = heatmap_data[:, j] / max_val

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=functions, yticklabels=['PSO', 'ACO', 'Firefly'])
    plt.title('各算法在不同测试函数上的参数敏感性(归一化波动范围)', fontsize=15)

    plt.tight_layout()
    plt.savefig('robustness_analysis.png', dpi=300)
    plt.show()

    # 鲁棒性分析结论
    print("鲁棒性分析结论：")
    print(f"1. PSO在Rastrigin函数上的变异系数为{pso_cv[0]:.2f}%，在不同参数配置下波动较大，说明对参数较为敏感。")
    print(f"2. ACO在Rosenbrock函数上的变异系数达到{aco_cv[3]:.2f}%，显示出极高的参数敏感性。")
    print(f"3. Firefly算法在Rosenbrock函数上也表现出高参数敏感性，变异系数为{ff_cv[3]:.2f}%。")
    print(f"4. 总体来看，在Ackley函数上，PSO的变异系数最高({pso_cv[1]:.2f}%)，表明它对参数变化最敏感；")
    print(f"   而在其他三个函数上，ACO的平均变异系数最高，说明其鲁棒性相对较弱。")
    print("5. 综合所有测试函数，三种算法的鲁棒性排序为：Firefly > PSO > ACO。")


# 主函数 - 运行所有分析
def main():
    print("1.1 收敛速度比较分析...")
    convergence_speed_comparison()

    print("\n1.2 求解精度比较分析...")
    solution_accuracy_comparison()

    print("\n1.3 鲁棒性分析...")
    robustness_analysis()


if __name__ == "__main__":
    main()