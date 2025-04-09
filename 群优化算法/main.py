import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from matplotlib.offsetbox import AnchoredText

# 导入你的模块
from test_functions import get_test_functions
from pso import ParticleSwarmOptimizer
from aco import AntColonyOptimizer
from firefly import FireflyAlgorithm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


def plot_function_with_result(func, bounds, best_position, title=None, resolution=100, save_path=None):
    """绘制函数三维图 + 最优点 + 二维投影图"""
    fig = plt.figure(figsize=(16, 7))

    # --- 三维图部分 ---
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    if len(bounds) >= 2:
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)

        fixed_values = np.array(best_position).copy()
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                pos = fixed_values.copy()
                pos[0] = X[i, j]
                pos[1] = Y[i, j]
                Z[i, j] = func(pos)

        surf = ax3d.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7,
                                 linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)

        ax3d.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap=cm.coolwarm, alpha=0.6)

        ax3d.scatter(best_position[0], best_position[1], func(best_position),
                     color='black', s=150, marker='*', label='最优点')
        ax3d.plot([best_position[0]] * 2,
                  [best_position[1]] * 2,
                  [Z.min(), func(best_position)],
                  'r--', linewidth=2)

        ax3d.set_xlabel('X轴', fontsize=12)
        ax3d.set_ylabel('Y轴', fontsize=12)
        ax3d.set_zlabel('f(x, y)', fontsize=12)
        ax3d.set_title(title or f"{func.__name__ if hasattr(func, '__name__') else '函数'} - 3D图", fontsize=14)
        ax3d.view_init(elev=30, azim=45)
        ax3d.legend()

    # --- 二维投影图部分（等高线图） ---
    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(X, Y, Z, cmap=cm.coolwarm, levels=50)
    fig.colorbar(contour, ax=ax2d)

    ax2d.plot(best_position[0], best_position[1], 'k*', markersize=15, label='最优点')
    ax2d.set_xlabel('X轴', fontsize=12)
    ax2d.set_ylabel('Y轴', fontsize=12)
    ax2d.set_title('函数二维等高线图', fontsize=14)
    ax2d.legend()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def compare_convergence(algorithms, func, num_runs=5):
    """比较不同算法的收敛速度"""
    convergence_data = {}
    best_positions = {}

    for alg_name, algorithm in algorithms.items():
        print(f"运行 {alg_name} 在 {func.name} 上...")
        all_fitnesses = []
        all_times = []
        all_best_positions = []

        for run in range(num_runs):
            # 重置算法状态，确保每次运行都是重新开始
            algorithm.reset() if hasattr(algorithm, 'reset') else None

            result = algorithm.optimize()

            # 确保每次迭代都记录最佳值，而非仅在发现新的更好值时记录
            # 处理不连续记录的问题
            if 'best_fitnesses' in result and len(result['best_fitnesses']) > 0:
                # 确保收敛曲线是单调的（不增加）
                monotonic_fitnesses = result['best_fitnesses'].copy() if isinstance(result['best_fitnesses'],
                                                                                    np.ndarray) else np.array(
                    result['best_fitnesses'])
                for i in range(1, len(monotonic_fitnesses)):
                    monotonic_fitnesses[i] = min(monotonic_fitnesses[i], monotonic_fitnesses[i - 1])
                all_fitnesses.append(monotonic_fitnesses)
            else:
                print(f"警告: {alg_name} 没有返回 'best_fitnesses'")
                all_fitnesses.append(np.array([]))

            all_times.append(result['run_time'] if 'run_time' in result else 0)
            all_best_positions.append(result['best_position'] if 'best_position' in result else None)

        # 对于长度不同的收敛曲线，使用插值而不是简单的填充
        if all(len(f) > 0 for f in all_fitnesses):
            max_len = max(len(f) for f in all_fitnesses)
            normalized_fitnesses = []

            for fitness in all_fitnesses:
                if len(fitness) < max_len:
                    # 使用线性插值填充缺失的点
                    x_original = np.linspace(0, 1, len(fitness))
                    x_new = np.linspace(0, 1, max_len)
                    interpolated = np.interp(x_new, x_original, fitness)
                    normalized_fitnesses.append(interpolated)
                else:
                    normalized_fitnesses.append(fitness)

            avg_fitnesses = np.mean(normalized_fitnesses, axis=0)
            std_fitnesses = np.std(normalized_fitnesses, axis=0)

            # 应用平滑处理，减少噪声和阶跃
            window_size = min(5, len(avg_fitnesses) // 10) if len(avg_fitnesses) > 10 else 1
            if window_size > 1:
                # 使用简单的滑动平均进行平滑处理
                avg_fitnesses = np.convolve(avg_fitnesses, np.ones(window_size) / window_size, mode='valid')
                # 为了保持原始长度，在两端填充
                pad_size = (max_len - len(avg_fitnesses)) // 2
                avg_fitnesses = np.pad(avg_fitnesses, (pad_size, max_len - len(avg_fitnesses) - pad_size),
                                       'edge')

                # 同样平滑标准差
                std_fitnesses = np.convolve(std_fitnesses, np.ones(window_size) / window_size, mode='valid')
                std_fitnesses = np.pad(std_fitnesses, (pad_size, max_len - len(std_fitnesses) - pad_size),
                                       'edge')
        else:
            print(f"警告: {alg_name} 的所有运行都没有产生有效的适应度历史")
            avg_fitnesses = np.array([])
            std_fitnesses = np.array([])

        convergence_data[alg_name] = {
            'avg_fitnesses': avg_fitnesses,
            'std_fitnesses': std_fitnesses,
            'avg_time': np.mean(all_times) if all_times else 0,
            'best_fitness': np.min([f[-1] for f in all_fitnesses if len(f) > 0]) if all_fitnesses else float('inf'),
        }

        # 找出性能最好的运行
        if all_best_positions and any(pos is not None for pos in all_best_positions):
            valid_runs = [(i, f) for i, f in enumerate(all_fitnesses) if len(f) > 0]
            if valid_runs:
                best_run_idx = min(valid_runs, key=lambda x: x[1][-1] if len(x[1]) > 0 else float('inf'))[0]
                best_positions[alg_name] = all_best_positions[best_run_idx]
            else:
                best_positions[alg_name] = next((pos for pos in all_best_positions if pos is not None), None)

    return convergence_data, best_positions


def plot_convergence_curves(convergence_data, func, save_path=None):
    """绘制收敛曲线"""
    plt.figure(figsize=(12, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':']

    for i, (alg_name, data) in enumerate(convergence_data.items()):
        if len(data['avg_fitnesses']) == 0:
            print(f"警告: 跳过 {alg_name} 的绘图，没有有效数据")
            continue

        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]

        # 使用平滑的线条，不使用标记点以避免视觉混乱
        plt.plot(data['avg_fitnesses'],
                 label=f"{alg_name} (最佳: {data['best_fitness']:.4e})",
                 color=color, linewidth=2, linestyle=line_style)

        # 控制标准差区域的透明度，使其不那么显眼
        if len(data['std_fitnesses']) == len(data['avg_fitnesses']):
            plt.fill_between(
                range(len(data['avg_fitnesses'])),
                data['avg_fitnesses'] - data['std_fitnesses'],
                data['avg_fitnesses'] + data['std_fitnesses'],
                alpha=0.15, color=color
            )

    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('适应度值 (对数刻度)', fontsize=14)
    plt.title(f"{func.name} - 算法收敛性对比", fontsize=16)

    # 使用对数刻度可以更好地展示收敛过程中的细微变化
    if any(data['avg_fitnesses'].min() > 0 for alg_name, data in convergence_data.items()
           if len(data['avg_fitnesses']) > 0):
        plt.yscale('log')

    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加表格展示每个算法的平均运行时间
    times_text = ""
    for alg_name, data in convergence_data.items():
        if 'avg_time' in data:
            times_text += f"{alg_name}: {data['avg_time']:.4f}秒\n"

    times_text = f"平均运行时间:\n{times_text}"
    at = AnchoredText(times_text, loc='lower right',  # 可选 'upper left', 'lower left', etc.
                      prop=dict(size=10), frameon=True)
    at.patch.set_boxstyle("round,pad=0.3")
    at.patch.set_alpha(0.5)
    at.patch.set_facecolor('lightgray')
    plt.gca().add_artist(at)

    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    dim = 3
    test_functions = get_test_functions(dim)
    max_iter = 100
    num_particles = 30
    verbose = True

    # 创建结果目录
    os.makedirs('results/plots', exist_ok=True)

    # 创建汇总结果
    summary_results = {}

    for func in test_functions:
        print(f"\n==== 正在处理函数: {func.name} ====")

        # 对于PSO算法，添加更好的参数
        pso = ParticleSwarmOptimizer(
            func, func.bounds, num_particles, max_iter, verbose,
            # 添加额外参数，如果你的实现支持的话
            # inertia_weight_decay=True,  # 惯性权重随迭代衰减
            # c1=2.0,  # 个体学习因子
            # c2=2.0   # 社会学习因子
        )

        # 对于ACO算法，添加更好的参数
        aco = AntColonyOptimizer(
            func, func.bounds, num_particles, max_iter, verbose,
            # 添加额外参数，如果你的实现支持的话
            # evaporation_rate=0.1,  # 较小的蒸发率使收敛更平滑
            # pheromone_factor=1.0,  # 信息素强度
            # smooth_update=True     # 使用平滑更新策略
        )

        # 对于萤火虫算法，添加更好的参数
        firefly = FireflyAlgorithm(
            func, func.bounds, num_particles, max_iter, verbose,
            # 添加额外参数，如果你的实现支持的话
            # alpha=0.5,          # 初始随机性参数
            # beta0=1.0,          # 初始吸引度
            # gamma=0.1,          # 光吸收系数
            # adaptive_params=True  # 使用自适应参数
        )

        algorithms = {
            'PSO': pso,
            'ACO': aco,
            'Firefly': firefly,
        }

        convergence_data, best_positions = compare_convergence(algorithms, func, num_runs=5)

        # 记录本函数的结果
        summary_results[func.name] = {
            'best_algorithm': min(
                convergence_data.items(),
                key=lambda x: x[1]['best_fitness']
            )[0],
            'results': {
                alg: {
                    'best_fitness': data['best_fitness'],
                    'avg_time': data['avg_time']
                }
                for alg, data in convergence_data.items()
            }
        }

        # 收敛曲线图
        plot_convergence_curves(
            convergence_data, func,
            save_path=f'results/plots/{func.name}_convergence.png'
        )

        # 三维目标函数图 + 每个算法的最优点
        for alg_name, best_pos in best_positions.items():
            if best_pos is not None:
                save_path = f'results/plots/{func.name}_{alg_name}_3D.png'
                plot_function_with_result(
                    func, func.bounds, best_pos,
                    title=f"{func.name} - {alg_name} 最优结果",
                    save_path=save_path
                )
                print(f"图像已保存到: {save_path}")

    # 输出汇总结果
    print("\n===== 优化结果汇总 =====")
    for func_name, result in summary_results.items():
        print(f"\n函数: {func_name}")
        print(f"最佳算法: {result['best_algorithm']}")
        print("各算法结果:")
        for alg, metrics in result['results'].items():
            print(f"  - {alg}: 最佳适应度 = {metrics['best_fitness']:.6e}, 平均时间 = {metrics['avg_time']:.4f}秒")


def main_PSO():
    dim = 10  # 使用固定维度进行参数测试
    test_functions = get_test_functions(dim)
    max_iter = 100
    verbose = False  # 关闭详细输出，以便更清晰展示结果

    # 创建结果目录
    os.makedirs('results/sensitivity', exist_ok=True)

    # PSO参数敏感性分析

    # 1. 粒子数量敏感性测试
    print("\n===== PSO 粒子数量敏感性分析 =====")
    particle_counts = [20, 30, 50, 100]

    # 准备结果表格
    results_table = {func.name: [] for func in test_functions}

    for num_particles in particle_counts:
        print(f"\n测试粒子数: {num_particles}")

        for func in test_functions:
            print(f"处理函数: {func.name}")

            # 运行PSO算法
            pso = ParticleSwarmOptimizer(
                func, func.bounds, num_particles, max_iter, verbose,
                w=0.7, c1=1.5, c2=1.5  # 使用固定的其他参数
            )

            # 多次运行取平均，评估稳定性
            avg_fitness = 0
            runs = 10
            for _ in range(runs):
                result = pso.optimize()
                best_pos = result["best_position"]
                best_fit = result["best_fitness"]
                avg_fitness += best_fit

            avg_fitness /= runs
            results_table[func.name].append(avg_fitness)
            print(f"- 平均适应度: {avg_fitness:.3e}")

    # 输出粒子数量敏感性结果表
    print("\n**粒子数量影响**:\n")
    print("| 粒子数 | Rastrigin | Ackley | Schwefel | Rosenbrock |")
    print("|--------|-----------|--------|----------|------------|")

    for i, count in enumerate(particle_counts):
        row = f"| {count}     |"
        for func in test_functions:
            row += f" {results_table[func.name][i]:.3e} |"
        print(row)

    # 2. 惯性权重敏感性测试
    print("\n===== PSO 惯性权重敏感性分析 =====")
    w_values = [0.4, 0.6, 0.7, 0.9]

    # 准备结果表格
    results_table = {func.name: [] for func in test_functions}

    for w in w_values:
        print(f"\n测试惯性权重: {w}")

        for func in test_functions:
            print(f"处理函数: {func.name}")

            # 运行PSO算法
            pso = ParticleSwarmOptimizer(
                func, func.bounds, 30, max_iter, verbose,  # 使用固定的粒子数
                w=w, c1=1.5, c2=1.5  # 使用固定的加速系数
            )

            # 多次运行取平均
            avg_fitness = 0
            runs = 10
            for _ in range(runs):
                result = pso.optimize()
                best_pos = result["best_position"]
                best_fit = result["best_fitness"]
                avg_fitness += best_fit

            avg_fitness /= runs
            results_table[func.name].append(avg_fitness)
            print(f"- 平均适应度: {avg_fitness:.3e}")

    # 输出惯性权重敏感性结果表
    print("\n**惯性权重影响**:\n")
    print("| 惯性权重w | Rastrigin | Ackley | Schwefel | Rosenbrock |")
    print("|-----------|-----------|--------|----------|------------|")

    for i, w in enumerate(w_values):
        row = f"| {w}       |"
        for func in test_functions:
            row += f" {results_table[func.name][i]:.3e} |"
        print(row)

    # 3. 加速系数敏感性测试
    print("\n===== PSO 加速系数敏感性分析 =====")
    c_values = [(1.0, 1.0), (1.5, 1.5), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)]

    # 准备结果表格
    results_table = {func.name: [] for func in test_functions}

    for c1, c2 in c_values:
        print(f"\n测试加速系数: c1={c1}, c2={c2}")

        for func in test_functions:
            print(f"处理函数: {func.name}")

            # 运行PSO算法
            pso = ParticleSwarmOptimizer(
                func, func.bounds, 30, max_iter, verbose,  # 使用固定的粒子数
                w=0.7, c1=c1, c2=c2  # 使用固定的惯性权重
            )

            # 多次运行取平均
            avg_fitness = 0
            runs = 10
            for _ in range(runs):
                result = pso.optimize()
                best_pos = result["best_position"]
                best_fit = result["best_fitness"]

                avg_fitness += best_fit

            avg_fitness /= runs
            results_table[func.name].append(avg_fitness)
            print(f"- 平均适应度: {avg_fitness:.3e}")

    # 输出加速系数敏感性结果表
    print("\n**加速系数(c1, c2)影响**:\n")
    print("| (c1, c2) | Rastrigin | Ackley | Schwefel | Rosenbrock |")
    print("|----------|-----------|--------|----------|------------|")

    for i, (c1, c2) in enumerate(c_values):
        row = f"| ({c1}, {c2}) |"
        for func in test_functions:
            row += f" {results_table[func.name][i]:.3e} |"
        print(row)

    # 还可以添加鲁棒性分析（多次运行结果的标准差）
    print("\n===== PSO 鲁棒性分析 =====")

    # 使用最佳参数设置
    for func in test_functions:
        print(f"\n函数: {func.name}")

        pso = ParticleSwarmOptimizer(
            func, func.bounds, 30, max_iter, verbose,
            w=0.7, c1=1.5, c2=1.5  # 最佳参数设置
        )

        # 运行多次记录结果
        results = []
        for _ in range(30):  # 30次独立运行
            result = pso.optimize()
            best_pos = result["best_position"]
            best_fit = result["best_fitness"]

            results.append(best_fit)

        # 计算统计量
        mean_fitness = np.mean(results)
        std_fitness = np.std(results)

        print(f"平均适应度: {mean_fitness:.3e}")
        print(f"标准差: {std_fitness:.3e}")
        print(f"鲁棒性指标(变异系数): {std_fitness / mean_fitness:.3f}")
def main_ACO():
    dim = 10
    test_functions = get_test_functions(dim)
    max_iter = 100
    verbose = False

    os.makedirs('results/sensitivity', exist_ok=True)

    # 1. 蚂蚁数量敏感性分析
    print("\n===== ACO 蚂蚁数量敏感性分析 =====")
    ant_counts = [20, 30, 50, 100]
    results_table = {func.name: [] for func in test_functions}

    for num_ants in ant_counts:
        print(f"\n测试蚂蚁数量: {num_ants}")

        for func in test_functions:
            print(f"处理函数: {func.name}")

            avg_fitness = 0
            runs = 10
            for _ in range(runs):
                aco = AntColonyOptimizer(
                    func, func.bounds, num_ants, max_iter, alpha=1.0, beta=2.0, rho=0.5, verbose=verbose
                )
                result = aco.optimize()
                avg_fitness += result["best_fitness"]

            avg_fitness /= runs
            results_table[func.name].append(avg_fitness)
            print(f"- 平均适应度: {avg_fitness:.3e}")

    print("\n**蚂蚁数量影响**:\n")
    print("| 蚂蚁数 | Rastrigin | Ackley | Schwefel | Rosenbrock |")
    print("|--------|-----------|--------|----------|------------|")
    for i, count in enumerate(ant_counts):
        row = f"| {count:<6} |"
        for func in test_functions:
            row += f" {results_table[func.name][i]:.3e} |"
        print(row)

    # 2. 参数 (α, β, ρ) 敏感性分析
    print("\n===== ACO 参数敏感性分析 (α, β, ρ) =====")
    param_combinations = [
        (0.5, 1.0, 0.3),
        (1.0, 2.0, 0.5),
        (1.5, 2.0, 0.5),
        (1.0, 3.0, 0.5),
        (1.0, 2.0, 0.7)
    ]
    results_table = {func.name: [] for func in test_functions}

    for alpha, beta, rho in param_combinations:
        print(f"\n测试参数: α={alpha}, β={beta}, ρ={rho}")

        for func in test_functions:
            print(f"处理函数: {func.name}")

            avg_fitness = 0
            runs = 10
            for _ in range(runs):
                aco = AntColonyOptimizer(
                    func, func.bounds, 30, max_iter,
                    alpha=alpha, beta=beta, rho=rho, verbose=verbose
                )
                result = aco.optimize()
                avg_fitness += result["best_fitness"]

            avg_fitness /= runs
            results_table[func.name].append(avg_fitness)
            print(f"- 平均适应度: {avg_fitness:.3e}")

    print("\n**ACO参数(α, β, ρ)影响**:\n")
    print("| (α, β, ρ)         | Rastrigin | Ackley | Schwefel | Rosenbrock |")
    print("|-------------------|-----------|--------|----------|------------|")
    for i, (alpha, beta, rho) in enumerate(param_combinations):
        row = f"| ({alpha}, {beta}, {rho}) |"
        for func in test_functions:
            row += f" {results_table[func.name][i]:.3e} |"
        print(row)

    # 3. 鲁棒性分析
    print("\n===== ACO 鲁棒性分析 =====")
    for func in test_functions:
        print(f"\n函数: {func.name}")

        aco = AntColonyOptimizer(
            func, func.bounds, 30, max_iter,
            alpha=1.0, beta=2.0, rho=0.5, verbose=verbose
        )
        results = []
        for _ in range(30):
            result = aco.optimize()
            results.append(result["best_fitness"])

        mean_fitness = np.mean(results)
        std_fitness = np.std(results)

        print(f"平均适应度: {mean_fitness:.3e}")
        print(f"标准差: {std_fitness:.3e}")
        print(f"鲁棒性指标(变异系数): {std_fitness / mean_fitness:.3f}")
def main_FFA():
    dim = 10  # 使用固定维度进行参数测试
    test_functions = get_test_functions(dim)  # 获取测试函数
    max_iter = 100  # 最大迭代次数
    verbose = False  # 关闭详细输出

    os.makedirs('results/sensitivity', exist_ok=True)  # 创建结果目录



    # 2. 参数 (β₀, γ, α) 敏感性分析
    print("\n===== FA 参数 (β₀, γ, α) 敏感性分析 =====")
    param_combinations = [
        (0.5, 1.0, 0.2),
        (1.0, 1.0, 0.2),
        (1.5, 1.0, 0.2),
        (1.0, 0.5, 0.2),
        (1.0, 2.0, 0.2),
        (1.0, 1.0, 0.1),
        (1.0, 1.0, 0.4)
    ]
    results_table = {func.name: [] for func in test_functions}

    for beta_0, gamma, alpha in param_combinations:
        print(f"\n测试参数: β₀={beta_0}, γ={gamma}, α={alpha}")

        for func in test_functions:
            print(f"处理函数: {func.name}")

            avg_fitness = 0
            runs = 10  # 每种情况运行10次取平均
            for _ in range(runs):
                fa = FireflyAlgorithm(
                    func, func.bounds, 30, max_iter,
                    alpha=alpha, beta0=beta_0, gamma=gamma, verbose=verbose
                )
                result = fa.optimize()
                avg_fitness += result["best_fitness"]

            avg_fitness /= runs
            results_table[func.name].append(avg_fitness)
            print(f"- 平均适应度: {avg_fitness:.3e}")

    print("\n**FA参数 (β₀, γ, α)影响**:\n")
    print("| (β₀, γ, α)         | Rastrigin | Ackley | Schwefel | Rosenbrock |")
    print("|--------------------|-----------|--------|----------|------------|")
    for i, (beta_0, gamma, alpha) in enumerate(param_combinations):
        row = f"| ({beta_0}, {gamma}, {alpha}) |"
        for func in test_functions:
            row += f" {results_table[func.name][i]:.3e} |"
        print(row)

    # 3. 鲁棒性分析
    print("\n===== FA 鲁棒性分析 =====")
    for func in test_functions:
        print(f"\n函数: {func.name}")

        fa = FireflyAlgorithm(
            func, func.bounds, 30, max_iter,
            alpha=0.5, beta0=1.0, gamma=1.0, verbose=verbose  # 默认参数
        )
        results = []
        for _ in range(30):  # 30次独立运行
            result = fa.optimize()
            results.append(result["best_fitness"])

        mean_fitness = np.mean(results)
        std_fitness = np.std(results)

        print(f"平均适应度: {mean_fitness:.3e}")
        print(f"标准差: {std_fitness:.3e}")
        print(f"鲁棒性指标(变异系数): {std_fitness / mean_fitness:.3f}")



if __name__ == '__main__':
    main()
