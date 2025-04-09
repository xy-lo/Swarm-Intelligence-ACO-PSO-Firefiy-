import numpy as np
import time


class AntColonyOptimizer:
    """蚁群算法的连续域优化版本（Continuous ACO）"""

    def __init__(self, objective_func, bounds, num_ants=30, max_iter=100,
                 alpha=1, beta=2.0, rho=0.7, q=1, verbose=False):

        """
        参数:
        - objective_func: 要优化的目标函数
        - bounds: 搜索空间边界 [(x1_min, x1_max), (x2_min, x2_max), ...]
        - num_ants: 蚂蚁数量
        - max_iter: 最大迭代次数
        - alpha: 信息素重要程度因子
        - beta: 启发因子重要程度因子
        - rho: 信息素挥发系数
        - q: 信息素更新的参数
        - verbose: 是否输出中间结果
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.verbose = verbose
        self.dim = len(bounds)

        # 搜索范围参数
        self.min_bounds = np.array([b[0] for b in bounds])
        self.max_bounds = np.array([b[1] for b in bounds])
        self.range = self.max_bounds - self.min_bounds

        # 记录每次迭代的最优位置和适应度
        self.best_positions = []
        self.best_fitnesses = []

        # 信息素矩阵初始化（使用高斯混合模型来表示）
        self.num_gaussians = 5  # 每个维度上的高斯分布数量
        self.initialize_gaussians()

    def initialize_gaussians(self):
        """初始化高斯混合模型用于信息素表示"""
        # 初始化每个维度上高斯分布的均值
        self.means = np.random.uniform(
            self.min_bounds[:, np.newaxis],
            self.max_bounds[:, np.newaxis],
            (self.dim, self.num_gaussians)
        )

        # 初始化每个高斯分布的标准差（初始设为搜索范围的10%）
        self.stds = np.ones((self.dim, self.num_gaussians)) * (self.range[:, np.newaxis] * 0.1)

        # 初始化每个高斯分布的权重
        self.weights = np.ones((self.dim, self.num_gaussians)) / self.num_gaussians

    def sample_ant_positions(self):
        """根据信息素分布采样蚂蚁位置"""
        positions = np.zeros((self.num_ants, self.dim))

        for i in range(self.num_ants):
            for j in range(self.dim):
                # 根据权重选择一个高斯分布
                gaussian_idx = np.random.choice(self.num_gaussians, p=self.weights[j])

                # 从选定的高斯分布采样
                positions[i, j] = np.random.normal(
                    self.means[j, gaussian_idx],
                    self.stds[j, gaussian_idx]
                )

            # 确保位置在搜索范围内
            positions[i] = np.clip(positions[i], self.min_bounds, self.max_bounds)

        return positions

    def update_gaussians(self, positions, fitness):
        """更新高斯混合模型参数（信息素更新）"""
        # 排序位置按适应度从好到坏
        sorted_indices = np.argsort(fitness)
        sorted_positions = positions[sorted_indices]

        # 选择适应度最好的前q%位置来更新高斯模型
        num_selected = max(1, int(self.q * self.num_ants))
        selected_positions = sorted_positions[:num_selected]

        # 更新高斯参数
        for j in range(self.dim):
            for k in range(self.num_gaussians):
                # 信息素挥发
                self.means[j, k] = (1 - self.rho) * self.means[j, k]
                self.stds[j, k] = (1 - self.rho) * self.stds[j, k] + self.rho * self.range[j] * 0.1

                # 计算每个选中位置对该高斯的贡献
                for pos in selected_positions:
                    # 计算位置与当前高斯的相似度
                    similarity = np.exp(-0.5 * ((pos[j] - self.means[j, k]) / self.stds[j, k]) ** 2)

                    # 根据相似度更新高斯参数
                    contribution = similarity * self.alpha
                    self.means[j, k] += contribution * (pos[j] - self.means[j, k]) / num_selected

                    # 更新标准差（防止过小）
                    std_update = abs(pos[j] - self.means[j, k]) * contribution / num_selected
                    self.stds[j, k] = max(self.stds[j, k] + std_update, self.range[j] * 0.01)

        # 重新标准化权重
        for j in range(self.dim):
            self.weights[j] = self.weights[j] / np.sum(self.weights[j])

    def optimize(self):
        """运行ACO算法优化过程"""
        start_time = time.time()

        # 初始化全局最优位置和适应度
        self.gbest_position = None
        self.gbest_fitness = float('inf')

        # 迭代优化
        for i in range(self.max_iter):
            # 根据高斯模型采样蚂蚁位置
            positions = self.sample_ant_positions()

            # 计算每个蚂蚁的适应度
            fitness = np.array([self.objective_func(p) for p in positions])

            # 更新全局最优
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.gbest_fitness:
                self.gbest_position = positions[best_idx].copy()
                self.gbest_fitness = fitness[best_idx]

            # 记录当前迭代的最优位置和适应度
            self.best_positions.append(self.gbest_position.copy())
            self.best_fitnesses.append(self.gbest_fitness)

            # 更新高斯模型（信息素更新）
            self.update_gaussians(positions, fitness)

            # 可选：打印中间结果
            if self.verbose and (i + 1) % 10 == 0:
                print(f"迭代 {i + 1}/{self.max_iter}, 最佳适应度: {self.gbest_fitness:.6f}")

        end_time = time.time()
        run_time = end_time - start_time

        result = {
            "best_position": self.gbest_position,
            "best_fitness": self.gbest_fitness,
            "best_positions": np.array(self.best_positions),
            "best_fitnesses": np.array(self.best_fitnesses),
            "run_time": run_time
        }

        return result