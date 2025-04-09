import numpy as np
import time


class FireflyAlgorithm:
    """萤火虫算法实现"""

    def __init__(self, objective_func, bounds, num_fireflies=50, max_iter=100,
                 alpha=0.4, beta0=1, gamma=1.0, alpha_reduction=1, verbose=False):
        """
        参数:
        - objective_func: 要优化的目标函数
        - bounds: 搜索空间边界 [(x1_min, x1_max), (x2_min, x2_max), ...]
        - num_fireflies: 萤火虫数量
        - max_iter: 最大迭代次数
        - alpha: 随机移动因子
        - beta0: 最大吸引度
        - gamma: 光强吸收系数
        - alpha_reduction: 随机移动因子的衰减率
        - verbose: 是否输出中间结果
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_fireflies = num_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.alpha_reduction = alpha_reduction
        self.beta0 = beta0
        self.gamma = gamma
        self.verbose = verbose
        self.dim = len(bounds)

        # 搜索范围参数
        self.min_bounds = np.array([b[0] for b in bounds])
        self.max_bounds = np.array([b[1] for b in bounds])
        self.range = self.max_bounds - self.min_bounds

        # 记录每次迭代的最优位置和适应度
        self.best_positions = []
        self.best_fitnesses = []

    def initialize(self):
        """初始化萤火虫位置"""
        # 在边界范围内随机初始化萤火虫位置
        self.positions = np.random.uniform(
            self.min_bounds, self.max_bounds, (self.num_fireflies, self.dim)
        )

        # 计算每个萤火虫的初始亮度（适应度）
        self.fitness = np.array([self.objective_func(p) for p in self.positions])

        # 初始化全局最优位置和适应度
        best_idx = np.argmin(self.fitness)
        self.gbest_position = self.positions[best_idx].copy()
        self.gbest_fitness = self.fitness[best_idx]

    def update_fireflies(self):
        """更新萤火虫位置"""
        for i in range(self.num_fireflies):
            for j in range(self.num_fireflies):
                # 如果j比i亮，则i向j移动
                if self.fitness[j] < self.fitness[i]:
                    # 计算距离
                    r = np.sqrt(np.sum((self.positions[i] - self.positions[j]) ** 2))

                    # 计算吸引度
                    beta = self.beta0 * np.exp(-self.gamma * r ** 2)

                    # 计算随机扰动
                    random_step = self.alpha * (np.random.random(self.dim) - 0.5) * self.range

                    # 移动萤火虫i
                    self.positions[i] += beta * (self.positions[j] - self.positions[i]) + random_step

                    # 限制位置在搜索范围内
                    self.positions[i] = np.clip(self.positions[i], self.min_bounds, self.max_bounds)

                    # 更新萤火虫i的适应度
                    self.fitness[i] = self.objective_func(self.positions[i])

                    # 更新全局最优
                    if self.fitness[i] < self.gbest_fitness:
                        self.gbest_position = self.positions[i].copy()
                        self.gbest_fitness = self.fitness[i]

        # 衰减随机移动因子
        self.alpha *= self.alpha_reduction

    def optimize(self):
        """运行萤火虫算法优化过程"""
        start_time = time.time()

        # 初始化萤火虫
        self.initialize()

        # 记录最佳位置和适应度
        self.best_positions.append(self.gbest_position.copy())
        self.best_fitnesses.append(self.gbest_fitness)

        # 迭代优化
        for i in range(self.max_iter):
            # 更新萤火虫位置
            self.update_fireflies()

            # 记录每次迭代的最佳位置和适应度
            self.best_positions.append(self.gbest_position.copy())
            self.best_fitnesses.append(self.gbest_fitness)

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