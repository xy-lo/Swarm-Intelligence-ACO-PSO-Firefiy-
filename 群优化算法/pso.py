import numpy as np
import time


class ParticleSwarmOptimizer:
    """粒子群优化算法实现"""

    def __init__(self, objective_func, bounds, num_particles=50, max_iter=100,
                 verbose=False,w=0.7, c1=1.5, c2=1.5):
        """
        参数:
        - objective_func: 要优化的目标函数
        - bounds: 搜索空间边界 [(x1_min, x1_max), (x2_min, x2_max), ...]
        - num_particles: 粒子数量
        - max_iter: 最大迭代次数
        - w: 惯性权重
        - c1: 个体学习因子
        - c2: 社会学习因子
        - verbose: 是否输出中间结果
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        self.dim = len(bounds)

        # 粒子位置和速度的范围
        self.min_bounds = np.array([b[0] for b in bounds])
        self.max_bounds = np.array([b[1] for b in bounds])
        self.velocity_bounds = np.array([(b[1] - b[0]) * 0.1 for b in bounds])

        # 记录每次迭代的最优位置和适应度
        self.best_positions = []
        self.best_fitnesses = []

    def initialize(self):
        """初始化粒子群的位置和速度"""
        # 在边界范围内随机初始化粒子位置
        self.positions = np.random.uniform(
            self.min_bounds, self.max_bounds, (self.num_particles, self.dim)
        )

        # 初始化粒子速度
        self.velocities = np.random.uniform(
            -self.velocity_bounds, self.velocity_bounds, (self.num_particles, self.dim)
        )

        # 计算每个粒子的初始适应度
        self.fitness = np.array([self.objective_func(p) for p in self.positions])

        # 初始化个体最优位置和适应度
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.fitness.copy()

        # 初始化全局最优位置和适应度
        gbest_idx = np.argmin(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitness[gbest_idx]

    def update_velocities(self):
        """更新粒子的速度"""
        r1 = np.random.random((self.num_particles, self.dim))
        r2 = np.random.random((self.num_particles, self.dim))

        # 更新速度
        cognitive_component = self.c1 * r1 * (self.pbest_positions - self.positions)
        social_component = self.c2 * r2 * (self.gbest_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive_component + social_component

        # 限制速度范围
        self.velocities = np.clip(self.velocities, -self.velocity_bounds, self.velocity_bounds)

    def update_positions(self):
        """更新粒子的位置"""
        # 更新位置
        self.positions = self.positions + self.velocities

        # 限制位置在搜索范围内
        self.positions = np.clip(self.positions, self.min_bounds, self.max_bounds)

        # 更新每个粒子的适应度
        self.fitness = np.array([self.objective_func(p) for p in self.positions])

        # 更新个体最优位置和适应度
        improved_idx = self.fitness < self.pbest_fitness
        self.pbest_positions[improved_idx] = self.positions[improved_idx].copy()
        self.pbest_fitness[improved_idx] = self.fitness[improved_idx].copy()

        # 更新全局最优位置和适应度
        gbest_idx = np.argmin(self.pbest_fitness)
        if self.pbest_fitness[gbest_idx] < self.gbest_fitness:
            self.gbest_position = self.pbest_positions[gbest_idx].copy()
            self.gbest_fitness = self.pbest_fitness[gbest_idx]

    def optimize(self):
        """运行PSO算法优化过程"""
        start_time = time.time()

        # 初始化粒子群
        self.initialize()

        # 记录最佳位置和适应度
        self.best_positions.append(self.gbest_position.copy())
        self.best_fitnesses.append(self.gbest_fitness)

        # 迭代优化
        for i in range(self.max_iter):
            # 更新速度和位置
            self.update_velocities()
            self.update_positions()

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