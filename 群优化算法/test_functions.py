import numpy as np


class TestFunction:
    """基础测试函数类"""

    def __init__(self, name, bounds, global_minimum=None):
        self.name = name
        self.bounds = bounds  # 搜索空间边界 [(x1_min, x1_max), (x2_min, x2_max), ...]
        self.global_minimum = global_minimum  # 全局最小值点和对应的函数值 (position, value)

    def __call__(self, x):
        """计算函数值"""
        raise NotImplementedError("子类必须实现此方法")


class Rastrigin(TestFunction):
    """
    Rastrigin函数
    f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)) for i=1:n
    全局最小值: f(0,...,0) = 0
    搜索范围: [-5.12, 5.12]^n
    """

    def __init__(self, dim=2):
        super().__init__(
            name="Rastrigin",
            bounds=[(-5.12, 5.12) for _ in range(dim)],
            global_minimum=(np.zeros(dim), 0.0)
        )
        self.dim = dim

    def __call__(self, x):
        x = np.asarray(x)
        return 10.0 * self.dim + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x))


class Ackley(TestFunction):
    """
    Ackley函数
    f(x) = -20*exp(-0.2*sqrt(1/n*sum(x_i^2))) - exp(1/n*sum(cos(2*pi*x_i))) + 20 + e
    全局最小值: f(0,...,0) = 0
    搜索范围: [-32.768, 32.768]^n
    """

    def __init__(self, dim=2):
        super().__init__(
            name="Ackley",
            bounds=[(-32.768, 32.768) for _ in range(dim)],
            global_minimum=(np.zeros(dim), 0.0)
        )
        self.dim = dim

    def __call__(self, x):
        x = np.asarray(x)
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
        term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
        return term1 + term2 + 20.0 + np.e


class Schwefel(TestFunction):
    """
    Schwefel函数
    f(x) = 418.9829*n - sum(x_i*sin(sqrt(|x_i|)))
    全局最小值: f(420.9687,...,420.9687) ≈ 0
    搜索范围: [-500, 500]^n
    """

    def __init__(self, dim=2):
        super().__init__(
            name="Schwefel",
            bounds=[(-500.0, 500.0) for _ in range(dim)],
            global_minimum=(np.ones(dim) * 420.9687, 0.0)
        )
        self.dim = dim

    def __call__(self, x):
        x = np.asarray(x)
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Rosenbrock(TestFunction):
    """
    Rosenbrock函数 (香蕉函数)
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    全局最小值: f(1,...,1) = 0
    搜索范围: [-5, 10]^n
    """

    def __init__(self, dim=2):
        super().__init__(
            name="Rosenbrock",
            bounds=[(-5.0, 10.0) for _ in range(dim)],
            global_minimum=(np.ones(dim), 0.0)
        )
        self.dim = dim

    def __call__(self, x):
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


# 创建所有测试函数的列表，默认3维
def get_test_functions(dim=3):
    return [
        Rastrigin(dim),
        Ackley(dim),
        Schwefel(dim),
        Rosenbrock(dim)
    ]