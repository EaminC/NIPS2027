"""事先定义好的分布。实验里直接调用 normal(...)、uniform(...) 等，不必再写抽样。"""

import math

import numpy as np
from scipy import stats


class Distribution:
    """一组已经绑好参数的分布。"""

    def __init__(self, name: str, params: dict, discrete: bool = False):
        self.name = name
        self.params = dict(params)
        self.discrete = discrete

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """从 rng 里接着抽 n 个。多次调用会消耗同一条随机数流。"""
        if n < 0:
            raise ValueError("n must be >= 0")
        if self.name == "normal":
            return rng.normal(self.params["mu"], self.params["sigma"], size=n)
        if self.name == "uniform":
            return rng.uniform(self.params["low"], self.params["high"], size=n)
        if self.name == "exponential":
            return rng.exponential(self.params["mean"], size=n)
        return (rng.random(n) < self.params["p"]).astype(np.int64)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """用种子重新抽 n 个。"""
        return self.draw(np.random.default_rng(seed), n)

    def theory_mean(self) -> float:
        if self.name == "normal":
            return self.params["mu"]
        if self.name == "uniform":
            return 0.5 * (self.params["low"] + self.params["high"])
        if self.name == "exponential":
            return self.params["mean"]
        return self.params["p"]

    def goodness_of_fit(self, values: np.ndarray) -> tuple[float, float]:
        """返回 (统计量, p 值)。伯努利用二项检验，其余用 KS。"""
        values = np.asarray(values)
        if self.name == "bernoulli":
            if values.size and np.any((values != 0) & (values != 1)):
                return math.inf, 0.0
            successes = int(np.sum(values == 1))
            pvalue = float(stats.binomtest(successes, values.size, self.params["p"]).pvalue)
            return float(successes), pvalue
        if self.name == "normal":
            cdf = stats.norm(loc=self.params["mu"], scale=self.params["sigma"]).cdf
        elif self.name == "uniform":
            width = self.params["high"] - self.params["low"]
            cdf = stats.uniform(loc=self.params["low"], scale=width).cdf
        else:
            cdf = stats.expon(scale=self.params["mean"]).cdf
        stat, pvalue = stats.kstest(values, cdf)
        return float(stat), float(pvalue)

    def window(self) -> tuple[float, float]:
        """画密度简图时使用的横轴范围。"""
        if self.name == "normal":
            mu, sigma = self.params["mu"], self.params["sigma"]
            return mu - 3.5 * sigma, mu + 3.5 * sigma
        if self.name == "uniform":
            low, high = self.params["low"], self.params["high"]
            pad = 0.15 * (high - low)
            return low - pad, high + pad
        if self.name == "exponential":
            return 0.0, 6.0 * self.params["mean"]
        raise ValueError("bernoulli has no continuous density")

    def density(self, xs: np.ndarray) -> np.ndarray:
        """理论密度。xs 是一串横坐标。"""
        xs = np.asarray(xs, dtype=float)
        if self.name == "normal":
            return stats.norm(self.params["mu"], self.params["sigma"]).pdf(xs)
        if self.name == "uniform":
            low, high = self.params["low"], self.params["high"]
            inside = (xs >= low) & (xs < high)
            return np.where(inside, 1.0 / (high - low), 0.0)
        if self.name == "exponential":
            return stats.expon(scale=self.params["mean"]).pdf(xs)
        raise ValueError("bernoulli has no continuous density")

    def __repr__(self) -> str:
        inner = ", ".join(f"{key}={value:g}" for key, value in self.params.items())
        return f"{self.name}({inner})"


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def normal(mu: float = 0.0, sigma: float = 1.0) -> Distribution:
    """正态分布 N(mu, sigma^2)。默认标准正态。"""
    mu = _finite("mu", mu)
    sigma = _finite("sigma", sigma)
    if sigma <= 0:
        raise ValueError("sigma must be a finite number greater than 0")
    return Distribution("normal", {"mu": mu, "sigma": sigma})


def uniform(low: float = 0.0, high: float = 1.0) -> Distribution:
    """均匀分布，默认 [0, 1)。"""
    low = _finite("low", low)
    high = _finite("high", high)
    if high <= low:
        raise ValueError("uniform needs high > low")
    return Distribution("uniform", {"low": low, "high": high})


def exponential(mean: float = 1.0) -> Distribution:
    """指数分布，mean 是均值。"""
    mean = _finite("mean", mean)
    if mean <= 0:
        raise ValueError("exponential needs mean > 0")
    return Distribution("exponential", {"mean": mean})


def bernoulli(p: float = 0.5) -> Distribution:
    """伯努利分布，样本是 0 或 1。"""
    p = _finite("p", p)
    if not 0 < p < 1:
        raise ValueError("bernoulli needs 0 < p < 1")
    return Distribution("bernoulli", {"p": p}, discrete=True)


FACTORIES = {
    "normal": normal,
    "uniform": uniform,
    "exponential": exponential,
    "bernoulli": bernoulli,
}
NAMES = tuple(FACTORIES)


def make(name: str, **params) -> Distribution:
    """按名字调用已经定义好的分布。值为 None 的参数会丢掉，改用该分布的默认值。"""
    try:
        factory = FACTORIES[name]
    except KeyError as exc:
        known = ", ".join(NAMES)
        raise ValueError(f"unknown distribution {name}. choices: {known}") from exc
    kwargs = {key: value for key, value in params.items() if value is not None}
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ValueError(f"bad parameters for {name}") from exc
