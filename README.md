# 流量层 AB / AA 实验室

这个仓库看一件事：指标 `S(i)` 本身可以几乎不动，但流量层的 hash 一旦定死，实验者就能靠反复留下最好的桶和最差的桶，把小流量上的 AB 做大。群体均值的前后变化（AA）不会跟着变大。每次进出都重新分配时，这套做法留不住极值。

本地页面和命令行用的是同一个 runner。策略、分布、时间过程和流量层规则都通过注册表替换。

## 设定

| 想法 | 代码 |
| --- | --- |
| hackable：层内 hash 一开始定死 | `system: sticky` |
| unhackable：每次进出流量层都重抽 | `system: reshuffle` |
| 诚实 baseline：不挑桶，每轮重新随机，A/B 事先指定 | `strategy: honest` |
| 五组各 20%，留下最好和最差，中间三组重分；更好就换上最好，更差就换上最差 | `strategy: extremes_keeper` |
| 理论上界 / 下界：当前 S 下，最好的 b% 减最差的 b% | `strategy: oracle`，方向 `upper` / `lower` |
| S(i) 的截面分布 | `metric`: `normal` / `lognormal` / `uniform` |
| S(i) 随时间的随机过程 | `process`: `static` / `ar1` / `random_walk` |
| 小流量 x% 再推到 100% | phase 的 `exposure`，从 `0.1` 到 `1.0` |
| 真正的参数效应（默认没有） | `effect`: `null` / `constant_lift` |

AB 是 `mean(treatment) - mean(control)`。`ab_claim` 用当前曝光样本，`ab_population` 用全体用户。AA 是群体均值相对开局的变化：`aa` 含测量层的处理效应，`aa_latent` 只含指标本身的时间过程。

上下界不看 hash。在当前这批 `S` 上，任意分成两组、每组占 `arm_fraction`，最大 AB 是 `oracle_upper`，最小是 `oracle_lower`。页面上的参照界固定用 20%，和五组 hack 对齐。诚实 baseline 是两组各 50%，它自己的界更紧，不和 20% 的界画在一起。

`oracle` 直接按 S 切尾部，流量层挡不住它，只作为参照。

多个维护者对应多个流量层。层间 hash 独立，这里固定看其中一层。

## 一轮里发生什么

同一次运行里的几条对比共用一条 S 的路径，以及同一套曝光顺序。小流量用户是全量用户的嵌套子集，推全只加人。每条对比只有自己的分桶随机数。

1. 若这个 phase 的 `evolve` 为真，先把 S 推进一步。
2. 若 `strategy_active`，策略提出动作，流量层决定改不改 hash，然后策略看到新分桶。
3. 策略给出 A/B。之后的保持或推全不再调用策略，分桶和 A/B 保持搜索结束时的样子。
4. 记录 AB、AA 和上下界。

动作和流量层：

- `new_layer`：新开一层，两种系统都整层重抽。诚实 baseline 每轮都发这个。
- `assign_all`：只给还没有 hash 的用户定桶。hackable 下再发一次不会改已有 hash。
- `freeze_reshuffle`：hackable 留下冻结桶，只重抽其余用户；unhackable 忽略冻结，整层重抽。
- `hold`：不动。

## 运行

```bash
pip install numpy
python3 server.py
```

打开 http://127.0.0.1:8765 。`--port` 可改端口。

页面默认用 3,000 位用户、10% 流量。五组 hack 里每组大约 60 个曝光用户，中间组重抽才容易出现更好或更差的桶，样本 AB 的抬升在图上看得清。`configs/` 里是 20,000 人：桶更大，同样的整桶替换抬不了多少，推全之后全体差值仍会回到 0 附近。

页面有两个预设：

- **小流量后推全**：S 不变，在 10% 上搜索，再推到 100%。hackable 上样本 AB 会被抬高，全体桶均值仍接近 0，推全后 claim 掉回去。AA 接近 0。
- **冻结后随时间消退**：搜索时 S 不动，然后保持分组并让 S 按 AR(1) 回归。被抬高的 AB 会消退，AA 跟着群体均值走。

命令行用同一套引擎，配置在 `configs/`：

```bash
python -m ab_aa_lab list
python -m ab_aa_lab run configs/static_selection.json
python -m ab_aa_lab run configs/temporal_fade.json --plot
python -m pytest tests -q
```

`--plot` 需要 matplotlib。图分三栏：可行策略的 AB、oracle 的 AB、AA。实线是样本 claim，虚线是全体差值。

全量上每个桶都很大，随机重分一次，桶均值会被压在总体均值附近，所以「整桶替换」到不了理论上界。它能做大的是小流量样本上的 AB。

## 热插拔一个策略

新策略是一个类，三个方法，用装饰器登记。参照 `examples/custom_strategy.py`。

```python
from ab_aa_lab.registry import strategies
from ab_aa_lab.strategies.base import Strategy
from ab_aa_lab.types import Action, Claim, Observation

@strategies.register("my_strategy")
class MyStrategy(Strategy):
    def __init__(self, n_buckets: int = 5):
        self.n_buckets = n_buckets

    @property
    def arm_fraction(self) -> float:
        return 1.0 / self.n_buckets

    def act(self, obs: Observation) -> Action:
        return Action(kind="assign_all", n_buckets=self.n_buckets)

    def observe(self, obs: Observation) -> None:
        return None

    def claim(self, obs: Observation) -> Claim:
        return Claim(control_bucket=0, treatment_bucket=1)
```

配置里加上文件路径，路径相对该 JSON 所在目录：

```json
{
  "plugins": ["../examples/custom_strategy.py"],
  "strategy": {"name": "peek_once", "n_buckets": 5}
}
```

`metric`、`process`、`policy`、`effect` 用同一套注册表。策略只能看到处理效应叠加之前的 S；`constant_lift` 只进入最后公布的 AB / AA。

一次 suite 里，各条对比只能改 `name`、`system`、`strategy`、`effect`。指标、时间过程和 phase 是共用的。
