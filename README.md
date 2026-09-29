# abhack

`src` defines users, layer hashing, and scoring from a distribution. `exp` wires those three steps into an experiment.

User `i` is the stable integer `user_id`. On a layer, the bucket depends only on `user_id` and the layer name, so the same user stays in the same bucket if you hash again or if you hash a subset and then the full list. A score has a column name you choose. It is drawn independently from a distribution and does not look at the bucket.

## Layout

```
src/abhack/
  users/          write and read user_id
  layer/          write the hash to CSV and check buckets
  metric/         draw scores and check them
  utils/hash.py           hash
  utils/distributions.py  distributions
  evaluate/       AB gain, AA gain, group bounds
  strategies/     sticky hash, fresh splits, keep-extremes
  process/        static, redraw, and AR(1) scores
  study/          compare the four methods on one score path
exp/run.py        run users, hashing, and scores from one file
exp/study.py      run the AB / AA comparison
exp/out/          default output directory
```

## Install

From the repository root:

```bash
pip install -e .
```

Without installing, prefix commands with `PYTHONPATH=src`. `python exp/run.py` adds `src` to the path itself.

Requires Python 3.10+, numpy, and scipy.

## Call the pieces

Run these from the repository root. The examples write `exp/out/`.

Write 100 users, `user_id` from `0` to `99`:

```python
from abhack.users import generate_users, write_csv, read_csv

ids = generate_users(100)
write_csv(ids, "exp/out/users.csv")
read_csv("exp/out/users.csv")[:8]
```

`write_csv(100, "exp/out/users.csv")` is the same call. A large count is written in chunks, so the ids do not all sit in memory first.

Hash those users on a layer. The bucket id comes from `utils.hash`. `layer` only writes the file and checks it:

```python
from abhack.utils.hash import hash_buckets
from abhack.layer import write_assignment, check_assignment_file

hash_buckets(ids[:8], n_buckets=2, layer="layer0")
write_assignment("exp/out/users.csv", "exp/out/layer0.csv", "layer0", 2)
check_assignment_file("exp/out/users.csv", "exp/out/layer0.csv", "layer0", 2)
```

The check recomputes the hash and reports the count in each bucket. The file has two columns: `user_id,bucket`.

Draw a score from a distribution. The column name is yours; here it is `ctr`:

```python
from abhack.utils.distributions import normal
from abhack.metric import write_metric, check_metric_file

score = normal(mu=0.0, sigma=1.0)
score.sample(5, seed=0)
write_metric("exp/out/users.csv", "exp/out/ctr.csv", "ctr", score, seed=0)
check_metric_file("exp/out/ctr.csv", score, seed=0, name="ctr")
```

The check does two things: replay the same seed in user-file order, and compare the sample with the distribution. Other families are `uniform(low=0, high=10)`, `exponential(mean=2)`, and `bernoulli(p=0.3)`. With no arguments, `normal()` is standard normal, `uniform()` is `[0, 1)`, `exponential()` has mean 1, and `bernoulli()` has `p` 0.5.

## Command line

```bash
python -m abhack.users -M 100000 -o exp/out/users.csv

python -m abhack.layer -i exp/out/users.csv -o exp/out/layer0.csv --layer layer0 -k 2
python -m abhack.layer --check -i exp/out/users.csv -b exp/out/layer0.csv --layer layer0 -k 2

python -m abhack.metric -i exp/out/users.csv -o exp/out/ctr.csv \
    --name ctr --dist normal --mu 0 --sigma 1 --seed 0
python -m abhack.metric --check -i exp/out/ctr.csv \
    --name ctr --dist normal --mu 0 --sigma 1 --seed 0
```

`--dist` is `normal`, `uniform`, `exponential`, or `bernoulli`. Add `--no-replay` to check the distribution without replaying the seed.

## One file per experiment

Experiments live in `exp/runs/`. One toml states the user count, the bucket count, named presets, and the scores to draw. A score can name a preset and override parameters, or it can set `dist` directly.

```toml
name = "demo"
users = 10000
buckets = 7
layer = "layer0"

[presets.standard_normal]
dist = "normal"
mu = 0
sigma = 1

[[scores]]
name = "s"
preset = "standard_normal"

[[scores]]
name = "shift"
preset = "standard_normal"
mu = 0.2
sigma = 1.5

[[scores]]
name = "click"
dist = "bernoulli"
p = 0.5
```

The families are still `normal`, `uniform`, `exponential`, and `bernoulli`, defined in `src/abhack/utils/distributions.py`.

```bash
python exp/run.py
python exp/run.py exp/runs/demo.toml
python exp/run.py exp/runs/demo.toml -M 10000 -N 3
```

`-M` and `-N` override the user count and the bucket count in the file. Results go to `exp/out/<experiment name>/`: `users.csv`, the bucket CSV, one CSV per score, and a combined `scores.csv`. The dash in flags is the ASCII `-`. The script exits non-zero if the bucket check or any score check fails. `python -m abhack...` needs `pip install -e .`, or `PYTHONPATH=src` in front of the command.

## AB, AA, and a sticky-hash hack

Scores stay iid from the distribution in the file. A process can then move them: `static` leaves them put, `redraw` draws a new sample each round, and `ar1` is a mean-reverting shock (`phi`, `shock`).

AB gain is mean(B) minus mean(A) on the current sample. AA gain is the population mean now minus the population mean at the start. The upper bound is the best current group's mean minus the worst, on that same sample. The lower bound is the reverse. Two-group AB is the default. The extremes hack uses five groups, 20% each.

Four methods share one score path:

- `honest` assigns a random arm every round and does not rename the better half to B.
- `hackable` hashes once. The better bucket is called B. The same user stays in that bucket when traffic goes from x% to 100%.
- `unhackable` draws a new split every time a user enters the layer. Picking the better side does not survive the next entry.
- `extremes` keeps the best group and the worst group, reshuffles the middle, and replaces an extreme when a new group beats it.

```bash
python exp/study.py
python exp/study.py exp/runs/hack.toml
python exp/study.py exp/runs/hack.toml -M 20000 --traffic 0.1
```

`exp/runs/hack.toml` sets users, rounds, traffic, the score, and the process. Rows land in `exp/out/<name>/study.csv` with `ab`, `ab_full`, `aa`, `upper`, and `lower`. `ab` is the sample claim. `ab_full` is the same arm labels on every user.
