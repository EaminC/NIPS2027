"""终端里的分桶条形图和分布简图。"""

import os
import shutil
import sys

import numpy as np

from .distributions import Distribution


def _color_on() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _paint(text: str, code: str, color: bool) -> str:
    if not color or text == "":
        return text
    return f"\033[{code}m{text}\033[0m"


def _width(preferred: int, margin: int) -> int:
    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(16, min(preferred, columns - margin))


def _tag(text: str, code: str, color: bool) -> str:
    return _paint(f"【{text}】", code, color)


def _hi(text: str, color: bool) -> str:
    return _paint(str(text), "1;36", color)


def _dim(text: str, color: bool) -> str:
    return _paint(str(text), "2", color)


def _num(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        return "n/a"
    if abs(value) >= 1000 or value == int(value):
        return f"{value:,.4g}"
    return f"{value:.4g}"


def render_run_header(name: str, config: str, out: str) -> str:
    """实验开头：配置和输出目录。"""
    color = _color_on()
    return "\n".join(
        [
            f"{_tag('Experiment', '1;35', color)}  {_hi(name, color)}",
            f"  {_dim('config', color)}  {_dim(config, color)}",
            f"  {_dim('output', color)}  {_dim(out, color)}",
        ]
    )


def render_run_done(name: str, n_scores: int, out: str, table: str) -> str:
    """实验结尾：写出了哪些文件。"""
    color = _color_on()
    return "\n".join(
        [
            f"{_tag('Done', '1;32', color)}  {_hi(name, color)}",
            f"  User list, bucket file, and {_hi(n_scores, color)} score columns are in {_dim(out, color)}",
            f"  {_dim('combined scores', color)}  {_dim(table, color)}",
        ]
    )


def render_users(n: int, path: str, start: int = 0) -> str:
    """用户文件的摘要。"""
    color = _color_on()
    span = "empty" if n <= 0 else f"{start:,} – {start + n - 1:,}"
    return "\n".join(
        [
            _tag("Users", "1;36", color),
            "  Write a user list. Bucketing and scoring both read it in this order.",
            f"  {_dim('users', color)}  {_hi(f'{n:,}', color)}    {_dim('ids', color)}  {_hi(span, color)}",
            f"  {_dim('file', color)}  {_dim(path, color)}",
        ]
    )


def render_buckets(
    counts: np.ndarray,
    layer: str,
    path: str,
    errors: list[str] | None = None,
) -> str:
    """每个桶一行条形图。errors 为 None 时不写检验结论。"""
    color = _color_on()
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    k = int(counts.size)
    expected = total / k if k else 0.0
    lines = [
        _tag("Buckets", "1;34", color),
        (
            f"  Layer {_hi(layer, color)} assigns each user to one of {_hi(k, color)} buckets. "
            "The same user always lands in the same bucket."
        ),
        f"  Each row is one bucket. Counts should sit near {_hi(_num(expected), color)}.",
        f"  {_dim('file', color)}  {_dim(path, color)}",
        "",
    ]
    lines.extend(_bucket_bars(counts, expected, color))
    if errors is not None:
        lines.append(
            _check_line(
                not errors,
                "Check",
                "bucket ids match a fresh hash of the same layer",
                errors,
                color,
            )
        )
    return "\n".join(lines)


def render_scores(
    name: str,
    dist: Distribution,
    values: np.ndarray | None = None,
    path: str | None = None,
    result: dict | None = None,
    seed: int | None = None,
    preset: str | None = None,
) -> str:
    """分布简图。有样本时叠上直方图，伯努利画成 0/1 两条。"""
    color = _color_on()
    values = None if values is None else np.asarray(values)
    n = 0 if values is None else int(values.size)
    detail = _hi(str(dist), color)
    if preset:
        detail += f"    {_dim('preset', color)}  {_paint(preset, '1;35', color)}"
    if seed is not None:
        detail += f"    {_dim('seed', color)}  {_hi(seed, color)}"
    lines = [
        f"{_tag('Score', '1;33', color)}  {_hi(name, color)}",
        "  Draw one score per user from this distribution. Buckets are ignored.",
        f"  {detail}",
    ]
    if path:
        lines.append(f"  {_dim('file', color)}  {_dim(path, color)}")
    if values is not None and n:
        mean_line = (
            f"  {_dim('sample mean', color)}  {_hi(_num(float(np.mean(values))), color)}"
            f"    {_dim('distribution mean', color)}  {_hi(_num(dist.theory_mean()), color)}"
        )
        if not dist.discrete:
            mean_line += f"    {_dim('sample sd', color)}  {_hi(_num(float(np.std(values))), color)}"
        lines.append(mean_line)
    lines.append("")
    if dist.discrete:
        lines.append("  Compare the share of 0 and of 1. The upper bar is theory, the lower bar is this sample.")
        lines.extend(_bernoulli_chart(dist, values, color))
    else:
        theory = _paint("dim ░", "2", color)
        sample = _paint("solid █", "1;36", color)
        both = _paint("overlap ▓", "1;37", color)
        lines.append(f"  {theory} is the theory, {sample} is this sample, {both} is both.")
        lines.extend(_density_chart(dist, values, color))
    if result is not None:
        lines.append("")
        errors = result.get("errors") or []
        if seed is not None:
            lines.append(
                _check_line(
                    bool(result.get("replay_ok")),
                    "Replay",
                    "redraw with the same seed; values match the file",
                    [e for e in errors if e.lower().startswith("replay")],
                    color,
                )
            )
        fit_errors = [e for e in errors if not e.lower().startswith("replay")]
        pvalue = result.get("pvalue")
        if pvalue == pvalue:
            fit_text = f"sample matches this distribution (p={pvalue:.4g}, larger is closer)"
        else:
            fit_text = "sample matches this distribution"
        lines.append(_check_line(not fit_errors, "Fit", fit_text, fit_errors, color))
    return "\n".join(lines)


def _bucket_bars(counts: np.ndarray, expected: float, color: bool) -> list[str]:
    k = int(counts.size)
    if k == 0:
        return [f"  {_dim('no buckets', color)}"]
    show = _index_window(k, limit=24)
    peak = int(counts.max()) if counts.size else 0
    width = _width(36, 28)
    lines = []
    previous = None
    for index in show:
        if previous is not None and index > previous + 1:
            lines.append("  …")
        count = int(counts[index])
        share = count / counts.sum() if counts.sum() else 0.0
        filled = 0 if peak == 0 else int(round(width * count / peak))
        bar = _paint("█" * filled, "1;36", color) + _paint("░" * (width - filled), "2", color)
        lines.append(
            f"  {_paint(f'{index:>4}', '1;33', color)}  {bar}  {_hi(f'{count:>10,}', color)}  {_paint(f'{share:6.1%}', '1;32', color)}"
        )
        previous = index
    return lines


def _index_window(k: int, limit: int) -> list[int]:
    if k <= limit:
        return list(range(k))
    head = limit // 2
    tail = limit - head
    return list(range(head)) + list(range(k - tail, k))


def _density_chart(dist: Distribution, values: np.ndarray | None, color: bool) -> list[str]:
    width = _width(48, 4)
    height = 8
    lo, hi = dist.window()
    edges = np.linspace(lo, hi, width + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    theory = dist.density(centers)
    sample_density = None
    outside = 0
    if values is not None and values.size:
        outside = int(np.sum((values < lo) | (values > hi)))
        hist, _ = np.histogram(values, bins=width, range=(lo, hi))
        bin_width = (hi - lo) / width
        sample_density = hist / (values.size * bin_width)
    theory_peak = float(theory.max()) if theory.size else 0.0
    if theory_peak > 0:
        peak = theory_peak
    elif sample_density is not None and float(sample_density.max()) > 0:
        peak = float(sample_density.max())
    else:
        peak = 1.0
    theory_n = np.clip(theory / peak, 0, 1)
    sample_n = None if sample_density is None else np.clip(sample_density / peak, 0, 1)

    rows = []
    for row in range(height, 0, -1):
        level = (row - 0.5) / height
        chars = []
        for i in range(width):
            theory_here = theory_n[i] >= level
            sample_here = sample_n is not None and sample_n[i] >= level
            if theory_here and sample_here:
                chars.append(_paint("▓", "1;37", color))
            elif sample_here:
                chars.append(_paint("█", "1;36", color))
            elif theory_here:
                chars.append(_paint("░", "34", color))
            else:
                chars.append(" ")
        rows.append("  " + "".join(chars))

    axis = "  " + _axis(lo, hi, width, color)
    lines = rows + [axis]
    if outside:
        lines.append(f"  {_paint(f'{outside:,} samples fall outside this axis', '33', color)}")
    return lines


def _bernoulli_chart(dist: Distribution, values: np.ndarray | None, color: bool) -> list[str]:
    width = _width(28, 36)
    p = float(dist.params["p"])
    theory = {0: 1.0 - p, 1: p}
    sample = None
    counts = {0: 0, 1: 0}
    if values is not None and values.size:
        ones = int(np.sum(values == 1))
        zeros = int(values.size - ones)
        counts = {0: zeros, 1: ones}
        sample = {0: zeros / values.size, 1: ones / values.size}
    lines = []
    for label in (0, 1):
        lines.append(f"  {_tag(str(label), '1;33', color)}")
        lines.append(
            f"    {_dim('theory', color)}  "
            + _fraction_bar(theory[label], width, color, "33")
            + f"  {_paint(f'{theory[label]:6.1%}', '33', color)}"
        )
        if sample is not None:
            lines.append(
                f"    {_paint('sample', '1;36', color)}  "
                + _fraction_bar(sample[label], width, color, "1;36")
                + f"  {_hi(f'{sample[label]:6.1%}', color)}   {_hi(f'{counts[label]:,}', color)} users"
            )
    return lines


def _fraction_bar(fraction: float, width: int, color: bool, code: str) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return _paint("█" * filled, code, color) + _paint("░" * (width - filled), "2", color)


def _axis(lo: float, hi: float, width: int, color: bool) -> str:
    left, mid, right = f"{lo:.4g}", f"{(lo + hi) / 2:.4g}", f"{hi:.4g}"
    if width <= len(left) + len(right) + 2:
        return _paint(left, "1;33", color)
    gap = width - len(left) - len(right)
    if gap > len(mid) + 2:
        pad_left = (gap - len(mid)) // 2
        pad_right = gap - len(mid) - pad_left
        return (
            _paint(left, "1;33", color)
            + " " * pad_left
            + _paint(mid, "1;33", color)
            + " " * pad_right
            + _paint(right, "1;33", color)
        )
    return _paint(left, "1;33", color) + " " * (width - len(left) - len(right)) + _paint(right, "1;33", color)


_METHOD_COLOR = {
    "honest": "1;32",
    "hackable": "1;31",
    "unhackable": "1;35",
    "extremes": "1;33",
}
_METHOD_NOTE = {
    "honest": "Random arm each round. The better half is not renamed to B.",
    "hackable": "Hash is fixed at the start. The better bucket is called B, and those users stay there when traffic goes to 100%.",
    "unhackable": "Every entry into the layer is a new random split. Calling the better side B does not survive the next entry.",
    "extremes": "Keep the best group and the worst group. Reshuffle the middle. A better new group replaces the best; a worse one replaces the worst.",
}
_SPARK = "▁▂▃▄▅▆▇█"


def render_study(name, dist, process, users, traffic, rounds, groups, path, rows) -> str:
    """四种做法的 AB、AA 和上下界。"""
    color = _color_on()
    by: dict[str, list] = {}
    for row in rows:
        by.setdefault(row.method, []).append(row)
    lines = [
        f"{_tag('Study', '1;35', color)}  {_hi(name, color)}",
        (
            f"  {_dim('users', color)}  {_hi(f'{users:,}', color)}"
            f"    {_dim('traffic', color)}  {_hi(f'{traffic:.0%}', color)}"
            f"    {_dim('rounds', color)}  {_hi(rounds, color)}"
            f"    {_dim('extreme groups', color)}  {_hi(groups, color)}"
        ),
        f"  {_dim('score', color)}  {_hi(dist, color)}    {_dim('process', color)}  {_hi(process, color)}",
        f"  {_dim('file', color)}  {_dim(path, color)}",
        "",
        f"  {_tag('AB', '1;36', color)}  {_dim('mean(B) - mean(A) on the current sample', color)}",
        f"  {_tag('AA', '1;32', color)}  {_dim('population mean now - population mean at the start', color)}",
        f"  {_tag('Upper', '1;33', color)}  {_dim('best current group - worst current group, on that sample', color)}",
        f"  {_tag('Lower', '1;34', color)}  {_dim('worst current group - best current group', color)}",
        "",
    ]
    for method in by:
        code = _METHOD_COLOR.get(method, "1;37")
        series = by[method]
        last = series[-1]
        lines.append(f"{_tag(method, code, color)}")
        lines.append(f"  {_METHOD_NOTE.get(method, '')}")
        lines.append(
            f"  {_dim('AB sample', color)}  {_paint(_num(last.ab), code, color)}"
            f"    {_dim('AB at 100%', color)}  {_paint(_num(last.ab_full), code, color)}"
            f"    {_dim('AA', color)}  {_paint(_num(last.aa), '1;32', color)}"
        )
        lines.append(
            f"  {_dim('upper', color)}  {_hi(_num(last.upper), color)}"
            f"    {_dim('lower', color)}  {_hi(_num(last.lower), color)}"
            f"    {_dim('AB over rounds', color)}  {_spark([row.ab for row in series], code, color)}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _spark(values, code: str, color: bool) -> str:
    vals = np.asarray(list(values), dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return _dim("n/a", color)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi == lo:
        chars = "▄" * vals.size
    else:
        idx = np.clip(np.rint((vals - lo) / (hi - lo) * (len(_SPARK) - 1)), 0, len(_SPARK) - 1)
        chars = "".join(" " if not np.isfinite(v) else _SPARK[int(i)] for v, i in zip(vals, idx))
    return _paint(chars, code, color)


def _check_line(ok: bool, name: str, meaning: str, errors: list[str], color: bool) -> str:
    code = "1;32" if ok else "1;31"
    mark = _paint("pass" if ok else "fail", code, color)
    lines = [f"  {_tag(name, code, color)}  {mark}    {_dim(meaning, color)}"]
    if not ok:
        lines.extend(_paint(f"        {error}", "31", color) for error in errors)
    return "\n".join(lines)
