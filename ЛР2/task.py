from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Callable, Iterable, List, Sequence, Tuple


A = 2.0
B = 5.0
N_VALUES = [100, 1_000, 10_000, 100_000]
STRATA_STEPS = [1.0, 0.5]
ROULETTE_R = [0.5, 0.75, 0.05]


def f(x: float) -> float:
    return x * x


def true_integral(a: float, b: float) -> float:
    return (b**3 - a**3) / 3.0


@dataclass
class Estimate:
    value: float
    stderr: float


def sample_mean_stderr(values: Sequence[float]) -> Estimate:
    n = len(values)
    mean = fmean(values)
    if n < 2:
        return Estimate(mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = math.sqrt(var / n)
    return Estimate(mean, stderr)


def mc_simple(a: float, b: float, n: int, rng: random.Random) -> Estimate:
    width = b - a
    samples = [width * f(rng.uniform(a, b)) for _ in range(n)]
    return sample_mean_stderr(samples)


def make_strata(a: float, b: float, step: float) -> List[Tuple[float, float]]:
    strata = []
    left = a
    while left < b - 1e-12:
        right = min(left + step, b)
        strata.append((left, right))
        left = right
    return strata


def split_counts(total_n: int, parts: int) -> List[int]:
    base = total_n // parts
    rest = total_n % parts
    return [base + (1 if i < rest else 0) for i in range(parts)]


def mc_stratified(a: float, b: float, n: int, step: float, rng: random.Random) -> Estimate:
    strata = make_strata(a, b, step)
    counts = split_counts(n, len(strata))

    estimate_sum = 0.0
    variance_sum = 0.0

    for (left, right), n_i in zip(strata, counts):
        if n_i <= 0:
            continue
        width = right - left
        fx = [f(rng.uniform(left, right)) for _ in range(n_i)]
        est = sample_mean_stderr(fx)
        estimate_sum += width * est.value
        variance_sum += (width**2) * (est.stderr**2)

    return Estimate(estimate_sum, math.sqrt(variance_sum))


def pdf_power_raw(x: float, k: int) -> float:
    return x**k


def pdf_power_norm_const(a: float, b: float, k: int) -> float:
    return (k + 1) / (b ** (k + 1) - a ** (k + 1))


def pdf_power(x: float, a: float, b: float, k: int) -> float:
    return pdf_power_norm_const(a, b, k) * pdf_power_raw(x, k)


def sample_power_pdf(a: float, b: float, k: int, rng: random.Random) -> float:
    u = rng.random()
    low = a ** (k + 1)
    high = b ** (k + 1)
    return (low + u * (high - low)) ** (1.0 / (k + 1))


def mc_importance(a: float, b: float, n: int, k: int, rng: random.Random) -> Estimate:
    values = []
    for _ in range(n):
        x = sample_power_pdf(a, b, k, rng)
        p = pdf_power(x, a, b, k)
        values.append(f(x) / p)
    return sample_mean_stderr(values)


def mc_mis(
    a: float,
    b: float,
    n: int,
    k1: int,
    k2: int,
    use_power_heuristic: bool,
    rng: random.Random,
) -> Estimate:
    n1 = n // 2
    n2 = n - n1

    s1 = []
    for _ in range(n1):
        x = sample_power_pdf(a, b, k1, rng)
        p1 = pdf_power(x, a, b, k1)
        p2 = pdf_power(x, a, b, k2)

        if use_power_heuristic:
            denom = (n1 * p1) ** 2 + (n2 * p2) ** 2
            w1 = (n1 * p1) ** 2 / denom
        else:
            denom = n1 * p1 + n2 * p2
            w1 = (n1 * p1) / denom

        s1.append(w1 * f(x) / p1)

    s2 = []
    for _ in range(n2):
        y = sample_power_pdf(a, b, k2, rng)
        p1 = pdf_power(y, a, b, k1)
        p2 = pdf_power(y, a, b, k2)

        if use_power_heuristic:
            denom = (n1 * p1) ** 2 + (n2 * p2) ** 2
            w2 = (n2 * p2) ** 2 / denom
        else:
            denom = n1 * p1 + n2 * p2
            w2 = (n2 * p2) / denom

        s2.append(w2 * f(y) / p2)

    est1 = sample_mean_stderr(s1) if s1 else Estimate(0.0, 0.0)
    est2 = sample_mean_stderr(s2) if s2 else Estimate(0.0, 0.0)

    value = est1.value + est2.value
    stderr = math.sqrt(est1.stderr**2 + est2.stderr**2)
    return Estimate(value, stderr)


def mc_russian_roulette(a: float, b: float, n: int, r: float, rng: random.Random) -> Estimate:
    survive_prob = 1.0 - r
    width = b - a
    values = []
    for _ in range(n):
        x = rng.uniform(a, b)
        contribution = width * f(x)
        if rng.random() < r:
            values.append(0.0)
        else:
            values.append(contribution / survive_prob)
    return sample_mean_stderr(values)


def format_row(
    method: str,
    n: int,
    est: Estimate,
    truth: float,
) -> str:
    abs_err = abs(est.value - truth)
    delta_i = truth / math.sqrt(n)
    return (
        f"{method:<26} {n:>8} "
        f"{est.value:>14.6f} {truth:>14.6f} {abs_err:>14.6f} {delta_i:>14.6f}"
    )


def run_experiment() -> None:
    rng = random.Random(42)
    truth = true_integral(A, B)

    print("Monte Carlo integration for f(x)=x^2 on [2, 5]")
    print(
    f"{'Method':<26} {'N':>8} {'I_est':>14} {'I_true':>14} "
    f"{'|err|':>14} {'Delta_I':>14}"
    )
    print("-" * 96)

    for n in N_VALUES:
        print(format_row("Simple MC", n, mc_simple(A, B, n, rng), truth))

        for step in STRATA_STEPS:
            method = f"Stratified (h={step:g})"
            print(format_row(method, n, mc_stratified(A, B, n, step, rng), truth))

        print(format_row("Importance p(x)~x", n, mc_importance(A, B, n, 1, rng), truth))
        print(format_row("Importance p(x)~x^2", n, mc_importance(A, B, n, 2, rng), truth))
        print(format_row("Importance p(x)~x^3", n, mc_importance(A, B, n, 3, rng), truth))

        print(format_row("MIS balance", n, mc_mis(A, B, n, 1, 3, False, rng), truth))
        print(format_row("MIS power(2)", n, mc_mis(A, B, n, 1, 3, True, rng), truth))

        for r in ROULETTE_R:
            method = f"Russian roulette R={r:g}"
            print(format_row(method, n, mc_russian_roulette(A, B, n, r, rng), truth))

        print("-" * 96)


if __name__ == "__main__":
    run_experiment()
