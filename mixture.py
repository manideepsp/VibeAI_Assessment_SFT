from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional


@dataclass(frozen=True)
class MixtureSpec:
    """Temperature-based mixture specification.

    Given dataset sizes n_i and temperature alpha in (0, 1], the sampling probability is:

        p_i = n_i^alpha / sum_j n_j^alpha

    This reduces dominance by large datasets when alpha < 1.
    """

    alpha: float
    seed: int


def temperature_mixture_probs(sizes: Mapping[str, int], alpha: float) -> Dict[str, float]:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    if not sizes:
        raise ValueError("sizes must be non-empty")

    weights: Dict[str, float] = {}
    for name, n in sizes.items():
        if n <= 0:
            raise ValueError(f"dataset '{name}' has non-positive size: {n}")
        weights[name] = float(n) ** float(alpha)

    z = sum(weights.values())
    if z <= 0.0 or math.isnan(z) or math.isinf(z):
        raise ValueError("invalid normalization constant for mixture")

    return {k: v / z for k, v in weights.items()}


class _IndexCycler:
    """Cycles through shuffled indices deterministically per RNG state."""

    def __init__(self, n: int, rng: random.Random):
        self._n = int(n)
        self._rng = rng
        self._indices = list(range(self._n))
        self._rng.shuffle(self._indices)
        self._pos = 0

    def next(self) -> int:
        if self._pos >= self._n:
            self._rng.shuffle(self._indices)
            self._pos = 0
        idx = self._indices[self._pos]
        self._pos += 1
        return idx


class TemperatureMixtureSampler:
    """Reproducible temperature-based sampler over multiple datasets.

    This sampler is intentionally framework-agnostic: it yields dict examples and can be wrapped
    by a torch IterableDataset later.

    Parameters
    - datasets: mapping name -> indexable dataset (supports __len__ and __getitem__)
    - alpha: mixture temperature in (0, 1]
    - seed: RNG seed (controls dataset selection and within-dataset shuffling)
    - rng: optional injected RNG. If provided, `seed` is ignored.
    - max_examples: optional cap for a finite iterator (useful for tests/epochs)

    Invariants / design intent
    - Each underlying dataset must be restartable via indexing (map-style).
    - Temperature controls *relative sampling frequency* across datasets, not within-dataset ordering.
    - This sampler is framework-agnostic; do not add torch dependencies here.

    Notes on leakage
    - This sampler does not inspect labels or text; it only uses dataset sizes and deterministic RNG.
    """

    def __init__(
        self,
        datasets: Mapping[str, Any],
        *,
        alpha: float,
        seed: int,
        rng: Optional[random.Random] = None,
        max_examples: Optional[int] = None,
    ) -> None:
        if not datasets:
            raise ValueError("datasets must be non-empty")

        self._datasets: Dict[str, Any] = dict(datasets)
        self._names = list(self._datasets.keys())
        self._sizes = {k: len(v) for k, v in self._datasets.items()}
        self._probs = temperature_mixture_probs(self._sizes, alpha)

        self._rng = rng if rng is not None else random.Random(seed)
        # Independent RNG streams for index cycling per dataset name (still deterministic).
        self._index_rngs = {k: random.Random(self._rng.randint(0, 2**31 - 1)) for k in self._names}
        self._cyclers = {k: _IndexCycler(self._sizes[k], self._index_rngs[k]) for k in self._names}

        self._max_examples = max_examples

        # Prepare cumulative distribution for dataset selection.
        cumulative = []
        total = 0.0
        for name in self._names:
            total += self._probs[name]
            cumulative.append((total, name))
        cumulative[-1] = (1.0, cumulative[-1][1])
        self._cdf = cumulative

    @property
    def probs(self) -> Dict[str, float]:
        return dict(self._probs)

    @property
    def sizes(self) -> Dict[str, int]:
        return dict(self._sizes)

    def _sample_dataset_name(self) -> str:
        r = self._rng.random()
        for cutoff, name in self._cdf:
            if r <= cutoff:
                return name
        return self._cdf[-1][1]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        remaining = self._max_examples
        while remaining is None or remaining > 0:
            ds_name = self._sample_dataset_name()
            ds = self._datasets[ds_name]
            idx = self._cyclers[ds_name].next()
            example = ds[idx]
            if not isinstance(example, dict):
                raise TypeError(f"Dataset '{ds_name}' returned non-dict example at idx={idx}")

            out = dict(example)
            out["mixture_source"] = ds_name

            yield out
            if remaining is not None:
                remaining -= 1


def sample_mixture_counts(sampler: TemperatureMixtureSampler, n: int) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in sampler.probs.keys()}
    for ex in sampler.take(n):
        counts[ex["mixture_source"]] += 1
    return counts


def _take_iter(it: Iterator[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    for _ in range(n):
        yield next(it)


def _sampler_take(self: TemperatureMixtureSampler, n: int) -> Iterator[dict[str, Any]]:
    return _take_iter(iter(self), n)


# Attach a small helper without making the class noisy.
TemperatureMixtureSampler.take = _sampler_take  # type: ignore[attr-defined]
