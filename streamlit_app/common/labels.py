"""Label naming + distribution helpers (matches etl/feature_engineer.py)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

# T+5 forward-return buckets produced by the feature engineer.
LABEL_NAMES: dict[int, str] = {0: "Down", 1: "Neutral", 2: "Up"}


def label_text(value: object) -> str:
    """Pretty label string; falls back to the raw repr when not an int we know."""
    if value is None:
        return "—"
    try:
        key = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    return LABEL_NAMES.get(key, str(key))


def count_label_distribution(neighbors: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Mỗi neighbor một đơn vị (phân bố đếm thuần)."""
    counter: Counter[str] = Counter()
    for row in neighbors:
        counter[label_text(row.get("label"))] += 1.0
    return dict(counter)


def weighted_label_distribution(
    neighbors: Iterable[Mapping[str, object]],
    *,
    eps: float = 0.02,
) -> dict[str, float]:
    """Trọng số ~ 1/(cosine_distance + eps): láng giềng gần đóng góp nhiều hơn."""
    acc: dict[str, float] = {}
    for row in neighbors:
        dist = float(row.get("cosine_distance", 0.0))
        weight = 1.0 / (max(dist, 0.0) + eps)
        key = label_text(row.get("label"))
        acc[key] = acc.get(key, 0.0) + weight
    return acc


def remap_label_keys(distribution: Mapping[object, object]) -> dict[str, float]:
    """Turn a ``{label_code: count}`` mapping into ``{pretty_label: float}``."""
    return {label_text(k): float(v) for k, v in distribution.items()}
