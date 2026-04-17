"""Formatting helpers for RAC evidence payloads."""

from __future__ import annotations

from dataclasses import dataclass

from rac.retriever import Neighbor


@dataclass(frozen=True)
class Evidence:
    neighbors: list[Neighbor]
    label_distribution: dict[str, int] | None
    confidence: float | None


def build_evidence(
    *,
    neighbors: list[Neighbor],
    label_distribution: dict[str, int] | None = None,
    confidence: float | None = None,
) -> Evidence:
    return Evidence(neighbors=neighbors, label_distribution=label_distribution, confidence=confidence)

