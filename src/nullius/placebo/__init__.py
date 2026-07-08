"""Skill placebos: generate skill-free nulls and locate the real metric against them."""

from __future__ import annotations

from .compare import compare_against_placebos
from .generators import (
    ar1_matched_placebo,
    random_uniform_placebo,
    shuffled_entries_placebo,
)

__all__ = [
    "ar1_matched_placebo",
    "random_uniform_placebo",
    "shuffled_entries_placebo",
    "compare_against_placebos",
]
