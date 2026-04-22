"""Episode-aware VLM prefix cache for VLA serving.

Designed for the moat: when `pi0.5` (or other VLAs) is deployed with the
state-out preprocessor, the language prompt is stable per episode while
images change every frame. A cache keyed on `(episode_id, lang_hash)`
hits ~99% of frames within an episode → ~9x per-chunk speedup (validated
by `scripts/modal_latency_pi05_decomposed.py` — 100ms → 11ms = 8.95x).

Design grounded in the vLLM block-hash pattern documented in
`reference/deep_dive_vllm_prefix_cache.md`. Differences vs vLLM:

- **Episode-keyed**, not request-keyed. A single episode produces N
  timesteps; all share the same VL prefix. vLLM caches per-request KV
  blocks for autoregressive generation; our cache holds one VLM
  output per episode.
- **No paged attention.** Action chunks are fixed-size (50 actions per
  forward); there's no growing token sequence. We store the entire
  `past_kv` tensor list per cache entry.
- **Image-agnostic on hit.** The cache key ignores image content
  entirely — within an episode, only proprio state changes (and that
  goes through `state_proj`, not the VLM). The expert_denoise still
  runs every step with the latest state.

The cache is referenced from `Pi05DecomposedInference` when
`cache_level="episode"`. Callers must pass `episode_id` to
`predict_action_chunk`. New episodes get a fresh VLM forward; subsequent
calls within the same episode reuse the cached `past_kv`.

## When to use

- `cache_level="episode"`: pi0.5 STATE-OUT student where lang is stable
  per episode. The 9x moat lives here.
- `cache_level="prefix"`: pi0 (state via state_proj, lang stable across
  episodes too) or SmolVLA. Existing single-slot cache suffices.
- `cache_level="action"`: pi0.5 DEFAULT student where lang drifts per
  frame; cache the entire action chunk on (image, lang) hash.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def lang_hash(lang_tokens: np.ndarray) -> bytes:
    """SHA256-truncated hash of the language token sequence.

    Truncated to 16 bytes — collision probability negligible for the
    ~10-1000 episodes a single Reflex serve process will see in its
    lifetime, and saves cache key memory.
    """
    return hashlib.sha256(lang_tokens.tobytes()).digest()[:16]


@dataclass
class EpisodePrefix:
    """One cached VLM forward for an episode."""
    episode_id: str
    lang_hash: bytes
    past_kv: list[np.ndarray]            # flat [k_0, v_0, k_1, v_1, ...]
    prefix_pad_masks: np.ndarray
    birth_time_ns: int = 0
    last_accessed_ns: int = 0
    hit_count: int = 0                   # how many timesteps reused this


@dataclass
class EpisodeCacheStats:
    """Cache metrics. Episode-grained: hits = timesteps that reused a
    cached prefix; episode_count = unique episodes seen."""
    hits: int = 0
    misses: int = 0
    episode_count: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": self.hit_rate,
            "episode_count": self.episode_count,
            "evictions": self.evictions,
        }


class EpisodeCache:
    """Episode-keyed VLM prefix cache.

    Storage: ordered dict keyed on (episode_id, lang_hash). LRU
    eviction once `max_episodes` is exceeded. Concurrent access is NOT
    safe — callers serialize via the parent inference class.

    Parameters
    ----------
    max_episodes
        How many episodes to retain at once. Each entry holds the full
        VLM `past_kv` (~few hundred MB on GPU for pi0.5). Default 8 fits
        comfortably in A100-80GB headroom; for Jetson Orin Nano (8GB) set
        to 1-2.
    """

    def __init__(self, max_episodes: int = 8):
        self.max_episodes = max_episodes
        self._cache: OrderedDict[tuple[str, bytes], EpisodePrefix] = OrderedDict()
        self.stats = EpisodeCacheStats()

    def lookup(
        self,
        episode_id: str,
        lang_tokens: np.ndarray,
    ) -> EpisodePrefix | None:
        """Return cached prefix for episode if present, else None."""
        key = (episode_id, lang_hash(lang_tokens))
        prefix = self._cache.get(key)
        if prefix is None:
            self.stats.misses += 1
            return None

        # Move to end (most-recently-used)
        self._cache.move_to_end(key)
        prefix.last_accessed_ns = time.monotonic_ns()
        prefix.hit_count += 1
        self.stats.hits += 1
        return prefix

    def insert(
        self,
        episode_id: str,
        lang_tokens: np.ndarray,
        past_kv: list[np.ndarray],
        prefix_pad_masks: np.ndarray,
    ) -> EpisodePrefix:
        """Insert a fresh VLM prefix for an episode. Evicts LRU if needed."""
        key = (episode_id, lang_hash(lang_tokens))
        now = time.monotonic_ns()

        # Evict LRU if at capacity
        while len(self._cache) >= self.max_episodes:
            evicted_key, _ = self._cache.popitem(last=False)
            self.stats.evictions += 1
            logger.debug("[episode-cache] evicted LRU episode %s", evicted_key[0])

        prefix = EpisodePrefix(
            episode_id=episode_id,
            lang_hash=key[1],
            past_kv=past_kv,
            prefix_pad_masks=prefix_pad_masks,
            birth_time_ns=now,
            last_accessed_ns=now,
            hit_count=0,
        )
        self._cache[key] = prefix
        self.stats.episode_count += 1
        return prefix

    def reset(self) -> None:
        """Drop all cached entries. Useful between LIBERO eval episodes
        when you want to ensure no cross-episode memory."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


__all__ = [
    "EpisodeCache",
    "EpisodeCacheStats",
    "EpisodePrefix",
    "lang_hash",
]
