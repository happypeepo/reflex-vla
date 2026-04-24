"""Reflex Python SDK — sync + async clients for `reflex serve`.

Designed against the production `/act`, `/health`, `/config` surface shipped
in src/reflex/runtime/server.py. Consumes:

- `/health` 6-state machine (initializing/loading/warming/ready/warmup_failed/
  degraded) — `ReflexClient.health()` returns the parsed state.
- `/act` 503 + Retry-After header on circuit-broken servers — retry loop
  respects the header.
- X-Reflex-Key bearer auth — `api_key` constructor arg sets the header.
- `episode_id` field on /act request — `client.episode()` context manager
  auto-generates + propagates so RTC adapter resets correctly between episodes.
- `guard_violations` + `guard_clamped` response fields (B.6 ActionGuard) —
  surfaced on the result object as fields, not silently swallowed.

Usage:

    from reflex.client import ReflexClient

    client = ReflexClient("http://localhost:8000", api_key="abc")
    result = client.act(image="frame.jpg", instruction="pick the cup", state=[...])
    print(result["actions"])

    # Episode tracking (auto episode_id, RTC reset):
    with client.episode() as ep:
        for frame in frames:
            ep.act(image=frame.image, state=frame.state)

    # Async:
    async with ReflexAsyncClient("http://localhost:8000") as client:
        result = await client.act(...)
"""

from reflex.client.client import (
    ReflexClient,
    ReflexAsyncClient,
    ReflexClientError,
    ReflexAuthError,
    ReflexServerDegradedError,
    ReflexServerNotReadyError,
    ReflexValidationError,
    encode_image,
)

__all__ = [
    "ReflexClient",
    "ReflexAsyncClient",
    "ReflexClientError",
    "ReflexAuthError",
    "ReflexServerDegradedError",
    "ReflexServerNotReadyError",
    "ReflexValidationError",
    "encode_image",
]
