"""Open X-Embodiment format converter (Phase 1 — skeleton).

Spec: features/08_curate/_curation/format-converters/openx-embodiment.md

OXE is the Google / DeepMind cross-robot dataset standard, built on top of
RLDS / TFRecord. Same dependency stack as RLDS (tensorflow_datasets) plus
the OXE-specific embodiment-id controlled vocabulary mapping.

Phase 1 ships this skeleton; full implementation lands alongside RLDS when
the first customer asks for OXE output. The OXE spec adds:
  - embodiment_id field per episode (Reflex slug → OXE name; e.g.
    "franka" → "franka_emika_panda")
  - Standardized observation/action key names per embodiment
  - Cross-embodiment normalization rules (preserved as-is in raw_actions
    per OXE convention)

Same API as the other converters; importing without [curate-rlds] extra
gives a clear install instruction.
"""
from __future__ import annotations

from pathlib import Path

from reflex.curate.format_converters.base import (
    ConversionResult,
    FormatConverter,
)


# Reflex embodiment slug → OXE canonical embodiment name.
# Sourced from the OXE robot taxonomy (sept 2024 version).
EMBODIMENT_OXE_MAP: dict[str, str] = {
    "franka": "franka_emika_panda",
    "so100": "so_arm100",
    "ur5": "ur5",
    "aloha": "aloha_static",
    "thor": "google_robot",
    "agx_orin": "google_robot",
}


class OpenXEmbodimentConverter(FormatConverter):
    """Phase 1 skeleton — raises on convert() until [curate-rlds] extra installed."""

    FORMAT_NAME = "openx-embodiment"

    def __init__(
        self,
        *,
        embodiment: str = "unknown",
    ):
        self.embodiment = embodiment
        self.oxe_embodiment = EMBODIMENT_OXE_MAP.get(
            embodiment.lower(), "unknown",
        )

    def convert(
        self,
        *,
        input_jsonl: str | Path | list[str | Path],
        output_dir: str | Path,
        min_quality: float | None = None,
        canonical_only: bool = False,
    ) -> ConversionResult:
        try:
            import tensorflow_datasets as tfds  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Open X-Embodiment conversion requires tensorflow_datasets: "
                "pip install 'reflex-vla[curate-rlds]'\n\n"
                "Phase 1 ships this converter as a skeleton. The full "
                "implementation lands when the first customer requests OXE "
                "output (~2-3 days work on top of RLDS; pattern is documented "
                "in the spec at features/08_curate/_curation/format-converters/"
                "openx-embodiment.md)."
            ) from exc

        # When tfds IS installed, full implementation goes here.
        raise NotImplementedError(
            "Open X-Embodiment converter Phase 1 ships as skeleton; full "
            "implementation deferred (sits on top of RLDS work)."
        )


__all__ = [
    "EMBODIMENT_OXE_MAP",
    "OpenXEmbodimentConverter",
]
