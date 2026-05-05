"""RLDS / TFRecord format converter (Phase 1 — skeleton).

Spec: features/08_curate/_curation/format-converters/rlds.md

RLDS is Google's reinforcement-learning dataset format built on TFRecord.
The full converter requires `tensorflow_datasets` (~80 MB install) which
is too heavy to ship in the base reflex-vla package.

Phase 1 ships this skeleton: importing RLDSConverter without the
[curate-rlds] extra fails loud with the install instruction. When a
customer needs RLDS output, they install the extra and the converter
works against pyarrow-converted parquet via tfds.builder_from_directory.

Code-completeness target for Phase 1.5 (when first customer asks for RLDS):
  - Episode → RLDS step protobuf
  - tfds.features schema
  - Sharded TFRecord writer with metadata
  - dataset_info.json + features.json + manifest

The skeleton API matches the other converters so callers don't need to
special-case the format — just `pip install` the extra and the same
`reflex curate convert --format rlds` works.
"""
from __future__ import annotations

from pathlib import Path

from reflex.curate.format_converters.base import (
    ConversionResult,
    FormatConverter,
)


class RLDSConverter(FormatConverter):
    """Phase 1 skeleton — raises on convert() until [curate-rlds] extra installed."""

    FORMAT_NAME = "rlds"

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
                "RLDS conversion requires tensorflow_datasets: "
                "pip install 'reflex-vla[curate-rlds]'\n\n"
                "Phase 1 ships this converter as a skeleton. The full "
                "implementation lands when the first customer requests RLDS "
                "output (~3-4 days work; pattern is documented in the spec at "
                "features/08_curate/_curation/format-converters/rlds.md)."
            ) from exc

        # When tfds IS installed, the implementation goes here.
        # Phase 1.5 work — tracked in features/08_curate/_curation/format-converters/rlds.md.
        raise NotImplementedError(
            "RLDS converter Phase 1 ships as skeleton; full implementation deferred."
        )


__all__ = [
    "RLDSConverter",
]
