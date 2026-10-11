"""Resolve normalization statistics belonging to the supplied checkpoint."""
from pathlib import Path


def resolve_norm_stats(checkpoint, configured_asset_id=None, explicit_asset_id=None):
    assets = Path(checkpoint).resolve() / "assets"
    candidates = sorted(p for p in assets.glob("*/norm_stats.json") if p.is_file())

    def for_asset(asset_id):
        if Path(asset_id).name != asset_id or asset_id in (".", ".."):
            raise ValueError("Normalization asset id must be a single directory name.")
        return assets / asset_id / "norm_stats.json"

    if explicit_asset_id:
        selected = for_asset(explicit_asset_id)
        if not selected.is_file():
            raise FileNotFoundError(f"Requested normalization statistics not found: {selected}")
        return selected
    if configured_asset_id:
        preferred = for_asset(configured_asset_id)
        if preferred.is_file():
            return preferred
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No normalization statistics found under {assets}/*/norm_stats.json")
    names = ", ".join(p.parent.name for p in candidates)
    raise ValueError(
        f"Ambiguous normalization statistics: {names}. "
        "Set PI05_NORM_ASSET_ID to the checkpoint asset directory matching the model."
    )
