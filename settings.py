"""Per-user processing settings, chosen in Telegram and persisted to disk.

Only the knobs a user should actually feel live here. Everything else — stage order,
face limits, detector threshold, thread count — stays in config as fixed operational
defaults, not something to expose.

A preset is just a named Settings value. Picking one sets every field at once; the
"advanced" tumblers (a later step) will edit fields individually, at which point the
preset simply reads as "custom".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """What a user gets to decide about their photo."""

    upscale: bool
    white_balance: bool
    # 0.0 turns face restoration off; otherwise it is the blend strength.
    face_restore_strength: float


# Two presets, differing essentially in upscaling — which is ~80% of the runtime, so
# "Speed" vs "Quality" really means "skip the slow stage or not" in language a user
# gets without knowing what an upscaler is.
PRESETS: dict[str, Settings] = {
    "speed": Settings(upscale=False, white_balance=False, face_restore_strength=0.5),
    "quality": Settings(upscale=True, white_balance=False, face_restore_strength=0.5),
}

PRESET_LABELS = {"speed": "⚡ Скорость", "quality": "💎 Качество"}

# New users start on Quality: this is a restoration bot, so the better result is the
# right first impression. Upscaling only kicks in for small files anyway, so large
# photos stay fast regardless of preset.
DEFAULT_PRESET = "quality"


class SettingsStore:
    """Maps user id -> chosen preset name, persisted as one JSON file.

    Persisted (not in-memory) so a restart or redeploy does not silently reset
    everyone's choice. A flat JSON file is enough at MVP scale; it lives on the data
    volume, kept apart from tmp/ which is wiped on startup.
    """

    def __init__(self, path: Path):
        self.path = path
        self._presets: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt settings file must not stop the bot — start fresh.
            logger.warning("could not read settings from %s; starting empty", self.path)
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot truncate the file.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._presets), encoding="utf-8")
        tmp.replace(self.path)

    def preset_name(self, user_id: int) -> str:
        name = self._presets.get(str(user_id), DEFAULT_PRESET)
        return name if name in PRESETS else DEFAULT_PRESET

    def settings_for(self, user_id: int) -> Settings:
        return PRESETS[self.preset_name(user_id)]

    def set_preset(self, user_id: int, preset: str) -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset: {preset}")
        self._presets[str(user_id)] = preset
        self._save()


def default_settings() -> Settings:
    return PRESETS[DEFAULT_PRESET]
