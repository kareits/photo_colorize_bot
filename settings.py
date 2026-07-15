"""Per-user processing settings, chosen in Telegram and persisted to disk.

Only the knobs a user should actually feel live here. Everything else — stage order,
face limits, detector threshold, thread count — stays in config as fixed operational
defaults, not something to expose.

Two ways to set them: a preset (Speed / Quality) sets every field at once, and the
advanced screen toggles fields individually. A combination that no longer matches a
preset is simply "custom" — the store keeps the full Settings, not a preset name, so
custom choices persist like any other.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """What a user gets to decide about their photo."""

    upscale: bool
    white_balance: bool
    # 0.0 turns face restoration off; otherwise it is the blend strength.
    face_restore_strength: float

    @classmethod
    def from_dict(cls, raw: dict) -> Settings:
        """Rebuild from stored JSON, tolerating missing/extra keys.

        A settings file written by an older or newer build must not crash the bot, so
        unknown keys are dropped and missing ones fall back to the default preset.
        """
        base = asdict(default_settings())
        base.update({k: raw[k] for k in base if k in raw})
        return cls(
            upscale=bool(base["upscale"]),
            white_balance=bool(base["white_balance"]),
            face_restore_strength=float(base["face_restore_strength"]),
        )


# Two presets, differing essentially in upscaling — which is ~80% of the runtime, so
# "Speed" vs "Quality" really means "skip the slow stage or not" in language a user
# gets without knowing what an upscaler is.
PRESETS: dict[str, Settings] = {
    "speed": Settings(upscale=False, white_balance=False, face_restore_strength=0.5),
    "quality": Settings(upscale=True, white_balance=False, face_restore_strength=0.5),
}

PRESET_LABELS = {"speed": "⚡ Скорость", "quality": "💎 Качество"}

# Face-restoration strengths the advanced screen cycles through, weakest first, with 0
# meaning off. These are the values compared side by side during tuning: 0.3 barely
# helps, 0.7 redraws good eyes, 0.5 is the balance.
FACE_LEVELS = [0.0, 0.3, 0.5, 0.7]
FACE_LABELS = {0.0: "Выкл", 0.3: "Слабо", 0.5: "Средне", 0.7: "Сильно"}

# New users start on Quality: this is a restoration bot, so the better result is the
# right first impression. Upscaling only kicks in for small files anyway, so large
# photos stay fast regardless of preset.
DEFAULT_PRESET = "quality"


def default_settings() -> Settings:
    return PRESETS[DEFAULT_PRESET]


def matching_preset(settings: Settings) -> str | None:
    """The preset this exactly equals, or None if it is a custom combination."""
    for name, preset in PRESETS.items():
        if preset == settings:
            return name
    return None


def next_face_level(current: float) -> float:
    """The next strength in the cycle, wrapping around."""
    try:
        idx = FACE_LEVELS.index(current)
    except ValueError:
        idx = FACE_LEVELS.index(0.5)  # unknown value snaps back to the middle
    return FACE_LEVELS[(idx + 1) % len(FACE_LEVELS)]


class SettingsStore:
    """Maps user id -> Settings, persisted as one JSON file.

    Persisted (not in-memory) so a restart or redeploy does not silently reset
    everyone's choice. A flat JSON file is enough at MVP scale; it lives on the data
    volume, kept apart from tmp/ which is wiped on startup.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt settings file must not stop the bot — start fresh.
            logger.warning("could not read settings from %s; starting empty", self.path)
            return {}
        # Back-compat: an earlier build stored a preset *name* (a string) per user.
        # Convert those to the full Settings dict on read.
        migrated: dict[str, dict] = {}
        for user, value in raw.items():
            if isinstance(value, str):
                migrated[user] = asdict(PRESETS.get(value, default_settings()))
            elif isinstance(value, dict):
                migrated[user] = value
        return migrated

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot truncate the file.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)

    def settings_for(self, user_id: int) -> Settings:
        raw = self._data.get(str(user_id))
        return default_settings() if raw is None else Settings.from_dict(raw)

    def save(self, user_id: int, settings: Settings) -> None:
        self._data[str(user_id)] = asdict(settings)
        self._save()

    def update(self, user_id: int, **changes) -> Settings:
        """Change individual fields, keeping the rest — for the advanced toggles."""
        updated = dataclasses.replace(self.settings_for(user_id), **changes)
        self.save(user_id, updated)
        return updated
