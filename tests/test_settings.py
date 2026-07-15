"""Tests for per-user settings: presets, the advanced toggles, and persistence."""
import dataclasses
import json

import pytest

import settings as S


class TestPresets:
    def test_speed_skips_upscale_quality_keeps_it(self):
        # The whole point of the two presets — Speed must not upscale, Quality must.
        assert S.PRESETS["speed"].upscale is False
        assert S.PRESETS["quality"].upscale is True

    def test_default_is_a_real_preset(self):
        assert S.DEFAULT_PRESET in S.PRESETS
        assert S.default_settings() == S.PRESETS[S.DEFAULT_PRESET]

    def test_matching_preset_recognises_each_preset(self):
        for name, preset in S.PRESETS.items():
            assert S.matching_preset(preset) == name

    def test_a_custom_combination_matches_no_preset(self):
        custom = dataclasses.replace(S.PRESETS["quality"], white_balance=True)
        assert S.matching_preset(custom) is None


class TestFaceLevelCycle:
    def test_cycle_advances_weakest_to_strongest_then_wraps(self):
        assert S.next_face_level(0.0) == 0.3
        assert S.next_face_level(0.3) == 0.5
        assert S.next_face_level(0.5) == 0.7
        assert S.next_face_level(0.7) == 0.0  # wraps back to off

    def test_an_unknown_level_snaps_back_into_the_cycle(self):
        # A value from a hand-edited file must not wedge the toggle.
        assert S.next_face_level(0.99) in S.FACE_LEVELS

    def test_every_level_has_a_label(self):
        for level in S.FACE_LEVELS:
            assert level in S.FACE_LABELS


class TestSettingsFromDict:
    def test_extra_keys_are_ignored(self):
        s = S.Settings.from_dict({"upscale": True, "white_balance": False,
                                  "face_restore_strength": 0.5, "junk": 123})
        assert s.upscale is True

    def test_missing_keys_fall_back_to_default(self):
        # A file from an older build lacking a field must still load.
        s = S.Settings.from_dict({"upscale": False})
        assert s.face_restore_strength == S.default_settings().face_restore_strength

    def test_types_are_coerced(self):
        s = S.Settings.from_dict({"upscale": 1, "white_balance": 0, "face_restore_strength": "0.3"})
        assert s.upscale is True and s.white_balance is False and s.face_restore_strength == 0.3


class TestSettingsStore:
    def test_a_new_user_gets_the_default(self, tmp_path):
        store = S.SettingsStore(tmp_path / "s.json")
        assert store.settings_for(999) == S.default_settings()

    def test_a_choice_survives_a_reload(self, tmp_path):
        """Persistence is the point — a restart must not reset anyone."""
        path = tmp_path / "s.json"
        S.SettingsStore(path).save(42, S.PRESETS["speed"])

        reloaded = S.SettingsStore(path)
        assert reloaded.settings_for(42) == S.PRESETS["speed"]

    def test_update_changes_one_field_and_keeps_the_rest(self, tmp_path):
        store = S.SettingsStore(tmp_path / "s.json")
        store.save(1, S.PRESETS["quality"])

        updated = store.update(1, white_balance=True)
        assert updated.white_balance is True
        assert updated.upscale is S.PRESETS["quality"].upscale  # unchanged
        assert S.matching_preset(updated) is None                # now custom

    def test_a_custom_combination_persists(self, tmp_path):
        path = tmp_path / "s.json"
        store = S.SettingsStore(path)
        store.update(7, upscale=False, face_restore_strength=0.0)

        s = S.SettingsStore(path).settings_for(7)
        assert s.upscale is False and s.face_restore_strength == 0.0

    def test_old_preset_name_format_migrates(self, tmp_path):
        """An earlier build stored a preset *name* string per user."""
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"7": "speed"}), encoding="utf-8")

        assert S.SettingsStore(path).settings_for(7) == S.PRESETS["speed"]

    def test_a_corrupt_file_does_not_crash_the_store(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{ this is not json", encoding="utf-8")

        store = S.SettingsStore(path)  # must not raise
        assert store.settings_for(1) == S.default_settings()

    def test_save_is_atomic_via_a_temp_file(self, tmp_path):
        # write-then-rename leaves no half-written primary file behind
        path = tmp_path / "s.json"
        store = S.SettingsStore(path)
        store.save(1, S.PRESETS["speed"])
        assert path.is_file()
        assert not path.with_suffix(".tmp").exists()
        assert json.loads(path.read_text(encoding="utf-8"))  # valid JSON
