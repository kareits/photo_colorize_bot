"""Tests for the bot's input guards.

The filename and the file itself both come from a stranger on the internet, so these
are the functions standing between that and the filesystem.
"""
import numpy as np
import pytest

import bot
import config


class TestSuffix:
    """The old code did `file_name.rsplit(".", 1)[-1]` and pasted the result into a
    path. Everything here is a way that goes wrong."""

    def test_ordinary_names_keep_their_extension(self):
        assert bot._suffix_for("photo.jpg", ".png") == ".jpg"
        assert bot._suffix_for("scan.TIFF", ".png") == ".tiff"

    def test_path_traversal_is_refused(self):
        """The attack the whitelist exists for.

        `Path("tmp") / "x.png/../../authorized_keys"` resolves outside tmp/, so the
        downloaded bytes would land wherever the sender pointed. Falling back to the
        default extension keeps the write inside tmp/ no matter what the name says.
        """
        evil = "photo.png/../../../../home/user/.ssh/authorized_keys"
        assert bot._suffix_for(evil, ".png") == ".png"

        # Separators must never survive into the suffix, whatever the flavour.
        for name in ("a.jpg/../../b", "a.jpg\\..\\..\\b", "../../../etc/passwd"):
            suffix = bot._suffix_for(name, ".png")
            assert "/" not in suffix and "\\" not in suffix and ".." not in suffix

    def test_unknown_and_missing_extensions_fall_back(self):
        assert bot._suffix_for("payload.exe", ".png") == ".png"
        assert bot._suffix_for("noextension", ".png") == ".png"
        assert bot._suffix_for(None, ".jpg") == ".jpg"

    def test_the_suffix_is_always_one_we_allow(self):
        for name in ("a.jpg", "a.exe", "a.png/../x", None, "", "..", "a.tar.gz"):
            assert bot._suffix_for(name, ".png") in bot.ALLOWED_SUFFIXES


class TestDecodedSize:
    """A file-size limit says nothing about memory: compression is the whole point."""

    def test_a_normal_photo_passes(self, tmp_path):
        import cv2
        path = tmp_path / "ok.png"
        cv2.imwrite(str(path), np.zeros((600, 800, 3), dtype=np.uint8))
        bot._check_decoded_size(path)  # must not raise

    def test_a_decompression_bomb_is_refused(self, tmp_path, monkeypatch):
        """A tiny file that unpacks into far too many pixels.

        Uniform PNG data compresses to almost nothing, so the file sails past any
        size check while the decoded array would be enormous. This is checked from
        the header, before a single pixel is allocated.
        """
        import cv2
        monkeypatch.setattr(config, "MAX_INPUT_PIXELS", 1_000_000)

        path = tmp_path / "bomb.png"
        cv2.imwrite(str(path), np.zeros((2000, 2000, 3), dtype=np.uint8))  # 4 MP
        assert path.stat().st_size < 200_000, "should compress to a small file"

        with pytest.raises(bot.RejectedError, match="Слишком большое"):
            bot._check_decoded_size(path)

    def test_a_corrupt_file_is_refused_not_crashed_on(self, tmp_path):
        path = tmp_path / "junk.png"
        path.write_bytes(b"this is not an image")
        with pytest.raises(bot.RejectedError):
            bot._check_decoded_size(path)


class TestResultName:
    """Ten photos must not all come back called "colorized.png"."""

    def test_the_result_is_named_after_the_original(self):
        assert bot._result_name("IMG_2790_bw.JPG") == "IMG_2790_bw_colorized.png"
        assert bot._result_name("дед 1943.jpeg") == "дед 1943_colorized.png"

    def test_a_compressed_photo_has_no_name_to_inherit(self):
        # Telegram sends compressed photos without a filename.
        assert bot._result_name(None) == "photo_colorized.png"
        assert bot._result_name("   ") == "photo_colorized.png"

    def test_a_hostile_name_cannot_escape_through_the_result(self):
        """The output filename is attacker-influenced too, so it gets the same care."""
        name = bot._result_name("../../etc/passwd")
        assert "/" not in name and "\\" not in name and ".." not in name


class TestQueue:
    def setup_method(self):
        bot._pending = 0
        bot._last_job_at.clear()

    def teardown_method(self):
        bot._pending = 0
        bot._last_job_at.clear()

    def test_positions_count_up(self, monkeypatch):
        monkeypatch.setattr(config, "USER_COOLDOWN_SEC", 0)
        assert bot._claim_slot(1) == 1
        assert bot._claim_slot(2) == 2
        assert bot._claim_slot(3) == 3

    def test_a_full_queue_is_refused(self, monkeypatch):
        monkeypatch.setattr(config, "USER_COOLDOWN_SEC", 0)
        monkeypatch.setattr(config, "MAX_QUEUE_SIZE", 2)
        bot._claim_slot(1)
        bot._claim_slot(2)
        with pytest.raises(bot.RejectedError, match="очереди"):
            bot._claim_slot(3)

    def test_one_user_cannot_flood_the_queue(self, monkeypatch):
        """Without a cooldown a single sender fills the queue and everyone else waits."""
        monkeypatch.setattr(config, "USER_COOLDOWN_SEC", 60)
        bot._claim_slot(42)
        with pytest.raises(bot.RejectedError, match="Подождите"):
            bot._claim_slot(42)

        # ...but a different person is unaffected.
        assert bot._claim_slot(43) == 2

    def test_releasing_frees_the_slot(self, monkeypatch):
        monkeypatch.setattr(config, "USER_COOLDOWN_SEC", 0)
        bot._claim_slot(1)
        bot._claim_slot(2)
        bot._release_slot()
        assert bot._pending == 1

    def test_release_cannot_drive_the_count_negative(self):
        bot._release_slot()
        bot._release_slot()
        assert bot._pending == 0
