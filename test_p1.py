"""Offline regression checks for the stage-one startup and persistence contracts."""

import hashlib
import json
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from app import ROOT
from meme_studio.config import load_config
from meme_studio.database import MIGRATIONS, connect, initialize
from meme_studio.errors import AppError
from meme_studio.logging_setup import configure_logging
from meme_studio.media_io import store_asset
from meme_studio.schemas import new_id, utc_now


class LocalFoundationTest(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "runtime" / "p1-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="tests-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "runtime"

    def test_settings_precedence_and_optional_key(self):
        settings = json.loads((ROOT / "settings.json").read_text(encoding="utf-8-sig"))
        settings["gallery_dir"] = "设置图库"
        (self.root / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        no_key = load_config(self.root, environ={})
        self.assertFalse(no_key.secrets.key_present)
        self.assertEqual(no_key.gallery_dir, self.root / "设置图库")
        self.assertEqual(no_key.runtime_dir, self.runtime)

        (self.root / ".env").write_text(
            "DASHSCOPE_API_KEY=synthetic-key-for-test\nMEME_STUDIO_GALLERY_DIR=文件图库\n", encoding="utf-8"
        )
        from_file = load_config(self.root, environ={})
        self.assertTrue(from_file.secrets.key_present)
        self.assertEqual(from_file.gallery_dir, self.root / "文件图库")
        self.assertNotIn("synthetic-key-for-test", repr(from_file))

        from_env = load_config(
            self.root, environ={"DASHSCOPE_API_KEY": "", "MEME_STUDIO_GALLERY_DIR": "环境图库"}
        )
        self.assertFalse(from_env.secrets.key_present)
        self.assertEqual(from_env.gallery_dir, self.root / "环境图库")
        with self.assertRaises(AppError) as failure:
            load_config(self.root, environ={"MEME_STUDIO_RUNTIME_DIR": "../外部目录"})
        self.assertEqual(failure.exception.code, "RUNTIME_PATH")

    def test_database_restarts_preserve_rows_and_enforce_foreign_keys(self):
        path = self.runtime / "app.sqlite3"
        self.assertEqual(initialize(path), len(MIGRATIONS))
        identity = new_id()
        self.assertEqual(UUID(identity).version, 4)
        with connect(path) as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            connection.execute(
                "INSERT INTO galleries(id, name, created_at) VALUES (?, ?, ?)", (identity, "合成图库", utc_now())
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(initialize, [path, path])), [len(MIGRATIONS)] * 2)
        with connect(path) as connection:
            self.assertEqual(connection.execute("SELECT name FROM galleries WHERE id=?", (identity,)).fetchone()[0], "合成图库")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0], len(MIGRATIONS))
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        with self.assertRaises(sqlite3.IntegrityError):
            with connect(path) as connection:
                connection.execute(
                    """INSERT INTO gallery_images(id, gallery_id, tags_json, import_status, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (new_id(), new_id(), '{"schema_version":1,"tags":[]}', "ready", utc_now()),
                )

    def test_failed_asset_validation_does_not_publish_file(self):
        original = b"synthetic file contents"

        def validate(path: Path) -> None:
            self.assertEqual(path.read_bytes(), original)

        first = store_asset(self.runtime, original, ".bin", validate)
        second = store_asset(self.runtime, original, ".bin", validate)
        self.assertEqual(first, second)
        self.assertEqual(first.sha256, hashlib.sha256(original).hexdigest())
        self.assertEqual((self.runtime / first.relative_path).read_bytes(), original)

        def reject(path: Path) -> None:
            raise AppError("REJECTED", "测试校验失败")

        with self.assertRaises(AppError):
            store_asset(self.runtime, b"broken file", ".bin", reject)
        with self.assertRaises(AppError):
            store_asset(self.runtime, original, "../bin", validate)
        self.assertEqual(len(list((self.runtime / "assets").iterdir())), 1)

    def test_log_redacts_sensitive_values(self):
        logger = configure_logging(self.runtime, "private-synthetic-key")
        handler = logger.handlers[0]
        try:
            logger.info(
                "failed key=%s header=%s body=%s url=%s",
                "private-synthetic-key",
                {"Authorization": "Bearer synthetic-token"},
                "data:image/png;base64,QUJDRA==",
                "https://example.invalid/path?signature=synthetic-signature",
            )
            logger.warning("base64=%s %s", "QUJD", "A1b+" * 64)
            handler.flush()
        finally:
            logger.removeHandler(handler)
            handler.close()
        log = (self.runtime / "app.log").read_text(encoding="utf-8")
        self.assertIn("failed", log)
        self.assertIn("[REDACTED]", log)
        for value in (
            "private-synthetic-key", "synthetic-token", "QUJD", "example.invalid", "synthetic-signature", "A1b+",
        ):
            self.assertNotIn(value, log)


if __name__ == "__main__":
    unittest.main()
