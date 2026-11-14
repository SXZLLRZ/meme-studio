"""Open SQLite connections and apply versioned migrations transactionally."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from .errors import AppError
from .schemas import utc_now


MIGRATIONS = ("0001_initial.sql", "0002_gallery.sql", "0003_editing.sql", "0004_static_jobs.sql")


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(path, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection


def _statements(script: str) -> Iterator[str]:
    lines: list[str] = []
    for line in script.splitlines(keepends=True):
        lines.append(line)
        statement = "".join(lines)
        if sqlite3.complete_statement(statement):
            yield statement
            lines.clear()
    if "".join(lines).strip():
        raise AppError("MIGRATION_SQL", "数据库迁移脚本不完整。")


def initialize(path: Path) -> int:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )"""
            )
            applied = [row[0] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
            if applied != list(range(1, len(applied) + 1)) or len(applied) > len(MIGRATIONS):
                raise AppError("MIGRATION_VERSION", "数据库版本与当前程序不兼容，请检查程序版本。")
            for version, name in enumerate(MIGRATIONS, start=1):
                if version in applied:
                    continue
                script = (Path(__file__).with_name("migrations") / name).read_text(encoding="utf-8")
                for statement in _statements(script):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )
            return len(MIGRATIONS)
    except (OSError, sqlite3.Error) as exc:
        raise AppError("DATABASE", "数据库初始化失败，请检查项目目录权限或数据库状态。") from exc
