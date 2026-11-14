
"""Build a portable course source archive from an explicit file allowlist."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "app.py", "start.cmd", ".env.example", ".gitignore", "README.md", "pyproject.toml",
    "requirements.lock", "requirements-models.lock",
)
SOURCE_FILES = {"src": {".py", ".sql"}, "tests": {".py"}, "docs": {".md"}, "scripts": {".py"}}
MEDIA_FILES = {
    "gallery-source.png": "demo/gallery/source.png",
    "p4-mask.png": "demo/media/p4-mask.png",
    "p4-static.png": "demo/media/p4-static.png",
    "static-caption.png": "demo/media/static-caption.png",
    "p7-live-caption.mp4": "demo/media/p7-live-caption.mp4",
    "p7-live-caption.gif": "demo/media/p7-live-caption.gif",
}
PERSONAL_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\Users\\[^\\\n]+(?:\\[^" + r'"\x60\n|]+' + r")?")
TOKEN = re.compile(r"(?i)(?:\bsk-[a-z0-9]{20,}|\b(?:x-oss-signature|ossaccesskeyid|x-amz-signature)=)")
BUNDLE = "meme-studio"


def source_bytes(path: Path, *, portable_docs: bool = False) -> bytes:
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError(f"Unsupported source entry: {path.name}")
    contents = path.read_text(encoding="utf-8-sig")
    if portable_docs:
        contents = PERSONAL_WINDOWS_PATH.sub("本机路径见实际环境", contents)
    if TOKEN.search(contents) or re.search(r"(?m)^DASHSCOPE_API_KEY=\s*\S", contents):
        raise ValueError(f"Private-looking data in {path.name}")
    return contents.encode("utf-8")


def files() -> dict[str, bytes]:
    entries = {name: source_bytes(ROOT / name, portable_docs=name.endswith(".md")) for name in ROOT_FILES}
    for folder, suffixes in SOURCE_FILES.items():
        for path in sorted((ROOT / folder).rglob("*")):
            relative = path.relative_to(ROOT)
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in relative.parts:
                entries[relative.as_posix()] = source_bytes(path, portable_docs=path.suffix == ".md")
    settings = json.loads((ROOT / "settings.json").read_text(encoding="utf-8-sig"))
    settings["gallery_dir"] = "demo/gallery"
    settings["runtime_dir"] = "runtime"
    settings.pop("editor_font", None)
    settings["api"].pop("verified_with_account", None)
    settings["api"].pop("verification_scope", None)
    entries["settings.json"] = (json.dumps(settings, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    source = ROOT / "runtime" / "p7-share-media"
    for local, target in MEDIA_FILES.items():
        path = source / local
        if not path.exists():
            if local.startswith("p7-live"):
                continue
            raise ValueError(f"Synthetic demonstration sample missing: {local}")
        if path.resolve() != path or path.stat().st_size > 16 * 1024**2:
            raise ValueError(f"Invalid synthetic demonstration sample: {local}")
        entries[target] = path.read_bytes()
    return entries


def build(destination: Path) -> int:
    entries = files()
    manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(entries.items()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_suffix(".tmp")
    with ZipFile(scratch, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(f"{BUNDLE}/{name}", data)
        archive.writestr(f"{BUNDLE}/MANIFEST.sha256", manifest)
    with ZipFile(scratch) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed")
        for name, data in entries.items():
            if archive.read(f"{BUNDLE}/{name}") != data:
                raise ValueError(f"Archive entry differs from manifest: {name}")
    scratch.replace(destination)
    return len(entries) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Verified entries: {build(args.destination.resolve())}")
