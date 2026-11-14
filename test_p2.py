"""Local gallery and cache regressions using synthetic photos."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
from PIL import Image

from app import ROOT
from meme_studio.database import connect, initialize
from meme_studio.errors import AppError
from meme_studio.gallery import GalleryStore
from meme_studio.model_manager import ModelSignature
from meme_studio.retrieval import GalleryRetrieval
from meme_studio.schemas import new_id, utc_now


class SyntheticEncoder:
    def __init__(self, signature="fake-v1"):
        self.preprocess = signature
        self.encoded = 0

    def signature(self):
        return ModelSignature("synthetic-test-encoder", "test-checkpoint", self.preprocess)

    def encode_images(self, paths):
        features = []
        for path in paths:
            with Image.open(path) as image:
                features.append(image.convert("RGB").getpixel((0, 0))[:2])
        self.encoded += len(features)
        return np.asarray(features, dtype=np.float32)

    def encode_text(self, text):
        return np.array([[1.0, 0.0]] if text == "红色" else [[0.0, 1.0]], dtype=np.float32)


class GalleryTest(TestCase):
    def setUp(self):
        scratch = ROOT / "runtime" / "p1-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="p2-tests-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.runtime = Path(temporary.name) / "运行数据"
        self.source = Path(temporary.name) / "中文 源图"
        self.source.mkdir()
        initialize(self.runtime / "app.sqlite3")
        self.gallery = GalleryStore(self.runtime)
        self.album = self.gallery.create("测试图库")

    def _color(self, name, color):
        path = self.source / name
        Image.new("RGB", (80, 60), color).save(path)
        return path

    def test_imports_are_read_only_deduplicated_and_survive_source_moves(self):
        source = self._color("红色 原图.png", "red")
        duplicate = self.source / "副本.png"
        copyfile(source, duplicate)
        (self.source / "坏图.png").write_bytes(b"bad png")
        (self.source / "未支持.heic").write_bytes(b"not yet supported")
        before = source.read_bytes()
        result = self.gallery.import_directory(self.album.id, self.source)
        self.assertEqual((result.added, result.duplicates, len(result.failed)), (1, 1, 2))
        self.assertEqual(source.read_bytes(), before)
        _, (photo,) = self.gallery.catalog(self.album.id)
        second = self.gallery.create("第二图库")
        self.assertEqual(self.gallery.import_files(second.id, [source]).added, 1)
        _, (other,) = self.gallery.catalog(second.id)
        self.assertEqual(other.asset_id, photo.asset_id)
        self.assertNotEqual(other.id, photo.id)
        moved = source.rename(self.source / "已移动.png")
        self.assertTrue(moved.exists())
        self.assertTrue(self.gallery.image_file(photo).is_file())
        self.assertTrue(self.gallery.image_file(photo, thumbnail=True).is_file())

        project = new_id()
        version = new_id()
        with connect(self.gallery.database) as db:
            db.execute(
                """INSERT INTO projects (id, title, source_image_id, created_at, updated_at)
                   VALUES (?, 'history fixture', ?, ?, ?)""", (project, photo.id, utc_now(), utc_now()),
            )
            db.execute(
                """INSERT INTO versions (id, project_id, kind, base_image_asset_id, text_layers_json,
                    constraints_json, operation_json, created_at) VALUES (?, ?, 'root', ?, '{}', '{}', '{}', ?)""",
                (version, project, photo.asset_id, utc_now()),
            )
        self.gallery.remove(self.album.id, photo.id)
        self.assertEqual(self.gallery.catalog(self.album.id)[1], ())
        self.assertEqual(self.gallery.get_photo(self.album.id, photo.id, include_removed=True).asset_id, photo.asset_id)
        with connect(self.gallery.database) as db:
            self.assertEqual(
                db.execute("SELECT base_image_asset_id FROM versions WHERE id = ?", (version,)).fetchone()[0],
                photo.asset_id,
            )
        self.assertTrue(self.gallery.image_file(photo).is_file())

    def test_orientation_transparency_and_limits(self):
        oriented = self.source / "有方向.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (11, 5), "red").save(oriented, exif=exif)
        transparent = self.source / "透明.png"
        Image.new("RGBA", (8, 7), (10, 200, 20, 0)).save(transparent)
        self.assertEqual(self.gallery.import_files(self.album.id, [oriented, transparent]).added, 2)
        _, photos = self.gallery.catalog(self.album.id)
        names = {photo.name: photo for photo in photos}
        with Image.open(self.gallery.image_file(names[oriented.name])) as image:
            self.assertEqual(image.size, (5, 11))
            self.assertFalse(image.getexif())
        with Image.open(self.gallery.image_file(names[transparent.name])) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((0, 0))[3], 0)
        album = self.gallery.create("尺寸限制")
        limited = GalleryStore(self.runtime, max_bytes=32, max_pixels=50)
        self.assertIn("文件超过", limited.import_files(album.id, [oriented]).failed[0][1])
        limited = GalleryStore(self.runtime, max_pixels=50)
        self.assertIn("像素", limited.import_files(album.id, [oriented]).failed[0][1])

    def test_incremental_index_reuse_tags_removal_and_model_key(self):
        red = self._color("红.png", "red")
        green = self._color("绿.png", "green")
        self.gallery.import_files(self.album.id, [red, green])
        encoder = SyntheticEncoder()
        retrieval = GalleryRetrieval(self.gallery, encoder)
        first = retrieval.ensure_index(self.album.id)
        self.assertEqual((first.encoded, first.dimension), (2, 2))
        self.assertEqual(self.gallery.import_files(self.album.id, [red, green]).duplicates, 2)
        other_encoder = SyntheticEncoder()
        reopened = GalleryRetrieval(GalleryStore(self.runtime), other_encoder)
        self.assertEqual(reopened.ensure_index(self.album.id).generation, first.generation)
        self.assertEqual(other_encoder.encoded, 0)
        ranked = reopened.search(self.album.id, "红色", 2)
        self.assertEqual([hit.photo.name for hit in ranked], ["红.png", "绿.png"])

        green_id = ranked[1].photo.id
        self.gallery.set_tags(self.album.id, green_id, ["我家"])
        filtered = reopened.search(self.album.id, "红色", 6, ("我家",))
        self.assertEqual([hit.photo.id for hit in filtered], [green_id])
        self.gallery.remove(self.album.id, green_id)
        with self.assertRaisesRegex(AppError, "尚无可用索引"):
            reopened.search(self.album.id, "红色", 6)
        self.assertEqual(reopened.ensure_index(self.album.id).encoded, 0)
        self.assertEqual([hit.photo.name for hit in reopened.search(self.album.id, "红色", 6)], ["红.png"])
        self.assertEqual(self.gallery.import_files(self.album.id, [green]).added, 1)
        self.assertEqual(reopened.ensure_index(self.album.id).encoded, 0)
        self.assertEqual(self.gallery.get_photo(self.album.id, green_id).tags, ("我家",))

        changed = GalleryRetrieval(self.gallery, SyntheticEncoder("fake-v2"))
        self.assertEqual(changed.ensure_index(self.album.id).encoded, 2)

    def test_index_damage_is_reported_and_publish_failure_keeps_previous(self):
        red = self._color("红色.png", "red")
        self.gallery.import_files(self.album.id, [red])
        retrieval = GalleryRetrieval(self.gallery, SyntheticEncoder())
        first = retrieval.ensure_index(self.album.id)
        matrix = self.runtime / "indexes" / first.generation / "matrix.npy"
        previous = matrix.read_bytes()
        with mock.patch.object(retrieval, "_switch", side_effect=AppError("TEST", "模拟切换失败")):
            with self.assertRaises(AppError):
                retrieval.ensure_index(self.album.id, force=True)
        self.assertEqual(retrieval.status(self.album.id).generation, first.generation)
        self.assertEqual(matrix.read_bytes(), previous)
        matrix.write_bytes(b"damaged index")
        with self.assertRaisesRegex(AppError, "损坏"):
            retrieval.search(self.album.id, "红色", 6)
        repaired = retrieval.ensure_index(self.album.id, force=True)
        self.assertTrue(repaired.repaired)
        self.assertEqual(repaired.encoded, 0)
        self.assertNotEqual(repaired.generation, first.generation)
        self.assertEqual(retrieval.search(self.album.id, "红色", 6)[0].photo.name, red.name)
