"""Offline regressions for immutable editing, masks, typography and PNG export."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from PIL import Image, ImageDraw

from app import ROOT
from meme_studio.database import connect, initialize
from meme_studio.editor import LocalEditor
from meme_studio.errors import AppError
from meme_studio.gallery import GalleryStore
from meme_studio.media_io import file_sha256
from meme_studio.regions import ViewTransform
from meme_studio.schemas import AssetKind, VersionKind
from meme_studio.text_layers import TextLayer, TextRenderer
from meme_studio.versions import ProjectStore


class EditingTest(TestCase):
    def setUp(self):
        scratch = ROOT / "runtime" / "p1-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="p3-tests-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.runtime = directory / "运行数据"
        self.source = directory / "测试 坐标.png"
        image = Image.new("RGBA", (1600, 900), (245, 238, 227, 255))
        ImageDraw.Draw(image).rectangle((0, 0, 140, 140), fill=(110, 160, 255, 100))
        image.save(self.source)
        initialize(self.runtime / "app.sqlite3")
        self.gallery = GalleryStore(self.runtime)
        self.album = self.gallery.create("合成图库")
        self.gallery.import_files(self.album.id, [self.source])
        self.photo = self.gallery.catalog(self.album.id)[1][0]
        self.projects = ProjectStore(self.gallery)
        self.editor = LocalEditor(self.projects, TextRenderer())
        self.root = self.projects.create(self.album.id, self.photo.id)

    def test_caption_versions_branch_and_never_request_network(self):
        source_hash = file_sha256(self.source)
        base = self.projects.asset(self.root.base_image_asset_id)
        clean_hash = file_sha256(self.projects.image_path(base))
        self.assertEqual(self.projects.undo(self.root.project_id, self.root.id), self.root)
        with mock.patch("socket.socket.connect", side_effect=AssertionError("unexpected network request")) as request:
            self.editor.preview_text(self.root, TextLayer("只是预览"))
            self.assertEqual(len(self.projects.history(self.root.project_id)), 1)
            first = self.editor.save_text(self.root, TextLayer("第一行\n第二行"))
            second = self.editor.save_text(first, TextLayer("另一个版本"))
            self.assertEqual(self.editor.save_text(second, TextLayer("另一个版本")).id, second.id)
            self.assertEqual(self.projects.undo(second.project_id, second.id).id, first.id)
            fork = self.editor.save_text(first, TextLayer("分支文字"))
            self.assertEqual(fork.parent_id, first.id)
            with self.assertRaisesRegex(AppError, "当前版本"):
                self.editor.save_text(second, TextLayer("过期写入"))
            self.assertEqual(self.projects.version(fork.project_id, second.id).id, second.id)
            reopened = LocalEditor(ProjectStore(GalleryStore(self.runtime)), TextRenderer())
            current = reopened.projects.current(fork.project_id)
            self.assertEqual(reopened.view(current).text.text, "分支文字")
            self.assertEqual(current.base_image_asset_id, base.id)
            request.assert_not_called()
        self.assertEqual(file_sha256(self.source), source_hash)
        self.assertEqual(file_sha256(self.projects.image_path(base)), clean_hash)
        with connect(self.projects.database) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM jobs").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM usage_events").fetchone()[0], 0)
        other = self.projects.create(self.album.id, self.photo.id)
        with self.assertRaisesRegex(AppError, "不属于当前项目"):
            self.projects.checkout(other.project_id, other.id, first.id)

    def test_box_paint_and_base_identity(self):
        transform = ViewTransform.for_base((1600, 900))
        self.assertEqual(transform.view_size, (1024, 576))
        with self.assertRaisesRegex(AppError, "留白"):
            self.editor.validate_point(self.root, (-1, 3))
        first, second = (290, 180), (110, 70)
        boxed = self.editor.save_box(self.root, first, second)
        region = self.projects.region(boxed)
        self.assertEqual(region.source_version_id, self.root.id)
        self.assertEqual(region.base_asset_hash, self.projects.asset(self.root.base_image_asset_id).sha256)
        mask = self.projects.load_image(self.projects.asset(region.mask_asset_id))
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getbbox(), (171, 109, 455, 283))
        current = self.editor.view(boxed)
        layer = current.brush_value["layers"][0]
        ImageDraw.Draw(layer).rectangle((140, 85, 185, 115), fill=(255, 62, 46, 0))
        painted = self.editor.save_paint(boxed, current.brush_value)
        painted_region = self.projects.region(painted)
        painted_mask = self.projects.load_image(self.projects.asset(painted_region.mask_asset_id))
        self.assertLess(painted_mask.histogram()[255], mask.histogram()[255])
        self.assertGreater(painted_mask.histogram()[255], 0)
        with self.assertRaisesRegex(AppError, "局部选区为空"):
            self.editor.save_paint(painted, {"background": current.brush_value["background"], "layers": []})
        caption = self.editor.save_text(painted, TextLayer("只改字"))
        self.assertEqual(caption.region_id, painted.region_id)
        new_base = self.projects.put_image(Image.new("RGB", (1600, 900), "purple"), AssetKind.IMAGE)
        child = self.projects.commit(caption, VersionKind.IMAGE_EDIT, base_image_asset_id=new_base.id)
        self.assertIsNone(child.region_id)
        restored = self.projects.checkout(child.project_id, child.id, caption.id)
        self.assertEqual(self.projects.region(restored).id, painted_region.id)
        whole = self.editor.clear_region(restored)
        self.assertIsNone(whole.region_id)
        self.assertEqual(self.editor.view(whole).text.text, "只改字")

    def test_text_safe_margin_missing_glyph_and_png_cache(self):
        renderer = TextRenderer()
        style = TextLayer("这是需要自动换行的一句长文案" * 10, size_ratio=0.35, x=1.0, y=0.0)
        overlay = renderer.render_overlay((500, 360), (style,))
        left, top, right, bottom = overlay.getchannel("A").getbbox()
        self.assertGreaterEqual(min(left, top), 18)
        self.assertLessEqual(right, 500 - 18)
        self.assertLessEqual(bottom, 360 - 18)
        with self.assertRaisesRegex(AppError, "缺少字符"):
            renderer.resolve_font(TextLayer(chr(0x10FFFF)))
        with self.assertRaisesRegex(AppError, "找不到可用字体"):
            TextRenderer(self.runtime / "missing.ttf").resolve_font(TextLayer("中文"))

        caption = self.editor.save_text(self.root, TextLayer("导出测试"))
        with mock.patch.object(self.projects, "put_image", side_effect=AppError("TEST", "模拟写入失败")):
            with self.assertRaises(AppError):
                self.editor.export_png(caption)
        self.assertEqual(self.projects.current(caption.project_id).id, caption.id)
        first = self.editor.export_png(caption)
        second = self.editor.export_png(caption)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(first.path, second.path)
        with Image.open(second.path) as png:
            self.assertEqual(png.size, (1600, 900))
            self.assertEqual(png.format, "PNG")
            self.assertEqual(png.getpixel((0, 0))[3], 100)
