
"""Video requests, immutable anchors and local media exports."""

import base64
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, SkipTest
from unittest.mock import patch

import httpx
from PIL import Image

from app import ROOT
from meme_studio.config import VideoDefaults
from meme_studio.constraints import read_constraints, save_constraint
from meme_studio.dashscope import DashScope, MEDIA_ENDPOINT, VIDEO_PATH
from meme_studio.database import connect, initialize
from meme_studio.editor import LocalEditor
from meme_studio.errors import AppError
from meme_studio.gallery import GalleryStore
from meme_studio.schemas import JobStatus, VersionKind
from meme_studio.text_layers import TextLayer, TextRenderer
from meme_studio.video_exports import GifOptions, VideoExports
from meme_studio.video_jobs import VIDEO_MODEL, VideoJobs, prepare_frame
from meme_studio.video_media import Ffmpeg, video_path
from meme_studio.versions import ProjectStore


RESULT_URL = "https://dashscope-test.oss-accelerate.aliyuncs.com/fake.mp4"
SCRATCH = ROOT / "runtime" / "p1-tmp"


class VideoInputTest(TestCase):
    def test_transparent_panorama_becomes_bounded_rgb_frame_without_stretch(self):
        image = Image.new("RGBA", (2200, 150), (0, 0, 0, 0))
        image.paste((220, 90, 20, 255), (50, 50, 2100, 100))
        data, transform = prepare_frame(image)
        with Image.open(BytesIO(data)) as frame:
            self.assertEqual((frame.mode, frame.size), ("RGB", (1440, 360)))
            self.assertEqual(frame.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(frame.getpixel((720, 180)), (220, 90, 20))
        self.assertEqual(transform["resized"], (1440, 98))
        self.assertEqual(transform["offset"], (0, 131))

    def test_gif_reduction_steps_are_bounded(self):
        self.assertEqual(GifOptions().sizes(), ((512, 12), (420, 12), (358, 10), (282, 8)))
        self.assertEqual(GifOptions(128, 6).sizes(), ((128, 6),))
        with self.assertRaises(AppError):
            GifOptions(fps=100)


class RemoteVideo:
    def __init__(self, data):
        self.data = data
        self.posts = []
        self.downloads = 0
        self.download_code = None
        self.next_post_timeout = False

    def __call__(self, request):
        if request.method == "POST":
            assert request.url.path == "/api/v1" + VIDEO_PATH
            assert request.headers["x-dashscope-async"] == "enable"
            self.posts.append(json.loads(request.content))
            if self.next_post_timeout:
                raise httpx.ReadTimeout("synthetic ambiguity", request=request)
            return httpx.Response(200, json={"output": {
                "task_id": f"video-{len(self.posts)}", "task_status": "PENDING",
            }})
        if request.url.host == "dashscope.aliyuncs.com":
            return httpx.Response(200, json={
                "output": {"task_id": request.url.path.rsplit("/", 1)[1], "task_status": "SUCCEEDED",
                           "video_url": RESULT_URL},
                "usage": {"duration": 3, "video_count": 1, "audio": False, "resolution": "720P"},
            })
        assert str(request.url) == RESULT_URL and "authorization" not in request.headers
        self.downloads += 1
        if self.download_code:
            return httpx.Response(self.download_code)
        return httpx.Response(200, content=self.data, headers={"content-type": "video/mp4"})


class VideoLifecycleTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = Ffmpeg(ROOT)
        if not cls.media.ready:
            raise SkipTest("local ffmpeg, ffprobe and libx264 required")
        SCRATCH.mkdir(parents=True, exist_ok=True)
        scratch = TemporaryDirectory(prefix="p6-video-fixture-", dir=SCRATCH)
        cls.addClassCleanup(scratch.cleanup)
        source = Path(scratch.name) / "fixture.mp4"
        cls.media.encode([
            "-f", "lavfi", "-i", "testsrc2=size=352x240:rate=24:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(source),
        ])
        cls.fixture = source.read_bytes()

    def setUp(self):
        temporary = TemporaryDirectory(prefix="p6-tests-", dir=SCRATCH)
        self.addCleanup(temporary.cleanup)
        runtime = Path(temporary.name)
        initialize(runtime / "app.sqlite3")
        gallery = GalleryStore(runtime)
        album = gallery.create("测试")
        file = runtime / "合成.png"
        Image.new("RGBA", (352, 240), (220, 90, 20, 100)).save(file)
        gallery.import_files(album.id, [file])
        self.projects = ProjectStore(gallery)
        self.renderer = TextRenderer()
        self.editor = LocalEditor(self.projects, self.renderer)
        self.exporter = VideoExports(self.projects, self.renderer, self.media)
        self.root = self.projects.create(album.id, gallery.catalog(album.id)[1][0].id)
        self.remote = RemoteVideo(self.fixture)
        self.client = httpx.Client(transport=httpx.MockTransport(self.remote))
        self.addCleanup(self.client.close)
        self.clock = 1000.
        self.jobs = self.reopen()

    def reopen(self):
        return VideoJobs(
            self.projects, DashScope(MEDIA_ENDPOINT, "synthetic-key", self.client), VIDEO_MODEL, self.media,
            VideoDefaults("720P", 3, False), clock=lambda: self.clock,
        )

    def submit(self, source, instruction):
        return self.jobs.submit(source.project_id, source.id, instruction)

    def finish(self):
        self.assertTrue(self.jobs.run_once())
        self.clock += 20
        self.assertTrue(self.jobs.run_once())
        self.assertTrue(self.jobs.run_once())
        return self.projects.current(self.root.project_id)

    def test_motion_reuses_original_anchor_and_video_caption_changes_no_source_asset(self):
        caption = self.editor.save_text(self.root, TextLayer("本机文字"))
        box = self.editor.save_box(caption, (5, 7), (120, 130))
        content, _ = save_constraint(box, None, category="motion", target="动作", text="缓慢眨眼",
                                     scope="video", duration="once")
        source = self.projects.commit(box, VersionKind.CONSTRAINT_EDIT, constraints_json=content)
        job_id = self.submit(source, "轻轻点头")
        self.assertEqual(job_id, self.submit(source, "轻轻点头"))
        first = self.finish()
        self.assertEqual(first.anchor_image_version_id, source.id)
        self.assertEqual(first.parent_id, source.id)
        self.assertEqual(first.kind, VersionKind.VIDEO)
        self.assertIsNone(first.region_id)
        self.assertEqual(read_constraints(first), ())
        self.assertEqual(len(self.remote.posts), 1)
        request = self.remote.posts[0]
        self.assertEqual(set(request["input"]), {"img_url", "prompt"})
        self.assertIn("缓慢眨眼", request["input"]["prompt"])
        self.assertNotIn("本机文字", request["input"]["prompt"])
        self.assertEqual({k: request["parameters"][k] for k in ("resolution", "duration", "audio")},
                         {"resolution": "720P", "duration": 3, "audio": False})
        with Image.open(BytesIO(base64.b64decode(request["input"]["img_url"].split(",", 1)[1]))) as frame:
            self.assertEqual(frame.mode, "RGB")
        edit = self.editor.save_text(first, TextLayer("改一行文字"))
        self.assertEqual(edit.base_video_asset_id, first.base_video_asset_id)
        self.assertEqual(edit.anchor_image_version_id, source.id)
        self.assertEqual(len(self.remote.posts), 1)
        self.submit(edit, "转头")
        second = self.finish()
        self.assertEqual(second.parent_id, edit.id)
        self.assertEqual(second.anchor_image_version_id, source.id)
        self.assertEqual(self.remote.posts[1]["input"]["img_url"], request["input"]["img_url"])
        self.assertNotIn("缓慢眨眼", self.remote.posts[1]["input"]["prompt"])
        self.assertEqual(self.projects.undo(second.project_id, second.id), edit)
        with connect(self.projects.database) as db:
            uses = db.execute("SELECT video_seconds, resolution, audio FROM usage_events ORDER BY rowid").fetchall()
            self.assertEqual([tuple(row) for row in uses], [(3, "720P", 0), (3, "720P", 0)])

    def test_reopen_retries_download_without_post_and_invalid_media_keeps_source(self):
        job_id = self.submit(self.root, "移动")
        self.assertTrue(self.jobs.run_once())
        self.jobs = self.reopen()
        self.clock += 20
        self.assertTrue(self.jobs.run_once())
        self.remote.download_code = 503
        self.assertTrue(self.jobs.run_once())
        self.assertEqual(self.jobs.list(self.root.project_id)[0].local_status, JobStatus.REMOTE_SUCCEEDED)
        self.clock += 20
        self.remote.download_code = None
        self.remote.data = b"broken mp4"
        self.assertTrue(self.jobs.run_once())
        job = self.jobs.list(self.root.project_id)[0]
        self.assertEqual((job.id, job.local_status, job.error_code), (job_id, JobStatus.FAILED, "VIDEO_FORMAT"))
        self.assertEqual(self.projects.current(self.root.project_id), self.root)
        self.assertEqual((len(self.remote.posts), self.remote.downloads), (1, 2))

    def test_ambiguous_submit_is_not_retried(self):
        self.remote.next_post_timeout = True
        first = self.submit(self.root, "点头")
        self.assertTrue(self.jobs.run_once())
        self.jobs = self.reopen()
        self.clock += 200
        self.assertEqual(self.submit(self.root, "点头"), first)
        self.assertFalse(self.jobs.run_once())
        self.assertEqual(self.jobs.list(self.root.project_id)[0].local_status, JobStatus.SUBMISSION_UNKNOWN)
        self.assertEqual(len(self.remote.posts), 1)

    def test_export_preserves_source_and_cache_and_gif_repeats(self):
        self.submit(self.root, "固定镜头")
        first = self.finish()
        snapshot = self.editor.save_text(first, TextLayer("两秒表情"))
        source_asset = self.projects.asset(snapshot.base_video_asset_id)
        path = video_path(self.projects, source_asset)
        mp4 = self.exporter.export(snapshot)
        gif = self.exporter.export(snapshot, GifOptions(224, 12, 5, .75))
        self.assertEqual((mp4.info.codec, mp4.info.width, mp4.info.height, mp4.info.audio), ("h264", 352, 240, False))
        self.assertEqual((gif.info.codec, gif.info.width, gif.info.frames), ("gif", 224, 9))
        with Image.open(gif.path) as loop:
            self.assertTrue(loop.is_animated)
            self.assertEqual(loop.info.get("loop"), 0)
            self.assertEqual(loop.n_frames, 9)
        with patch.object(self.media, "encode", side_effect=AssertionError("cache must bypass encode")):
            self.assertEqual(self.exporter.export(snapshot).path, mp4.path)
            self.assertTrue(self.exporter.export(snapshot, GifOptions(224, 12, 5, .75)).cached)
        with patch.object(self.media, "encode", side_effect=AppError("VIDEO_ENCODE", "synthetic encoder failure")):
            with self.assertRaises(AppError):
                self.exporter.export(snapshot, GifOptions(225, 12, 5, .75))
        self.assertEqual(path.read_bytes(), self.fixture)
        self.assertEqual(len(self.remote.posts), 1)
