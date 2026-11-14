"""Exercise static jobs against HTTP doubles, including ambiguous paid submissions."""

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import httpx
from PIL import Image

from app import ROOT
from meme_studio.dashscope import DashScope, MEDIA_ENDPOINT
from meme_studio.database import connect, initialize
from meme_studio.editor import LocalEditor
from meme_studio.gallery import GalleryStore
from meme_studio.image_files import png_bytes
from meme_studio.schemas import JobStatus as Status
from meme_studio.static_jobs import LEASE_SECONDS, StaticJobs, WAIT_SECONDS
from meme_studio.text_layers import TextLayer, TextRenderer
from meme_studio.versions import ProjectStore


URL = "https://dashscope-5859.oss-cn-wulanchabu-acdr-1.aliyuncs.com/result.png?signature=private-test"
MODEL = "wanx2.1-imageedit"


class Remote:
    def __init__(self):
        self.posts = []
        self.polls = []
        self.downloads = []
        self.post_error = None
        self.query_codes = []
        self.result_url = URL
        self.download_error = None
        self.image_size = None
        self.pause_barrier = None

    def __call__(self, request):
        if request.method == "POST":
            assert request.headers["x-dashscope-async"] == "enable"
            assert request.headers["authorization"] == "Bearer fake-test-key"
            self.posts.append(json.loads(request.content))
            if self.post_error == "lost":
                raise httpx.ReadTimeout("synthetic lost response", request=request)
            if self.post_error:
                return httpx.Response(self.post_error, json={"code": "Arrearage" if self.post_error == 402 else "Rejected"})
            return httpx.Response(200, json={"output": {"task_id": f"task-{len(self.posts)}", "task_status": "PENDING"}})
        if request.url.host == "dashscope.aliyuncs.com":
            self.polls.append(request.url.path)
            if self.pause_barrier is not None:
                enter, leave = self.pause_barrier
                enter.set()
                if not leave.wait(10):
                    raise RuntimeError("test barrier")
                self.pause_barrier = None
                return httpx.Response(200, json={"output": {"task_id": "task-1", "task_status": "RUNNING"}})
            if self.query_codes:
                return httpx.Response(self.query_codes.pop(0))
            return httpx.Response(200, json={
                "output": {"task_id": "task-1", "task_status": "SUCCEEDED", "results": [{"url": self.result_url}]},
                "usage": {"image_count": 1},
            })
        assert "authorization" not in request.headers
        self.downloads.append(str(request.url))
        if self.download_error == "broken":
            return httpx.Response(200, content=b"not png", headers={"content-type": "image/png"})
        if self.download_error:
            return httpx.Response(self.download_error)
        source = _image(self.posts[0]["input"]["base_image_url"])
        result = Image.new("RGB", self.image_size or source.size, "#0044ff")
        return httpx.Response(200, content=png_bytes(result), headers={"content-type": "image/png"})


def _image(uri):
    return Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))


class StaticJobTest(TestCase):
    def setUp(self):
        scratch = ROOT / "runtime" / "p1-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="p4-tests-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.runtime = Path(temporary.name)
        initialize(self.runtime / "app.sqlite3")
        gallery = GalleryStore(self.runtime)
        album = gallery.create("测试图库")
        source = self.runtime / "合成.png"
        Image.new("RGB", (1024, 260), "#eab369").save(source)
        gallery.import_files(album.id, [source])
        photo = gallery.catalog(album.id)[1][0]
        self.projects = ProjectStore(gallery)
        self.editor = LocalEditor(self.projects, TextRenderer())
        self.root = self.projects.create(album.id, photo.id)
        self.boxed = self.editor.save_box(self.root, (120, 50), (450, 170))
        self.source = self.editor.save_text(self.boxed, TextLayer("本地配字"))
        self.remote = Remote()
        self.client = httpx.Client(transport=httpx.MockTransport(self.remote))
        self.addCleanup(self.client.close)
        self.now = 1000.0
        self.jobs = self.reopen()

    def reopen(self):
        return StaticJobs(self.projects, DashScope(MEDIA_ENDPOINT, "fake-test-key", self.client), MODEL,
                          clock=lambda: self.now)

    def submit(self, local=True, protect=False, prompt="把选区内改成蓝色"):
        return self.jobs.submit(self.source.project_id, self.source.id, prompt, local, protect)

    def advance(self, seconds=20):
        self.now += seconds
        return self.jobs.run_once()

    def state(self):
        return self.jobs.list(self.source.project_id)[0]

    def test_mask_preserves_outside_and_next_request_uses_clean_base(self):
        job_id = self.submit(protect=True)
        self.assertEqual(self.submit(protect=True), job_id)
        self.assertTrue(self.jobs.run_once())
        self.assertEqual(self.state().remote_task_id, "task-1")
        body = self.remote.posts[0]
        self.assertEqual(body["parameters"]["n"], 1)
        self.assertEqual(body["input"]["function"], "description_edit_with_mask")
        self.assertNotIn("本地配字", body["input"]["prompt"])
        with _image(body["input"]["base_image_url"]) as base, _image(body["input"]["mask_image_url"]) as mask:
            self.assertEqual(base.size, mask.size)
            self.assertEqual(base.size, (1024, 512))
            self.assertEqual(mask.getbbox(), (120, 176, 451, 297))
            self.assertEqual(base.getpixel((200, 200)), (234, 179, 105))

        self.assertTrue(self.advance())
        self.assertEqual(self.state().local_status, Status.REMOTE_SUCCEEDED)
        self.assertIsNone(self.state().result_version_id)
        self.assertTrue(self.jobs.run_once())
        saved = self.projects.current(self.source.project_id)
        self.assertEqual(saved.parent_id, self.source.id)
        self.assertIsNone(saved.region_id)
        self.assertEqual(self.editor.view(saved).text.text, "本地配字")
        asset = self.projects.asset(saved.base_image_asset_id)
        pixels = self.projects.load_image(asset)
        self.assertEqual(pixels.getpixel((0, 0)), (234, 179, 105))
        self.assertEqual(pixels.getpixel((200, 100)), (0, 68, 255))
        self.assertEqual(pixels.size, (1024, 260))
        self.assertFalse(self.advance())
        with connect(self.projects.database) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM versions WHERE job_id = ?", (job_id,)).fetchone()[0], 1)
            use = db.execute("SELECT image_count, estimated_cost FROM usage_events").fetchall()
            self.assertEqual([tuple(row) for row in use], [(1, None)])
            self.assertIsNone(db.execute("SELECT result_url FROM jobs WHERE id=?", (job_id,)).fetchone()[0])

        self.jobs.submit(saved.project_id, saved.id, "调整色调", False)
        self.assertTrue(self.jobs.run_once())
        whole = self.remote.posts[1]["input"]
        self.assertEqual(whole["function"], "description_edit")
        self.assertNotIn("mask_image_url", whole)
        with _image(whole["base_image_url"]) as next_base:
            self.assertEqual(next_base.getpixel((200, 200)), (0, 68, 255))
        self.assertNotIn("本地配字", whole["prompt"])

    def test_concurrent_clicks_and_ambiguous_post_do_not_resubmit(self):
        self.remote.post_error = "lost"
        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(lambda _: self.submit(), range(2)))
        self.assertEqual(ids[0], ids[1])
        self.assertTrue(self.jobs.run_once())
        self.assertEqual(self.state().local_status, Status.SUBMISSION_UNKNOWN)
        self.assertEqual(len(self.remote.posts), 1)
        self.jobs = self.reopen()
        self.assertEqual(self.submit(), ids[0])
        self.assertFalse(self.advance(1000))
        self.assertEqual(len(self.remote.posts), 1)
        self.assertEqual(self.projects.current(self.source.project_id).id, self.source.id)

    def test_interrupted_claim_cannot_be_submitted_after_restart(self):
        self.submit()
        claimed = self.jobs._claim()
        self.assertEqual(claimed["local_status"], Status.SUBMITTING)
        self.jobs = self.reopen()
        self.assertFalse(self.advance(LEASE_SECONDS + 1))
        self.assertEqual(self.state().local_status, Status.SUBMISSION_UNKNOWN)
        self.assertEqual(len(self.remote.posts), 0)

    def test_definite_rejections_and_ambiguous_5xx(self):
        for code, wanted, state in (
            (401, "AUTH", Status.FAILED), (403, "AUTH", Status.FAILED),
            (402, "BALANCE", Status.FAILED), (429, "RATE_LIMIT", Status.FAILED),
            (503, "SUBMIT_UNKNOWN", Status.SUBMISSION_UNKNOWN),
        ):
            self.remote.post_error = code
            self.submit(prompt=f"测试 {code}")
            self.assertTrue(self.jobs.run_once())
            self.assertEqual((self.state().local_status, self.state().error_code), (state, wanted))
        self.assertEqual(len(self.remote.posts), 5)

    def test_query_backoff_download_only_retry_and_version_switch(self):
        self.remote.query_codes = [429, 503]
        self.submit()
        self.assertTrue(self.jobs.run_once())
        self.assertTrue(self.advance())
        self.assertEqual(self.state().error_code, "RATE_LIMIT")
        self.assertFalse(self.jobs.run_once())
        self.jobs = self.reopen()
        self.assertTrue(self.advance())
        self.assertEqual(self.state().remote_task_id, "task-1")
        self.assertTrue(self.advance())
        self.remote.download_error = "broken"
        self.assertTrue(self.jobs.run_once())
        self.assertEqual(self.state().error_code, "RESULT_DECODE")

        # Going away and returning to the same source still counts as a user checkout.
        self.projects.checkout(self.source.project_id, self.source.id, self.root.id)
        self.projects.checkout(self.source.project_id, self.root.id, self.source.id)
        self.remote.download_error = None
        self.assertTrue(self.advance())
        result = self.projects.version(self.source.project_id, self.state().result_version_id)
        self.assertEqual(result.parent_id, self.source.id)
        self.assertEqual(self.projects.current(self.source.project_id).id, self.source.id)
        self.assertEqual(len(self.remote.downloads), 2)
        self.assertEqual(len(self.remote.polls), 3)
        self.assertEqual(len(self.remote.posts), 1)

    def test_pause_during_query_survives_response_and_timeout(self):
        job_id = self.submit()
        self.assertTrue(self.jobs.run_once())
        enter, leave = threading.Event(), threading.Event()
        self.remote.pause_barrier = enter, leave
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.advance)
            self.assertTrue(enter.wait(10))
            try:
                self.jobs.set_paused(self.source.project_id, job_id, True)
            finally:
                leave.set()
            self.assertTrue(future.result(timeout=10))
        self.assertTrue(self.state().paused)
        self.assertFalse(self.advance(WAIT_SECONDS + 1))
        self.jobs.set_paused(self.source.project_id, job_id, False)
        self.assertFalse(self.advance(WAIT_SECONDS + 1))
        self.assertEqual(self.state().error_code, "WAIT_TIMEOUT")
        self.jobs.set_paused(self.source.project_id, job_id, False)
        self.assertTrue(self.jobs.run_once())
        self.assertTrue(self.jobs.run_once())
        self.assertEqual(self.state().local_status, Status.SUCCEEDED)
        self.assertEqual(len(self.remote.posts), 1)

    def test_expired_link_requeries_same_id_and_rejects_unsafe_redirects(self):
        self.submit()
        self.assertTrue(self.jobs.run_once())
        self.assertTrue(self.advance())
        self.remote.download_error = 403
        self.assertTrue(self.jobs.run_once())
        self.remote.result_url = "https://127.0.0.1/private"
        self.assertTrue(self.jobs.run_once())
        self.assertTrue(self.jobs.run_once())
        self.assertEqual(self.state().error_code, "RESULT_URL")
        self.assertEqual(len(self.remote.downloads), 1)
        self.assertEqual(len(self.remote.posts), 1)
        self.assertEqual(self.remote.polls, ["/api/v1/tasks/task-1"] * 2)

    def test_incompatible_output_size_keeps_original_when_protected(self):
        self.remote.image_size = (768, 768)
        self.submit(protect=True)
        self.assertTrue(self.jobs.run_once())
        self.assertTrue(self.advance())
        self.assertTrue(self.jobs.run_once())
        self.assertEqual(self.state().error_code, "RESULT_GEOMETRY")
        self.assertEqual(self.state().remote_status, "SUCCEEDED")
        self.assertEqual(self.state().image_count, 1)
        self.assertEqual(self.projects.current(self.source.project_id).id, self.source.id)
