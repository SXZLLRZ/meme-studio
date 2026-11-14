
"""P5 regression coverage. No tests submit a real cloud request."""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import httpx
import numpy as np
from PIL import Image, ImageDraw

from app import ROOT
from meme_studio.constraints import delete_constraint, read_constraints, save_constraint
from meme_studio.database import initialize
from meme_studio.dashscope import DashScope, MEDIA_ENDPOINT
from meme_studio.editor import LocalEditor
from meme_studio.errors import AppError
from meme_studio.gallery import GalleryStore
from meme_studio.image_files import png_bytes
from meme_studio.image_requests import compile_image_prompt
from meme_studio.intent import IntentContext, IntentParser, local_plan, validate_plan
from meme_studio.regions import ViewTransform
from meme_studio.schemas import IntentAction as Action, IntentPlan, JobStatus, VersionKind
from meme_studio.segmentation import Candidate, SegmentationDraft, add_hint, binding, save_candidate
from meme_studio.static_jobs import StaticJobs
from meme_studio.text_layers import TextLayer, TextRenderer
from meme_studio.versions import ProjectStore
from test_p4 import MODEL, Remote


ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
COMBINED = "找一张橘猫，给它加墨镜"
GALLERY = IntentContext("gallery", False)


def payload(**fields):
    data = asdict(IntentPlan(Action.SEARCH_THEN_EDIT, search_query="橘猫", image_instruction="给它加墨镜"))
    data.pop("schema_version")
    return {**data, **fields}


class IntentTest(TestCase):
    def test_local_routes_distinguish_existing_and_new_sunglasses(self):
        existing = local_plan("找戴墨镜的橘猫", GALLERY)
        editing = local_plan(COMBINED, GALLERY)
        self.assertEqual((existing.action, existing.search_query, existing.image_instruction),
                         (Action.SEARCH, "戴墨镜的橘猫", ""))
        self.assertEqual((editing.action, editing.search_query, editing.image_instruction),
                         (Action.SEARCH_THEN_EDIT, "橘猫", "给它加墨镜"))
        self.assertEqual(local_plan("找一张橘猫，做成戴墨镜的得意表情包", GALLERY).search_query, "橘猫")
        self.assertEqual(local_plan("找一张橘猫再给它加墨镜", GALLERY).search_query, "橘猫")
        self.assertEqual(local_plan("不要再保留条纹", GALLERY).constraint_remove, ("条纹",))
        self.assertEqual(local_plan("眼泪再大一点", IntentContext("static", True)).action, Action.EDIT_IMAGE)

    def test_model_json_is_strict(self):
        expected = validate_plan(json.dumps(payload(), ensure_ascii=False))
        self.assertEqual(expected.action, Action.SEARCH_THEN_EDIT)
        missing = payload()
        del missing["caption"]
        for data in (missing, payload(action="run"), payload(region_required=1), payload(path="secret"),
                     payload(search_query=""), payload(constraint_add=[{"text": "a"}]),
                     payload(constraint_remove=[""]), payload(needs_clarification=True)):
            with self.subTest(data=data), self.assertRaises((ValueError, AppError)):
                validate_plan(json.dumps(data))
        with self.assertRaises(ValueError):
            validate_plan('{"action":"search","action":"undo"}')

    def test_cloud_is_explicit_minimal_and_validates_response(self):
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(request.url.path, "/compatible-mode/v1/chat/completions")
            message = json.loads(request.content)
            self.assertEqual(message["response_format"], {"type": "json_object"})
            self.assertFalse(message["enable_thinking"])
            self.assertEqual(json.loads(message["messages"][1]["content"]), {
                "page": "gallery", "selected": False, "version_kind": "", "constraints": [], "instruction": COMBINED,
            })
            return httpx.Response(200, json={"choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(payload())}},
            ]})

        parser = IntentParser(ENDPOINT, "qwen-plus", "fake", transport=httpx.MockTransport(respond))
        self.assertEqual(parser.draft(COMBINED, GALLERY).plan.action, Action.SEARCH_THEN_EDIT)
        for text, action in (("导出 PNG", Action.EXPORT), ("撤销", Action.UNDO), ("配字改成我太难了", Action.EDIT_TEXT)):
            self.assertEqual(parser.draft(text, GALLERY, cloud=True).plan.action, action)
        self.assertEqual(parser.draft("字" * 601, GALLERY, cloud=True).plan.action, Action.CLARIFY)
        self.assertEqual(calls, [])
        result = parser.draft(COMBINED, GALLERY, cloud=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.plan.search_query, "橘猫")
        self.assertIn("云端拆解完成", result.note)

    def test_cloud_failures_fall_back_once(self):
        invalid = (
            httpx.Response(200, json={"choices": ["invalid"]}),
            httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]}),
            httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}),
            httpx.Response(200, content=b"x" * 25_000),
            httpx.Response(302, headers={"Location": "https://untrusted.invalid/"}),
            httpx.Response(503),
            None,
        )
        reasons = ("INTENT_ENVELOPE", "INTENT_SCHEMA", "INTENT_FINISH", "INTENT_RESPONSE_SIZE",
                   "HTTP 302", "HTTP 503", "ReadTimeout")
        for answer, reason in zip(invalid, reasons, strict=True):
            with self.subTest(reason=reason):
                calls = []

                def respond(request):
                    calls.append(request)
                    if answer is None:
                        raise httpx.ReadTimeout("mock timeout", request=request)
                    return answer

                parser = IntentParser(ENDPOINT, "qwen-plus", "fake", transport=httpx.MockTransport(respond))
                with self.assertLogs("meme_studio.intent", level="INFO") as logs:
                    result = parser.draft(COMBINED, GALLERY, cloud=True)
                self.assertTrue(logs.output[0].endswith(reason))
                self.assertIn("已回退到本机", result.note)
                self.assertEqual(result.plan.image_instruction, "给它加墨镜")
                self.assertEqual(len(calls), 1)


class VersionsTest(TestCase):
    def setUp(self):
        scratch = ROOT / "runtime" / "p1-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="p5-tests-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.runtime = Path(temporary.name)
        initialize(self.runtime / "app.sqlite3")
        gallery = GalleryStore(self.runtime)
        album = gallery.create("测试")
        source = self.runtime / "source.png"
        Image.new("RGB", (1400, 490), "#d39a5b").save(source)
        gallery.import_files(album.id, [source])
        self.album = album.id
        self.photo = gallery.catalog(album.id)[1][0].id
        self.projects = ProjectStore(gallery)
        self.editor = LocalEditor(self.projects, TextRenderer())
        self.root = self.projects.create(self.album, self.photo)

    def requirement(self, parent, target, text, scope="image", duration="persistent", category="subject", selected=None):
        content, replaced = save_constraint(parent, selected, category=category, target=target, text=text,
                                             scope=scope, duration=duration)
        return self.projects.commit(parent, VersionKind.CONSTRAINT_EDIT, constraints_json=content), replaced

    def test_scope_conflict_and_target_replacement(self):
        saved, _ = self.requirement(self.root, "背景", "背景保持不变", "media", category="region")
        saved, _ = self.requirement(saved, "条纹", "保留猫的条纹")
        saved, _ = self.requirement(saved, "配字", "文案标记", "caption", category="text")
        with self.assertRaises(AppError) as conflict:
            compile_image_prompt(saved, "背景完全不变并换成海边", False)
        self.assertEqual(conflict.exception.code, "CONSTRAINT_CONFLICT")
        with self.assertRaises(AppError) as remove:
            compile_image_prompt(saved, "不要再保留条纹", False)
        self.assertEqual(remove.exception.code, "CONSTRAINT_REMOVE")
        with self.assertRaises(AppError):
            self.requirement(saved, "氛围", "把背景换成蓝色", category="style")
        prompt, plan = compile_image_prompt(saved, "把背景换成蓝色，眼泪再大一点", True)
        self.assertIn("白色蒙版", prompt)
        self.assertIn("保留猫的条纹", prompt)
        self.assertNotIn("文案标记", prompt)
        self.assertNotIn("背景保持不变", prompt)
        self.assertEqual([entry.target for entry in plan.replaced], ["背景"])
        self.assertEqual([entry.scope for entry in plan.remaining if entry.target == "背景"], ["video"])
        updated, overwritten = self.requirement(saved, "背景图", "背景为蓝色", category="region")
        self.assertEqual(overwritten, ("背景保持不变",))
        entries = read_constraints(updated)
        item = next(entry for entry in entries if entry.target == "背景图")
        self.assertEqual(item.source_version_id, saved.id)
        revised, _ = self.requirement(updated, "背景", "背景为深蓝色", category="region", selected=item.id)
        self.assertIn("背景为深蓝色", revised.constraints_json)
        removed = delete_constraint(revised, item.id)
        self.assertNotIn("背景为深蓝色", removed)
        self.assertIn("背景为深蓝色", revised.constraints_json)

    def test_only_success_consumes_applicable_once_and_history_restores(self):
        saved, _ = self.requirement(self.root, "毛色", "保留橘色毛发")
        saved, _ = self.requirement(saved, "条纹", "保留条纹", duration="once")
        saved, _ = self.requirement(saved, "背景", "背景保持不变", category="region")
        saved, _ = self.requirement(saved, "动作", "缓慢摇头", "video", "once", "motion")
        saved, _ = self.requirement(saved, "文案", "用两行文字", "caption", "once", "text")
        source = self.editor.save_text(saved, TextLayer("本机文字"))
        self.assertEqual({entry.target for entry in read_constraints(source)}, {"毛色", "条纹", "背景", "动作"})
        remote = Remote()
        with httpx.Client(transport=httpx.MockTransport(remote)) as client:
            clock = [1000.0]
            jobs = StaticJobs(self.projects, DashScope(MEDIA_ENDPOINT, "fake-test-key", client), MODEL,
                              clock=lambda: clock[0])
            remote.post_error = 402
            jobs.submit(source.project_id, source.id, "把背景换成紫色", False)
            self.assertTrue(jobs.run_once())
            self.assertEqual(jobs.list(source.project_id)[0].local_status, JobStatus.FAILED)
            self.assertEqual(self.projects.current(source.project_id), source)
            remote.posts.clear()  # The shared fake returns task-1 for its single polled task.
            remote.post_error = None
            jobs.submit(source.project_id, source.id, "把背景换成蓝色", False)
            self.assertTrue(jobs.run_once())
            clock[0] += 20
            self.assertTrue(jobs.run_once())
            self.assertTrue(jobs.run_once())
            result = self.projects.current(source.project_id)
            self.assertEqual(result.parent_id, source.id)
            self.assertEqual({entry.target for entry in read_constraints(result)}, {"毛色", "动作"})
            operation = json.loads(result.operation_json)
            self.assertIn("保留条纹", operation["prompt"])
            self.assertNotIn("本机文字", operation["prompt"])
            self.assertNotIn("缓慢摇头", operation["prompt"])
            self.assertNotIn("背景保持不变", operation["prompt"])
            self.assertEqual(self.projects.undo(result.project_id, result.id), source)
            self.assertEqual(self.projects.checkout(result.project_id, source.id, result.id), result)

    def test_saved_sam_uses_binary_original_grid_and_is_bound_to_version(self):
        asset = self.projects.asset(self.root.base_image_asset_id)
        transform = ViewTransform.for_base((asset.width, asset.height))
        draft = SegmentationDraft(binding(self.root, asset))
        draft = add_hint(draft, transform, (180, 60), "positive")
        draft = add_hint(draft, transform, (40, 80), "negative")
        draft = add_hint(draft, transform, (160, 40), "box")
        draft = add_hint(draft, transform, (210, 90), "box")
        self.assertEqual(draft.points, ((180, 60, 1), (40, 80, 0)))
        with self.assertRaises(AppError):
            add_hint(draft, transform, (-1, 0), "positive")
        mask = Image.new("L", transform.view_size)
        ImageDraw.Draw(mask).rectangle((160, 40, 210, 90), fill=255)
        draft = SegmentationDraft(draft.binding, draft.points, draft.corners, (Candidate(png_bytes(mask), .92),))
        foreign = self.projects.create(self.album, self.photo)
        with self.assertRaises(AppError):
            save_candidate(self.editor, foreign, draft, 0)
        child = save_candidate(self.editor, self.root, draft, 0)
        region = self.projects.region(child)
        stored = self.projects.load_image(self.projects.asset(region.mask_asset_id))
        self.assertEqual(stored.size, (1400, 490))
        self.assertEqual(set(np.unique(np.array(stored))), {0, 255})
        self.assertEqual(json.loads(region.prompts_json)["mode"], "sam2")
        with self.assertRaises(AppError):
            save_candidate(self.editor, child, draft, 0)
        view = self.editor.view(child)
        view.brush_value["layers"][0].putpixel((180, 60), (0, 0, 0, 0))
        revised = self.editor.save_paint(child, view.brush_value)
        revised_region = self.projects.region(revised)
        self.assertNotEqual(revised_region.mask_asset_id, region.mask_asset_id)
        self.assertEqual(json.loads(revised_region.prompts_json)["mode"], "paint")
