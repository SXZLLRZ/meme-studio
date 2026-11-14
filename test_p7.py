
"""Cross-page editing state must stay isolated even for the same project."""

import asyncio
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from app import ROOT
from meme_studio.config import Secrets, load_config
from meme_studio.database import initialize
from meme_studio.dashscope import DashScope
from meme_studio.gallery import GalleryStore
from meme_studio.static_jobs import StaticJobs
from meme_studio.versions import ProjectStore
from meme_studio.ui.app import build_page


class PageIsolationTest(TestCase):
    def test_box_drafts_are_session_local_and_stale_save_is_rejected(self):
        import gradio as gr
        from gradio.state_holder import SessionState

        scratch = ROOT / "runtime" / "p1-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="p7-page-", dir=scratch) as temporary:
            runtime = Path(temporary)
            config = replace(load_config(ROOT, {}), runtime_dir=runtime, gallery_dir=runtime, secrets=Secrets(""))
            status = SimpleNamespace(python_version="3.11", sqlite_version="test", schema_version=4,
                                     api_key_present=False)
            initialize(runtime / "app.sqlite3")
            store = GalleryStore(runtime)
            album = store.create("合成样例")
            source = runtime / "source.png"
            Image.new("RGB", (640, 480), "ivory").save(source)
            store.import_files(album.id, [source])
            projects = ProjectStore(store)
            version = projects.create(album.id, store.catalog(album.id)[1][0].id)
            jobs = StaticJobs(projects, DashScope(config.api.media_base_url, ""), config.api.image_model)
            self.addCleanup(jobs.close)
            cache = {"GRADIO_TEMP_DIR": str(runtime / "gradio-cache"), "GRADIO_ANALYTICS_ENABLED": "False"}
            with patch.dict(os.environ, cache):
                async def scenario():
                    page = build_page(config, status, jobs)
                    states = [SessionState(page), SessionState(page)]
                    endpoints = {function.api_name: function for function in page.fns.values() if function.api_name}

                    async def api(name, session, *args, point=None):
                        event = gr.SelectData(None, {"index": list(point), "value": None}) if point else None
                        return await page.process_api(
                            endpoints[name], list(args), state=states[session], event_data=event,
                            session_hash=f"p7-session-{session}",
                        )

                    project = version.project_id
                    for session in (0, 1):
                        await api("open_edit_project", session, project)
                    await api("pick_box_corner", 0, project, None, None, point=(12, 15))
                    await api("pick_box_corner", 1, project, None, None, point=(321, 212))
                    await api("pick_box_corner", 0, project, None, None, point=(81, 87))
                    child = projects.current(project)
                    region = projects.region(child)
                    mask = projects.load_image(projects.asset(region.mask_asset_id))
                    self.assertEqual(mask.getbbox(), (12, 15, 82, 88))
                    with self.assertRaises(gr.Error):
                        await api("pick_box_corner", 1, project, None, None, point=(383, 280))
                    self.assertEqual(projects.current(project).id, child.id)
                    self.assertEqual(jobs.list(project), [])
                    await api("open_edit_project", 1, project)
                    await api("pick_box_corner", 1, project, None, None, point=(370, 250))
                    self.assertEqual(projects.current(project).id, child.id)
                    await api("pick_box_corner", 1, project, None, None, point=(383, 280))
                    revised = projects.region(projects.current(project))
                    revised_mask = projects.load_image(projects.asset(revised.mask_asset_id))
                    self.assertEqual(revised_mask.getbbox(), (370, 250, 384, 281))

                asyncio.run(scenario())
