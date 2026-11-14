
"""Video projects, durable jobs, local captions and exports."""

import json

import gradio as gr

from ..errors import AppError
from ..schemas import VersionKind
from ..text_layers import TextLayer
from ..video_exports import GifOptions, VideoExports
from ..video_jobs import anchor_for
from .editor_panel import KIND_LABELS, _style, _style_fields, _ui_action
from .static_panel import ERRORS, STATES


def video_panel(page, tab, jobs, editor_binding):
    editor, editor_project, editor_outputs, editor_bundle, project_choices = editor_binding
    projects = jobs.projects
    exports = VideoExports(projects, editor.renderer, jobs.media)
    initial = project_choices()
    project = gr.Dropdown(initial, value=initial[0][1] if initial else None, label="动态项目")
    current = gr.State("")
    with gr.Row():
        version = gr.Textbox(label="当前版本与静态锚点", interactive=False, lines=3)
        refresh_version = gr.Button("刷新当前版本")
    anchor_image = gr.Image(label="静态无字锚点", format="png", interactive=False, buttons=[], height=240)
    with gr.Row():
        history = gr.Dropdown(label="项目历史（包含静态版本）")
        restore = gr.Button("打开历史版本")
        undo = gr.Button("撤销一步")
    note = gr.Textbox(label="动态操作状态", interactive=False)

    with gr.Tabs():
        with gr.Tab("生成与任务"):
            settings = jobs.defaults
            audio = "有声" if settings.audio else "无声"
            gr.Markdown(
                f"提交时将发送静态锚点和动作描述，生成 1 段 {settings.duration} 秒、{settings.resolution}、{audio}视频。"
                "在视频版本修改动作时，会用同一静态锚点生成一个新版本。静态选区属于静态编辑。"
            )
            motion = gr.Textbox(label="动作描述", lines=3, placeholder="例如 固定镜头，主体轻轻眨眼")
            submit = gr.Button("提交 1 段视频（可能计费）", variant="primary", interactive=jobs.ready)
            if not jobs.ready:
                gr.Markdown("生成需要本机 FFmpeg、ffprobe、H.264 编码器，以及已配置的媒体服务。")
            task = gr.Dropdown(label="视频任务")
            job_note = gr.Textbox(label="任务状态", interactive=False, lines=5)
            with gr.Row():
                update_jobs = gr.Button("刷新任务")
                pause = gr.Button("停止等待")
                resume = gr.Button("继续查询")
                open_result = gr.Button("打开结果版本")
            gr.Markdown("停止等待会暂停本机查询。云端任务仍可能继续运行并计费。")
            timer = gr.Timer(2)

        with gr.Tab("视频配字和导出"):
            gr.Markdown("文字固定在画面上，保存配字、预览和导出都在本机处理。PNG 位于「编辑与历史」。")
            text = gr.Textbox(label="视频配字", lines=3)
            with gr.Row():
                size = gr.Slider(3, 40, value=12, step=1, label="字号（短边百分比）")
                x = gr.Slider(0, 100, value=50, step=1, label="横向位置")
                y = gr.Slider(0, 100, value=100, step=1, label="纵向位置")
            with gr.Row():
                fill = gr.ColorPicker(value="#FFFFFF", label="文字颜色")
                stroke_fill = gr.ColorPicker(value="#000000", label="描边颜色")
                stroke = gr.Slider(0, 3, value=.6, step=.1, label="描边宽度（短边百分比）")
            save = gr.Button("保存视频配字版本", variant="primary")
            mp4 = gr.Button("预览并导出 MP4", interactive=jobs.media.ready)
            video = gr.Video(label="MP4 预览", interactive=False, height=360)
            mp4_file = gr.File(label="下载 MP4", interactive=False)
            with gr.Accordion("GIF 参数与导出", open=False):
                with gr.Row():
                    edge = gr.Slider(128, 720, value=512, step=1, label="GIF 长边上限")
                    fps = gr.Slider(6, 20, value=12, step=1, label="GIF 目标帧率")
                    mib = gr.Slider(1, 20, value=5, step=.5, label="目标体积（MiB）")
                    seconds = gr.Slider(0, 15, value=0, step=.1, label="从开头截取（秒，0 为全长）")
                gr.Markdown("GIF 普通重复播放，音轨只保留在 MP4。若超过目标，最多尝试三次降低尺寸或帧率。")
                gif_button = gr.Button("导出 GIF", interactive=jobs.media.ready)
                gif_preview = gr.Image(label="GIF 预览", format="gif", interactive=False, buttons=[], height=300)
                gif_file = gr.File(label="下载 GIF", interactive=False)
            export_note = gr.Textbox(label="导出结果", interactive=False)

    style_fields = [text, size, x, y, fill, stroke_fill, stroke]
    form = [current, version, anchor_image, history, motion, *style_fields, video, mp4_file,
            gif_preview, gif_file, export_note, note]

    def snapshot_view(snapshot, message):
        anchor = anchor_for(projects, snapshot)
        entries = projects.history(snapshot.project_id)
        labels = [
            (f"{KIND_LABELS.get(item.kind, item.kind)} · {item.id[:8]}", item.id)
            for item in entries
        ]
        layers = json.loads(snapshot.text_layers_json)["layers"]
        style = TextLayer(**layers[0]) if layers else TextLayer()
        instruction = ""
        if snapshot.base_video_asset_id is not None:
            by_id = {item.id: item for item in entries}
            node = snapshot
            while node.parent_id and node.kind != VersionKind.VIDEO:
                node = by_id[node.parent_id]
            instruction = json.loads(node.operation_json).get("instruction", "")
        source = f"静态锚点 {anchor.id[:8]}"
        if snapshot.base_video_asset_id is not None:
            asset = projects.asset(snapshot.base_video_asset_id)
            source += f"\n无字视频 · {asset.width}×{asset.height} · {asset.duration:.2f} 秒"
        return (
            snapshot.id, f"{KIND_LABELS.get(snapshot.kind, snapshot.kind)} · {snapshot.id[:8]}\n{source}",
            editor.clean_view(anchor), gr.update(choices=labels, value=snapshot.id), instruction,
            *_style_fields(style), None, None, None, None, "", message,
        )

    def empty():
        return ("", "", None, gr.update(choices=[], value=None), "", *_style_fields(TextLayer()),
                None, None, None, None, "", "先从图库创建或选择项目。")

    @_ui_action
    def open_project(project_id):
        selection = gr.update(choices=project_choices(), value=project_id)
        return (selection, *snapshot_view(projects.current(project_id), "已读取当前版本。")) if project_id else (selection, *empty())

    def version_outputs(snapshot, message):
        return (*snapshot_view(snapshot, message), *editor_bundle(snapshot, message))

    @_ui_action
    def refresh_current(project_id):
        if not project_id:
            return (*empty(), *(gr.skip() for _ in editor_outputs))
        return version_outputs(projects.current(project_id), "已读取当前版本。")

    @_ui_action
    def enter_tab(project_id, expected_id):
        if project_id and projects.current(project_id).id == expected_id:
            return [gr.skip()] * (len(form) + 1)
        return open_project(project_id)

    @_ui_action
    def use_history(project_id, version_id, target):
        snapshot = projects.checkout(project_id, version_id, target)
        return version_outputs(snapshot, "已打开历史版本。")

    @_ui_action
    def undo_history(project_id, version_id):
        return version_outputs(projects.undo(project_id, version_id), "已返回上一步。")

    def render_jobs(project_id, task_id):
        items = jobs.list(project_id) if project_id else []
        choices = [(f"{item.id[:8]} · {STATES.get(item.local_status, item.local_status)}", item.id) for item in items]
        item = next((entry for entry in items if entry.id == task_id), None) or (items[0] if items else None)
        if item is None:
            return gr.update(choices=choices, value=None), "当前项目暂无视频任务。"
        state = STATES.get(item.local_status, item.local_status)
        lines = [f"本机：{'暂停等待 · ' if item.paused else ''}{state}",
                 f"平台：{item.remote_status or '待确认'}", f"任务 ID：{item.remote_task_id or '待确认'}"]
        if item.video_seconds is not None:
            lines.append(f"平台计量：{item.video_seconds:g} 秒"
                         + (f" · {item.resolution}" if item.resolution else "")
                         + (f" · {'有声' if item.audio else '无声'}" if item.audio is not None else ""))
        if item.error_code:
            lines.append(ERRORS.get(item.error_code, f"状态码：{item.error_code}"))
        return gr.update(choices=choices, value=item.id), "\n".join(lines)

    @_ui_action
    def start(project_id, version_id, instruction):
        job_id = jobs.submit(project_id, version_id, instruction or "")
        return render_jobs(project_id, job_id)

    @_ui_action
    def set_waiting(project_id, task_id, paused):
        jobs.set_paused(project_id, task_id, paused)
        return render_jobs(project_id, task_id)

    @_ui_action
    def open_generated(project_id, version_id, task_id):
        item = next((entry for entry in jobs.list(project_id) if entry.id == task_id), None)
        if item is None or item.result_version_id is None:
            raise AppError("VIDEO_RESULT", "请选择一个已保存本机结果的视频任务。")
        snapshot = projects.current(project_id)
        if snapshot.id != item.result_version_id:
            snapshot = projects.checkout(project_id, version_id, item.result_version_id)
        return version_outputs(snapshot, "已打开生成结果。")

    @_ui_action
    def save_text(project_id, version_id, *fields):
        snapshot = projects.require_current(project_id, version_id)
        exports.source(snapshot)
        child = editor.save_text(snapshot, _style(*fields))
        return version_outputs(child, "视频配字已保存。")

    @_ui_action
    def export_mp4(project_id, version_id):
        snapshot = projects.require_current(project_id, version_id)
        result = exports.export(snapshot)
        return str(result.path), str(result.path), result.label()

    @_ui_action
    def export_gif(project_id, version_id, max_edge, frame_rate, target, length):
        snapshot = projects.require_current(project_id, version_id)
        result = exports.export(snapshot, GifOptions(int(max_edge), int(frame_rate), target, length))
        return str(result.path), str(result.path), result.label()

    serial = {"concurrency_id": "editor", "concurrency_limit": 1}
    local_encode = {"concurrency_id": "video-export", "concurrency_limit": 1}
    active = [project, current]
    version_changes = [*form, *editor_outputs]
    job_outputs = [task, job_note]
    page.load(open_project, editor_project, [project, *form], api_name=False, **serial)
    tab.select(enter_tab, [editor_project, current], [project, *form], api_name=False, **serial)
    editor_project.change(open_project, editor_project, [project, *form], api_name=False, **serial)
    project.input(lambda value: gr.update(choices=project_choices(), value=value), project, editor_project,
                  api_name=False, **serial)
    refresh_version.click(refresh_current, project, version_changes, api_name="open_video_project", **serial)
    restore.click(use_history, [*active, history], version_changes, api_name="restore_video_version", **serial)
    undo.click(undo_history, active, version_changes, api_name="undo_video_version", **serial)
    submit.click(start, [*active, motion], job_outputs, api_name="submit_video", **serial)
    timer.tick(render_jobs, [project, task], job_outputs, api_name=False, queue=False, show_progress="hidden")
    update_jobs.click(render_jobs, [project, task], job_outputs, api_name="video_job_status", **serial)
    pause.click(lambda p, t: set_waiting(p, t, True), [project, task], job_outputs, api_name="pause_video_job", **serial)
    resume.click(lambda p, t: set_waiting(p, t, False), [project, task], job_outputs, api_name="resume_video_job", **serial)
    open_result.click(open_generated, [*active, task], version_changes, api_name="open_video_result", **serial)
    save.click(save_text, [*active, *style_fields], version_changes, api_name="save_video_caption", **serial)
    mp4.click(export_mp4, active, [video, mp4_file, export_note], api_name="export_video_mp4", **local_encode)
    gif_button.click(export_gif, [*active, edge, fps, mib, seconds], [gif_preview, gif_file, export_note],
                     api_name="export_video_gif", **local_encode)
