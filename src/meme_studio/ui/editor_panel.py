"""Project editor bindings. Browser state holds only version IDs and draft coordinates."""

import json
from functools import wraps

import gradio as gr

from ..config import Config
from ..editor import LocalEditor
from ..errors import AppError
from ..static_jobs import StaticJobs
from ..text_layers import TextLayer, TextRenderer
from .constraint_panel import ConstraintPanel
from .sam_panel import SamPanel
from .static_panel import static_panel


KIND_LABELS = {"root": "原图", "text_edit": "配字", "region_edit": "选区", "image_edit": "画面", "video": "视频",
               "constraint_edit": "保留要求"}


def _ui_action(callback):
    @wraps(callback)
    def run(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except AppError as exc:
            raise gr.Error(exc.user_message) from None

    return run


def _style_fields(style: TextLayer) -> tuple:
    return (
        style.text, round(style.size_ratio * 100, 2), round(style.x * 100, 2), round(style.y * 100, 2),
        style.fill, style.stroke_fill, round(style.stroke_ratio * 100, 2),
    )


def _style(text, size, x, y, fill, stroke_fill, stroke) -> TextLayer:
    return TextLayer(text or "", size / 100, x / 100, y / 100, fill, stroke_fill, stroke / 100)


def editor_panel(
    config: Config, page: gr.Blocks, tabs: gr.Tabs, gallery, selected_photo, create_button,
    pending, jobs: StaticJobs, segmenter, parser,
):
    projects = jobs.projects
    editor = LocalEditor(projects, TextRenderer(config.font_path))

    def project_choices():
        return [(f"{entry.title} · {entry.id[:8]}", entry.id) for entry in projects.projects()]

    choices = project_choices()
    expected = gr.State("")
    corner = gr.State(None)
    with gr.Row():
        project = gr.Dropdown(choices=choices, value=choices[0][1] if choices else None, label="编辑项目")
        reload_project = gr.Button("重新打开项目")
    version = gr.Textbox(label="当前版本", interactive=False, elem_id="p3-current-version")
    message = gr.Textbox(label="编辑状态", elem_id="p3-edit-message", value="从图库选择一张图片，再创建编辑项目。", interactive=False)
    gr.Markdown("保存时创建版本。切换项目、保存或恢复版本时，未保存的文字和笔画草稿会清空。")

    with gr.Tabs() as editor_tabs:
        with gr.Tab("本地配字", id="caption"):
            preview = gr.Image(label="成品预览", interactive=False, format="png", buttons=[], height=400)
            text = gr.Textbox(label="文案", elem_id="p3-caption", lines=3, max_lines=10, placeholder="可换行，留空表示隐藏文字")
            with gr.Row():
                size = gr.Slider(3, 40, value=12, step=1, label="字号（短边百分比）")
                x = gr.Slider(0, 100, value=50, step=1, label="横向位置（左到右）")
                y = gr.Slider(0, 100, value=100, step=1, label="纵向位置（上到下）")
            with gr.Row():
                fill = gr.ColorPicker(value="#FFFFFF", label="文字颜色")
                stroke_fill = gr.ColorPicker(value="#000000", label="描边颜色")
                stroke = gr.Slider(0, 3, value=0.6, step=0.1, label="描边宽度（短边百分比）")
            with gr.Row():
                preview_button = gr.Button("预览配字")
                save = gr.Button("保存文字版本", variant="primary")
        with gr.Tab("局部选区"):
            region_label = gr.Textbox(label="已保存选区", interactive=False, elem_id="p3-region-state")
            clear = gr.Button("整图 / 清空选区")
            with gr.Tabs():
                with gr.Tab("两角框选"):
                    gr.Markdown("在无字底图上点击矩形的两个对角。第二次点击后保存选区，红色为实际蒙版。")
                    box_view = gr.Image(
                        label="框选底图", interactive=False, format="png", buttons=[], height=420,
                        elem_id="p3-box-image",
                    )
                with gr.Tab("画笔与橡皮"):
                    gr.Markdown("在画布上用画笔增加红色选区，用橡皮擦除。完成后点「保存涂抹选区」。")
                    brush = gr.ImageEditor(
                        label="涂抹画布", type="pil", image_mode="RGBA", format="png",
                        sources=[], transforms=[], fixed_canvas=False,
                        layers=gr.LayerOptions(allow_additional_layers=False),
                        brush=gr.Brush(colors=["#FF3E2E"], color_mode="fixed", default_size=24),
                        eraser=gr.Eraser(default_size=24), buttons=[], height=440,
                        elem_id="p3-brush-editor",
                    )
                    paint = gr.Button("保存涂抹选区", variant="primary")
                with gr.Tab("SAM 点选主体"):
                    sam = SamPanel()
        with gr.Tab("保留要求", id="constraints"):
            requirements = ConstraintPanel()
        with gr.Tab("历史与导出"):
            history = gr.Dropdown(label="版本历史")
            constraints = gr.Textbox(label="当前版本的保留要求", interactive=False)
            generated_prompt = gr.Textbox(label="当前生成版本的实际提示词", interactive=False, lines=3)
            with gr.Row():
                undo = gr.Button("撤销一步")
                restore = gr.Button("恢复所选版本")
            export = gr.Button("导出 PNG", variant="primary")
            download = gr.File(label="PNG 下载", interactive=False, elem_id="p3-download")
            gr.Markdown("PNG 使用静态画面和配字。当前是视频版本时，PNG 来自其静态锚点。")

        with gr.Tab("静态生成", id="static") as static_tab:
            pass

    style_inputs = [text, size, x, y, fill, stroke_fill, stroke]
    outputs = [
        expected, corner, version, history, constraints, preview, box_view, brush, region_label,
        *style_inputs, download, *sam.outputs, *requirements.outputs, generated_prompt, message,
    ]

    def bundle(snapshot, status: str):
        view = editor.view(snapshot)
        options = []
        for item in projects.history(snapshot.project_id):
            kind = KIND_LABELS.get(item.kind, item.kind)
            parent = f" · 父版本 {item.parent_id[:8]}" if item.parent_id else ""
            options.append((f"{kind} · {item.id[:8]}{parent}", item.id))
        current_kind = KIND_LABELS.get(snapshot.kind, snapshot.kind)
        return (
            snapshot.id, None, f"{current_kind} · {snapshot.id}",
            gr.update(choices=options, value=snapshot.id), view.constraint_summary,
            view.preview, view.region_preview, view.brush_value, view.region_summary,
            *_style_fields(view.text), None, *sam.cleared(view.brush_value['background']),
            *requirements.view(snapshot), json.loads(snapshot.operation_json).get("prompt", ""), status,
        )

    @_ui_action
    def open_project(project_id):
        if not project_id:
            return (
                "", None, "", gr.update(choices=[], value=None), "", None, None, None, "",
                *_style_fields(TextLayer()), None, *sam.cleared(None), *requirements.view(),
                "", "从图库选择一张图片，再创建编辑项目。",
            )
        return bundle(projects.current(project_id), "已打开项目，恢复当前版本。")

    @_ui_action
    def create_project(gallery_id, image_id, instruction):
        if not image_id:
            raise AppError("PHOTO_REQUIRED", "请先点击图库中的一张图片。")
        root = projects.create(gallery_id, image_id)
        instruction = (instruction or "").strip()
        return (
            gr.update(choices=project_choices(), value=root.project_id), gr.update(selected="edit"),
            instruction, "", gr.update(selected="static" if instruction else "caption"),
        )

    def require_static(project_id, version_id):
        snapshot = projects.require_current(project_id, version_id)
        if snapshot.base_video_asset_id is not None:
            raise AppError("STATIC_VERSION", "当前是视频版本。请在「动态与导出」配字，或打开静态锚点编辑图片。")
        return snapshot

    @_ui_action
    def show_preview(project_id, version_id, *fields):
        snapshot = require_static(project_id, version_id)
        return editor.preview_text(snapshot, _style(*fields)), "已预览。点击「保存文字版本」写入历史。"

    @_ui_action
    def save_caption(project_id, version_id, *fields):
        snapshot = require_static(project_id, version_id)
        updated = editor.save_text(snapshot, _style(*fields))
        status = "文字版本已保存。" if updated.id != snapshot.id else "文字内容没有变化。"
        return bundle(updated, status)

    @_ui_action
    def choose_corner(project_id, version_id, first, event: gr.SelectData):
        snapshot = require_static(project_id, version_id)
        if not isinstance(event.index, (list, tuple)):
            raise AppError("REGION_POINT", "请点击底图中的一个像素位置。")
        point = editor.validate_point(snapshot, tuple(event.index))
        if first is None:
            updates = [gr.skip()] * len(outputs)
            updates[1] = point
            updates[-1] = f"已记录第一角 {point}，请点击对角。"
            return updates
        return bundle(editor.save_box(snapshot, tuple(first), point), "两角框选已保存。")

    @_ui_action
    def save_strokes(project_id, version_id, editor_value):
        snapshot = require_static(project_id, version_id)
        return bundle(editor.save_paint(snapshot, editor_value), "涂抹选区已保存。")

    @_ui_action
    def clear_selection(project_id, version_id):
        snapshot = require_static(project_id, version_id)
        return bundle(editor.clear_region(snapshot), "已清空选区，使用整图。")

    @_ui_action
    def undo_version(project_id, version_id):
        snapshot = projects.undo(project_id, version_id)
        status = "已撤销到父版本。" if snapshot.id != version_id else "已经是项目原图。"
        return bundle(snapshot, status)

    @_ui_action
    def restore_version(project_id, version_id, target_id):
        return bundle(projects.checkout(project_id, version_id, target_id), "已恢复所选版本。")

    @_ui_action
    def export_png(project_id, version_id):
        snapshot = projects.require_current(project_id, version_id)
        output = editor.export_png(snapshot)
        status = "已复用上次 PNG 导出。" if output.cached else "PNG 已生成，可以下载。"
        return str(output.path), status

    with static_tab:
        instruction = static_panel(jobs, editor, project, expected, outputs, bundle, editor_tabs, parser)

    options = {"concurrency_id": "editor", "concurrency_limit": 1}
    active = [project, expected]
    sam.bind(editor, segmenter, active, outputs, bundle, _ui_action, options)
    requirements.bind(projects, active, outputs, bundle, _ui_action, options)
    create_button.click(
        create_project, [gallery, selected_photo, pending], [project, tabs, instruction, pending, editor_tabs],
        api_name="create_edit_project", **options,
    )
    project.input(lambda: "", outputs=instruction, api_name=False, **options)
    page.load(open_project, project, outputs, api_name=False, **options)
    project.change(open_project, project, outputs, api_name="open_edit_project", **options)
    reload_project.click(open_project, project, outputs, api_name=False, **options)
    preview_button.click(show_preview, [*active, *style_inputs], [preview, message], api_name="preview_local_caption", **options)
    save.click(save_caption, [*active, *style_inputs], outputs, api_name="save_local_caption", **options)
    box_view.select(choose_corner, [*active, corner], outputs, api_name="pick_box_corner", **options)
    paint.click(save_strokes, [*active, brush], outputs, api_name="save_local_strokes", **options)
    clear.click(clear_selection, active, outputs, api_name="clear_local_region", **options)
    undo.click(undo_version, active, outputs, api_name="undo_edit_version", **options)
    restore.click(restore_version, [*active, history], outputs, api_name="restore_edit_version", **options)
    export.click(export_png, active, [download, message], api_name="export_local_png", **options)
    return editor, project, outputs, bundle, project_choices
