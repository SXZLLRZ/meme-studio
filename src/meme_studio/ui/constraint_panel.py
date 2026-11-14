
"""Edit one versioned requirement at a time and display the resulting list."""

import gradio as gr

from ..constraints import (
    CATEGORIES, DURATIONS, SCOPES, constraint_summary, delete_constraint, read_constraints, save_constraint,
)
from ..errors import AppError
from ..schemas import VersionKind


class ConstraintPanel:
    def __init__(self):
        gr.Markdown(
            "按目标保存要求，如「毛色」「条纹」或「背景」。同一目标的新要求会替换旧条目。"
            "画面要求由模型尽力遵循。要固定选区外原像素，请使用静态生成中的局部合成选项。"
            "配字要求供本机排版时参考。"
        )
        self.summary = gr.Textbox(label="此版本的保留要求", lines=4, interactive=False)
        self.selected = gr.Dropdown(label="已有条目")
        with gr.Row():
            self.load = gr.Button("载入所选")
            self.delete = gr.Button("移除所选")
        with gr.Row():
            self.category = gr.Dropdown([(label, key) for key, label in CATEGORIES.items()],
                                        label="类别", value="subject")
            self.target = gr.Textbox(label="目标", placeholder="例如 背景")
        self.text = gr.Textbox(label="具体要求", placeholder="例如 保持原来的背景", lines=2)
        with gr.Row():
            self.scope = gr.Dropdown([(label, key) for key, label in SCOPES.items()], label="作用域", value="image")
            self.duration = gr.Dropdown([(label, key) for key, label in DURATIONS.items()],
                                        label="持续时间", value="persistent")
        with gr.Row():
            self.add = gr.Button("新增 / 替换同目标", variant="primary")
            self.update = gr.Button("更新所选条目")
        self.fields = [self.category, self.target, self.text, self.scope, self.duration]
        self.outputs = [self.summary, self.selected, *self.fields]

    def view(self, snapshot=None):
        entries = read_constraints(snapshot) if snapshot else ()
        choices = [(f"{entry.target} · {SCOPES[entry.scope]}", entry.id) for entry in entries]
        return (constraint_summary(snapshot) if snapshot else "", gr.update(choices=choices, value=None),
                "subject", "", "", "image", "persistent")

    def bind(self, projects, active, editor_outputs, bundle, ui_action, options):
        @ui_action
        def load(project_id, version_id, selected):
            snapshot = projects.require_current(project_id, version_id)
            item = next((entry for entry in read_constraints(snapshot) if entry.id == selected), None)
            if item is None:
                raise AppError("CONSTRAINT_SELECTION", "请先选择当前版本的一项要求。")
            return item.category, item.target, item.text, item.scope, item.duration

        def save(project_id, version_id, selected, category, target, text, scope, duration):
            snapshot = projects.require_current(project_id, version_id)
            content, replaced = save_constraint(snapshot, selected, category=category, target=target, text=text,
                                                 scope=scope, duration=duration)
            child = snapshot if content == snapshot.constraints_json else projects.commit(
                snapshot, VersionKind.CONSTRAINT_EDIT, constraints_json=content,
            )
            status = "保留要求已保存。"
            if replaced:
                status += " 已替换：" + "；".join(replaced)
            return bundle(child, status)

        @ui_action
        def add(project_id, version_id, *fields):
            return save(project_id, version_id, None, *fields)

        @ui_action
        def update(project_id, version_id, selected, *fields):
            if not selected:
                raise AppError("CONSTRAINT_SELECTION", "请先选择要更新的条目。")
            return save(project_id, version_id, selected, *fields)

        @ui_action
        def delete(project_id, version_id, selected):
            snapshot = projects.require_current(project_id, version_id)
            content = delete_constraint(snapshot, selected)
            child = projects.commit(snapshot, VersionKind.CONSTRAINT_EDIT, constraints_json=content)
            return bundle(child, "已移除所选要求。")

        self.load.click(load, [*active, self.selected], self.fields, api_name="load_constraint", **options)
        self.add.click(add, [*active, *self.fields], editor_outputs, api_name="add_constraint", **options)
        self.update.click(update, [*active, self.selected, *self.fields], editor_outputs,
                          api_name="update_constraint", **options)
        self.delete.click(delete, [*active, self.selected], editor_outputs, api_name="remove_constraint", **options)
