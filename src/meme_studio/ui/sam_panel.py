
"""Point and box hints and candidate selection for the local segmenter."""

import gradio as gr

from ..errors import AppError
from ..regions import ViewTransform
from ..segmentation import (
    SegmentationDraft, add_hint, binding, candidate_preview, preview_hints, save_candidate,
)


class SamPanel:
    def __init__(self):
        gr.Markdown(
            "在无字底图上点选。绿点表示保留在选区内的主体，红点表示排除；也可点两角加蓝色提示框。"
            "生成候选后可继续加点。保存候选后，可到「画笔与橡皮」修正。"
        )
        self.draft = gr.State(None)
        self.image = gr.Image(label="SAM 点选底图", interactive=False, format="png", buttons=[], height=420)
        self.mode = gr.Radio(
            [("正点", "positive"), ("负点", "negative"), ("两角提示框", "box")],
            value="positive", label="点击方式",
        )
        with gr.Row():
            self.predict = gr.Button("生成本地蒙版候选", variant="primary")
            self.reset = gr.Button("清空点、框和候选")
        self.status = gr.Textbox(label="点选状态", interactive=False)
        self.cards = gr.Gallery(label="候选蒙版预览", columns=3, interactive=False, buttons=[], format="png")
        with gr.Row():
            self.choice = gr.Dropdown(label="选择候选蒙版")
            self.save = gr.Button("保存所选蒙版", variant="primary")
        self.outputs = [self.draft, self.image, self.status, self.cards, self.choice]

    def cleared(self, clean):
        return (None, clean, "请添加正点或提示框。" if clean is not None else "", [],
                gr.update(choices=[], value=None))

    def bind(self, editor, segmenter, active, editor_outputs, bundle, ui_action, options):
        projects = editor.projects

        def read(project_id, version_id, draft):
            snapshot = projects.require_current(project_id, version_id)
            if snapshot.base_video_asset_id is not None:
                raise AppError("STATIC_VERSION", "请打开静态锚点后再点选图片区域。")
            asset = projects.asset(snapshot.base_image_asset_id)
            key = binding(snapshot, asset)
            if draft is None or draft.binding != key:
                draft = SegmentationDraft(key)
            transform = ViewTransform.for_base((asset.width, asset.height))
            return snapshot, transform, draft

        def render(clean, draft, status, cards=(), choices=()):
            return (draft, preview_hints(clean, draft), status, list(cards),
                    gr.update(choices=list(choices), value=choices[0][1] if choices else None))

        @ui_action
        def click(project_id, version_id, draft, mode, event: gr.SelectData):
            snapshot, transform, draft = read(project_id, version_id, draft)
            if not isinstance(event.index, (tuple, list)):
                raise AppError("REGION_POINT", "请点击图片以内的位置。")
            draft = add_hint(draft, transform, tuple(event.index), mode)
            positives = sum(point[2] for point in draft.points)
            return render(editor.clean_view(snapshot), draft,
                          f"正点 {positives} · 负点 {len(draft.points) - positives} · 框角 {len(draft.corners)}/2")

        @ui_action
        def predict(project_id, version_id, draft):
            snapshot, _, draft = read(project_id, version_id, draft)
            clean = editor.clean_view(snapshot)
            draft = segmenter.segment(clean, draft)
            choices = [(f"候选 {i + 1} · 分数 {item.score:.3f}", i) for i, item in enumerate(draft.candidates)]
            cards = [(candidate_preview(clean, item), choices[i][0]) for i, item in enumerate(draft.candidates)]
            return render(clean, draft, f"本机生成 {len(choices)} 个候选，请观察红色蒙版并选择。", cards, choices)

        @ui_action
        def reset(project_id, version_id):
            snapshot = projects.require_current(project_id, version_id)
            return self.cleared(editor.clean_view(snapshot))

        @ui_action
        def save(project_id, version_id, draft, index):
            snapshot = projects.require_current(project_id, version_id)
            child = save_candidate(editor, snapshot, draft, index)
            return bundle(child, "SAM 选区已保存，可在「画笔与橡皮」继续修正。")

        self.image.select(click, [*active, self.draft, self.mode], self.outputs, api_name="sam_add_hint", **options)
        self.predict.click(predict, [*active, self.draft], self.outputs, api_name="sam_candidates", **options)
        self.reset.click(reset, active, self.outputs, api_name="sam_clear_hints", **options)
        self.save.click(save, [*active, self.draft, self.choice], editor_outputs, api_name="sam_save_candidate", **options)
