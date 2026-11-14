"""Gradio gallery controls. Search results and selections use stable photo IDs."""

from pathlib import Path
import re

import gradio as gr

from ..config import Config
from ..errors import AppError
from ..gallery import GalleryStore, Photo
from ..image_files import EXTENSIONS
from ..intent import IntentContext, IntentParser, describe_draft
from ..model_manager import ModelManager
from ..retrieval import GalleryRetrieval, IndexResult
from ..schemas import IntentAction as Action


def _index_label(result: IndexResult) -> str:
    if result.generation is None:
        return "当前图库为空。"
    repaired = "已修复损坏的索引。 " if result.repaired else ""
    return f"{repaired}索引可用 · {result.count} 张 · {result.dimension} 维 · 本次新增编码 {result.encoded} 张"


def _tags(text: str | None) -> list[str]:
    return re.split(r"[,，\n]+", text or "")


def gallery_panel(config: Config, model: ModelManager, parser: IntentParser):
    store = GalleryStore(config.runtime_dir, config.max_image_bytes, config.max_image_pixels)
    retrieval = GalleryRetrieval(store, model)
    galleries = store.galleries()
    if not galleries:
        galleries = [store.create("默认图库", config.gallery_dir)]
    first = galleries[0]

    def cards(photos: list[tuple[Photo, float | None]]) -> list[tuple[str, str]]:
        previews = []
        for photo, score in photos:
            try:
                thumbnail = store.image_file(photo, thumbnail=True)
            except AppError:
                continue
            details = f" · 余弦 {score:.3f}" if score is not None else ""
            previews.append((str(thumbnail), f"{photo.name}{details}\nID {photo.id}"))
        return previews

    def choices():
        return [(item.name, item.id) for item in store.galleries()]

    def index_message(gallery_id: str, *, update: bool = False, force: bool = False) -> str:
        try:
            result = retrieval.ensure_index(gallery_id, force=force) if update else retrieval.status(gallery_id)
            return _index_label(result)
        except AppError as exc:
            return f"{exc.user_message} 可继续浏览和手动选图。"

    def browse(gallery_id: str, tags: list[str] | None = None):
        _, photos = store.catalog(gallery_id)
        labels = sorted({tag for photo in photos for tag in photo.tags})
        selected = [tag for tag in tags or [] if tag in labels]
        visible = [photo for photo in photos if set(selected).issubset(photo.tags)]
        return cards([(photo, None) for photo in visible]), labels, selected

    initial_cards, initial_tags, _ = browse(first.id)
    with gr.Row():
        gallery = gr.Dropdown(choices=choices(), value=first.id, label="当前图库", scale=2)
        name = gr.Textbox(label="新图库名称", placeholder="例如 我的猫", scale=2)
        create = gr.Button("新建图库", scale=1)

    with gr.Row():
        source = gr.Textbox(label="图库目录", value=first.source_dir or str(config.gallery_dir), scale=3)
        directory_import = gr.Button("导入目录", scale=1)
    with gr.Row():
        files = gr.File(label="选择图片文件", type="filepath", file_count="multiple", file_types=sorted(EXTENSIONS))
        file_import = gr.Button("导入已选文件")
    index_state = gr.Textbox(value=index_message(first.id), label="索引状态", interactive=False)
    rebuild = gr.Button("更新/重建索引")

    with gr.Accordion("合并指令", open=False):
        raw = gr.Textbox(label="原句", lines=2, placeholder="找一张橘猫，给它加墨镜")
        with gr.Row():
            local_plan = gr.Button("本机拆解")
            cloud_plan = gr.Button("云端拆解文字（可能计费）", interactive=parser.ready)
        gr.Markdown("云端拆解发送原句与当前页面上下文。拆解只填草稿，检索、选图和生成分别由你操作。")
        intent_status = gr.Textbox(label="拆解结果", interactive=False, lines=3)
        pending = gr.Textbox(label="选图后的编辑要求", lines=2,
                             placeholder="创建项目时带入静态生成页，可先修改")

    gr.Markdown("用中文描述画面，或按标签缩小范围。余弦分数只用于排序。")
    with gr.Row():
        query = gr.Textbox(label="检索描述", placeholder="例如 橘猫趴着", scale=3)
        top_k = gr.Slider(1, 50, value=6, step=1, label="候选数", scale=1)
    filter_tags = gr.Dropdown(choices=initial_tags, value=[], multiselect=True, label="标签过滤（满足全部）")
    with gr.Row():
        search = gr.Button("中文检索", variant="primary")
        show_all = gr.Button("浏览全部")
    message = gr.Textbox(label="操作结果", value="选择目录或文件后导入。", interactive=False, lines=2)
    results = gr.Gallery(
        value=initial_cards, label="图库图片", elem_id="p3-gallery-results", columns=4, object_fit="contain",
        buttons=["fullscreen"], interactive=False,
    )
    with gr.Row():
        selected_name = gr.Textbox(label="已选图片", interactive=False)
        selected_id = gr.Textbox(label="图片 ID", interactive=False)
    with gr.Row():
        tag_text = gr.Textbox(label="选中图片的标签（逗号分隔）", scale=3)
        save_tags = gr.Button("保存标签", scale=1)
        remove = gr.Button("移出当前图库", scale=1)
    gr.Markdown("移出仅取消图库关联。原文件和项目中的图片副本仍可用于历史。")
    edit_project = gr.Button("用所选图创建编辑项目", variant="primary")

    def refreshed(gallery_id: str, selected_tags: list[str] | None = None):
        items, labels, active = browse(gallery_id, selected_tags)
        return items, gr.update(choices=labels, value=active)

    def switch_gallery(gallery_id: str):
        info, _ = store.catalog(gallery_id)
        return (info.source_dir or str(config.gallery_dir), *refreshed(gallery_id),
                index_message(gallery_id), f"已打开「{info.name}」。", "", "", "")

    def create_gallery(title: str):
        try:
            new = store.create(title or "")
            return gr.update(choices=choices(), value=new.id), ""
        except AppError as exc:
            raise gr.Error(exc.user_message) from None

    def import_photos(gallery_id: str, directory: str | None = None, uploads: list[str] | None = None):
        try:
            if directory is not None:
                folder = directory.strip().strip('"').strip()
                if not folder:
                    raise AppError("GALLERY_DIRECTORY", "请填写图库目录。")
                result = store.import_directory(gallery_id, Path(folder))
            else:
                if not uploads:
                    raise AppError("UPLOAD_EMPTY", "请先选择要导入的图片文件。")
                cache = (config.runtime_dir / "gradio-cache").resolve()
                paths = [Path(path).resolve() for path in uploads]
                if any(not path.is_relative_to(cache) for path in paths):
                    raise AppError("UPLOAD_PATH", "只接受通过当前页面上传的文件。")
                result = store.import_files(gallery_id, paths)
        except AppError as exc:
            raise gr.Error(exc.user_message) from None

        summary = f"新增 {result.added} 张，重复 {result.duplicates} 张，失败 {len(result.failed)} 张。"
        if result.failed:
            summary += "\n" + "\n".join(f"{label} · {reason}" for label, reason in result.failed)
        return (*refreshed(gallery_id), index_message(gallery_id, update=True), summary, "", "", "")

    def browse_action(gallery_id: str, tags: list[str] | None):
        items, _, _ = browse(gallery_id, tags)
        text = f"当前显示 {len(items)} 张，可直接点击图片。" if items else "当前没有符合条件的图片。"
        return items, text, "", "", ""

    def search_action(gallery_id: str, description: str, limit: int, tags: list[str] | None):
        selected = list(tags or [])
        note = ""
        description = description or ""
        if "我家" in description:
            _, photos = store.catalog(gallery_id)
            if any("我家" in photo.tags for photo in photos):
                selected.append("我家")
                description = description.replace("我家", "").strip()
                note = "已按「我家」标签过滤。"
            else:
                note = "图库没有「我家」标签，按画面匹配，无法判断归属。"
        if "上次" in description or "那个人" in description:
            note += "检索无法推断历史或身份，请用画面描述或标签缩小范围。"
        try:
            hits = retrieval.search(gallery_id, description, int(limit), tuple(selected))
            return cards([(hit.photo, hit.score) for hit in hits]), f"返回 {len(hits)} 张候选。{note}", "", "", ""
        except AppError as exc:
            items, _, _ = browse(gallery_id, selected)
            return items, f"{exc.user_message}{note}", "", "", ""

    def select_photo(gallery_id: str, event: gr.SelectData):
        if not event.selected:
            return "", "", ""
        caption = event.value.get("caption", "") if isinstance(event.value, dict) else ""
        _, separator, image_id = caption.rpartition("\nID ")
        if not separator:
            raise gr.Error("无法读取图片 ID，请刷新图库。")
        try:
            photo = store.get_photo(gallery_id, image_id)
        except AppError as exc:
            raise gr.Error(exc.user_message) from None
        return photo.name, photo.id, "，".join(photo.tags)

    def update_tags(gallery_id: str, image_id: str, text: str):
        try:
            photo = store.set_tags(gallery_id, image_id, _tags(text))
        except AppError as exc:
            raise gr.Error(exc.user_message) from None
        return *refreshed(gallery_id), "已保存标签。", "，".join(photo.tags)

    def remove_photo(gallery_id: str, image_id: str):
        try:
            store.remove(gallery_id, image_id)
        except AppError as exc:
            raise gr.Error(exc.user_message) from None
        return (*refreshed(gallery_id), index_message(gallery_id, update=True),
                "已移出当前图库。", "", "", "")

    def split(raw_text, photo_id, cloud):
        draft = parser.draft(raw_text, IntentContext("gallery", bool(photo_id)), cloud=cloud)
        plan = draft.plan
        search_text = plan.search_query if plan.action in {Action.SEARCH, Action.SEARCH_THEN_EDIT} else gr.skip()
        if plan.action == Action.CLARIFY:
            edit_text = gr.skip()
        else:
            edit_text = plan.image_instruction if plan.action in {Action.SEARCH_THEN_EDIT, Action.EDIT_IMAGE} else ""
        return search_text, edit_text, describe_draft(draft)

    selection = [selected_name, selected_id, tag_text]
    view_outputs = [results, filter_tags, index_state, message, *selection]
    page_options = {"concurrency_id": "gallery", "concurrency_limit": 1}
    create.click(create_gallery, name, [gallery, name], api_name="create_gallery", **page_options)
    gallery.change(switch_gallery, gallery, [source, *view_outputs], api_name="switch_gallery", **page_options)
    directory_import.click(
        lambda gallery_id, folder: import_photos(gallery_id, directory=folder or ""),
        [gallery, source], view_outputs, api_name="import_directory", **page_options,
    )
    file_import.click(
        lambda gallery_id, uploads: import_photos(gallery_id, uploads=uploads),
        [gallery, files], view_outputs, api_name="import_images", **page_options,
    )
    rebuild.click(
        lambda gallery_id: index_message(gallery_id, update=True, force=True), gallery, index_state,
        api_name="rebuild_gallery_index", **page_options,
    )
    search_outputs = [results, message, *selection]
    search.click(search_action, [gallery, query, top_k, filter_tags], search_outputs, api_name="search_gallery", **page_options)
    query.submit(search_action, [gallery, query, top_k, filter_tags], search_outputs, api_name=False, **page_options)
    show_all.click(browse_action, [gallery, filter_tags], search_outputs, api_name="browse_gallery", **page_options)
    results.select(select_photo, gallery, selection, api_name="select_gallery_image", **page_options)
    save_tags.click(update_tags, [gallery, selected_id, tag_text], [results, filter_tags, message, tag_text],
                    api_name="save_gallery_tags", **page_options)
    remove.click(remove_photo, [gallery, selected_id], view_outputs, api_name="remove_gallery_image", **page_options)
    local_plan.click(lambda text, photo: split(text, photo, False), [raw, selected_id],
                     [query, pending, intent_status], api_name="split_instruction_locally", **page_options)
    cloud_plan.click(lambda text, photo: split(text, photo, True), [raw, selected_id],
                     [query, pending, intent_status], api_name="split_instruction_in_cloud", **page_options)
    return gallery, selected_id, edit_project, pending
