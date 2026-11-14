"""User-initiated static editing and local job status."""

import gradio as gr

from ..constraints import read_constraints
from ..errors import AppError
from ..image_requests import compile_image_prompt
from ..intent import IntentContext, describe_draft
from ..schemas import IntentAction as Action
from ..static_jobs import StaticJobs


STATES = {
    "CREATED": "本机待提交", "SUBMITTING": "正在提交", "QUEUED": "平台排队", "RUNNING": "平台处理中",
    "REMOTE_SUCCEEDED": "平台已生成，等待本机下载", "DOWNLOADING": "正在下载",
    "SUCCEEDED": "结果已保存", "FAILED": "任务未完成", "RESULT_EXPIRED": "结果已过期",
    "SUBMISSION_UNKNOWN": "提交未知，请核对平台控制台",
}
ERRORS = {
    "SUBMIT_UNKNOWN": "无法确认平台是否收到了请求。请核对平台控制台。",
    "AUTH": "平台拒绝鉴权或模型权限，请核对本机密钥和模型开通状态。",
    "BALANCE": "平台余额或可用额度不足。",
    "RATE_LIMIT": "平台限流。",
    "WAIT_TIMEOUT": "等待已达 10 分钟，可继续查询原任务。",
    "RESULT_GEOMETRY": "返回尺寸与发送副本不同，无法可靠对齐选区，原版本已保留。",
    "RESULT_EXPIRED": "原任务未返回有效的新结果地址。",
    "RESULT_LINK_EXPIRED": "尝试从原任务刷新下载地址。",
    "CONNECT": "未连接到平台，请检查网络。",
}


def static_panel(jobs: StaticJobs, editor, project, expected, editor_outputs, bundle, editor_tabs, parser):
    gr.Markdown(
        "点提交会把当前无字底图发往千问。局部模式会同时发送当前版本已保存的蒙版。"
        "每次请求一张图，可能计费，单价待配置。配字会在本机叠回结果。"
    )
    instruction = gr.Textbox(label="修改说明", lines=3, placeholder="例如：给这只猫加上一副墨镜")
    with gr.Accordion("拆解复杂要求", open=False):
        gr.Markdown("云端拆解发送原句、版本类型和此轮适用的画面要求。")
        with gr.Row():
            local_plan = gr.Button("本机拆解当前文字")
            cloud_plan = gr.Button("云端拆解当前文字（可能计费）", interactive=parser.ready)
        intent_status = gr.Textbox(label="拆解结果", interactive=False, lines=3)
    mode = gr.Radio([("整图", "whole"), ("使用已保存选区", "mask")], value="whole", label="编辑范围")
    protect = gr.Checkbox(label="保留选区外原像素（仅局部）", value=False)
    gr.Markdown("画面保留要求是发给模型的文字。固定选区外像素依赖本机合成，需要生成图可与发送副本对齐。")
    prompt_preview = gr.Button("预览本轮请求")
    request_summary = gr.Textbox(label="上次请求预览", interactive=False, lines=5)
    checked = gr.State(None)
    submit = gr.Button("提交 1 张（可能计费）", variant="primary", interactive=jobs.ready)
    if not jobs.ready:
        gr.Markdown("配置本机密钥及已验证的媒体地址和模型后，即可提交静态生成。")
    task = gr.Dropdown(label="静态任务")
    status = gr.Textbox(label="任务状态", interactive=False, lines=5)
    with gr.Row():
        refresh = gr.Button("刷新任务")
        pause = gr.Button("停止等待")
        resume = gr.Button("继续查询")
    gr.Markdown("停止等待只暂停本机后续查询。已经提交的云端任务可能继续运行并计费。")
    preview = gr.Image(label="生成结果预览", format="png", interactive=False, buttons=[])
    with gr.Row():
        result = gr.Button("查看结果")
        open_result = gr.Button("打开结果版本")
    timer = gr.Timer(2)

    def find_task(project_id, job_id):
        item = next((item for item in jobs.list(project_id) if item.id == job_id), None)
        if item is None:
            raise gr.Error("请选择当前项目的一个任务。")
        return item

    def render(project_id, job_id):
        entries = jobs.list(project_id) if project_id else []
        item = next((item for item in entries if item.id == job_id), None)
        item = item or (entries[0] if entries else None)
        choices = [(f"{entry.id[:8]} · {STATES.get(entry.local_status, entry.local_status)}", entry.id)
                   for entry in entries]
        if item is None:
            return gr.update(choices=choices, value=None), "当前项目暂无静态任务。"
        lines = [
            f"本机：{'暂停等待 · ' if item.paused else ''}{STATES.get(item.local_status, item.local_status)}",
            f"平台：{item.remote_status or '待确认'}",
            f"任务 ID：{item.remote_task_id or '待确认'}",
            f"成功图片数：{item.image_count if item.image_count is not None else '待平台返回'}　单价：待配置",
        ]
        if item.error_code:
            lines.append(ERRORS.get(item.error_code, f"状态码：{item.error_code}"))
        return gr.update(choices=choices, value=item.id), "\n".join(lines)

    def summarize(project_id, version_id, text, scope, keep):
        snapshot = jobs.projects.require_current(project_id, version_id)
        local = scope == "mask"
        prompt, requirements = compile_image_prompt(snapshot, text or "", local)
        if local and jobs.projects.region(snapshot) is None:
            raise AppError("REGION_REQUIRED", "请先保存一个局部选区。")
        if keep and not local:
            raise AppError("PROTECT_MODE", "保留选区外像素仅用于局部蒙版编辑。")
        return prompt, requirements

    def inspect_request(project_id, version_id, text, scope, keep):
        try:
            prompt, requirements = summarize(project_id, version_id, text, scope, keep)
        except AppError as exc:
            return exc.user_message, None
        lines = [prompt, "本机固定选区外像素：" + ("启用" if keep else "未启用")]
        if requirements.replaced:
            lines.append("本轮采用新的修改说明。成功后移除旧的静态要求：" + "；".join(entry.text for entry in requirements.replaced))
        return "\n".join(lines), (project_id, version_id, text, scope, keep)

    def split_editor(project_id, version_id, text, cloud):
        try:
            snapshot = jobs.projects.require_current(project_id, version_id)
        except AppError as exc:
            raise gr.Error(exc.user_message) from None
        active = tuple(entry.text for entry in read_constraints(snapshot) if entry.applies("image"))
        draft = parser.draft(text, IntentContext("static", True, snapshot.kind, active), cloud=cloud)
        plan = draft.plan
        next_text = plan.image_instruction if plan.action == Action.EDIT_IMAGE else gr.skip()
        return next_text, describe_draft(draft)

    def start(project_id, version_id, text, scope, keep, reviewed):
        try:
            _, requirements = summarize(project_id, version_id, text, scope, keep)
            if requirements.replaced and reviewed != (project_id, version_id, text, scope, keep):
                raise AppError("REVIEW_REPLACEMENTS", "这轮修改会替换保留要求，请先点「预览本轮请求」核对。")
            job_id = jobs.submit(project_id, version_id, text or "", scope == "mask", bool(keep))
            return (*render(project_id, job_id), None)
        except AppError as exc:
            raise gr.Error(exc.user_message) from None

    def waiting(project_id, job_id, paused):
        try:
            jobs.set_paused(project_id, job_id, paused)
            return render(project_id, job_id)
        except AppError as exc:
            raise gr.Error(exc.user_message) from None

    def get_result(project_id, job_id):
        item = find_task(project_id, job_id)
        if item.result_version_id is None:
            raise gr.Error("任务还没有保存到本机的结果。")
        return jobs.projects.version(project_id, item.result_version_id)

    def show(project_id, job_id):
        try:
            return editor.view(get_result(project_id, job_id)).preview
        except AppError as exc:
            raise gr.Error(exc.user_message) from None

    def open_version(project_id, version_id, job_id):
        try:
            target = get_result(project_id, job_id)
            current = jobs.projects.current(project_id)
            if current.id != target.id:
                target = jobs.projects.checkout(project_id, version_id, target.id)
            return (*bundle(target, "已打开生成结果版本。"), gr.update(selected="caption"))
        except AppError as exc:
            raise gr.Error(exc.user_message) from None

    options = {"concurrency_id": "editor", "concurrency_limit": 1}
    request_inputs = [project, expected, instruction, mode, protect]
    prompt_preview.click(inspect_request, request_inputs, [request_summary, checked],
                         api_name="preview_static_request", **options)
    local_plan.click(lambda project_id, version_id, text: split_editor(project_id, version_id, text, False),
                     [project, expected, instruction], [instruction, intent_status],
                     api_name="split_editor_instruction_locally", **options)
    cloud_plan.click(lambda project_id, version_id, text: split_editor(project_id, version_id, text, True),
                     [project, expected, instruction], [instruction, intent_status],
                     api_name="split_editor_instruction_in_cloud", **options)
    submit.click(start, [*request_inputs, checked], [task, status, preview],
                 api_name="submit_static_image", **options)
    timer.tick(render, [project, task], [task, status], api_name=False, queue=False, show_progress="hidden")
    refresh.click(render, [project, task], [task, status], api_name="static_job_status", **options)
    project.change(lambda project_id: (*render(project_id, None), None), project, [task, status, preview], api_name=False)
    pause.click(lambda project_id, job_id: waiting(project_id, job_id, True), [project, task], [task, status],
                api_name="pause_static_job", **options)
    resume.click(lambda project_id, job_id: waiting(project_id, job_id, False), [project, task], [task, status],
                 api_name="resume_static_job", **options)
    result.click(show, [project, task], preview, api_name="view_static_result", **options)
    open_result.click(open_version, [project, expected, task], [*editor_outputs, editor_tabs],
                      api_name="open_static_result", **options)
    return instruction
