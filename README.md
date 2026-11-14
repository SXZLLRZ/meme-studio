# 个性化静态与动态表情包系统

当前实现 **P7 整合验收与课程交付**。真实静态到动态路径和云端文字复测已通过，见阶段记录。

## 首次安装

使用 Windows x64 和 Python 3.11。在下载并解压的仓库根目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip check
.\start.cmd --check
.\start.cmd
```

基础安装可以导入图片、手工选区、本机配字、管理历史和导出 PNG。中文语义检索、SAM 点选分割和视频处理分别需要本地模型及 FFmpeg，准备方法见 [安装复现](docs/安装复现.md)。权重和 FFmpeg 不随源码上传。

云端功能需要自己的密钥。先复制空模板，再在本机编辑 `.env`：

```powershell
Copy-Item -LiteralPath .env.example -Destination .env
```

已有 `.env` 时直接使用原文件，不重复复制。未配置密钥也可使用本地功能。

## 启动

在此项目目录双击 `start.cmd`，或在 PowerShell 中运行：

```powershell
.\start.cmd
```

默认仅监听本机，启动时会尝试打开浏览器。也可以打开控制台显示的地址，通常是 http://127.0.0.1:7860。运行期间保留控制台，按 Ctrl+C 停止。

如果需要指定端口或手动打开浏览器：

```powershell
.\start.cmd --port 7861 --no-browser
```

打开页面的「本地状态」查看 Python、SQLite、数据库版本及密钥是否已配置。缺少模型、视频工具或密钥时，页面会显示相应提示。

## 使用图库

1. 在「图库与检索」打开默认图库，点击「导入目录」读取仓库自带的 `demo/gallery` 合成图片，或更改为自己的图库目录。也可多选本地文件后点「导入已选文件」。
2. 查看导入统计和索引状态，输入中文描述，点击「中文检索」。点击卡片查看所选图片 ID。
3. 可给图片保存标签、按标签过滤、移出当前图库或建立另一册。缺少模型时可以用「浏览全部」手动选图。
4. 用同一个目录再导入一次，正常应显示新增 0，索引新增编码 0。重启后仍可使用已建索引。

模型安装、尺寸上限、复测记录和重建方法见 [P2 阶段记录](docs/P2-阶段记录.md)。

## 使用本地编辑

1. 在图库点击已导入的照片，再点击「用所选图创建编辑项目」。
2. 在「本地配字」填写文案，预览后点「保存文字版本」。在「局部选区」可点击两个对角，或用画笔和橡皮修改蒙版。
3. 在「历史与导出」撤销或恢复版本，点击「导出 PNG」。从旧版本再次保存会创建分支。
4. 服务重启后，在「编辑与历史」打开原项目，确认当前文字和选区恢复。

坐标、字体和 PNG 复测方法见 [P3 阶段记录](docs/P3-阶段记录.md)。

## 使用静态生成

在编辑项目中进入「静态生成」。局部修改需先保存选区，并选择「使用已保存选区」。填写说明后明确提交 1 张，平台可能计费。任务保存成功后可查看或打开结果版本。需要停止等待时可暂停，稍后继续查询原任务。

详见 [P4 阶段记录](docs/P4-阶段记录.md)。

## 使用 SAM 点选和合并指令

- 在「局部选区 → SAM 点选主体」标正点、负点或两角框，生成候选并保存，可用画笔和橡皮进一步修正。
- 在「图库与检索 → 合并指令」输入一句话，点「本机拆解」，分别核对检索描述和编辑要求。选图创建项目后，编辑要求会带入静态生成草稿。云端文字拆解由独立按钮触发。
- 在「保留要求」管理随版本保存的条目，在「静态生成」点击「预览本轮请求」核对范围、适用要求与替换。模型安装、规则和复测步骤见 [P5 阶段记录](docs/P5-阶段记录.md)。

## 使用动态生成和视频导出

- 在「动态与导出 → 生成与任务」核对静态锚点，输入动作后明确提交 1 段视频。默认 3 秒、720P、无声。
- 打开已完成的结果版本，在「视频配字和导出」保存文案，预览或下载 MP4，也可调整 GIF 的长边、帧率、目标体积和截取时长。
- 只改文字时复用原无字视频。更新动作时，从同一个静态锚点生成新版本。详见 [P6 阶段记录](docs/P6-阶段记录.md)。

## 测试

在项目目录运行：

```powershell
.\start.cmd --check
.\start.cmd --check --gpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

- 第一个命令检查必需和可选依赖，初始化或打开 SQLite，并输出本地状态。
- 第二个命令额外运行 CUDA 张量运算和 torchvision 算子。通过时输出 RTX 4060 Laptop、CUDA 12.8、平方和 30 和 NMS 索引 [0, 1]。
- 33 项离线测试覆盖图库、任务恢复、指令解析、动态导出和双页面选区隔离。测试使用临时数据、合成媒体和 HTTP 模拟接口。

启动、图库检索、配字、选区、导出和自动测试在本机完成。静态生成由用户明确提交。

## 配置与数据

- 普通设置保存在 `settings.json`。相对目录以项目根目录为基准。默认图库为 `demo/gallery`，包含可分享的合成示例，可在页面或配置中更换。
- 同名配置按环境变量、`.env`、`settings.json` 的顺序读取。可用 MEME_STUDIO_GALLERY_DIR 指定图库，用 MEME_STUDIO_RUNTIME_DIR 指定项目内的数据子目录。
- API 密钥保存在本地 `.env`，或通过环境变量 DASHSCOPE_API_KEY 提供。启动只显示是否存在密钥。
- 数据库保存在 `runtime/app.sqlite3`，应用日志在 `runtime/logs/app.log`。分享源码时排除 `.env`、`.venv/`、`runtime/` 和模型权重。

## 重建独立环境

在项目目录用 Windows x64 Python 3.11 创建虚拟环境。锁定版本及可选工具的安装方法见 [安装复现](docs/安装复现.md)。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\start.cmd --check
```

若虚拟环境在安装 pip 时遇到系统临时目录权限问题，可将当前 PowerShell 的 TEMP 和 TMP 指向项目内的 `runtime/` 下的临时目录，再运行 `.\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip`。CUDA 配对以 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v280) 为准。

## 开发文档与既有验证

- [开发总说明](docs/开发总说明.md)
- [开发任务清单](docs/开发任务清单.md)
- [P1 交接记录](docs/P1-阶段记录.md)

P0 至 P7 的本机运行记录保存在开发机 `runtime/`，不随公开仓库上传。仓库提供阶段说明及 [合成演示素材](demo/README.md)，历史媒体只作已完成产物展示。P7 详见 [阶段记录](docs/P7-阶段记录.md)、[课程报告](docs/课程报告.md) 和 [演示与答辩](docs/演示与答辩.md)。

## 目录与源码交付

```text
app.py                 应用入口
start.cmd              Windows 启动与检查
.env.example           空密钥模板
settings.json          可移植的普通配置
src/meme_studio/        检索 分割 编辑 版本 云端任务和导出
tests/                 P1 至 P7 的 33 项离线测试
docs/                  安装 技术说明 阶段记录与演示步骤
demo/gallery/          合成测试原图
demo/media/            合成素材的蒙版 静态结果与历史视频
scripts/               源码打包脚本
```

公开仓库不包含真实密钥、个人图库、运行数据库、虚拟环境和模型权重。`.gitignore` 排除这些内容及 Python 缓存。

在项目目录运行以下命令，可用仓库文件直接生成带 SHA-256 清单的源码包：

```powershell
.\.venv\Scripts\python.exe scripts/package_release.py dist/meme-studio-source.zip
```

当前仓库整理与验证范围见 [仓库交付记录](docs/仓库交付记录.md)。课程组别、成员及实际分工由提交者在正式结课报告中填写。
