"""Start the local meme studio or print environment diagnostics."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="个人图库表情包工作台")
    parser.add_argument("--check", action="store_true", help="检查本地环境并初始化数据库")
    parser.add_argument("--gpu", action="store_true", help="随 --check 实测 CUDA")
    parser.add_argument("--port", type=int, help="指定本机端口")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()
    if args.gpu and not args.check:
        parser.error("--gpu 需和 --check 一起使用")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")

    from meme_studio.errors import AppError

    try:
        from meme_studio.config import load_config
        from meme_studio.startup import check_gpu, prepare, render_diagnostics

        config = load_config(ROOT)
        status = prepare(config)
        if args.check:
            print(render_diagnostics(status, check_gpu() if args.gpu else None))
        else:
            from meme_studio.ui.app import launch

            launch(config, status, args.port, open_browser=not args.no_browser)
        return 0
    except AppError as exc:
        print(f"[{exc.code}] {exc.user_message}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        print(f"项目依赖尚未装齐（{exc.name}）。请检查 README 中的安装步骤。", file=sys.stderr)
        return 1
    except OSError:
        print("无法启动本地服务。请检查项目目录权限，或用 --port 指定另一个端口。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
