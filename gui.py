import argparse
import locale
import os
import platform
import subprocess
import sys
import threading
import webbrowser
from auto_lora import MyApp

from mikazuki.launch_utils import prepare_environment, base_dir_path
from mikazuki.log import log

parser = argparse.ArgumentParser(description="GUI for stable diffusion training")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=28000, help="Port to run the server on")
parser.add_argument("--listen", action="store_true")
parser.add_argument("--skip-prepare-environment", action="store_true")
parser.add_argument("--disable-tensorboard", action="store_true")
parser.add_argument("--disable-tageditor", action="store_true")
parser.add_argument(
    "--tensorboard-host",
    type=str,
    default="127.0.0.1",
    help="Port to run the tensorboard",
)
parser.add_argument(
    "--tensorboard-port", type=int, default=6006, help="Port to run the tensorboard"
)
parser.add_argument("--localization", type=str)
parser.add_argument("--dev", action="store_true")


def run_tensorboard():
    log.info("Starting tensorboard...")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            "logs",
            "--host",
            args.tensorboard_host,
            "--port",
            str(args.tensorboard_port),
        ]
    )


def run_tag_editor():
    log.info("Starting tageditor...")
    cmd = [
        sys.executable,
        base_dir_path() / "mikazuki/dataset-tag-editor/scripts/launch.py",
        "--port",
        "28001",
        "--shadow-gradio-output",
        "--root-path",
        "/proxy/tageditor",
    ]
    if args.localization:
        cmd.extend(["--localization", args.localization])
    elif locale.getdefaultlocale()[0].startswith("zh"):
        cmd.extend(["--localization", "zh-Hans"])
    subprocess.Popen(cmd)


def run_uvicorn():
    import uvicorn

    uvicorn.run("mikazuki.app:app", host=args.host, port=args.port, log_level="error")

    


def launch():
    log.info("Starting SD-Trainer Mikazuki GUI...")
    log.info(f"Base directory: {base_dir_path()}, Working directory: {os.getcwd()}")
    log.info(f"{platform.system()} Python {platform.python_version()} {sys.executable}")

    if not args.skip_prepare_environment:
        prepare_environment()

    os.environ["MIKAZUKI_HOST"] = args.host
    os.environ["MIKAZUKI_PORT"] = str(args.port)
    os.environ["MIKAZUKI_TENSORBOARD_HOST"] = args.tensorboard_host
    os.environ["MIKAZUKI_TENSORBOARD_PORT"] = str(args.tensorboard_port)

    if args.listen:
        args.host = "0.0.0.0"
        args.tensorboard_host = "0.0.0.0"

    if not args.disable_tageditor:
        run_tag_editor()

    if not args.disable_tensorboard:
        run_tensorboard()

    import uvicorn

    log.info(f"Server started at http://{args.host}:{args.port}")
    # uvicorn.run("mikazuki.app:app", host=args.host, port=args.port, log_level="error")

    # 创建并启动 UVICORN 服务器的线程
    uvicorn_thread = threading.Thread(target=run_uvicorn)
    uvicorn_thread.start()

    print("打开ui")
    app = MyApp()
    # app.frame.Bind(wx.EVT_CLOSE, on_close)
    app.MainLoop()


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    launch()
