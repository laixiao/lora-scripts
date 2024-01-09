import argparse
import locale
import os
import platform
import subprocess
import sys
import threading
import time
import webbrowser
from auto_lora import MyApp
import requests

from mikazuki.launch_utils import prepare_environment, base_dir_path
from mikazuki.log import log

import uvicorn
import wx

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
    log.info(f"Server started at http://{args.host}:{args.port}")
    uvicorn.run("mikazuki.app:app", host=args.host, port=args.port, log_level="error")


def launch():
    global bootSuccess
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

    # import uvicorn
    # log.info(f"Server started at http://{args.host}:{args.port}")
    # uvicorn.run("mikazuki.app:app", host=args.host, port=args.port, log_level="error")

    # 创建并启动 UVICORN 服务器的线程
    # 检测已启动则不再启动
    try:
        response = requests.post(
            "http://127.0.0.1:28000/api/tasks",
            headers={
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
                "Content-Type": "application/json",
                "Cookie": "_ga=GA1.1.787707760.1702920485; _ga_R1FN4KJKJH=GS1.1.1704301128.3.0.1704301128.0.0.0",
                "Origin": "http://127.0.0.1:28000",
                "Referer": "http://127.0.0.1:28000/lora/master.html",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            },
            data="",
        )
        bootSuccess = True
        print(response)
        print("启动成功")
        print("请拖入图片文件夹开始训练")
    except Exception as e:
        # 请求不通才创建
        if not args.disable_tageditor:
            run_tag_editor()

        if not args.disable_tensorboard:
            run_tensorboard()

        uvicorn_thread = threading.Thread(target=run_uvicorn)
        uvicorn_thread.start()


bootSuccess = False


def on_close(self):
    uvicorn.Server.should_exit = True
    # 关闭UI窗口
    self.Destroy()
    # 退出整个应用程序
    sys.exit()


def delayed_print():
    time.sleep(1)  # 等待3秒
    print("启动成功")
    print("请拖入图片文件夹开始训练")


if __name__ == "__main__":
    args, _ = parser.parse_known_args()

    launch()

    if bootSuccess:
        # 创建并启动线程
        threading.Thread(target=delayed_print).start()

    print("打开ui")
    app = MyApp()
    # app.frame.Bind(wx.EVT_CLOSE, on_close)
    app.MainLoop()
