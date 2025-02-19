""" 
安装依赖：
.\venv\Scripts\activate

.\venv\Scripts\python.exe -m pip install wxPython

.\venv\Scripts\python.exe gui.py

"""

import json
import os
import queue
import subprocess
import threading
import time
import requests
import wx
import sys
import toml
import os


def find_default_toml_in_parent_of_current_project():
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)

    # 获取当前脚本所在的目录（即项目目录）
    project_dir = os.path.dirname(current_script_path)

    # 获取项目目录的上一层目录
    parent_dir = os.path.dirname(project_dir)

    # 检查在上一层目录中是否存在default.toml文件
    default_toml_path = os.path.join(parent_dir, "default.toml")
    if os.path.isfile(default_toml_path):
        return default_toml_path
    else:
        return None


# 调用函数
path_to_default_toml = find_default_toml_in_parent_of_current_project()


def read_config_from_toml(file_path):
    try:
        # 打开并读取TOML文件
        with open(file_path, "r", encoding="utf-8") as file:
            config = toml.load(file)
            return config
    except FileNotFoundError:
        print("默认配置参数文件未找到")
        return None
    except toml.TomlDecodeError:
        print("默认配置参数TOML文件格式错误")
        return None


# 使用示例
tomlConfig = read_config_from_toml(path_to_default_toml)

# if config is not None:
#     print(config)  # 打印配置信息


def boot():
    app = MyApp()
    app.MainLoop()


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame("Lora训练工具", (50, 60), (960, 640))
        self.frame.Show()
        return True


class FileDrop(wx.FileDropTarget):
    def __init__(self, window):
        super(FileDrop, self).__init__()
        self.window = window

    def OnDropFiles(self, x, y, folderPaths):
        success = False
        for folderPath in folderPaths:
            if os.path.isdir(folderPath):
                self.handleFolder(folderPath)
                success = True
            else:
                wx.MessageBox(f"{folderPath} 不是文件夹", "错误", wx.ICON_ERROR)
        return success

    def handleFolder(self, folderPath):
        hasImages = False
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    hasImages = True
                    break
            if hasImages:
                self.window.OutputFolder(root)
                hasImages = False  # 重置标志以检查下一个文件夹

        self.window.log_message(f"--- ")
        self.window.start_training_folder()


# 监听文件是否存在，存在则返回1
# # 使用示例
# def file_found(result):
#   print("文件找到了，返回值：", result)
# watcher = FileWatcher("train_data_dir", "fileName", self.file_found)
# watcher.start()
class FileWatcher:
    def __init__(self, directory, filename, callback):
        self.directory = directory
        self.filename = filename
        self.callback = callback
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        while self.running:
            full_path = os.path.join(self.directory, self.filename)
            if os.path.exists(full_path):
                # 5s后再执行下一个
                time.sleep(5)
                self.callback(1)
                self.running = False
            time.sleep(1)

    def stop(self):
        self.running = False
        self.thread.join()


class TrainingClient:
    def __init__(self, custom_data=None):
        global tomlConfig

        self.url = "http://127.0.0.1:28000/api/run"
        self.headers = {
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
        }

        self.data = {
            "model_train_type": tomlConfig.get("model_train_type", "sd-lora"),
            "pretrained_model_name_or_path": tomlConfig.get(
                "pretrained_model_name_or_path",
                "./sd-models/absolutereality_v16.safetensors",
            ),
            "v2": tomlConfig.get("v2", "false"),
            "train_data_dir": tomlConfig.get("train_data_dir", "./train/test"),
            "prior_loss_weight": tomlConfig.get("prior_loss_weight", 1),
            "resolution": tomlConfig.get("resolution", "512,512"),
            "enable_bucket": tomlConfig.get("enable_bucket", "true"),
            "min_bucket_reso": tomlConfig.get("min_bucket_reso", 256),
            "max_bucket_reso": tomlConfig.get("max_bucket_reso", 1024),
            "bucket_reso_steps": tomlConfig.get("bucket_reso_steps", 64),
            "output_name": tomlConfig.get("output_name", "aki"),
            "output_dir": tomlConfig.get("output_dir", "./output"),
            "save_model_as": tomlConfig.get("save_model_as", "safetensors"),
            "save_precision": tomlConfig.get("save_precision", "fp16"),
            "save_every_n_epochs": tomlConfig.get("save_every_n_epochs", 2),
            "max_train_epochs": tomlConfig.get("max_train_epochs", 10),
            "train_batch_size": tomlConfig.get("train_batch_size", 1),
            "gradient_checkpointing": tomlConfig.get("gradient_checkpointing", "false"),
            "network_train_unet_only": tomlConfig.get(
                "network_train_unet_only", "false"
            ),
            "network_train_text_encoder_only": tomlConfig.get(
                "network_train_text_encoder_only", "false"
            ),
            "learning_rate": tomlConfig.get("learning_rate", 0.0001),
            "unet_lr": tomlConfig.get("unet_lr", 0.0001),
            "text_encoder_lr": tomlConfig.get("text_encoder_lr", 0.00001),
            "lr_scheduler": tomlConfig.get("lr_scheduler", "cosine_with_restarts"),
            "lr_warmup_steps": tomlConfig.get("lr_warmup_steps", 0),
            "lr_scheduler_num_cycles": tomlConfig.get("lr_scheduler_num_cycles", 1),
            "optimizer_type": tomlConfig.get("optimizer_type", "AdamW8bit"),
            "network_module": tomlConfig.get("network_module", "networks.lora"),
            "network_dim": tomlConfig.get("network_dim", 32),
            "network_alpha": tomlConfig.get("network_alpha", 32),
            "log_with": tomlConfig.get("log_with", "tensorboard"),
            "logging_dir": tomlConfig.get("logging_dir", "./logs"),
            "caption_extension": tomlConfig.get("caption_extension", ".txt"),
            "shuffle_caption": tomlConfig.get("shuffle_caption", "true"),
            "keep_tokens": tomlConfig.get("keep_tokens", 0),
            "max_token_length": tomlConfig.get("max_token_length", 255),
            "seed": tomlConfig.get("seed", 1337),
            "clip_skip": tomlConfig.get("clip_skip", 2),
            "mixed_precision": tomlConfig.get("mixed_precision", "fp16"),
            "xformers": tomlConfig.get("xformers", "true"),
            "lowram": tomlConfig.get("lowram", "false"),
            "cache_latents": tomlConfig.get("cache_latents", "true"),
            "cache_latents_to_disk": tomlConfig.get("cache_latents_to_disk", "true"),
            "cache_text_encoder_outputs": tomlConfig.get(
                "cache_text_encoder_outputs", "false"
            ),
            "cache_text_encoder_outputs_to_disk": tomlConfig.get(
                "cache_text_encoder_outputs_to_disk", "false"
            ),
            "persistent_data_loader_workers": tomlConfig.get(
                "persistent_data_loader_workers", "true"
            ),
            "gpu_ids": tomlConfig.get("gpu_ids", ["0"]),
        }

        if custom_data:
            self.data.update(custom_data)

        print("=======训练参数=======")
        for key, value in self.data.items():
            print(key, value)

    def post_request(self):
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(self.data)
        )
        return response

    def start_training(self):
        thread = threading.Thread(target=self.send_training_request)
        thread.start()

    def send_training_request(self):
        try:
            response = requests.post(
                self.url, headers=self.headers, data=json.dumps(self.data)
            )
            if response.status_code == 200:
                print("训练状态：", response.json())
            else:
                print("开启训练失败：", response.status_code)

        except Exception as e:
            print(f"开始训练失败：{e}")


class TextCtrlRedirector:
    def __init__(self, text_ctrl):
        self.text_ctrl = text_ctrl

    def write(self, text):
        self.text_ctrl.AppendText(text)

    def flush(self):
        pass

    def isatty(self):
        return False


class MyFrame(wx.Frame):
    def __init__(self, title, pos, size):
        super(MyFrame, self).__init__(None, -1, title, pos, size)

        self.folderQueue = queue.Queue()  # 在这里定义队列

        # # 创建菜单栏
        # menuBar = wx.MenuBar()
        # settingsMenu = wx.Menu()
        # item = settingsMenu.Append(wx.ID_ANY, "训练参数", "配置默认训练参数")
        # self.Bind(wx.EVT_MENU, self.OnSettings, item)
        # menuBar.Append(settingsMenu, "选项")
        # self.SetMenuBar(menuBar)

        # 设置拖放
        dropTarget = FileDrop(self)
        self.SetDropTarget(dropTarget)

        # 创建一个多行的TextCtrl，用作日志输出
        self.logTextCtrl = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL
        )
        # 使用sizer来管理布局
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.logTextCtrl, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)

        # 拖入图片文件夹开始训练Lora
        # self.helloText = wx.StaticText(self, label="拖入图片文件夹开始训练Lora", pos=(10, 100))

        self.log_message(f"请稍后，启动中...")

        # 将命令行输出重定向到logTextCtrl
        sys.stdout = TextCtrlRedirector(self.logTextCtrl)

    def log_message(self, message):
        """向日志控件中添加一条消息，并滚动到最新消息"""
        self.logTextCtrl.AppendText(message + "\n")
        self.logTextCtrl.ShowPosition(self.logTextCtrl.GetLastPosition())

    def OutputFolder(self, folderPath):
        self.log_message(f"排队-lora: {folderPath}")
        self.folderQueue.put(folderPath)  # 将文件夹路径添加到队列中

    def start_training_folder(self):
        if not self.folderQueue.empty():
            folderPath = self.folderQueue.get()
            folderName = os.path.basename(folderPath)
            self.log_message(f"开始训练Lora: {folderPath}")
            # print(f"Lora训练中: {folderPath}")
            # 启动训练逻辑...
            # 训练完成后需要调用self.start_training_folder()来处理下一个文件夹
            custom_data = {
                "train_data_dir": folderPath,
                "output_dir": folderPath,
                "output_name": folderName,
                "save_model_as": "safetensors",
            }
            client = TrainingClient(custom_data=custom_data)
            client.start_training()

            watcher = FileWatcher(
                custom_data.get("output_dir"),
                custom_data.get("output_name") + "." + custom_data.get("save_model_as"),
                self.file_found,
            )
            watcher.start()
        else:
            print("全部Lora训练完成")
            self.log_message(f"全部Lora训练完成")
            self.log_message(f"再次拖入图片文件夹开始训练Lora")
            self.log_message(f"-----------------")

    def file_found(self, result):
        print(f"Lora训练完成：{result}")
        self.log_message(f"Lora训练完成：{result}")
        wx.CallAfter(self.start_training_folder)  # 处理队列中的下一个文件夹

    def OnSettings(self, event):
        wx.MessageBox("配置默认训练参数", "设置")


# if __name__ == "__main__":
#     app = MyApp()
#     app.MainLoop()
