import asyncio
import mimetypes
import os
import webbrowser
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mikazuki.utils.devices import check_torch_gpu
from mikazuki.app.api import router as api_router
from mikazuki.app.proxy import router as proxy_router

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


async def app_startup():
    await asyncio.to_thread(check_torch_gpu)
    if sys.platform == "win32":
        print(f'启动成功： http://{os.environ["MIKAZUKI_HOST"]}:{os.environ["MIKAZUKI_PORT"]}')
        print("请拖入图片文件夹开始训练")
        # webbrowser.open(f'http://{os.environ["MIKAZUKI_HOST"]}:{os.environ["MIKAZUKI_PORT"]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)


cors_config = os.environ.get("MIKAZUKI_APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8004"]
    else:
        cors_config = cors_config.split(";")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response

app.include_router(api_router, prefix="/api")


@app.get("/")
async def index():
    return FileResponse("./frontend/dist/index.html")


app.mount("/", StaticFiles(directory="frontend/dist"), name="static")
