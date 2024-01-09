@echo off

call .\venv\Scripts\activate
set HF_HOME=huggingface
set PYTHONUTF8=1

python gui.py

pause