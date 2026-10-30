@echo off
rem Viewer.py 用 MomoGraph.xxx 的包路径导入，必须把项目根目录加进 PYTHONPATH，
rem 否则在 DashBoard 目录下直接跑会报 No module named 'MomoGraph'
set "ROOT=E:\Study\GitProject"
set "PYTHONPATH=%ROOT%"
"%ROOT%\MomoGraph\.venv\Scripts\python.exe" "%ROOT%\MomoGraph\DashBoard\Viewer.py" %*
pause
