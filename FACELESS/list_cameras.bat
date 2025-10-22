@echo off
echo Scanning for available cameras...
echo.
call C:\Users\miche\miniconda3\Scripts\activate.bat facerec
python recognize_live.py --list-cameras
pause
