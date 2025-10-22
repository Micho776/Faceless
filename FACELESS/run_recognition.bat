@echo off
echo ============================================================
echo FACELESS - Live Facial Recognition (Auto-Improving)
echo ============================================================
echo Features: Confidence scoring, Multi-encoding, Logging, Learning
echo.
echo Controls:
echo   'q' - Quit
echo   's' - Save current frame for training
echo.
echo Starting recognition...
echo ============================================================
echo.

call C:\Users\miche\miniconda3\Scripts\activate.bat facerec
python recognize_live.py --model hog --track --fps --scale 0.5 --confidence 0.6 --log --learn

echo.
echo ============================================================
echo Recognition session ended
echo ============================================================
echo.

REM Check if learning samples exist
if exist "learning_samples\*" (
    echo [INFO] Learning samples detected!
    echo [INFO] Auto-improving model...
    echo.
    python improve_model.py --auto
    echo.
) else (
    echo [INFO] No new learning samples to process
)

echo ============================================================
echo Session complete!
echo ============================================================
pause
