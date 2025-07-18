@echo off
echo Starting AI Event Digest...
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
)

REM Run the Python script
python meetup_ai_notifier.py

REM Log completion
echo Digest completed at %date% %time%
pause
