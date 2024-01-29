@echo off
python .\Code_and_model\Program\train.py --data kdd --usingMultiprocess

if %ERRORLEVEL% == 0 (
    python Code_and_model\Program\train.py --data cic --usingMultiprocess --n_Process 4
) else (
    echo First script failed, second script not run.
)
pause
