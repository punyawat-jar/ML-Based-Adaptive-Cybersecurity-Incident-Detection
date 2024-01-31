@echo off
python Code_and_model\Program\train.py --data cic --n_Process 2 --multiProcess
if %errorlevel% neq 0 exit /b %errorlevel%
python Code_and_model\Program\test.py --data cic
