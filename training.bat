@echo off
python .\Code_and_model\kdd\classical_ML\classical_ML_mix.py

if %ERRORLEVEL% == 0 (
    python .\Code_and_model\cic\classical_ML\classical_ML_mix.py
) else (
    echo First script failed, second script not run.
)
pause
