@echo off
REM Reproduces every table, figure and LaTeX number macro (about 8 minutes). Run from this folder.
cd /d "%~dp0"
if not exist ..\logs mkdir ..\logs
for %%s in (00_build_dataset 05_changepoint 10_q1_lesson_learning 20_q2_exit_rates 21_q2_hazard 22_q2_where 30_q3_persistence_return 31_q3_concentration 40_make_figures 50_make_report_inputs) do (
  echo === %%s
  python %%s.py > ..\logs\%%s.txt 2>&1
  if errorlevel 1 (
    echo FAILED at %%s - see ..\logs\%%s.txt
    exit /b 1
  )
)
echo ALL DONE
