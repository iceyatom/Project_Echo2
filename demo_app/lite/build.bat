@echo off
REM ===========================================================================
REM  Build the LITE (3-booster blend) income predictor as a standalone .exe.
REM
REM  PREREQUISITE: model.pkl must already exist in THIS folder. Generate it with:
REM      python make_model.py        (needs the full pkl at the repo root)
REM
REM  Run from inside demo_app\lite\ with the project venv active:
REM      ..\..\.venv\Scripts\activate
REM      build.bat
REM
REM  The model is NOT embedded -- after building, keep model.pkl in the SAME
REM  FOLDER as the .exe (dist\). Ship the .exe + model.pkl together.
REM  Output: dist\ProjectEcho_IncomeLite.exe  (~330 MB)
REM ===========================================================================
REM matplotlib/PIL/IPython get pulled in by sklearn/catboost but are never used
REM at predict time, so excluding them trims the .exe.
pyinstaller --clean --noconfirm --onefile --windowed ^
  --collect-all xgboost ^
  --collect-all lightgbm ^
  --collect-all catboost ^
  --collect-all sklearn ^
  --collect-all scipy ^
  --hidden-import=joblib ^
  --exclude-module matplotlib ^
  --exclude-module PIL ^
  --exclude-module IPython ^
  --name "ProjectEcho_IncomeLite" app.py

echo.
echo ============================================================
echo  Build finished.  ->  dist\ProjectEcho_IncomeLite.exe
echo  Now copy model.pkl next to the .exe:
echo      copy model.pkl dist\
echo  Then ship BOTH files (.exe + model.pkl) together.
echo ============================================================
copy /Y model.pkl dist\model.pkl >nul 2>&1
