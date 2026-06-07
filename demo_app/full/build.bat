@echo off
REM ===========================================================================
REM  Build the FULL exp4 Stacking ensemble income predictor (--onedir = a folder).
REM
REM  The ~6.7 GB model is NOT embedded. After building, COPY the trained pkl into
REM  the output folder, next to the .exe:
REM      copy ..\..\acs2024v1.1_income_models_exp4.pkl dist\ProjectEcho_IncomeFull\
REM  (the app loads model.pkl OR acs2024v1.1_income_models_exp4.pkl from beside the exe).
REM
REM  Run from inside demo_app\full\ with the project venv active:
REM      ..\..\.venv\Scripts\activate
REM      build.bat
REM  Output: dist\ProjectEcho_IncomeFull\ProjectEcho_IncomeFull.exe  (+ the pkl you copy in)
REM ===========================================================================
pyinstaller --clean --noconfirm --onedir --windowed ^
  --collect-all xgboost ^
  --collect-all lightgbm ^
  --collect-all catboost ^
  --collect-all sklearn ^
  --collect-all scipy ^
  --hidden-import=joblib ^
  --exclude-module matplotlib ^
  --exclude-module PIL ^
  --exclude-module IPython ^
  --name "ProjectEcho_IncomeFull" app.py

echo.
echo ============================================================
echo  Build finished.  ->  dist\ProjectEcho_IncomeFull\
echo  NOW copy the trained model beside the .exe, e.g.:
echo    copy ..\..\acs2024v1.1_income_models_exp4.pkl dist\ProjectEcho_IncomeFull\
echo  Then ship the ENTIRE folder; run ProjectEcho_IncomeFull.exe inside it.
echo ============================================================
copy /Y "..\..\acs2024v1.1_income_models_exp4.pkl" "dist\ProjectEcho_IncomeFull\" >nul 2>&1
