@echo off
setlocal
cd /d "%~dp0.."

if "%LLAMATUNER_PORT%"=="" set "LLAMATUNER_PORT=4187"

echo Stopping any LlamaTuner on port %LLAMATUNER_PORT% (and its llama.cpp children)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%LLAMATUNER_PORT% ^| findstr LISTENING') do taskkill /F /T /PID %%a >nul 2>&1

echo Building fresh frontend + type-checking...
call bun run build
if errorlevel 1 (
  echo Build failed - not restarting. Fix the errors above and run this again.
  exit /b 1
)

echo Starting LlamaTuner...
start "LlamaTuner" /min cmd /c "bun server/index.ts >> .llamatuner-server.log 2>&1"

echo.
echo LlamaTuner restarted at http://127.0.0.1:%LLAMATUNER_PORT%
echo (hard-refresh the browser with Ctrl+F5 to drop the cached bundle)
endlocal
