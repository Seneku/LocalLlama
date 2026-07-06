@echo off
setlocal

cd /d "%~dp0.."
set "LLAMATUNER_PORT=4187"

bun server/index.ts >> ".llamatuner-server.log" 2>&1

endlocal
