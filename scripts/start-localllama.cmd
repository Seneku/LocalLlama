@echo off
setlocal

cd /d "%~dp0.."
set "LOCALLLAMA_PORT=4187"

bun server/index.ts >> ".localllama-server.log" 2>&1

endlocal
