@echo off
echo ========================================
echo Starting acacia.chat Remote Dev Services
echo ========================================
echo.

echo [0/3] Stopping existing services...
taskkill /F /IM ttyd.exe 2>nul
taskkill /F /IM caddy.exe 2>nul
taskkill /F /IM cloudflared.exe 2>nul
echo Killing persistent PID 5056...
taskkill /F /PID 5056 2>nul
echo Killing any Caddy admin processes on port 2020...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :2020') do taskkill /F /PID %%a 2>nul
echo Checking for port conflicts...
netstat -ano | findstr :80
netstat -ano | findstr :2020
echo Waiting for processes to terminate...
timeout /t 5 /nobreak > nul

echo [1/3] Starting ttyd (Web Terminal)...
echo Checking port 7682...
netstat -ano | findstr 7682
echo Killing any processes using port 7682...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7682') do taskkill /F /PID %%a 2>nul
timeout /t 2 /nobreak > nul
echo Starting ttyd with correct WSL distribution name...
echo Using Ubuntu-24.04 distribution...
start /B C:\Users\ttyr6\ttyd.exe -W -p 7682 -i 127.0.0.1 wsl.exe -d Ubuntu-24.04 bash -l
timeout /t 3 /nobreak > nul
echo Checking if ttyd started successfully...
netstat -ano | findstr 7682 | findstr LISTENING
if %ERRORLEVEL% NEQ 0 (
    echo ttyd failed to start on 7682, trying fallback...
    start /B C:\Users\ttyr6\ttyd.exe -W -p 7682 -i 127.0.0.1 wsl.exe bash
    timeout /t 2 /nobreak > nul
)

echo [2/3] Starting Caddy (Reverse Proxy)...
cd /d C:\Users\ttyr6
echo Checking and killing existing Caddy processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :2020') do (
    echo Killing Caddy process PID %%a...
    taskkill /F /PID %%a 2>nul
)
timeout /t 2 /nobreak > nul
echo Starting Caddy with error handling...
start /B C:\Users\ttyr6\caddy\caddy.exe run --config C:\Users\ttyr6\Caddyfile --adapter caddyfile
timeout /t 3 /nobreak > nul
echo Verifying Caddy started...
netstat -ano | findstr :2020

echo [3/3] Restarting Cloudflared with updated config...
taskkill /F /IM cloudflared.exe 2>nul
timeout /t 2 /nobreak > nul
powershell -Command "Start-Process -FilePath 'C:\Users\ttyr6\cloudflared.exe' -ArgumentList 'tunnel', 'run', 'acacia-dev-tunnel' -WindowStyle Hidden"
echo Cloudflared restarted with new port configuration!

echo.
echo ========================================
echo All services started successfully!
echo ========================================
echo.
echo Access your development environment at:
echo - Code Editor: https://code.acacia.chat
echo - Terminal: https://terminal.acacia.chat
echo - Dev Page: https://dev.acacia.chat
echo.
echo IMPORTANT: Access terminal.acacia.chat FIRST to start WSL!
echo.
pause