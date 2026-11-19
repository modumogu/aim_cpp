@echo off
setlocal

rem One-click: build then run.
rem   run.bat
rem   run.bat --expo -6
rem   run.bat --video D:\path\video.mp4 --save out.png

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

echo ============================================
echo   1/2  building...
echo ============================================
call "%SRC%\build.bat"
if errorlevel 1 (
  echo.
  echo [ERROR] build failed, see messages above
  goto :end
)

echo.
echo ============================================
echo   2/2  running %SRC%\build\aim.exe
echo ============================================
"%SRC%\build\aim.exe" %*

:end
echo.
echo -- done --
pause
