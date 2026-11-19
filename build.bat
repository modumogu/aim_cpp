@echo off
setlocal

rem ---- paths: edit these if your install location differs ----
set "VCVARS=D:\Apps\VSstudio\VC\Auxiliary\Build\vcvars64.bat"
set "OPENCV_DIR_PATH=D:/Apps/opencv/opencv/build/x64/vc16/lib"

if not exist "%VCVARS%" (
  echo [ERROR] vcvars64.bat not found: "%VCVARS%"
  exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 (
  echo [ERROR] failed to run vcvars64.bat
  exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cl.exe not found after vcvars
  exit /b 1
)

rem ---- project root = folder of this script (strip trailing backslash) ----
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

set "BUILD=%SRC%\build"

rem ---- drop a cache left over from a failed configure ----
if exist "%BUILD%\CMakeCache.txt" (
  findstr /C:"CMAKE_CXX_COMPILER:FILEPATH=NOTFOUND" "%BUILD%\CMakeCache.txt" >nul 2>nul
  if not errorlevel 1 (
    echo [WARN] stale cache from a failed configure, removing build folder
    rmdir /s /q "%BUILD%"
  )
)

cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="%OPENCV_DIR_PATH%"
if errorlevel 1 (
  echo [WARN] configure failed, cleaning build folder and retrying once...
  rmdir /s /q "%BUILD%"
  cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="%OPENCV_DIR_PATH%"
  if errorlevel 1 (
    echo [ERROR] cmake configure failed after clean retry
    exit /b 1
  )
)

cmake --build "%BUILD%"
if errorlevel 1 (
  echo [ERROR] build failed
  exit /b 1
)

echo.
echo [OK] build finished: "%BUILD%\aim.exe"
endlocal
