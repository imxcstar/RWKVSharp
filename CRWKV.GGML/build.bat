@echo off
set MSBuildProjectDirectory="%~1"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist %MSBuildProjectDirectory%\obj\rwkv (
    mkdir %MSBuildProjectDirectory%\obj\rwkv
)

cd %MSBuildProjectDirectory%\obj\rwkv

cmake ..\..\..\libs\rwkv.cpp\ -G "Visual Studio 17 2022"
msbuild rwkv.cpp.sln /p:Configuration=Release

set RWKVDllPath="%MSBuildProjectDirectory%\obj\rwkv\bin\Release\rwkv.dll"
copy %RWKVDllPath:"=% %MSBuildProjectDirectory%