@echo off
set MSBuildProjectDirectory="%~1"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist %MSBuildProjectDirectory%\obj\rwkv (
    mkdir %MSBuildProjectDirectory%\obj\rwkv
)

cd %MSBuildProjectDirectory%\obj\rwkv

cmake ..\..\..\libs\rwkv.cpp\
cmake --build . --config Release

set RWKVDllPath="%MSBuildProjectDirectory%\obj\rwkv\bin\Release\rwkv.dll"
set GGMLDllPath="%MSBuildProjectDirectory%\obj\rwkv\bin\Release\ggml.dll"

copy /V /Y %RWKVDllPath:"=% %MSBuildProjectDirectory%
copy /V /Y %GGMLDllPath:"=% %MSBuildProjectDirectory%