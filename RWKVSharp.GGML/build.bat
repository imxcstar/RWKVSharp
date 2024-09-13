@echo off
set MSBuildProjectDirectory="%~1"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist %MSBuildProjectDirectory%\obj\rwkv (
    mkdir %MSBuildProjectDirectory%\obj\rwkv
)

cd %MSBuildProjectDirectory%\obj\rwkv

if not exist %MSBuildProjectDirectory%\obj\rwkv\avx (
    mkdir %MSBuildProjectDirectory%\obj\rwkv\avx
)
if not exist %MSBuildProjectDirectory%\obj\rwkv\avx2 (
    mkdir %MSBuildProjectDirectory%\obj\rwkv\avx2
)
if not exist %MSBuildProjectDirectory%\obj\rwkv\avx512 (
    mkdir %MSBuildProjectDirectory%\obj\rwkv\avx512
)

cd avx
cmake -DRWKV_AVX2=OFF ..\..\..\..\libs\rwkv.cpp\
cmake --build . --config Release

cd ..
cd avx2
cmake ..\..\..\..\libs\rwkv.cpp\
cmake --build . --config Release

cd ..
cd avx512
cmake -DRWKV_AVX512=ON ..\..\..\..\libs\rwkv.cpp\
cmake --build . --config Release
