@echo off
setlocal

echo Building Rust Native Module for Vault Reconstruct...
echo.
echo IMPORTANT: TDM-GCC has known incompatibilities with modern Rust on Windows,
echo causing Error 193 when building build-scripts, and dlltool errors with spaces in paths.
echo This script temporarily adjusts the environment to bypass these issues.
echo.

:: 1. Remove TDM-GCC from PATH to prevent rustc from using its broken DLLs or gcc.exe for build scripts.
set "PATH=%PATH:C:\TDM-GCC-64\bin;=%"
set "PATH=%PATH:C:\TDM-GCC-64\bin=%"

:: 2. Provide a shim for dlltool.exe and as.exe (copied from TDM-GCC)
:: We created c:\dlltool_shim with dlltool.exe and as.exe
set "PATH=%PATH%;c:\dlltool_shim"

:: 3. Use a target directory WITHOUT spaces to prevent TDM-GCC's dlltool.exe from failing.
set "CARGO_TARGET_DIR=C:\cargo_target\reconstruct_rust"

:: 4. Explicitly tell the cc crate to use TDM-GCC for compiling C/C++ dependencies (like ring, etc).
set "CC=C:\TDM-GCC-64\bin\gcc.exe"
set "CXX=C:\TDM-GCC-64\bin\g++.exe"
set "AR=C:\TDM-GCC-64\bin\ar.exe"

:: 5. Tell rustc to use its internal lld linker since gcc is out of PATH
set "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=rust-lld.exe"
set "CARGO_BUILD_LINKER=rust-lld.exe"

echo Environment configured successfully.
echo Running maturin develop...
uv run maturin develop %*
