@echo off

setlocal enabledelayedexpansion

REM Read the links.txt file line by line and download each dataset
for /F "delims=" %%i in (links.txt) do (
    curl -O "%%i"
)
