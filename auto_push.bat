@echo off
setlocal enabledelayedexpansion

:: ======== Auto Push Script for https://github.com/SinceTodayL/AI-Learning ========

:: Change to local repo directory
cd /d "E:\AI Learning"

:: Stage all changes
git add .

:: Check if there are any staged changes
git diff --cached --quiet
if errorlevel 1 (
    :: Get current date in yyyyMMdd format
    for /f %%i in ('powershell -command "Get-Date -Format \"yyyyMMdd\""') do set timestamp=%%i

    :: Commit with timestamp
    git commit -m "commit at !timestamp!"
    
    :: Push to main branch
    git push origin main
    echo [✓] Changes detected. Committed and pushed to main at !timestamp!.
) else (
    echo [–] No changes detected. Nothing to commit.
)
