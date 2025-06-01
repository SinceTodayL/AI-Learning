@echo off
setlocal enabledelayedexpansion

:: 设置最大文件大小（单位为字节）100MB = 104857600
set "MAX_SIZE=104857600"

:: 切换到本地仓库目录
cd /d "E:\AI Learning"

:: 添加所有更改
git add .

:: 检查是否有超过 100MB 的暂存文件
for /f "tokens=*" %%f in ('git diff --cached --name-only') do (
    for %%A in ("%%f") do (
        set "FILESIZE=%%~zA"
        if !FILESIZE! gtr %MAX_SIZE% (
            echo [×] 文件 "%%f" 的大小为 !FILESIZE! 字节，已超过 100MB 限制！
            echo [！] 请移除该文件或使用 Git LFS。
            git reset "%%f"
            goto :end
        )
    )
)

:: 检查是否有暂存更改
git diff --cached --quiet
if errorlevel 1 (
    for /f %%i in ('powershell -command "Get-Date -Format \"yyyyMMdd\""') do set timestamp=%%i
    git commit -m "update at !timestamp!"
    git push origin main
    echo [✓] 提交成功并已推送到 main 分支（时间：!timestamp!）。
) else (
    echo [–] 没有变更内容，未提交。
)

:end
pause
