param([string]$Root)
$ErrorActionPreference='Stop'
try {
    Write-Host 'Posegraph Live 开发版安装' -ForegroundColor Cyan
    Write-Host '请先在插件管理器停用原版 Posegraph，再退出游戏。Brio 保持安装。'
    if (Get-Process ffxiv_dx11 -ErrorAction SilentlyContinue) { throw '请完全退出游戏，再重新运行本安装器。' }
    if (-not $Root) {
        Add-Type -AssemblyName System.Windows.Forms
        $dialog=New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description='选择已有 Posegraph 后端根目录（里面有 python 和 backend）；新用户选择准备安装的目录'
        if ($dialog.ShowDialog() -ne 'OK') { exit }
        $Root=$dialog.SelectedPath
    }
    $python=Join-Path $Root 'python\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host '首次安装：将启动原项目安装器，按提示选择显卡和网络方式。'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'base-installer\Install-Posegraph.ps1') -Root $Root -NoWait -SkipPluginConfig
        if ($LASTEXITCODE -ne 0) { throw '基础后端安装失败，请检查日志后重试。' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Root 'backend\src\posegraph\app.py'))) { throw '选中的目录不是完整 Posegraph 后端。' }
    & $python (Join-Path $PSScriptRoot 'install_live.py') --root $Root
    if ($LASTEXITCODE -ne 0) { throw 'Live 安装未完成。请保留错误信息。' }
    Write-Host '安装完成。启动游戏，使用 /posegraph，点击 Live 图标。' -ForegroundColor Green
} catch { Write-Host $_ -ForegroundColor Red; exit 1 }
