param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
& $Python -m venv (Join-Path $PSScriptRoot '.venv')
if ($LASTEXITCODE) { throw '创建训练环境失败' }
& (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE) { throw '安装训练依赖失败' }
