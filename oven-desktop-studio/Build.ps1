param(
    [Parameter(Mandatory = $true)]
    [string]$OriginalExe
)

$ErrorActionPreference = 'Stop'
$OriginalExe = (Resolve-Path -LiteralPath $OriginalExe).Path
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$build = Join-Path $PSScriptRoot 'build'
New-Item -ItemType Directory -Path $build -Force | Out-Null
$assembly = [Reflection.Assembly]::Load([IO.File]::ReadAllBytes($OriginalExe))
foreach ($name in @('OvenDesk.DesktopAI.zip', 'OvenDesk.WebP.zip')) {
    $resource = $assembly.GetManifestResourceStream($name)
    if ($null -eq $resource) { throw "Missing resource: $name" }
    try {
        $file = [IO.File]::Create((Join-Path $build $name))
        try { $resource.CopyTo($file) } finally { $file.Dispose() }
    } finally { $resource.Dispose() }
}
$aiZip = Join-Path $build 'OvenDesk.DesktopAI.zip'
$archive = [IO.Compression.ZipFile]::Open($aiZip, 'Update')
try {
    foreach ($name in @('infer_desktop.py', 'camera_worker.py', 'food.json',
                         'oven_food/__init__.py', 'oven_food/augment.py', 'oven_food/manifest.py')) {
        $entry = $archive.GetEntry($name)
        if ($null -ne $entry) { $entry.Delete() }
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, (Join-Path $PSScriptRoot $name), $name
        ) | Out-Null
    }
} finally { $archive.Dispose() }

$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$output = Join-Path $build 'OvenDesk-Optimized.exe'
$sources = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'desktop') -Filter '*.cs' |
    ForEach-Object FullName)
$references = @('System.Windows.Forms.dll', 'System.Drawing.dll', 'System.Web.Extensions.dll', 'System.Net.Http.dll',
                'System.IO.Compression.dll', 'System.IO.Compression.FileSystem.dll') |
    ForEach-Object { "/reference:$_" }
& $compiler /nologo /target:winexe /platform:x64 /codepage:65001 "/out:$output" @references "/resource:$(Join-Path $PSScriptRoot 'food.json'),OvenDesk.Food.json" `
    "/resource:$aiZip,OvenDesk.DesktopAI.zip" `
    "/resource:$(Join-Path $build 'OvenDesk.WebP.zip'),OvenDesk.WebP.zip" @sources
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }
Write-Host $output

