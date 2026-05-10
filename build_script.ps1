$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$VersionFile = Join-Path $ProjectRoot "VERSION"
if (-not (Test-Path $VersionFile)) {
    Set-Content -Path $VersionFile -Value "3.0.0" -NoNewline
}

# Read and auto-bump patch version, matching the ModBox21 build flow.
$VersionStr = (Get-Content $VersionFile -Raw).Trim()
$VersionParts = $VersionStr -split '\.'
if ($VersionParts.Count -lt 3) {
    throw "Invalid VERSION file contents: '$VersionStr'"
}
$Major = [int]$VersionParts[0]
$Minor = [int]$VersionParts[1]
$Patch = [int]$VersionParts[2] + 1
$Version = "$Major.$Minor.$Patch"
Set-Content -Path $VersionFile -Value $Version -NoNewline
Write-Host "Version: $Version" -ForegroundColor Cyan

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "Starting build process..." -ForegroundColor Cyan

# Run PyInstaller with the spec file
Write-Host "`n[1/3] Running PyInstaller..." -ForegroundColor Yellow
$PyInstallerStart = $Stopwatch.Elapsed
pyinstaller --clean --noconfirm (Join-Path $ProjectRoot "FallTalk.spec") --log-level INFO
$pyinstallerTime = $Stopwatch.Elapsed - $PyInstallerStart
Write-Host "PyInstaller completed in $($pyinstallerTime.ToString('hh\:mm\:ss'))" -ForegroundColor Green

# File operations
Write-Host "`n[2/3] Copying resources..." -ForegroundColor Yellow
$TargetDir = Join-Path $ProjectRoot "dist\FallTalk"
$CopyStart = $Stopwatch.Elapsed

# Create the resource directory inside the target directory if it doesn't exist
$ResourceTargetDir = Join-Path $TargetDir "resource"
if (-not (Test-Path -Path $ResourceTargetDir)) {
    New-Item -ItemType Directory -Path $ResourceTargetDir | Out-Null
}

# Copy runtime version tracking and bundled executables into the target directory
Copy-Item -Path $VersionFile -Destination $TargetDir -Force
Copy-Item -Path (Join-Path $ProjectRoot "ffmpeg.exe") -Destination $TargetDir -Force
Copy-Item -Path (Join-Path $ProjectRoot "ffprobe.exe") -Destination $TargetDir -Force

# Copy the contents of the resource directory
Copy-Item -Path (Join-Path $ProjectRoot "resource\*") -Destination $ResourceTargetDir -Recurse -Force
$copyTime = $Stopwatch.Elapsed - $CopyStart
Write-Host "Resources copied in $($copyTime.ToString('hh\:mm\:ss'))" -ForegroundColor Green

# Compression
Write-Host "`n[3/3] Creating 7z archive..." -ForegroundColor Yellow
$ArchivePath = Join-Path $ProjectRoot "FallTalk-$Version.7z"
if (Test-Path $ArchivePath) {
    Remove-Item $ArchivePath -Force
}
$CompressionStart = $Stopwatch.Elapsed
Push-Location -Path $TargetDir
7z a -mx=9 -mmt=32 $ArchivePath *
Pop-Location
$compressionTime = $Stopwatch.Elapsed - $CompressionStart

# Final summary
$Stopwatch.Stop()
Write-Host "`nBuild completed successfully!" -ForegroundColor Green
Write-Host "`nTime Summary:" -ForegroundColor Cyan
Write-Host "- PyInstaller: $($pyinstallerTime.ToString('hh\:mm\:ss'))"
Write-Host "- File Copying: $($copyTime.ToString('hh\:mm\:ss'))"
Write-Host "- Compression: $($compressionTime.ToString('hh\:mm\:ss'))"
Write-Host "`nTotal Time: $($Stopwatch.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
