# Start the timer
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "Starting build process..." -ForegroundColor Cyan

# Run PyInstaller with the spec file
Write-Host "`n[1/3] Running PyInstaller..." -ForegroundColor Yellow
pyinstaller --clean --noconfirm FallTalk.spec --log-level INFO
$pyinstallerTime = $Stopwatch.Elapsed.ToString('hh\:mm\:ss')
Write-Host "PyInstaller completed in $pyinstallerTime" -ForegroundColor Green

# File operations
Write-Host "`n[2/3] Copying resources..." -ForegroundColor Yellow
$TARGET_DIR = "dist\FallTalk"

# Create the resource directory inside the target directory if it doesn't exist
$RESOURCE_TARGET_DIR = "$TARGET_DIR\resource"
if (-Not (Test-Path -Path $RESOURCE_TARGET_DIR)) {
    New-Item -ItemType Directory -Path $RESOURCE_TARGET_DIR | Out-Null
}

# Copy ffmpeg.exe and ffprobe.exe to the target directory
Copy-Item -Path "ffmpeg.exe" -Destination $TARGET_DIR -Force
Copy-Item -Path "ffprobe.exe" -Destination $TARGET_DIR -Force

# Copy the contents of the resource directory
Copy-Item -Path "resource\*" -Destination $RESOURCE_TARGET_DIR -Recurse -Force
Write-Host "Resources copied in $($Stopwatch.Elapsed - $pyinstallerTime)" -ForegroundColor Green

# Compression
Write-Host "`n[3/3] Creating 7z archive..." -ForegroundColor Yellow
Set-Location -Path "dist\FallTalk"
7z a -mx=9 -mmt=32 ..\FallTalk_v2.0.0.7z *
Set-Location -Path "..\.."
$compressionTime = $Stopwatch.Elapsed.ToString('hh\:mm\:ss')

# Final summary
$Stopwatch.Stop()
Write-Host "`nBuild completed successfully!" -ForegroundColor Green
Write-Host "`nTime Summary:" -ForegroundColor Cyan
Write-Host "- PyInstaller: $pyinstallerTime"
Write-Host "- File Copying: $($Stopwatch.Elapsed - $pyinstallerTime - ($Stopwatch.Elapsed - $compressionTime))"
Write-Host "- Compression: $($Stopwatch.Elapsed - $compressionTime)"
Write-Host "`nTotal Time: $($Stopwatch.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
