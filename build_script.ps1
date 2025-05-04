# Run PyInstaller with the spec file
pyinstaller --noconfirm FallTalk.spec --log-level INFO

$TARGET_DIR = "dist\FallTalk"

# Create the resource directory inside the target directory if it doesn't exist
$RESOURCE_TARGET_DIR = "$TARGET_DIR\resource"
if (-Not (Test-Path -Path $RESOURCE_TARGET_DIR)) {
    New-Item -ItemType Directory -Path $RESOURCE_TARGET_DIR
}

# Copy ffmpeg.exe and ffprobe.exe to the target directory
Copy-Item -Path "ffmpeg.exe" -Destination $TARGET_DIR -Force
Copy-Item -Path "ffprobe.exe" -Destination $TARGET_DIR -Force

# Copy the contents of the resource directory to the target directory's resource folder
Copy-Item -Path "resource\*" -Destination $RESOURCE_TARGET_DIR -Recurse -Force

Set-Location -Path "dist"
# Compress the FallTalk directory into FallTalk.7z using 7-Zip
# Ensure 7-Zip/NANA is installed and 7z.exe is in your PATH
#$FALLTALK_DIR = "FallTalk"
#7z a -mx=9 FallTalk_v2.0.0.7z $FALLTALK_DIR
#
## Change back to the original directory
#Set-Location -Path ".."