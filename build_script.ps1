# Run PyInstaller with the spec file
pyinstaller --noconfirm FallTalk.spec

$TARGET_DIR = "dist\FallTalk"

# Copy ffmpeg.exe and ffprobe.exe to the target directory
Copy-Item -Path "ffmpeg.exe" -Destination $TARGET_DIR
Copy-Item -Path "ffprobe.exe" -Destination $TARGET_DIR

# Create the resource directory inside the target directory if it doesn't exist
$RESOURCE_TARGET_DIR = "$TARGET_DIR\resource"
if (-Not (Test-Path -Path $RESOURCE_TARGET_DIR)) {
    New-Item -ItemType Directory -Path $RESOURCE_TARGET_DIR
}

# Copy the contents of the resource directory to the target directory's resource folder
Copy-Item -Path "resource\*" -Destination $RESOURCE_TARGET_DIR -Recurse

Set-Location -Path "dist"
# Compress the FallTalk directory into FallTalk.7z using 7-Zip
# Ensure 7-Zip is installed and 7z.exe is in your PATH
$FALLTALK_DIR = "FallTalk"
7z a FallTalk_v1.1.4.7z $FALLTALK_DIR

# Change back to the original directory
Set-Location -Path ".."