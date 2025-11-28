@echo off
echo Converting H.264 stream with proper timestamps to MP4...
echo Method 1: Using explicit framerate and codec copy
ffmpeg -y -framerate 30 -i timestamped_output.h264 -c copy -movflags +faststart output_with_sei.mp4
if %ERRORLEVEL% EQU 0 (
    echo Conversion successful without timestamp warnings!
    echo.
    echo Analyzing the result:
    ffprobe -v quiet -select_streams v:0 -show_entries packet=pts_time,dts_time,duration_time -of csv=p=0 output_with_sei.mp4 | head -5
    echo.
    echo Playing with ffplay...
    ffplay -autoexit output_with_sei.mp4
) else (
    echo Conversion failed.
)
pause
