@echo off
REM Jalankan di Windows (butuh Python 3.10-3.12). Hasil: dist\YTClipper\YTClipper.exe
py -3.11 -m venv .venv || goto :err
call .venv\Scripts\activate
pip install -r requirements.txt pyinstaller || goto :err
if not exist models mkdir models
if not exist models\face_detection_yunet_2023mar.onnx curl -L -o models\face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
pyinstaller --noconfirm --windowed --name YTClipper --collect-all faster_whisper --collect-all ctranslate2 --collect-all imageio_ffmpeg --collect-all yt_dlp --collect-data cv2 --add-data "models;models" app.py || goto :err
echo Selesai: dist\YTClipper\YTClipper.exe
exit /b 0
:err
echo Build gagal & exit /b 1
