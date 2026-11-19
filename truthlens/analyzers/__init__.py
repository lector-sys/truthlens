from . import image, video, audio, text

BY_EXT = {
    **dict.fromkeys((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"), image),
    **dict.fromkeys((".mp4", ".mov", ".mkv", ".webm", ".avi"), video),
    **dict.fromkeys((".wav", ".mp3", ".m4a", ".flac", ".ogg"), audio),
}
