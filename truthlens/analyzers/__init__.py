from . import image, video, audio, text

# Pillow reads most of these; the rest are converted to PNG by media_meta.readable_image (ffmpeg).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".avif", ".heic", ".heif", ".jxl", ".jp2")
TEXT_EXTS = (".txt", ".md")  # text evidence (a message, post, article): claims extracted and cross-checked like a caption

# Videos also go through audio: audio.analyze decodes the track itself, so A/V flags can be linked.
BY_EXT = {
    **dict.fromkeys(IMAGE_EXTS, (image,)),
    **dict.fromkeys((".mp4", ".mov", ".mkv", ".webm", ".avi"), (video, audio)),
    **dict.fromkeys((".wav", ".mp3", ".m4a", ".flac", ".ogg"), (audio,)),
    **dict.fromkeys(TEXT_EXTS, (text,)),
}
