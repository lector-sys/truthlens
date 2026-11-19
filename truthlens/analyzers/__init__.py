from . import image, video, audio, text

# Videos also go through audio: audio.analyze decodes the track itself, so A/V flags can be linked.
BY_EXT = {
    **dict.fromkeys((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"), (image,)),
    **dict.fromkeys((".mp4", ".mov", ".mkv", ".webm", ".avi"), (video, audio)),
    **dict.fromkeys((".wav", ".mp3", ".m4a", ".flac", ".ogg"), (audio,)),
}
