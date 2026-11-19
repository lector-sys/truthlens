# Limitations

## Trained AI-generation detectors (learned_synthetic_image / _audio / _video)

Each finding names its model version and what it was trained on (see MODEL_CARD.md). They are weak on generators not
in training (e.g. Sora, Kling, Pika, Midjourney), and there is no face-swap training data. They were trained on studio
speech, ImageNet photos and stock footage, so phone processing and messenger recompression are out of distribution: on
our own untouched files 2 of 3 were flagged (an iPhone voice memo at 0.90, a WhatsApp video at 0.69). Their weight in
the label is calibrated from the server's held-out test predictions plus our own files; with few own files, that weight
mostly reflects the training distribution. A trained-detector finding is a reason to look further, never a conclusion
on its own.
