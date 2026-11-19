"""The one shared contract. Every analyzer returns list[Indicator]."""
from dataclasses import dataclass, field, asdict

RELIABILITY_WEIGHT = {"low": 0.3, "medium": 0.6, "high": 0.9}


@dataclass
class Indicator:
    file: str                      # vault filename
    modality: str                  # image | video | audio | text | cross
    type: str                      # e.g. "ela_hotspot", "audio_splice"
    score: float                   # indicator strength 0..1
    reliability: str               # low | medium | high (how much to trust this detector)
    explanation: str
    location: dict = field(default_factory=dict)   # {"bbox":[x,y,w,h]} | {"t":12.4,"frame":312} | {"start":3.1,"end":4.0} | {"span":"..."}
    artifact: str | None = None    # path to heatmap / thumbnail
    alt_explanations: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)  # other files (cross-modal links)

    def to_dict(self):
        return asdict(self)
