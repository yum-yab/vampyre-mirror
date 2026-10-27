import urllib.request
import warnings
from pathlib import Path

import torch

RVM_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
    "rvm_mobilenetv3_fp16.torchscript"
)
CACHE_DIR = Path.home() / ".cache" / "vampyre-mirror"


def _model_path() -> Path:
    path = CACHE_DIR / RVM_URL.rsplit("/", 1)[1]
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading matting model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(RVM_URL, tmp)
        tmp.rename(path)
    return path


class PersonMatte:
    """Robust Video Matting (MobileNetV3): soft, temporally stable alpha for the whole person."""

    def __init__(self, device: torch.device, downsample_ratio: float = 0.5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.model = torch.jit.load(str(_model_path()), map_location=device).eval()
        self.downsample_ratio = downsample_ratio
        self.reset()

    def reset(self) -> None:
        self.rec = [None] * 4

    @torch.inference_mode()
    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [1,3,H,W] float16 in 0..1 -> alpha [1,1,H,W] float16."""
        _fgr, pha, *self.rec = self.model(rgb, *self.rec, self.downsample_ratio)
        return pha
