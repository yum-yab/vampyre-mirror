import torch
import torch.nn.functional as F


# Square max / box filters are separable: two 1-D passes are much cheaper for large radii.


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    mask = F.max_pool2d(mask, (k, 1), stride=1, padding=(radius, 0))
    return F.max_pool2d(mask, (1, k), stride=1, padding=(0, radius))


def feather(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    mask = F.avg_pool2d(mask, (k, 1), stride=1, padding=(radius, 0), count_include_pad=False)
    return F.avg_pool2d(mask, (1, k), stride=1, padding=(0, radius), count_include_pad=False)


class BackgroundModel:
    """Clean plate of the static scene.

    Starts from an average of calibration frames, then keeps adapting (slow EMA) only where
    no person is visible, so standing still never makes you part of the background.
    Areas that were occluded during calibration heal once you move away from them.
    """

    def __init__(self, rate: float = 0.02, person_threshold: float = 0.05, margin: int = 15):
        self.rate = rate
        self.person_threshold = person_threshold
        self.margin = margin
        self.bg: torch.Tensor | None = None
        self._acc: torch.Tensor | None = None
        self._count = 0

    def add_calibration_frame(self, rgb: torch.Tensor) -> None:
        frame = rgb.float()
        self._acc = frame if self._acc is None else self._acc + frame
        self._count += 1

    def finish_calibration(self) -> None:
        self.bg = self._acc / self._count  # float32: tiny EMA steps vanish in fp16
        self._acc, self._count = None, 0

    def update(self, rgb: torch.Tensor, person_alpha: torch.Tensor) -> None:
        person = dilate((person_alpha > self.person_threshold).to(torch.float16), self.margin)
        self.bg += self.rate * (1 - person.float()) * (rgb.float() - self.bg)
