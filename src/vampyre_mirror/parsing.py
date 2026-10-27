import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# ATR labels of this model that are the body itself (everything else, e.g. clothes,
# hats, sunglasses, bags, stays visible in the mirror).
BODY_LABELS = ("Hair", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg")
LEG_LABELS = ("Left-leg", "Right-leg")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BodyParser:
    """SegFormer human parsing: which pixels are bare body (skin + hair) rather than clothing."""

    def __init__(self, device: torch.device, short_side: int = 384, keep_legs: bool = False):
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, use_safetensors=True)
            .to(device=device, dtype=torch.float16)
            .eval()
        )
        self.short_side = short_side
        label2id = self.model.config.label2id
        names = [n for n in BODY_LABELS if not (keep_legs and n in LEG_LABELS)]
        self.body_ids = torch.tensor([label2id[n] for n in names], device=device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float16).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """rgb: [1,3,H,W] float16 in 0..1 -> (body mask, clothes mask, labels [H,W]).

        Masks are [1,1,H,W] float16; clothes is everything worn (neither background nor body).
        """
        h, w = rgb.shape[-2:]
        scale = self.short_side / min(h, w)
        x = F.interpolate(rgb, scale_factor=scale, mode="bilinear", antialias=True)
        x = (x - self.mean) / self.std
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0]
        body = torch.isin(labels, self.body_ids)
        clothes = (labels != 0) & ~body
        return body[None, None].to(torch.float16), clothes[None, None].to(torch.float16), labels
