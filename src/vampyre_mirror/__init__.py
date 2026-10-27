import argparse
import time

import cv2
import numpy as np
import torch

from .background import BackgroundModel, dilate, feather
from .capture import Camera
from .matting import PersonMatte
from .parsing import BodyParser

WINDOW = "Vampyre Mirror"
RESOLUTIONS = {"360p": (640, 360), "480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
# Pixel-sized options are given for 480p and scaled with the frame height.
REFERENCE_HEIGHT = 480


def resolution(value: str) -> tuple[int, int]:
    if value in RESOLUTIONS:
        return RESOLUTIONS[value]
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"use one of {', '.join(RESOLUTIONS)} or WIDTHxHEIGHT") from None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A webcam mirror in which your body has no reflection.")
    p.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    p.add_argument(
        "--resolution",
        type=resolution,
        default="480p",
        help=f"camera resolution: {', '.join(RESOLUTIONS)} or WIDTHxHEIGHT (default 480p)",
    )
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--debug", action="store_true", help="start with the debug view (toggle with 'd')")
    p.add_argument("--keep-legs", action="store_true", help="don't erase bare legs")
    p.add_argument("--parse-size", type=int, default=384, help="short side for the body parser")
    p.add_argument("--grow", type=int, default=12, help="px @480p to grow the body mask by (not into clothes)")
    p.add_argument("--hair-reach", type=int, default=25, help="px @480p around the body where the matte adds strands")
    p.add_argument("--soften", type=int, default=6, help="px @480p of edge feathering")
    p.add_argument("--calibration", type=float, default=3.0, help="seconds to step out of frame")
    return p.parse_args()


def to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """HxWx3 uint8 BGR -> [1,3,H,W] float16 RGB in 0..1 on the GPU."""
    t = torch.from_numpy(frame_bgr).to(device, non_blocking=True)
    return t.flip(-1).permute(2, 0, 1)[None].to(torch.float16) / 255


def to_image(rgb: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] (or [1,1,H,W]) in 0..1 -> HxWx3 uint8 BGR, mirrored horizontally."""
    if rgb.shape[1] == 1:
        rgb = rgb.expand(-1, 3, -1, -1)
    img = (rgb[0].flip(0, 2).permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8)
    return img.cpu().numpy()


def label_colors(labels: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    palette = torch.rand(32, 3, generator=gen).to(labels.device)
    palette[0] = 0
    return palette[labels].permute(2, 0, 1)[None]


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def debug_grid(tiles: dict[str, np.ndarray], stats: str) -> np.ndarray:
    imgs = []
    for name, img in tiles.items():
        h, w = img.shape[:2]
        if h > REFERENCE_HEIGHT:
            img = cv2.resize(img, (w * REFERENCE_HEIGHT // h, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            img = img.copy()
        put_text(img, name, (10, 25))
        imgs.append(img)
    grid = np.vstack([np.hstack(imgs[:2]), np.hstack(imgs[2:])])
    put_text(grid, stats, (10, grid.shape[0] - 15))
    return grid


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available - the mirror needs an NVIDIA GPU.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    width, height = args.resolution
    cam = Camera(args.camera, width, height)
    if cam.size != (width, height):
        print(f"Camera does not support {width}x{height}, using {cam.size[0]}x{cam.size[1]}")
    scale = cam.size[1] / REFERENCE_HEIGHT
    grow, hair_reach, soften = (round(v * scale) for v in (args.grow, args.hair_reach, args.soften))

    print("Loading models ...")
    # RVM works on a downsampled copy internally; keep that at ~240 px high regardless of input size.
    matte = PersonMatte(device, downsample_ratio=min(1.0, 0.5 / scale))
    parser = BodyParser(device, short_side=args.parse_size, keep_legs=args.keep_legs)
    background = BackgroundModel(margin=round(15 * scale))

    # TorchScript and cuDNN optimise on the first calls (seconds) - do it before the mirror is live.
    dummy = torch.rand(1, 3, cam.size[1], cam.size[0], device=device, dtype=torch.float16)
    for _ in range(5):
        matte(dummy)
        parser(dummy)
    matte.reset()
    print(f"Camera running at {cam.size[0]}x{cam.size[1]}. Keys: q quit, b recapture background, d debug view")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    debug = args.debug
    calib_until = time.perf_counter() + args.calibration
    body_hold: torch.Tensor | None = None
    seq = 0
    fps = 0.0
    last = time.perf_counter()

    try:
        while True:
            seq, frame = cam.read(seq)
            t0 = time.perf_counter()
            rgb = to_tensor(frame, device)

            if calib_until is not None:
                # Step out of frame: average the last second of frames as the clean plate.
                remaining = calib_until - t0
                if remaining <= 1.0:
                    background.add_calibration_frame(rgb)
                if remaining <= 0:
                    background.finish_calibration()
                    matte.reset()
                    calib_until = None
                img = to_image(rgb)
                put_text(img, f"Step out of the frame... {max(remaining, 0):.1f}", (20, 40), 0.8)
                cv2.imshow(WINDOW, img)
            else:
                pha = matte(rgb)
                t_matte = time.perf_counter()
                body, clothes, labels = parser(rgb)
                # Fast attack, slow release: moving hands are masked at once, never trail a ghost.
                body_hold = body if body_hold is None else torch.maximum(body, 0.6 * body_hold)
                t_parse = time.perf_counter()

                # Outside the person the plate matches the frame anyway, so grow generously -
                # but never into clothes. The matte adds hair strands / fingers the parser missed.
                grown = dilate(body_hold, grow)
                strands = (pha > 0.02) * dilate(body_hold, hair_reach)
                erase = torch.maximum(grown, strands) * (1 - clothes)
                alpha = feather(erase, soften)
                bg = background.bg.to(torch.float16)
                out = alpha * bg + (1 - alpha) * rgb
                background.update(rgb, pha)
                img = to_image(out)

                if debug:
                    torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    stats = (
                        f"{fps:5.1f} fps | matte {1000 * (t_matte - t0):4.1f} ms | "
                        f"parse {1000 * (t_parse - t_matte):4.1f} ms | "
                        f"total {1000 * (t_end - t0):4.1f} ms"
                    )
                    img = debug_grid(
                        {
                            "mirror": img,
                            "erase mask": to_image(alpha),
                            "body parts": to_image(label_colors(labels)),
                            "background": to_image(bg),
                        },
                        stats,
                    )
                cv2.imshow(WINDOW, img)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("b"):
                calib_until = time.perf_counter() + args.calibration
            elif key == ord("d"):
                debug = not debug
    finally:
        cam.close()
        cv2.destroyAllWindows()
