# Vampyre Mirror

A webcam mirror in which you have no reflection. Your face, hair, hands and other bare skin are
replaced by the background behind you, but your clothes, glasses and hat stay visible.

## How it works

Each frame goes through two models on the GPU:

1. **Robust Video Matting** (MobileNetV3, TorchScript, ~7 ms) gives a soft alpha matte of the whole
   person. It is temporally stable and handles fine hair and fingers.
2. **SegFormer-B2 human parsing** (`mattmdjaga/segformer_b2_clothes`, ~15 ms) labels which pixels are
   hair, face, arms or legs and which are clothing.

The parsed body parts are grown by a generous margin, but never into clothing. Within reach of the body,
everything the matte sees as person (stray hair strands, fingers) is added. The result is feathered and
replaced by a background plate. Outside the person the plate matches the live frame, so the margin is
invisible and no rim of the erased parts stays behind. At startup the plate is the average of frames taken while you step
out of the frame. After that it keeps adapting slowly, but only where no person is visible, so standing
still never fades you into the background.

## Usage

```sh
uv sync
uv run vampyre-mirror                # windowed
uv run vampyre-mirror --fullscreen
uv run vampyre-mirror --resolution 720p   # 360p, 480p (default), 720p, 1080p or e.g. 960x720
uv run vampyre-mirror --debug        # 2×2 view: mirror / erase mask / body parts / background + timings
```

At startup the mirror waits 3 s (`--calibration`). **Step out of the frame** until the countdown ends.

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `b` | recapture the background (step out of the frame again) |
| `d` | toggle the debug view |

Useful flags: `--keep-legs` (don't erase bare legs), `--grow` / `--hair-reach` / `--soften` (mask
margin, matte strand reach and edge feathering, in px at 480p and scaled with the resolution), `--parse-size` (parser resolution; lower is faster), `--camera N`.

## Tips

- Use a static background and steady lighting. If your camera's auto exposure or white balance drifts,
  fix it with `v4l2-ctl -c auto_exposure=1 -c exposure_time_absolute=...`.
- On startup the app sets `exposure_dynamic_framerate=0` with `v4l2-ctl`, if it is installed, so the
  camera does not drop to a low frame rate in dim light.
- Models are downloaded on first run: the matting model goes to `~/.cache/vampyre-mirror`, and the
  parser goes to the Hugging Face cache.
