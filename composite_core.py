"""
composite_core.py — processing logic for the Green Screen Compositor.
Called by app.py (Streamlit UI). No CONFIG block — all params passed as arguments.
"""

import cv2
import numpy as np
import subprocess
import os


def hex2bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.float32)


def _build_crop(bvW, bvH, locked_w, locked_h, zoom, pan_x, pan_y):
    """
    Compute crop coordinates in the replacement video frame.
    zoom > 1 zooms in (shows less of the bg video, bigger-looking content).
    Returns (cx1, cy1, cx2, cy2) — pixel coordinates in the bg frame.
    """
    scale  = max(locked_w / bvW, locked_h / bvH) * zoom
    crop_w = locked_w / scale
    crop_h = locked_h / scale
    crop_x = (bvW - crop_w) * pan_x
    crop_y = (bvH - crop_h) * pan_y
    cx1 = max(0, int(round(crop_x)))
    cy1 = max(0, int(round(crop_y)))
    cx2 = min(bvW, int(round(crop_x + crop_w)))
    cy2 = min(bvH, int(round(crop_y + crop_h)))
    return cx1, cy1, cx2, cy2


def _composite_frame(src_frame, bg_frm,
                     locked_w, locked_h,
                     cx1, cy1, cx2, cy2,
                     rect_x, rect_y,
                     key_bgr, tol, soft, spill_amt, spill_ch, other_chs,
                     W, H):
    """
    Composite a single source frame + bg frame → output frame (BGR uint8).
    """
    rx1 = max(0, rect_x)
    ry1 = max(0, rect_y)
    rx2 = min(W, rect_x + locked_w)
    ry2 = min(H, rect_y + locked_h)

    bg_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if bg_frm is not None and rx2 > rx1 and ry2 > ry1:
        cropped   = bg_frm[cy1:cy2, cx1:cx2]
        scaled_bg = cv2.resize(cropped, (locked_w, locked_h),
                               interpolation=cv2.INTER_LINEAR)
        bx1 = rx1 - rect_x
        by1 = ry1 - rect_y
        bx2 = bx1 + (rx2 - rx1)
        by2 = by1 + (ry2 - ry1)
        bg_canvas[ry1:ry2, rx1:rx2] = scaled_bg[by1:by2, bx1:bx2]

    src_f = src_frame.astype(np.float32)
    diff  = src_f - key_bgr
    dist  = np.sqrt((diff ** 2).sum(axis=2))

    if soft > 0:
        alpha = np.clip((dist - tol) / soft, 0.0, 1.0)
    else:
        alpha = (dist >= tol).astype(np.float32)

    if spill_amt > 0 and rx2 > rx1 and ry2 > ry1:
        roi_src   = src_f[ry1:ry2, rx1:rx2]
        roi_alpha = alpha[ry1:ry2, rx1:rx2]
        is_source = roi_alpha > 0
        ch        = roi_src[:, :, spill_ch].copy()
        other_max = np.maximum(roi_src[:, :, other_chs[0]],
                               roi_src[:, :, other_chs[1]])
        excess    = np.maximum(0.0, ch - other_max)
        suppress  = is_source & (excess > 0)
        if suppress.any():
            roi_src[:, :, spill_ch] = np.where(
                suppress, ch - excess * spill_amt * roi_alpha, ch)
            src_f[ry1:ry2, rx1:rx2] = roi_src

    alpha3 = alpha[:, :, np.newaxis]
    out    = np.clip(src_f * alpha3 + bg_canvas.astype(np.float32) * (1 - alpha3),
                     0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Preview helpers
# ─────────────────────────────────────────────────────────────────────────────

def prepare_preview(src_path, bg_path,
                    key_hex="#00b140", tolerance=60, min_pix_frac=0.005):
    """
    Quick scan of the source video (up to ~150 evenly-spaced frames) to find
    the frame where the placeholder rectangle is largest (fully visible).

    Returns a dict:
        {
          'src_frame': ndarray (H×W×3 BGR),
          'bg_frame' : ndarray (bvH×bvW×3 BGR),
          'locked_w' : int,
          'locked_h' : int,
          'rect_x'   : int,   # true left edge
          'rect_y'   : int,
          'W'        : int,   # source video width
          'H'        : int,   # source video height
          'bvW'      : int,   # bg video width
          'bvH'      : int,   # bg video height
        }
    Or None if no placeholder found.
    """
    src_cap = cv2.VideoCapture(src_path)
    if not src_cap.isOpened():
        return None
    total = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_cap = cv2.VideoCapture(bg_path)
    if not bg_cap.isOpened():
        src_cap.release()
        return None
    bvW = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bvH = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    key_bgr = hex2bgr(key_hex)
    tol     = float(tolerance)

    # Sample up to 150 frames evenly
    sample_n = min(150, total)
    step     = max(1, total // sample_n)
    indices  = list(range(0, total, step))

    best_area  = 0
    best_frame = None
    best_rect  = None
    best_idx   = 0

    for fi in indices:
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = src_cap.read()
        if not ret:
            continue
        diff  = frame.astype(np.float32) - key_bgr
        dist  = np.sqrt((diff ** 2).sum(axis=2))
        mask  = dist < tol
        count = int(mask.sum())
        if count > W * H * min_pix_frac:
            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            w, h   = x2 - x1 + 1, y2 - y1 + 1
            area   = w * h
            if area > best_area:
                best_area  = area
                best_frame = frame.copy()
                best_rect  = {'x': x1, 'y': y1, 'w': w, 'h': h}
                best_idx   = fi

    src_cap.release()

    if best_frame is None or best_rect is None:
        bg_cap.release()
        return None

    locked_w = best_rect['w']
    locked_h = best_rect['h']

    # Infer true left edge (same logic as main scan)
    rx1, rx2 = best_rect['x'], best_rect['x'] + locked_w - 1
    if rx1 > 2:
        true_x = rx1
    elif rx2 < W - 5:
        true_x = rx2 + 1 - locked_w
    else:
        true_x = rx1

    # Grab matching bg frame
    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # always use first bg frame for preview
    ret, bg_frame = bg_cap.read()
    bg_cap.release()
    if not ret:
        bg_frame = np.zeros((bvH, bvW, 3), dtype=np.uint8)

    return {
        'src_frame': best_frame,
        'bg_frame' : bg_frame,
        'locked_w' : locked_w,
        'locked_h' : locked_h,
        'rect_x'   : true_x,
        'rect_y'   : best_rect['y'],
        'W'        : W,
        'H'        : H,
        'bvW'      : bvW,
        'bvH'      : bvH,
    }


def composite_single_frame(preview,
                            key_hex="#00b140", tolerance=60, softness=20,
                            spill=0.8, zoom=1.0,
                            pan_x=0.0, pan_y=0.0,
                            offset_x=0, offset_y=0):
    """
    Composite a single preview frame using current slider values.
    `preview` is the dict returned by prepare_preview().
    Returns a BGR uint8 ndarray (H×W×3), or None on error.
    """
    if preview is None:
        return None

    src_frame = preview['src_frame']
    bg_frm    = preview['bg_frame']
    locked_w  = preview['locked_w']
    locked_h  = preview['locked_h']
    rect_x    = preview['rect_x'] + offset_x
    rect_y    = preview['rect_y'] + offset_y
    W         = preview['W']
    H         = preview['H']
    bvW       = preview['bvW']
    bvH       = preview['bvH']

    key_bgr   = hex2bgr(key_hex)
    spill_ch  = int(np.argmax(key_bgr))
    other_chs = [i for i in range(3) if i != spill_ch]

    cx1, cy1, cx2, cy2 = _build_crop(bvW, bvH, locked_w, locked_h, zoom, pan_x, pan_y)

    return _composite_frame(
        src_frame, bg_frm,
        locked_w, locked_h,
        cx1, cy1, cx2, cy2,
        rect_x, rect_y,
        key_bgr, float(tolerance), float(softness), float(spill),
        spill_ch, other_chs,
        W, H,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    src_path,
    bg_path,
    out_path,
    key_hex      = "#00b140",
    tolerance    = 60,
    softness     = 20,
    spill        = 0.8,
    zoom         = 1.0,
    pan_x        = 0.0,
    pan_y        = 0.0,
    offset_x     = 0,
    offset_y     = 0,
    snap_frac    = 0.01,
    min_pix_frac = 0.005,
    progress_cb  = None,   # progress_cb(phase: int, pct: float, msg: str)
):
    """
    Composite a replacement video into a sliding green-screen rectangle.

    Args:
        src_path   : path to source video (contains the green placeholder)
        bg_path    : path to replacement video
        out_path   : where to write the output MP4
        key_hex    : chroma key color as hex string (e.g. "#00b140")
        tolerance  : color-distance threshold (0–180)
        softness   : feather width in pixels (0 = hard cut)
        spill      : spill suppress strength (0–1)
        zoom       : zoom multiplier for replacement video (1.0 = fit, >1 zooms in)
        pan_x/y    : crop position in replacement video (0=left/top, 1=right/bottom)
        offset_x/y : fine-tune placement of replacement (pixels)
        snap_frac  : snap-to-stable threshold as fraction of rect width
        min_pix_frac: min green pixel fraction to count as rect detection
        progress_cb: optional callback(phase, pct_0_to_1, message)

    Returns:
        out_path on success. Raises RuntimeError on failure.
    """

    def _cb(phase, pct, msg):
        if progress_cb:
            progress_cb(phase, pct, msg)

    # ── Open videos ───────────────────────────────────────────────────────────
    src_cap = cv2.VideoCapture(src_path)
    if not src_cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {src_path}")

    fps   = src_cap.get(cv2.CAP_PROP_FPS)
    total = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_cap   = cv2.VideoCapture(bg_path)
    if not bg_cap.isOpened():
        raise RuntimeError(f"Cannot open replacement video: {bg_path}")
    bvW      = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bvH      = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg_total = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_cap.release()

    key_bgr   = hex2bgr(key_hex)
    tol       = float(tolerance)
    spill_ch  = int(np.argmax(key_bgr))
    other_chs = [i for i in range(3) if i != spill_ch]

    # ── Phase 1: Scan every frame ─────────────────────────────────────────────
    _cb(1, 0.0, f"Phase 1/2: Scanning {total} frames…")

    raw_pos         = []
    max_area        = 0
    best_rect_frame = 0
    best_rect       = None

    for fi in range(total):
        ret, frame = src_cap.read()
        if not ret:
            raw_pos.append({'valid': False})
            break

        diff  = frame.astype(np.float32) - key_bgr
        dist  = np.sqrt((diff ** 2).sum(axis=2))
        mask  = dist < tol
        count = int(mask.sum())

        if count > W * H * min_pix_frac:
            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            w, h   = x2 - x1 + 1, y2 - y1 + 1
            raw_pos.append({'x': x1, 'y': y1, 'w': w, 'h': h, 'valid': True})
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = {'x': x1, 'y': y1, 'w': w, 'h': h}
                best_rect_frame = fi
        else:
            raw_pos.append({'valid': False})

        if fi % 30 == 0:
            pct = fi / total
            _cb(1, pct, f"Phase 1/2: Scanning frame {fi}/{total}…")

    src_cap.release()

    if not best_rect:
        raise RuntimeError(
            "No placeholder rectangle found. "
            "Check Key Color and Tolerance — the color must exactly match the placeholder."
        )

    locked_w = best_rect['w']
    locked_h = best_rect['h']

    # Compute trueX per frame (infer negative left edge during slide-out)
    true_pos = []
    for p in raw_pos:
        if not p['valid']:
            true_pos.append({'x': None, 'y': None, 'valid': False})
            continue
        right = p['x'] + p['w'] - 1
        if p['x'] > 2:
            tx = p['x']
        elif right < W - 5:
            tx = right + 1 - locked_w
        else:
            tx = p['x']
        true_pos.append({'x': tx, 'y': p['y'], 'valid': True})

    bp       = true_pos[best_rect_frame]
    stable_x = bp['x'] if bp['valid'] else 0
    stable_y = raw_pos[best_rect_frame]['y']

    snap_thresh = max(4, locked_w * snap_frac)
    for p in true_pos:
        if p['valid'] and p['x'] is not None:
            if abs(p['x'] - stable_x) <= snap_thresh:
                p['x'] = stable_x
                p['y'] = stable_y

    rect_first_frame = next((i for i, p in enumerate(true_pos) if p['valid']), 0)

    # Cover-fit + zoom crop for replacement video
    cx1, cy1, cx2, cy2 = _build_crop(bvW, bvH, locked_w, locked_h, zoom, pan_x, pan_y)

    # ── Phase 2: Composite + encode ───────────────────────────────────────────
    _cb(2, 0.0, "Phase 2/2: Compositing and encoding…")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', 'pipe:0',
        '-i', src_path,
        '-map', '0:v',
        '-map', '1:a?',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        out_path
    ]
    try:
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install it: brew install ffmpeg  (Mac) or apt install ffmpeg  (Linux)")

    src_cap = cv2.VideoCapture(src_path)
    bg_cap  = cv2.VideoCapture(bg_path)
    bg_cap_pos = 0

    def read_bg_frame(target_idx):
        nonlocal bg_cap_pos
        target_idx = target_idx % max(1, bg_total)
        if target_idx != bg_cap_pos:
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            bg_cap_pos = target_idx
        ret, frm = bg_cap.read()
        if ret:
            bg_cap_pos += 1
        return frm if ret else None

    soft      = float(softness)
    spill_amt = float(spill)

    for fi in range(total):
        ret, src_frame = src_cap.read()
        if not ret:
            break

        pos = true_pos[fi] if fi < len(true_pos) else {'valid': False}

        if not pos['valid']:
            ffmpeg.stdin.write(src_frame.tobytes())
            if fi % 30 == 0:
                _cb(2, fi / total, f"Phase 2/2: Compositing frame {fi}/{total}…")
            continue

        rect_x = int(round(pos['x'])) + offset_x
        rect_y = int(round(pos['y'])) + offset_y

        bg_idx = max(0, fi - rect_first_frame)
        bg_frm = read_bg_frame(bg_idx)

        out = _composite_frame(
            src_frame, bg_frm,
            locked_w, locked_h,
            cx1, cy1, cx2, cy2,
            rect_x, rect_y,
            key_bgr, tol, soft, spill_amt, spill_ch, other_chs,
            W, H,
        )
        ffmpeg.stdin.write(out.tobytes())

        if fi % 30 == 0:
            _cb(2, fi / total, f"Phase 2/2: Compositing frame {fi}/{total}…")

    src_cap.release()
    bg_cap.release()
    ffmpeg.stdin.close()
    ret = ffmpeg.wait()

    if ret != 0:
        raise RuntimeError(f"FFmpeg encoding failed (exit code {ret})")

    _cb(2, 1.0, "Done!")
    return out_path
