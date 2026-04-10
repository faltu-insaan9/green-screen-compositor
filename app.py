"""
Green Screen Compositor — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import tempfile, os
import cv2
import numpy as np
from composite_core import process_video, prepare_preview, composite_single_frame

st.set_page_config(page_title="Green Screen Compositor", layout="wide")
st.title("🎬 Green Screen Compositor")

# ── Session state defaults ────────────────────────────────────────────────────
for k, v in [
    ("reframe_done",      False),
    ("output_ready",      False),
    ("preview_image",     None),
    ("preview_data",      None),
    ("preview_cache_key", ""),
    ("output_path",       None),
    ("file_key",          ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── 4-column layout ───────────────────────────────────────────────────────────
col_upload, col_settings, col_preview, col_output = st.columns([1.3, 2.2, 1.8, 2.3])

# ══════════════════════════════════════════════════════════════════════════════
# Column 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════
with col_upload:
    st.subheader("Upload")
    src_file = st.file_uploader(
        "① Source video",
        type=["mp4", "mov", "avi", "mkv"],
        help="The video containing the solid-colour placeholder rectangle.",
    )
    bg_file = st.file_uploader(
        "② Replacement video",
        type=["mp4", "mov", "avi", "mkv"],
        help="The video that will be composited into the placeholder.",
    )

# Detect file changes → reset downstream state
_fkey = (
    f"{src_file.name}_{src_file.size}" if src_file else "none"
    f"_{bg_file.name}_{bg_file.size}" if bg_file else "_none"
)
if _fkey != st.session_state["file_key"]:
    st.session_state.update({
        "reframe_done":  False,
        "output_ready":  False,
        "preview_image": None,
        "output_path":   None,
        "file_key":      _fkey,
    })

# ══════════════════════════════════════════════════════════════════════════════
# Column 2 — Key settings + Framing + Reframe button
# ══════════════════════════════════════════════════════════════════════════════
with col_settings:
    st.subheader("🎨 Key settings")

    KEY_PRESETS = {
        "Green  (#00b140)": "#00b140",
        "Blue   (#0047FF)": "#0047FF",
        "Custom":            None,
    }
    preset_choice = st.selectbox(
        "Placeholder colour",
        list(KEY_PRESETS.keys()),
        help=(
            "Colour of the solid rectangle in your source video.\n"
            "• Green / Blue are standard chroma-key colours\n"
            "• Custom: enter any hex code (use imagecolorpicker.com to sample it)"
        ),
    )
    if KEY_PRESETS[preset_choice] is not None:
        key_color = KEY_PRESETS[preset_choice]
        st.caption(f"Using `{key_color}`")
    else:
        key_color = st.text_input(
            "Custom hex code", value="#00b140",
            help="e.g. #00b140  — must start with #",
        )

    tolerance = st.slider(
        "Tolerance", 0, 180, 60,
        help=(
            "How far a pixel can be from the key colour and still be replaced.\n"
            "• Range: 0–180\n"
            "• Low → colour patches may remain  |  High → nearby colours also removed\n"
            "• Default 60"
        ),
    )
    softness = st.slider(
        "Softness", 0, 80, 20,
        help=(
            "Feather width at placeholder edges (pixels).\n"
            "• Range: 0–80 px\n"
            "• 0 = hard cut  |  Higher = smoother blend\n"
            "• Default 20"
        ),
    )
    spill = st.slider(
        "Spill suppress", 0.0, 1.0, 0.8, step=0.05,
        help=(
            "Reduces colour cast bleeding onto objects in front of the placeholder.\n"
            "• Range: 0.0–1.0\n"
            "• 0 = off  |  1 = maximum (may desaturate edges slightly)\n"
            "• Default 0.8"
        ),
    )

    with st.expander("✂️ Framing  (zoom · pan · offset)"):
        c1, c2 = st.columns(2)
        zoom = c1.slider(
            "Zoom", 1.0, 4.0, 1.0, step=0.1,
            help="Zoom into the replacement clip (1× = fit, 2× = 2× crop in). Use Pan to pick the visible area.",
        )
        pan_x = c1.slider(
            "Pan X", 0.0, 1.0, 0.0, step=0.01,
            help="Horizontal crop position (0 = left edge, 1 = right edge). Active when Zoom > 1.",
        )
        pan_y = c2.slider(
            "Pan Y", 0.0, 1.0, 0.0, step=0.01,
            help="Vertical crop position (0 = top, 1 = bottom). Active when Zoom > 1.",
        )
        offset_x = c1.slider(
            "Offset X (px)", -200, 200, 0,
            help="Nudge replacement left (−) or right (+) inside the placeholder. Range: −200–+200 px.",
        )
        offset_y = c2.slider(
            "Offset Y (px)", -200, 200, 0,
            help="Nudge replacement up (−) or down (+) inside the placeholder. Range: −200–+200 px.",
        )

    st.write("")  # spacer before button
    both_uploaded = bool(src_file and bg_file)
    reframe_clicked = st.button(
        "🔍  1. Reframe",
        use_container_width=True,
        disabled=not both_uploaded,
        help="Generate a preview frame with current settings."
              if both_uploaded else "Upload both videos to enable.",
    )

# ── Reframe logic (before rendering preview column) ───────────────────────────
if reframe_clicked and src_file and bg_file:
    cache_key = f"{src_file.name}_{src_file.size}_{bg_file.name}_{bg_file.size}"

    if st.session_state["preview_cache_key"] != cache_key:
        tmp = tempfile.mkdtemp()
        sp = os.path.join(tmp, src_file.name)
        bp = os.path.join(tmp, bg_file.name)
        src_file.seek(0); bg_file.seek(0)
        with open(sp, "wb") as f: f.write(src_file.read())
        with open(bp, "wb") as f: f.write(bg_file.read())

        with st.spinner("Scanning for best preview frame…"):
            pv = prepare_preview(sp, bp, key_hex=key_color, tolerance=tolerance)

        st.session_state["preview_data"]      = pv
        st.session_state["preview_cache_key"] = cache_key
    else:
        pv = st.session_state["preview_data"]

    if pv:
        composed = composite_single_frame(
            pv,
            key_hex=key_color, tolerance=tolerance, softness=softness,
            spill=spill, zoom=zoom, pan_x=pan_x, pan_y=pan_y,
            offset_x=offset_x, offset_y=offset_y,
        )
        if composed is not None:
            st.session_state["preview_image"] = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
            st.session_state["reframe_done"]  = True
    else:
        st.session_state["preview_image"] = None
        st.warning("No placeholder found. Check Key Colour and Tolerance.")

# ══════════════════════════════════════════════════════════════════════════════
# Column 3 — Framing preview
# ══════════════════════════════════════════════════════════════════════════════
with col_preview:
    st.subheader("📐 Preview")
    img = st.session_state["preview_image"]
    if img is not None:
        st.image(img, use_container_width=True,
                 caption="Reference frame · not the final video")
    elif both_uploaded:
        st.info("Click **1. Reframe** to preview your framing here.")
    else:
        st.caption("Upload both videos, then click **1. Reframe**.")

# ══════════════════════════════════════════════════════════════════════════════
# Column 4 — Process + Export + Output
# ══════════════════════════════════════════════════════════════════════════════
with col_output:

    btn_l, btn_r = st.columns(2)

    # ── Process Video button ──────────────────────────────────────────────────
    reframe_done = st.session_state["reframe_done"]
    process_clicked = btn_l.button(
        "▶  2. Process Video",
        use_container_width=True,
        type="primary",
        disabled=not reframe_done,
        help="Render the full composited video."
              if reframe_done else "Click 1. Reframe first to enable this.",
    )

    # ── Export button ─────────────────────────────────────────────────────────
    output_ready = st.session_state["output_ready"]
    out_path     = st.session_state["output_path"]

    if output_ready and out_path and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            btn_r.download_button(
                "⬇️  Export",
                data=f,
                file_name="composited_output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        btn_r.button(
            "⬇️  Export",
            use_container_width=True,
            disabled=True,
            help="Process the video first to enable export.",
        )

    st.divider()

    # ── Processing ────────────────────────────────────────────────────────────
    if process_clicked:
        tmp      = tempfile.mkdtemp()
        src_path = os.path.join(tmp, src_file.name)
        bg_path  = os.path.join(tmp, bg_file.name)
        new_out  = os.path.join(tmp, "output.mp4")

        src_file.seek(0); bg_file.seek(0)
        with open(src_path, "wb") as f: f.write(src_file.read())
        with open(bg_path,  "wb") as f: f.write(bg_file.read())

        status    = st.empty()
        bar       = st.progress(0)
        log_box   = st.empty()
        log_lines = []

        def progress_cb(phase, pct, msg):
            bar.progress(min(0.5 * pct if phase == 1 else 0.5 + 0.5 * pct, 1.0))
            status.info(msg)
            log_lines.append(msg)
            log_box.text("\n".join(log_lines[-4:]))

        try:
            process_video(
                src_path=src_path, bg_path=bg_path, out_path=new_out,
                key_hex=key_color, tolerance=tolerance, softness=softness,
                spill=spill, zoom=zoom, pan_x=pan_x, pan_y=pan_y,
                offset_x=offset_x, offset_y=offset_y,
                progress_cb=progress_cb,
            )
            bar.progress(1.0)
            status.success("✓ Done! Download using the Export button above.")
            log_box.empty()
            st.session_state["output_path"]  = new_out
            st.session_state["output_ready"] = True
            st.rerun()   # refresh so Export button activates immediately

        except Exception as e:
            status.error(f"Error: {e}")

    # ── Output video ──────────────────────────────────────────────────────────
    if st.session_state["output_ready"] and st.session_state["output_path"]:
        st.video(st.session_state["output_path"])
    else:
        st.caption("Your rendered video will appear here after processing.")
