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
st.caption("Replace a sliding coloured placeholder in a video with a replacement clip.")

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    src_file = st.file_uploader("① Source video  (contains the coloured placeholder)",
                                 type=["mp4", "mov", "avi", "mkv"])
    bg_file  = st.file_uploader("② Replacement video  (fills the placeholder)",
                                 type=["mp4", "mov", "avi", "mkv"])

    st.subheader("🎨 Key settings")
    KEY_PRESETS = {
        "Green  (#00b140)": "#00b140",
        "Blue   (#0047FF)": "#0047FF",
        "Custom": None,
    }
    preset_choice = st.selectbox(
        "Placeholder colour",
        list(KEY_PRESETS.keys()),
        help=(
            "Choose the colour of the solid rectangle in your source video.\n"
            "• Green / Blue are standard chroma-key colours\n"
            "• Custom: type any hex code (use imagecolorpicker.com to sample the exact colour)"
        ),
    )
    if KEY_PRESETS[preset_choice] is not None:
        key_color = KEY_PRESETS[preset_choice]
        st.caption(f"Using `{key_color}`")
    else:
        key_color = st.text_input(
            "Custom hex code",
            value="#00b140",
            help="Enter any hex colour, e.g. #00b140. Must start with #.",
        )
    tolerance = st.slider(
        "Tolerance",
        0, 180, 60,
        help=(
            "How far (in RGB Euclidean distance) a pixel can be from the key colour "
            "and still be replaced.\n"
            "• Range: 0–180\n"
            "• Too low → placeholder colour not fully removed (green patches remain)\n"
            "• Too high → nearby colours also get replaced\n"
            "• Default 60 works well for a clean solid-colour placeholder."
        ),
    )
    softness = st.slider(
        "Softness",
        0, 80, 20,
        help=(
            "Width of the feathered transition zone at placeholder edges (pixels).\n"
            "• Range: 0–80 px\n"
            "• 0 = hard, crisp cut — best for a perfectly solid placeholder\n"
            "• Higher = smoother blend — helps if the placeholder has slight antialiasing\n"
            "• Default 20."
        ),
    )
    spill = st.slider(
        "Spill suppress",
        0.0, 1.0, 0.8, step=0.05,
        help=(
            "Reduces colour cast (spill) from the key colour that bleeds onto "
            "objects sitting in front of the placeholder.\n"
            "• Range: 0.0–1.0\n"
            "• 0 = no correction\n"
            "• 1 = maximum correction (may slightly desaturate edges)\n"
            "• Default 0.8."
        ),
    )

    with st.expander("✂️ Framing (optional)"):
        col1, col2 = st.columns(2)
        zoom = col1.slider(
            "Zoom",
            1.0, 4.0, 1.0, step=0.1,
            help=(
                "Zoom into the replacement video.\n"
                "• Range: 1.0–4.0×\n"
                "• 1.0 = fit the whole replacement clip into the placeholder\n"
                "• 2.0 = show half the clip (cropped in)\n"
                "• Use Pan X/Y to choose which part is visible after zooming."
            ),
        )
        col2.write("")   # spacer

        pan_x = col1.slider(
            "Pan X",
            0.0, 1.0, 0.0, step=0.01,
            help=(
                "Horizontal crop position in the replacement video.\n"
                "• Range: 0.0 (left edge) → 1.0 (right edge)\n"
                "• Only visible when Zoom > 1.0 or the replacement clip is wider than the placeholder."
            ),
        )
        pan_y = col2.slider(
            "Pan Y",
            0.0, 1.0, 0.0, step=0.01,
            help=(
                "Vertical crop position in the replacement video.\n"
                "• Range: 0.0 (top) → 1.0 (bottom)\n"
                "• Only visible when Zoom > 1.0 or the replacement clip is taller than the placeholder."
            ),
        )
        offset_x = col1.slider(
            "Offset X (px)",
            -200, 200, 0,
            help=(
                "Nudge the replacement image left or right inside the placeholder.\n"
                "• Range: −200 → +200 px\n"
                "• Positive = shift right, Negative = shift left\n"
                "• Useful to correct tiny mis-registration."
            ),
        )
        offset_y = col2.slider(
            "Offset Y (px)",
            -200, 200, 0,
            help=(
                "Nudge the replacement image up or down inside the placeholder.\n"
                "• Range: −200 → +200 px\n"
                "• Positive = shift down, Negative = shift up."
            ),
        )

    process = st.button("▶  Process Video", type="primary", use_container_width=True)

# ── Right column: preview + output ─────────────────────────────────────────────
with right:

    # ── Live preview ──────────────────────────────────────────────────────────
    if src_file and bg_file:
        # Save uploads to stable temp paths keyed by file names + sizes
        # so we only re-scan when files actually change.
        cache_key = f"{src_file.name}_{src_file.size}_{bg_file.name}_{bg_file.size}"

        if st.session_state.get("preview_cache_key") != cache_key:
            # Files changed — save to disk and re-scan
            tmp_prev = tempfile.mkdtemp()
            src_prev_path = os.path.join(tmp_prev, src_file.name)
            bg_prev_path  = os.path.join(tmp_prev, bg_file.name)
            src_file.seek(0)
            bg_file.seek(0)
            with open(src_prev_path, "wb") as f: f.write(src_file.read())
            with open(bg_prev_path,  "wb") as f: f.write(bg_file.read())

            with st.spinner("Scanning for preview frame…"):
                pv = prepare_preview(src_prev_path, bg_prev_path,
                                     key_hex=key_color,
                                     tolerance=tolerance)

            st.session_state["preview_data"]      = pv
            st.session_state["preview_cache_key"] = cache_key
            st.session_state["preview_src_path"]  = src_prev_path
            st.session_state["preview_bg_path"]   = bg_prev_path
        else:
            pv = st.session_state.get("preview_data")

        if pv is not None:
            st.subheader("🖼 Framing Preview")
            st.caption("Updates live as you move the Framing sliders.")

            composed = composite_single_frame(
                pv,
                key_hex=key_color,
                tolerance=tolerance,
                softness=softness,
                spill=spill,
                zoom=zoom,
                pan_x=pan_x,
                pan_y=pan_y,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            if composed is not None:
                # Convert BGR → RGB for st.image
                st.image(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
        else:
            st.info("ℹ️ No placeholder found with current Key Color / Tolerance — "
                    "adjust them and the preview will appear here.")

    # ── Processing output ─────────────────────────────────────────────────────
    st.subheader("Output")

    if process:
        if not src_file:
            st.error("Please upload a source video.")
        elif not bg_file:
            st.error("Please upload a replacement video.")
        else:
            tmp = tempfile.mkdtemp()

            src_path = os.path.join(tmp, src_file.name)
            bg_path  = os.path.join(tmp, bg_file.name)
            out_path = os.path.join(tmp, "output.mp4")

            src_file.seek(0)
            bg_file.seek(0)
            with open(src_path, "wb") as f: f.write(src_file.read())
            with open(bg_path,  "wb") as f: f.write(bg_file.read())

            status    = st.empty()
            bar       = st.progress(0)
            log_box   = st.empty()
            log_lines = []

            def progress_cb(phase, pct, msg):
                overall = pct / 2 if phase == 1 else 0.5 + pct / 2
                bar.progress(min(overall, 1.0))
                status.info(msg)
                log_lines.append(msg)
                log_box.text("\n".join(log_lines[-6:]))

            try:
                process_video(
                    src_path=src_path, bg_path=bg_path, out_path=out_path,
                    key_hex=key_color, tolerance=tolerance, softness=softness,
                    spill=spill, zoom=zoom,
                    pan_x=pan_x, pan_y=pan_y,
                    offset_x=offset_x, offset_y=offset_y,
                    progress_cb=progress_cb,
                )
                bar.progress(1.0)
                status.success("✓ Done!")

                st.video(out_path)

                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️  Download output",
                        data=f,
                        file_name="composited_output.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

            except Exception as e:
                status.error(f"Error: {e}")
