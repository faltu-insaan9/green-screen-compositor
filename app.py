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

# ── Workflow guide ─────────────────────────────────────────────────────────────
st.info(
    "**How to use:** &nbsp; ① Upload both videos &nbsp;→&nbsp; "
    "② Adjust key settings &nbsp;→&nbsp; "
    "③ *(Optional)* Open **Framing** to preview & adjust crop &nbsp;→&nbsp; "
    "④ Click **▶ Process Video** to render the final output",
    icon="👆",
)

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([11, 9])

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
            "• Too low → placeholder colour not fully removed (colour patches remain)\n"
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

    # ── Framing expander (with live preview inside) ───────────────────────────
    with st.expander("✂️ Framing  *(optional — open to adjust crop & preview)*"):
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
        col2.write("")

        pan_x = col1.slider(
            "Pan X",
            0.0, 1.0, 0.0, step=0.01,
            help=(
                "Horizontal crop position in the replacement video.\n"
                "• Range: 0.0 (left edge) → 1.0 (right edge)\n"
                "• Only has effect when Zoom > 1.0."
            ),
        )
        pan_y = col2.slider(
            "Pan Y",
            0.0, 1.0, 0.0, step=0.01,
            help=(
                "Vertical crop position in the replacement video.\n"
                "• Range: 0.0 (top) → 1.0 (bottom)\n"
                "• Only has effect when Zoom > 1.0."
            ),
        )
        offset_x = col1.slider(
            "Offset X (px)",
            -200, 200, 0,
            help=(
                "Nudge the replacement image left or right inside the placeholder.\n"
                "• Range: −200 → +200 px\n"
                "• Positive = shift right, Negative = shift left."
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

        # ── Live preview (inside framing expander) ────────────────────────────
        if src_file and bg_file:
            st.divider()
            st.caption("📐 **Reference frame** — shows how your replacement clip will be framed. "
                       "Adjust sliders above until it looks right, then close this and click ▶ Process Video.")

            cache_key = f"{src_file.name}_{src_file.size}_{bg_file.name}_{bg_file.size}"

            if st.session_state.get("preview_cache_key") != cache_key:
                tmp_prev      = tempfile.mkdtemp()
                src_prev_path = os.path.join(tmp_prev, src_file.name)
                bg_prev_path  = os.path.join(tmp_prev, bg_file.name)
                src_file.seek(0)
                bg_file.seek(0)
                with open(src_prev_path, "wb") as f: f.write(src_file.read())
                with open(bg_prev_path,  "wb") as f: f.write(bg_file.read())

                with st.spinner("Scanning for best preview frame…"):
                    pv = prepare_preview(src_prev_path, bg_prev_path,
                                         key_hex=key_color, tolerance=tolerance)

                st.session_state["preview_data"]      = pv
                st.session_state["preview_cache_key"] = cache_key
            else:
                pv = st.session_state.get("preview_data")

            if pv is not None:
                composed = composite_single_frame(
                    pv,
                    key_hex=key_color, tolerance=tolerance,
                    softness=softness, spill=spill,
                    zoom=zoom, pan_x=pan_x, pan_y=pan_y,
                    offset_x=offset_x, offset_y=offset_y,
                )
                if composed is not None:
                    # Show at fixed pixel width so it doesn't dominate the screen
                    st.image(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB),
                             width=520, caption="Preview frame (not the final video)")
            else:
                st.warning("No placeholder found with current Key Colour / Tolerance — "
                           "adjust them and reopen this section.")
        else:
            st.caption("Upload both videos to see the framing preview here.")

    st.button("▶  Process Video", type="primary", use_container_width=True, key="process_btn")

# ── Right column: output only ──────────────────────────────────────────────────
with right:
    st.subheader("Output")

    process = st.session_state.get("process_btn", False)

    if process:
        if not src_file:
            st.error("Please upload a source video.")
        elif not bg_file:
            st.error("Please upload a replacement video.")
        else:
            tmp      = tempfile.mkdtemp()
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
                log_box.text("\n".join(log_lines[-4:]))

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
                status.success("✓ Done! Watch the video below or download it.")
                log_box.empty()

                # Show video at constrained size
                vid_col, _ = st.columns([3, 1])
                vid_col.video(out_path)

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
    else:
        st.caption("Your rendered video will appear here after you click ▶ Process Video.")
