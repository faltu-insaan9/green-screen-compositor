# Green Screen Compositor — v1

Replaces a sliding coloured placeholder rectangle in a video with a replacement video clip. The replacement video tracks the placeholder exactly as it slides in, holds, and slides out.

---

## Option A — Run online (HuggingFace Spaces)

No installation needed. Your team just opens a URL.

**One-time setup (5 minutes, free):**

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click **New Space** → give it a name → set SDK to **Gradio** → Hardware: **CPU Basic (free)**
3. Upload these four files to the Space:
   - `app.py`
   - `composite_core.py`
   - `requirements.txt`
   - `packages.txt`
4. Wait ~2 minutes for it to build
5. Share the Space URL with your team — that's it

> **Note:** Videos you upload are processed on HuggingFace's servers. If your content is confidential, use Option B instead.

---

## Option B — Run locally

Your team needs Python installed once, then just one command each time.

**Install (one time):**
```bash
pip install -r requirements.txt
```

> Also needs ffmpeg: `brew install ffmpeg` (Mac) or `sudo apt install ffmpeg` (Linux)

**Run:**
```bash
python app.py
```
Opens automatically at **http://localhost:7860** in your browser.

---

## How to use

1. **Upload source video** — the video containing the coloured placeholder rectangle
2. **Upload replacement video** — the clip you want to fill the placeholder
3. **Key Settings:**
   - *Placeholder colour* — must exactly match the colour of the rectangle in the source video. Click the colour swatch to open a picker, or type the hex code
   - *Tolerance* — how strict the colour matching is (default 60 works for most cases)
   - *Softness* — smoothness of the edge (0 = sharp, higher = feathered)
   - *Spill suppress* — reduces colour cast on edges of objects overlaid on the placeholder
4. **Framing** (optional) — use Pan X/Y to choose which part of the replacement video is visible, and Offset X/Y to nudge its position
5. Click **Process Video**
6. Wait for both phases to complete (progress bar shows status)
7. Preview the result and click the download button below the video

---

## Processing time

| Video length | Approximate time |
|---|---|
| 15 seconds | ~3–5 min |
| 30 seconds | ~6–10 min |
| 60 seconds | ~12–20 min |

Times are for a typical laptop CPU. HuggingFace free tier is similar.
