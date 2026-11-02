"""SnapFix web app - a thin page over snapfix.py.

Upload a photo or short video -> the proven snapfix.py pipeline runs
-> diagnosis, verified highlight, automatic brand/model identification,
a checkout list with the part AND every tool, a step-by-step
illustrated tutorial (one image per fix step, generated automatically),
and a chat box for follow-up questions about the repair.

The page shows the user only the diagnosis - engine cost and debug
lines are stripped before anything is displayed.
"""

import base64
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="SnapFix", page_icon="🔧")
st.title("🔧 SnapFix")
st.write(
    "Upload a photo or a short video of the broken appliance. SnapFix "
    "diagnoses the fault, highlights the failed part, identifies the "
    "brand and model, lists every material to check out, and walks "
    "you through the fix step by step, with a picture for every step. "
    "If a repair is dangerous, SnapFix says so clearly and shows the "
    "safe way."
)

VIDEO_EXTS = (".mp4", ".mov", ".webm")
RESULT = Path("snapfix_result.jpg")
UPLOAD = Path("upload.jpg")

if "last_run" not in st.session_state:
    st.session_state.last_run = ""
if "run_failed" not in st.session_state:
    st.session_state.run_failed = False
if "chat" not in st.session_state:
    st.session_state.chat = []
if "guide" not in st.session_state:
    st.session_state.guide = {}
if "guide_attempted" not in st.session_state:
    st.session_state.guide_attempted = False
if "appliance_id" not in st.session_state:
    st.session_state.appliance_id = ""


def best_frame(video_path: Path, still_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best, best_score = None, -1.0
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, frame = cap.read()
        if not ok:
            continue
        score = cv2.Laplacian(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var()
        if score > best_score:
            best, best_score = frame, score
    cap.release()
    if best is None:
        return False
    cv2.imwrite(str(still_path), best)
    return True


def field(text: str, name: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def steps_of(text: str) -> list[str]:
    steps, on = [], False
    for line in text.splitlines():
        if line.strip().startswith("FIX_STEPS:"):
            on = True
            continue
        if on:
            m = re.match(r"\s*\d+[.)]\s*(.*)", line)
            if m and m.group(1).strip():
                steps.append(m.group(1).strip())
    return steps


def clean_output(raw: str) -> str:
    """Keep ONLY the diagnosis block for display - no cost lines, no
    engine debug lines."""
    lines = raw.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.startswith("DIAGNOSIS:")), None)
    if start is None:
        return ""
    kept = []
    for l in lines[start:]:
        if re.match(r"^(Highlighting:|\[session|saved |Attempt \d|"
                    r"Showing the highlight|No masks|No highlighting)",
                    l):
            break
        kept.append(l)
    return "\n".join(kept).strip()


def api_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=os.environ["MODEL_API_KEY"],
    )


def read_appliance_id(image_bytes: bytes) -> str:
    """Read brand + model automatically from a photo of the appliance
    front, logo, control panel or model plate. Transcribes only text
    actually visible - never guesses."""
    b64 = base64.b64encode(image_bytes).decode()
    r = api_client().responses.create(
        model="muse-spark-1.3",
        instructions=(
            "Read the appliance brand and model from this photo. "
            "Transcribe ONLY text you can actually read on a badge, "
            "logo, control panel or model plate - never guess. Reply "
            "with exactly two lines:\n"
            "BRAND: <brand, or NOT READABLE>\n"
            "MODEL: <model, or NOT READABLE>"
        ),
        input=[{"role": "user", "content": [
            {"type": "input_image",
             "image_url": f"data:image/jpeg;base64,{b64}"}]}])
    brand = field(r.output_text, "BRAND")
    model = field(r.output_text, "MODEL")
    parts = [p for p in (brand, model)
             if p and p.upper() != "NOT READABLE"]
    return " ".join(parts)


media = st.file_uploader(
    "Photo or short video of the problem",
    type=["jpg", "jpeg", "png", "mp4", "mov", "webm"],
)
symptom = st.text_input("What do you see or hear? (optional)", "")

if media is not None:
    if media.name.lower().endswith(VIDEO_EXTS):
        st.video(media)
    else:
        st.image(media, caption="Your photo", use_container_width=True)

if st.button("Diagnose", type="primary", disabled=media is None):
    if media.name.lower().endswith(VIDEO_EXTS):
        Path("upload_video.mp4").write_bytes(media.getvalue())
        if not best_frame(Path("upload_video.mp4"), UPLOAD):
            st.error("Could not read a frame from that video.")
            st.stop()
    else:
        UPLOAD.write_bytes(media.getvalue())

    RESULT.unlink(missing_ok=True)
    for old in Path(".").glob("guide_step_*.jpg"):
        old.unlink(missing_ok=True)
    with st.status("SnapFix is working...", expanded=True) as status:
        st.write("Reading your photo and diagnosing the fault...")
        st.write("Highlighting the faulty part and verifying it...")
        result = subprocess.run(
            [sys.executable, "snapfix.py", "upload.jpg"],
            input=symptom + "\n\n\n",
            capture_output=True, text=True, timeout=600,
        )
        status.update(label="Diagnosis complete", state="complete")
    raw = result.stdout or result.stderr
    st.session_state.last_run = clean_output(raw)
    st.session_state.run_failed = not st.session_state.last_run
    st.session_state.chat = []
    st.session_state.guide = {}
    st.session_state.guide_attempted = False
    st.session_state.appliance_id = ""
    st.rerun()

out = st.session_state.last_run
if out:
    st.code(out)

    verdict = field(out, "VERDICT")
    v = verdict.upper()
    fixable = ("DIY" in v or "RISKY" in v) and "STOP" not in v

    if RESULT.exists():
        st.image(str(RESULT), caption="Faulty part highlighted",
                 use_container_width=True)

    # --- automatic brand / model identification ----------------------
    appliance = field(out, "APPLIANCE")
    if "not readable" in appliance.lower() \
            and not st.session_state.appliance_id:
        st.info(
            "No brand or model badge is visible in that photo. Add "
            "one photo of the front of the appliance or its model "
            "plate and SnapFix reads the brand and model "
            "automatically - nothing to type."
        )
        plate = st.file_uploader(
            "Photo of the front panel or model plate",
            type=["jpg", "jpeg", "png"], key="plate",
        )
        if plate is not None:
            with st.spinner("Reading the badge..."):
                found = read_appliance_id(plate.getvalue())
            if found:
                st.session_state.appliance_id = found
                st.rerun()
            else:
                st.warning("The badge in that photo is not readable "
                           "either - the checkout list below uses the "
                           "part name alone.")
    effective_appliance = st.session_state.appliance_id or appliance
    if st.session_state.appliance_id:
        st.success(f"Appliance identified: "
                   f"**{st.session_state.appliance_id}**")

    # --- checkout list: the part AND every tool ----------------------
    raw_part = field(out, "PART_NEEDED")
    has_part = bool(raw_part) and not raw_part.lower().startswith("none")
    backup_only = has_part and "backup" in raw_part.lower()
    tool_list = [t.strip() for t in field(out, "TOOLS_NEEDED").split(",")
                 if t.strip() and t.strip().lower() != "none"]
    if fixable and (has_part or tool_list):
        st.subheader("🧾 Checkout — materials for this fix")
        if effective_appliance:
            st.write(f"Appliance: **{effective_appliance}**")
        estimate = field(out, "ESTIMATE")
        if estimate:
            st.write(f"Estimated cost & time: **{estimate}**")
        n = 0
        if has_part:
            n += 1
            label = "Backup part — buy only if the old one is damaged: " \
                if backup_only else "Part: "
            st.markdown(f"**{n}. {label}{raw_part}**")
            q = quote_plus((effective_appliance + " "
                            if effective_appliance else "") + raw_part)
            st.link_button("Find this part on Amazon.ca",
                           f"https://www.amazon.ca/s?k={q}")
            st.link_button(
                "Find this part on PartSelect.ca",
                "https://www.google.com/search?q="
                + quote_plus(f"site:partselect.ca "
                             f"{effective_appliance} {raw_part}"))
        else:
            st.write("✅ **No part to buy** — the existing part can be "
                     "cleaned and reused.")
        for tool in tool_list:
            n += 1
            st.markdown(f"**{n}. Tool: {tool}**")
            st.link_button(
                f"Find {tool} on Amazon.ca",
                "https://www.amazon.ca/s?k=" + quote_plus(tool),
            )
        st.caption("One list with everything the fix needs. You "
                   "confirm and pay at the retailer — SnapFix never "
                   "checks out for you.")

    # --- step-by-step tutorial, one image per step, automatic --------
    steps = steps_of(out)
    guide_base = RESULT if RESULT.exists() else UPLOAD
    if fixable and steps and guide_base.exists() \
            and not st.session_state.guide \
            and not st.session_state.guide_attempted:
        st.session_state.guide_attempted = True
        total = min(len(steps), 8)
        bar = st.progress(0.0, text="Drawing your step-by-step guide...")
        for i, step in enumerate(steps[:8], start=1):
            prompt = (
                "Create a clean, beginner-friendly instructional "
                f"illustration for step {i} of this repair: "
                f"{step} Base it on this photo of the user's actual "
                "appliance and keep the parts consistent with it: "
                f"{field(out, 'SUSPECT_PARTS')}. Show hands and "
                "tools performing exactly this action. Flat "
                "instructional style, high contrast, "
                "uncluttered. No text or words in the image."
            )
            try:
                with guide_base.open("rb") as f:
                    made = api_client().images.edit(
                        model="muse-image-1.0", image=f, prompt=prompt)
                item = made.data[0]
                path = Path(f"guide_step_{i}.jpg")
                if item.b64_json:
                    path.write_bytes(base64.b64decode(item.b64_json))
                else:
                    path.write_bytes(requests.get(
                        item.url, timeout=60).content)
                st.session_state.guide[i] = str(path)
            except Exception as e:
                st.warning(f"Step {i} image failed: {e}")
            bar.progress(i / total, text=f"Step {i} of {total}...")
        st.rerun()

    if steps and st.session_state.guide:
        st.subheader("🧭 Your step-by-step guide")
        for i, step in enumerate(steps[:8], start=1):
            st.markdown(f"**Step {i}.** {step}")
            p = st.session_state.guide.get(i)
            if p and Path(p).exists():
                st.image(p, use_container_width=True)

    st.subheader("💬 Ask SnapFix about this repair")
    for role, msg in st.session_state.chat:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg)
    q = st.chat_input("Ask about this repair...")
    if q:
        convo = list(st.session_state.chat)
        convo.append(("user", q))

        def as_input(role: str, msg: str) -> dict:
            part = {"type": "input_text" if role == "user"
                    else "output_text", "text": msg}
            return {"role": role, "content": [part]}

        with st.chat_message("user"):
            st.markdown(q)
        try:
            with st.spinner("SnapFix is thinking..."):
                r = api_client().responses.create(
                    model="muse-spark-1.3",
                    instructions=(
                        "You are SnapFix, a repair guide for beginners. "
                        "Below is the diagnosis you already gave for the "
                        "user's photo or video. Answer follow-up questions "
                        "about THIS repair in plain, simple words. If a "
                        "step is dangerous, say so clearly and give the "
                        "safe way to do it.\n\n" + out
                    ),
                    input=[as_input(role, msg) for role, msg in convo],
                )
            st.session_state.chat = convo + [("assistant",
                                              r.output_text)]
        except Exception as e:
            st.session_state.chat = convo
            st.warning(f"SnapFix could not answer just now: {e}")
        st.rerun()
elif st.session_state.run_failed:
    st.error("The diagnosis did not complete. Please try again.")
