"""SnapFix - diagnose a broken appliance from a photo.

Muse Spark diagnoses from the photo first (questions only when the
photo cannot answer), SAM 3.1 segments the suspect parts, a second
Spark pass verifies each painted region before you see it, and the
survivors are highlighted on your own photo with their names.

Safety: SnapFix always explains how a fault is fixed. Dangerous
repairs get a RISKY verdict with a DANGER note as step 1. Only an
active emergency (gas smell right now, sparks, flooding) returns
STOP with emergency actions.
"""

import asyncio
import base64
import os
import re
import sys
from contextlib import aclosing

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from meta_sam_parser import (
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)

load_dotenv()

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.environ["MODEL_API_KEY"],
)
aclient = AsyncOpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.environ["MODEL_API_KEY"],
)

SPARK = "muse-spark-1.3"
SAM = "sam-3.1"
PRICE_IN, PRICE_OUT = 1.25, 4.25  # USD per 1M tokens (muse-spark-1.3)
SAM_PRICE = 2.50 / 1000           # USD per image

session_total = 0.0

INSTRUCTIONS = """You are SnapFix, an appliance repair expert for total beginners who know nothing about repairs.

RULES:
1. Diagnose from the PHOTO first. Ask a question ONLY when the answer cannot be seen in the photo. Ask at most 2 questions in total, one at a time. If you can already see the fault, ask ZERO questions and output the full result block.
2. Use plain, simple words. The first time you use a technical word, explain it in the same sentence.
3. ALWAYS explain how the fault is fixed - that is the whole point of SnapFix. Never refuse to explain a repair.
4. VERDICT is one of:
   - DIY: a beginner can do this fix safely.
   - RISKY - a licensed technician is recommended: the repair involves gas parts, opening an electrical panel, a sealed refrigerant system, or parts that store dangerous energy (like a microwave capacitor). STILL give the full fix steps, and make step 1 of FIX_STEPS a clear DANGER note that starts with the word DANGER and gives the safety preparation: shut off the gas or the breaker, verify the power is off, discharge stored energy where that applies.
   - STOP - emergency: ONLY when the user reports an active emergency happening right now - smelling gas right now, sparks or smoke, or water flooding onto live electrics. Then FIX_STEPS contains only: leave the area, and call the gas utility or emergency services. No repair steps, PART_NEEDED none, TOOLS_NEEDED none.
   - NO FAULT VISIBLE: the appliance in the photo looks healthy and nothing is clearly broken. CONFIDENCE low, PART_NEEDED none, TOOLS_NEEDED none, SUSPECT_PARTS none, FIX_STEPS explains what was checked.
5. APPLIANCE: name the appliance type and brand if you can read them in the photo. Include the model only if you can actually READ it in the photo - never guess a model number. If the brand is not readable, say "brand and model not readable in photo".
6. PART_NEEDED: if the old part can simply be reconnected, cleaned or adjusted, say so and name a replacement only as a backup. Never sell a part the photo shows is fine.
7. FIX_STEPS (except STOP): numbered steps a beginner can follow, safety preparation first, each step one clear action.

When you are ready - immediately if the photo is enough - output ONLY this block, no other text:

DIAGNOSIS: <what is broken and why it causes the problem, 1-2 short sentences>
CONFIDENCE: <0-100, how sure you are>
VERDICT: <DIY / RISKY - a licensed technician is recommended / STOP - emergency / NO FAULT VISIBLE>
APPLIANCE: <type, brand if readable, model only if readable>
ESTIMATE: <part cost range and time the fix takes>
PART_NEEDED: <exact part name, or none>
TOOLS_NEEDED: <comma-separated short list, or none>
SUSPECT_PARTS: <comma-separated short names of the physical parts in the photo the fault involves - these get highlighted, or none>
FIX_STEPS:
1. <step>
2. <step>"""


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def bill(usage) -> float:
    cost = (usage.input_tokens * PRICE_IN
            + usage.output_tokens * PRICE_OUT) / 1_000_000
    global session_total
    session_total += cost
    return cost


def diagnose(image_b64: str, symptom: str) -> str:
    content = [{"type": "input_text",
                "text": symptom or "The user gave no description - "
                                   "diagnose from the photo alone."},
               {"type": "input_image", "image_url":
                f"data:image/jpeg;base64,{image_b64}"}]
    text = ""
    for _ in range(4):
        r = client.responses.create(model=SPARK, instructions=INSTRUCTIONS,
                                    input=[{"role": "user",
                                            "content": content}])
        print(f"[turn ${bill(r.usage):.4f} | total ${session_total:.4f}]")
        text = r.output_text.strip()
        if text.startswith("DIAGNOSIS:"):
            return text
        print(f"\nSnapFix: {text}")
        answer = ask("You: ")
        content = [{"type": "input_text",
                    "text": answer or "(no answer given)"}]
    return text


def field(block: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.+)$", block, re.MULTILINE)
    return m.group(1).strip() if m else ""


def concepts_of(block: str) -> list[str]:
    raw = field(block, "SUSPECT_PARTS")
    if not raw or raw.lower() == "none":
        return []
    return [c.strip() for c in raw.split(",") if c.strip()]


# --- SAM plumbing: adapt() feeds the parser what the SAM API omits --

_KNOWN_TYPES = {"response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
                "response.failed",
                "response.incomplete"}


def _ev_type(ev):
    if isinstance(ev, dict):
        return ev.get("type", "")
    return getattr(ev, "type", "") or ""


def _to_dict(ev):
    if isinstance(ev, dict):
        return ev
    dump = getattr(ev, "model_dump", None)
    if callable(dump):
        try:
            out = dump()
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    return {"type": _ev_type(ev)}


async def adapt(stream):
    lane = {"item_id": None, "output_index": 0, "content_index": 0}
    parts = []
    already_done = False
    async for ev in stream:
        t = _ev_type(ev)
        if t not in _KNOWN_TYPES or t == "response.output_text.done":
            continue
        d = _to_dict(ev)
        if t == "response.output_text.delta":
            for k in ("item_id", "output_index", "content_index"):
                if d.get(k) is not None:
                    lane[k] = d[k]
            parts.append(d.get("delta", ""))
        if t == "response.content_part.done":
            part = d.get("part") or {}
            if part.get("type") == "output_text":
                already_done = True
        yield d
    if parts and not already_done and lane["item_id"]:
        yield {"type": "response.output_text.done",
               "item_id": lane["item_id"],
               "output_index": lane["output_index"],
               "content_index": lane["content_index"],
               "text": "".join(parts)}


async def segment(image_path: str, concept: str) -> list[dict]:
    global session_total
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    stream = await aclient.responses.create(
        model=SAM, stream=True,
        input=[{"role": "user", "content": [
            {"type": "input_image",
             "image_url": f"data:image/jpeg;base64,{b64}"},
            {"type": "input_text", "text": concept}]}])
    session_total += SAM_PRICE
    parsed = parse_responses_stream(adapt(stream),
                                    image_segmentation_format())
    async with aclosing(parsed) as p:
        result = await p.final_result()
    try:
        await stream.close()
    except Exception:
        pass
    boxes = {r.object_id: r for r in result.records
             if type(r).__name__ == "SegmentationBoxRecord"}
    out = []
    for rec in result.records:
        if type(rec).__name__ != "SegmentationMaskRecord":
            continue
        box = boxes.get(rec.object_id)
        left = box.left if box else rec.bounds.left
        top = box.top if box else rec.bounds.top
        right = box.right if box else rec.bounds.right
        bottom = box.bottom if box else rec.bounds.bottom
        out.append({"mask": rec.mask,
                    "rel_box": [round(left, 1), round(top, 1),
                                round(right, 1), round(bottom, 1)]})
    return out


# --- verification: Spark vetoes painted regions that miss the part --

PALETTE = [(255, 60, 60), (60, 220, 80), (60, 120, 255),
           (255, 170, 40), (200, 80, 255), (40, 220, 220)]


def overlay(image_path: str, items: list[dict],
            out_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    rgba = img.convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for k, it in enumerate(items):
        mask = it["mask"]
        raster = decode_mask_to_raster(mask)
        m = Image.frombytes("L", (mask.width, mask.height), raster)
        b = it["rel_box"]
        x0, y0 = int(b[0] / 1000 * W), int(b[1] / 1000 * H)
        x1, y1 = int(b[2] / 1000 * W), int(b[3] / 1000 * H)
        m = m.resize((max(x1 - x0, 1), max(y1 - y0, 1)), Image.NEAREST)
        color = PALETTE[k % len(PALETTE)] + (165,)
        solid = Image.new("RGBA", m.size, color)
        layer.paste(solid, (x0, y0), m)
        edge = ImageChops.subtract(m.filter(ImageFilter.MaxFilter(5)), m)
        rim = Image.new("RGBA", m.size,
                        PALETTE[k % len(PALETTE)] + (255,))
        layer.paste(rim, (x0, y0), edge)
    rgba = Image.alpha_composite(rgba, layer)
    draw = ImageDraw.Draw(rgba)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, W // 45))
    except Exception:
        font = ImageFont.load_default()
    for k, it in enumerate(items):
        b = it["rel_box"]
        x = int(b[0] / 1000 * W)
        y = max(0, int(b[1] / 1000 * H) - (font.size + 12))
        label = f"{it['num']} - {it['part']}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x, y, x + tw + 14, y + font.size + 10],
                       fill=(0, 0, 0, 235))
        draw.text((x + 7, y + 5), label, fill=(255, 255, 255), font=font)
    rgba.convert("RGB").save(out_path)
    return rgba.convert("RGB")


def verify(orig_b64: str, check_b64: str, items: list[dict]) -> set[int]:
    lines = "\n".join(
        f"{it['num']}: {it['part']}" for it in items)
    r = client.responses.create(
        model=SPARK,
        instructions=(
            "You verify repair highlights. You are given TWO images of "
            "the same appliance: FIRST the original photo, SECOND the "
            "same photo with numbered colored regions painted on it. "
            "Compare them. Each chip on the painted image names its "
            "region, like '1 - drain hose'. For each number, decide: "
            "KEEP if the paint substantially overlaps the named part "
            "and follows its shape - some paint touching the surface "
            "the part lies on (a grate, the floor, a cabinet wall) is "
            "fine. VETO only if the paint lies mostly on a different "
            "object, floats beside the part, or is just a thin trace "
            "along its outline while sitting off the part. Answer "
            "with ONLY one line per number, exactly: <number>: KEEP "
            "or <number>: VETO"),
        input=[{"role": "user", "content": [
            {"type": "input_text",
             "text": f"Numbered regions:\n{lines}"},
            {"type": "input_image", "image_url":
             f"data:image/jpeg;base64,{orig_b64}"},
            {"type": "input_image", "image_url":
             f"data:image/jpeg;base64,{check_b64}"}]}])
    print(f"[verify ${bill(r.usage):.4f}]")
    keep = set()
    for line in r.output_text.splitlines():
        m = re.match(r"\s*(\d+)\s*:\s*(KEEP|VETO)", line, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if m.group(2).upper() == "KEEP":
                keep.add(n)
            print(f"verify {n} ({items[n - 1]['part']}): "
                  f"{m.group(2).upper()}")
    return keep


async def highlight(image_path: str, concepts: list[str]) -> None:
    with open(image_path, "rb") as f:
        orig_b64 = base64.b64encode(f.read()).decode()
    items: list[dict] = []
    survivors: list[dict] = []
    for attempt in (1, 2):
        items = []
        for concept in concepts:
            segs = await segment(image_path, concept)
            print(f"{concept!r}: masks={len(segs)}")
            for s in segs:
                items.append({"part": concept, "mask": s["mask"],
                              "rel_box": s["rel_box"],
                              "num": len(items) + 1})
        if not items:
            print("No masks found - nothing to highlight.")
            return
        overlay(image_path, items, "numbered_check.jpg")
        with open("numbered_check.jpg", "rb") as f:
            check_b64 = base64.b64encode(f.read()).decode()
        keep = verify(orig_b64, check_b64, items)
        survivors = [it for it in items if it["num"] in keep]
        if survivors:
            break
        print(f"Attempt {attempt}: verification vetoed every region.")
    verified = bool(survivors)
    shown = survivors if verified else items
    if not verified:
        print("Showing the highlight WITHOUT verification - "
              "treat it as approximate.")
    final = overlay(image_path, shown, "snapfix_result.jpg")
    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, final.size[0] // 45))
    except Exception:
        font = ImageFont.load_default()
    seen = {}
    for it in shown:
        seen.setdefault(it["part"], it)
    y = 10
    for part, it in seen.items():
        color = PALETTE[(it["num"] - 1) % len(PALETTE)]
        tw = draw.textlength(part, font=font)
        draw.rectangle([10, y, 24 + tw + 10, y + font.size + 10],
                       fill=(0, 0, 0, 220))
        draw.rectangle([14, y + 5, 28, y + font.size + 5], fill=color)
        draw.text((36, y + 5), part, fill=(255, 255, 255), font=font)
        y += font.size + 16
    final.save("snapfix_result.jpg")
    print("saved snapfix_result.jpg")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python snapfix.py <photo.jpg>")
        return
    image_path = sys.argv[1]
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    symptom = ask("Describe the problem: ")
    print("\nSnapFix:", end=" ")
    block = diagnose(image_b64, symptom)
    print(block)
    verdict = field(block, "VERDICT").upper()
    fixable = ("DIY" in verdict or "RISKY" in verdict) \
        and "STOP" not in verdict
    concepts = concepts_of(block) if fixable else []
    if concepts:
        print(f"\nHighlighting: {concepts}")
        asyncio.run(highlight(image_path, concepts))
    else:
        print("\nNo highlighting for this verdict.")
    print(f"[session total ≈ ${session_total:.4f}]")


if __name__ == "__main__":
    main()
