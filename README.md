# SnapFix — an agentic repair copilot

SnapFix helps a total beginner fix a broken appliance. Upload a photo
or a short video of the problem and SnapFix:

1. **Diagnoses from the media first** (Muse Spark) — it asks at most
   two short questions, and only when the photo or video cannot answer.
2. **Names the faulty part and highlights it on your own photo**
   (SAM 3.1 segmentation), after a second Spark pass verifies every
   painted region — regions that miss the part are vetoed before you
   see them.
3. **Gives a verdict**: DIY, RISKY (dangerous work — a licensed
   technician is recommended, a DANGER note comes first, and the full
   fix is still explained), STOP (active emergency only), or
   NO FAULT VISIBLE.
4. **Builds the buy card**: the exact part *and every tool the fix
   needs*, with retailer search links (Amazon.ca, PartSelect.ca).
   You confirm and pay at the retailer — SnapFix never checks out.
5. **Walks you through the fix step by step**, one Muse Image
   illustration per step, grounded in your annotated photo, with a
   chat box for follow-up questions about your repair.

Built for the Meta AI Developer Hackathon (Oct 26 – Nov 4, 2026),
Indie track, theme: Push the Frontier. All code in this repository
was written during the event window.

## Meta models used

- **Muse Spark (`muse-spark-1.3`)** — the agent: media-first
  diagnosis, the technician-style interview, and the verification
  pass that vetoes bad highlights.
- **SAM 3.1 (`sam-3.1`)** — open-vocabulary segmentation of the
  suspect parts Spark names, painted back onto the user's photo.
- **Muse Image (`muse-image-1.0`)** — the per-step instructional
  illustrations, edited from the user's own annotated photo.

## Run it locally

```bash
uv sync            # or: pip install -r requirements.txt
cp .env.example .env   # put your MODEL_API_KEY in .env
uv run python snapfix.py washer.jpg     # terminal version
uv run streamlit run app.py             # web app
```

## Deploy (Google Cloud Run)

```bash
gcloud run deploy snapfix --source . \
  --region northamerica-northeast1 \
  --allow-unauthenticated --memory 2Gi --timeout 900 \
  --set-env-vars MODEL_API_KEY=<your key>
```

(For anything beyond a demo, store the key in Secret Manager and pass
`--set-secrets MODEL_API_KEY=<secret-name>:latest` instead.)
