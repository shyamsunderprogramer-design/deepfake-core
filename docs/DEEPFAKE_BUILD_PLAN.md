# Deepfake Face-Swap Feature — End-to-End Build Plan

**Target platform:** macOS desktop app on Apple Silicon (M1/M2/M3/M4)
**Primary goal:** Face-swapping with high identity-matching accuracy, in two modes:
1. **Offline/high-quality mode** — DeepFaceLab-style trained-model pipeline (best accuracy, hours of training per face pair).
2. **Real-time mode** — single-photo swap on live video (DeepFaceLive / Deep Live Cam style, 30–60 FPS).

**Document status:** v1.0 — research synthesis + architecture + implementation roadmap.

---

## Table of Contents

1. [Scope & Use Cases](#1-scope--use-cases)
2. [Responsible Use — Mandatory Requirements](#2-responsible-use--mandatory-requirements)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Pipeline Stages (Detailed)](#4-pipeline-stages-detailed)
5. [Mode A — Offline Trained-Model Pipeline (DeepFaceLab-style)](#5-mode-a--offline-trained-model-pipeline)
6. [Mode B — Real-Time Single-Photo Swap](#6-mode-b--real-time-single-photo-swap)
7. [Apple Silicon Performance Strategy](#7-apple-silicon-performance-strategy)
8. [Data Acquisition & Quality](#8-data-acquisition--quality)
9. [Training Configurations](#9-training-configurations)
10. [Blending, Masking & Enhancement Stack](#10-blending-masking--enhancement-stack)
11. [Accuracy Evaluation & Metrics](#11-accuracy-evaluation--metrics)
12. [Tech Stack & Dependencies](#12-tech-stack--dependencies)
13. [Project Structure](#13-project-structure)
14. [Milestones & Deliverables](#14-milestones--deliverables)
15. [Risks & Mitigations](#15-risks--mitigations)
16. [References](#16-references)

---

## 1. Scope & Use Cases

### 1.1 What we are building

A macOS application that:

- Accepts a **source identity** (a photo set or video of person A) and a **target** (video or live camera of person B).
- Produces video where person B's face is replaced with person A's identity, preserving B's pose, expression, and lighting.
- Runs in two modes:
  - **Offline:** train a per-pair model for maximum fidelity (film/VFX-quality output).
  - **Real-time:** swap from a single reference photo at 30–60 FPS for streaming, preview, and rapid iteration.

### 1.2 Legitimate use cases (in scope)

- Film/VFX pre-visualization and post-production with signed talent releases.
- Consented creative content (parody/satire where lawful, with disclosure).
- Corporate/educational media produced with documented consent.
- Accessibility and dubbing localization (actor consents to face+lipsync).
- Personal/artistic projects using your own likeness or licensed likenesses.

### 1.3 Explicitly out of scope (hard exclusions)

- Non-consensual intimate imagery (criminal in most jurisdictions; e.g., UK Sexual Offences Act, US state laws such as CA AB 602 and TX SB 751).
- Impersonation for fraud, authentication bypass, or deception.
- Evasion of deepfake detectors or provenance systems (C2PA, watermark removers).
- Political disinformation or election-related manipulation.
- Any scraping/use of a person's likeness without documented consent.

These are architectural constraints, not policy text: consent verification and watermarking are built into the pipeline (Section 2), not bolted on.

---

## 2. Responsible Use — Mandatory Requirements

Every output of this system must satisfy all of the following. Treat these as release gates.

### 2.1 Consent ledger

- A signed consent record for every source identity is stored in `consent/<identity_id>.json` with: identity reference hash, scope (projects/uses permitted), expiry, revocation contact.
- The pipeline **refuses to run** a source identity without a valid, unexpired consent record.
- Revocation immediately invalidates trained models derived from that identity (models are deleted, not just disabled).

### 2.2 Provenance & watermarking

- **Visible disclosure:** rendered output carries an unobtrusive but human-visible "synthetic media" badge by default (toggleable only for licensed VFX deliveries that carry separate contractual disclosure).
- **Invisible watermark:** embed a robust provenance mark (e.g., C2PA content credentials via `c2pa-rs`; fallback: frequency-domain watermark surviving re-encode/recompress).
- **Metadata:** every output file embeds generator ID, model version, consent record ID, and creation timestamp.

### 2.3 Legal compliance checklist

| Region | Relevant law (examples) | Obligation |
|---|---|---|
| UK | Sexual Offences Act (intimate image offences), Online Safety Act | No intimate deepfakes; platform duties if hosting |
| US (CA) | AB 602 (deceptive audiovisual media), AB 730 (elections) | Disclosure + consent |
| US (TX) | SB 751 | Election-period restrictions |
| EU | AI Act (deepfake transparency obligations, in force 2026) | Mandatory labeling of synthetic content |
| India | DPDP Act; election deepfake advisory | Consent for biometric-like likeness use |

> This table is a starting point, not legal advice. Before any commercial release, get counsel review for distribution regions.

### 2.4 Defensive features (ship by default)

- A **detector self-check** tool: run a third-party open deepfake detector against our own output to confirm watermarks/provenance survive transcoding (this validates durability — it is not used to evade detection; outputs must remain detectable).
- Export restrictions: no export path that strips provenance metadata.

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        macOS App (Swift/SwiftUI shell)          │
│  ┌───────────────┐   ┌──────────────────────────────────────┐  │
│  │ Consent &     │   │        Python inference core          │  │
│  │ provenance    │──▶│  (ONNX Runtime + CoreML EP)           │  │
│  │ gate          │   │                                      │  │
│  └───────────────┘   │  detect → align → swap → mask →      │  │
│                      │  blend → restore → color → encode    │  │
│                      └──────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Offline trainer (DeepFaceLab-style, PyTorch MPS / CUDA)   │ │
│  │   dataset prep → SAEHD/Quick96 training → XSeg masks →    │ │
│  │   merge/export → .dfm / .onnx                             │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Evaluation harness (ArcFace ID sim, FID, LPIPS, FVD, ...) │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**

| Decision | Choice | Why |
|---|---|---|
| Inference runtime | ONNX Runtime + CoreML Execution Provider | Best Apple Silicon performance; uses ANE + GPU; beats raw PyTorch MPS for these models |
| Training framework | PyTorch (MPS locally, CUDA for serious training) | DeepFaceLab ecosystem compatibility; MPS fine for experiments, CUDA box/cloud for long runs |
| Model format for shipping | ONNX (single .onnx per stage) | Portable across CoreML EP, DirectML, CUDA EP |
| Trained-model interchange | .dfm (DeepFaceLive) + exported .onnx | Lets us train offline and run trained models in real-time mode |
| App shell | Swift/SwiftUI calling Python core via embedded interpreter or local socket | Native capture/encode via AVFoundation; GPU-side compositing |

---

## 4. Pipeline Stages (Detailed)

Every frame (offline or real-time) flows through these stages:

```
frame → [1] detect → [2] align/landmarks → [3] swap → [4] mask
      → [5] blend → [6] restore/enhance → [7] color match → [8] encode/out
```

### Stage 1 — Face detection
- **Model:** SCRFD-10G / RetinaFace (ONNX).
- Notes: det_10g benefits hugely from CoreML constant-folding of Shape/Gather ops (21 ms → 4 ms on M-series per Deep-Live-Cam's `onnx_optimize.py` rewrites).
- Real-time optimization: detection runs on a **separate thread with time-based caching** — detect every ~100 ms, interpolate boxes between detections. This decoupling alone takes a naive pipeline from ~25 FPS to ~45 FPS on M4 Pro.

### Stage 2 — Alignment & landmarks
- 68-point or 478-point landmarks (MediaPipe face mesh or FAN, ONNX-exported).
- Similarity-transform alignment to a canonical crop (128×128 for inswapper-class models; up to 640 for SAEHD head mode).
- Keep the transform matrix — needed to paste back.

### Stage 3 — Identity swap (the core)
Two interchangeable swap engines (Section 5 & 6 detail training/model choices):
- **Real-time:** inswapper_128-class single-photo swapper (identity encoder = pretrained ArcFace R101, unit-normalized 512-d embedding).
- **Offline:** trained SAEHD/AMP decoder pair, or a research model (SimSwap / FaceShifter / DiffSwap) fine-tuned per identity pair.

### Stage 4 — Masking
- **XSeg U-Net** (DeepFaceLab's trainable masker) for precise learned face-region masks — handles hair boundary, occlusion by hands/objects after per-pair training.
- Fallback/RT default: learned occluder mask from inswapper, or landmark-convex-hull + feather.
- XSeg internals worth copying: U-Net with FRNorm2D layers (filter response norm — no batch norm dependency issues), TLU activations, BlurPool downsampling for anti-aliasing; exports cleanly to ONNX.

### Stage 5 — Blending
- Poisson (seamless clone) OR learned mask feathering + Laplacian pyramid blend.
- Mask gaussian blur weighted by mask MSE (mirrors DeepFaceLab's merged loss weighting).
- GPU-side paste back through the inverse alignment transform (zero-copy with `CVPixelBuffer`/Metal).

### Stage 6 — Restoration / enhancement
- GFPGAN v1.4 (default) or CodeFormer (w parameter trades fidelity vs. quality).
- Graph rewrite note: GFPGAN's Split→Slice conversion drops 155 ms → 89 ms on CoreML EP.
- Used in both modes; at real-time settings run at 128–256 face resolution, offline at 512+.

### Stage 7 — Color & tone matching
- DeepFaceLab's `ct_mode` options: `rct` (Reinhard), `mkl` (Monge-Kantorovich), `idt` (iterative distribution transfer), `sot` (sliced OT) — implement as post-LUT on the swapped region before final composite.
- Exposure/white-balance histogram match per frame, clamped temporally to avoid flicker.

### Stage 8 — Encode / output
- AVFoundation (H.264/HEVC, hardware-accelerated VideoToolbox).
- Real-time mode: virtual camera (OBS-compatible) or direct preview window.
- Provenance metadata + watermark embedded here (Section 2) as the last non-optional step.

---

## 5. Mode A — Offline Trained-Model Pipeline

DeepFaceLab-style training gives the highest identity-matching accuracy. Architecture notes are distilled from iperov/DeepFaceLab (branch e4b7543f internals).

### 5.1 Model choices

| Model | When to use | Key facts |
|---|---|---|
| **Quick96** | Smoke tests, data debugging | Fixed 96×96, FULL face type; ae_dims=128, e_dims=64, d_dims=64, d_mask_dims=16; shared Encoder → Inter → two Decoders (src/dst); RMSprop lr=2e-4, lr_dropout=0.3, batch 4; loss = DSSIM+MSE on masked region + gaussian-weighted mask MSE |
| **SAEHD** | Main quality model | Res 64–640 (start 128, scale to 224–320); face_type: `h` (half), `mf`, `f` (full), `wf` (whole-face), `head`; archi `df` or `liae` (+ `-u`/`-d` variants); options: eyes_mouth_prio, uniform_yaw, random_warp, gan_power, true_face_power, ct_mode, clipgrad, pretrain |
| **AMP** | Photoreal experiments, morph effects | Default 224 res, `wf`; shared Encoder/Decoder + dual Inter with morph_factor 0.1–0.5 latent blending; AdaBelief lr=5e-5; 300× eyes+mouth priority loss weight; exports to .dfm via tf2onnx for DeepFaceLive |

**Recommendation:** start with SAEHD @ 128, `f` face type, `liae` archi; graduate to 224 `wf`; use AMP only if morphing or DeepFaceLive export of a trained identity is a requirement.

### 5.2 Data pipeline (per identity pair)

1. **Extract frames** from source and target video (every Nth frame, dedupe by perceptual hash).
2. **Detect + align** faces; store BGR crop + warp matrix + FACE_MASK samples.
3. **Filter:** drop blurry (Laplacian variance), occluded, extreme-yaw, low-res faces; balance yaw/pitch bins (`uniform_yaw` helps residual skew).
4. **XSeg labeling:** label ~100–200 representative frames per side in the XSeg editor, train XSeg, apply to full sets, hand-fix failure cases.
5. **Sample generation** (SampleGeneratorFace-equivalent): warped/unwarped BGR + masks, channel-shuffle/color-transfer augmentation, random flip, uniform yaw sampling.

### 5.3 Progressive training schedule

| Phase | Settings | Stop criterion |
|---|---|---|
| 1. Structure | Low res (96–128), `random_warp=on`, batch max for VRAM | Loss plateau; identity recognizable in previews |
| 2. Detail | Raise res (→224), warp still on, `eyes_mouth_prio=on` | Preview textures sharpen |
| 3. Refinement | `random_warp=off`, `lr_dropout=0.3`, optional `true_face_power` | 3–10k iters without preview improvement |
| 4. GAN polish | Enable `gan_power≈0.1` (start low; GAN destabilizes easily) | Grain detail matches skin; stop if artifacts grow |

- Track identity similarity every 1k iterations with the eval harness (Section 11) — do not rely on eyeballing previews.

### 5.4 Merge & export

- Merge with learned XSeg mask, chosen `ct_mode`, GFPGAN/CodeFormer pass as needed.
- Export trained decoder to ONNX; for live reuse of the trained identity, export AMP `.dfm` via tf2onnx.

---

## 6. Mode B — Real-Time Single-Photo Swap

### 6.1 Reference stack (proven on Apple Silicon)

The Deep Live Cam ecosystem demonstrates 45–60 FPS on M-series:

```
camera (MJPEG) → [thread A: SCRFD detect, cached ~100ms]
frame queue   → [thread B: inswapper_128 on ANE via CoreML EP]
              → [thread B: GFPGAN lite → GPU paste_back]
              → zero-copy display / virtual camera
```

- **Swap model:** `inswapper_128.onnx` (ArcFace R101 ID encoder + generator). Apply the CoreML graph rewrites from Deep-Live-Cam commit f65aeae: replace `Pad(reflect)` with `Slice+Concat` — collapses 14 CoreML partitions to 1 (partition boundaries force CPU↔ANE memory copies; each is milliseconds).
- **Detection routing:** run SCRFD on the GPU partition deliberately, leaving the ANE free for the swap model (ANE is the swap bottleneck).
- **Reference implementation:** Swap Lense (aniruddhha/swap-lense) — minimal M1/M2/M3 codebase with OBS Virtual Camera output; good first scaffold before growing into full app shell.

### 6.2 Accuracy upgrades over stock inswapper

Stock inswapper_128 is the accuracy floor (128×128 output). To push identity matching in real time:

1. **Better identity encoder input:** aggregate the ArcFace embedding from the top-k sharpest, frontal-iest frames of the source (embedding averaging beats single-photo ID fidelity).
2. **Higher-res swap models:** evaluate SimSwap-512 or DiffSwap-512 exports — DiffSwap reports ID retrieval 98.54% / FID 2.16 on FF++ vs FaceShifter 90.04/3.50 and SimSwap 93.07/3.04; but diffusion inference cost likely breaks 30 FPS on ANE — benchmark before committing; SimSwap++ (TPAMI 2024) is the more realistic high-accuracy RT candidate.
3. **Restore at face-native size:** run GFPGAN at 256 on the swapped face only (not full frame).

### 6.3 DeepFaceLive integration path

DeepFaceLive (GPL-3.0 — note license obligations if shipping derivative work) contributes:
- DFM pipeline: a trained AMP model runs live under Insight-style alignment — highest accuracy achievable in real-time mode.
- Face Animator module (pose/expression transfer without identity swap).
- Lesson for our app: their architecture proves trained-model + live camera is viable on desktop; we replicate with CoreML EP instead of DirectX 12.

---

## 7. Apple Silicon Performance Strategy

### 7.1 Hard-won rules (from Deep-Live-Cam graph-rewrite work)

1. **ONNX Runtime + CoreML EP** — not PyTorch MPS — for shipped inference. MPS is for training experiments only.
2. **Eliminate CoreML partition splits.** Every partition boundary is a sync + memory copy. Target: 1 partition per model. Achieve via graph surgery (Pad→Slice+Concat, Split→Slice, Shape/Gather constant-folding).
3. **Assign models to units deliberately:** ANE for the big swap model; GPU for detection and enhancement; CPU only for scheduling/IO.
4. **Zero-copy display:** keep frames in `CVPixelBuffer` end to end; paste-back and blend in Metal.
5. **Decouple detection from swap:** parallel threads + cached detections. (~25 → ~45 FPS on M4 Pro.)
6. **MJPEG capture** over raw YUV when CPU-bound on decode.

### 7.2 Performance budgets (M-series targets)

| Stage | Budget (1080p frame) | Model |
|---|---|---|
| Detect | 4 ms (cached → ~0.4 ms amortized) | SCRFD det_10g optimized |
| Align | 1 ms | landmarks ONNX |
| Swap | 8–12 ms | inswapper_128, 1 partition, ANE |
| Mask | reuse swap mask (0 ms) | — |
| Restore | 20–30 ms (optional at RT; run every Nth frame or at 256px) | GFPGAN Split→Slice build |
| Blend/color/encode | 3 ms | Metal compositor |
| **Total** | **~16 ms core → 60 FPS** w/o restore; ~30 FPS with periodic restore | |

### 7.3 Training on Mac vs cloud

- MPS training of SAEHD is possible (~2–4× slower than a 4090-class GPU) — acceptable for 96–128 res experiments.
- For 224+ resolution and GAN phases, use a CUDA cloud instance; sync the workspace; export ONNX/dfm back to the Mac.

---

## 8. Data Acquisition & Quality

### 8.1 Consent-first sourcing (mandatory)

- Only faces with a signed consent record (Section 2.1). For VFX: talent releases. For research experiments: properly licensed datasets — FF++, Celeb-DF, DFD, or self-recorded actors.

### 8.2 Capture guidelines (for recording your own actors)

| Parameter | Guideline |
|---|---|
| Resolution | ≥1080p source; 4K preferred for offline training |
| Frame rate | 30–60 fps, constant |
| Lighting | Flat frontal + fill; avoid hard shadows across face |
| Coverage | Full yaw -90°…+90°, pitch -30°…+45°, blink/smile/talking range |
| Duration | 2–5 min per side minimum for Quick96 sanity runs; 10–20 min for production SAEHD |
| Lens | 35–85 mm equivalent; avoid wide-angle distortion |

### 8.3 Quality filters (automated)

- Blur: Laplacian variance threshold per crop.
- Pose: bin by yaw/pitch; rebalance to near-uniform for `uniform_yaw`.
- Occlusion: XSeg hand-labeled classes; drop or mask.
- Duplicates: perceptual-hash dedupe to keep effective dataset diversity high.

---

## 9. Training Configurations

### 9.1 Quick96 (sanity runs)

```text
resolution: 96 (fixed)     face_type: FULL
ae_dims: 128   e_dims: 64  d_dims: 64  d_mask_dims: 16
optimizer: RMSprop lr=2e-4, lr_dropout=0.3
batch: 4
loss: DSSIM+MSE(masked) + MSE(mask, gaussian-weighted)
```

### 9.2 SAEHD (production)

```text
resolution: 128 → 224 (progressive)
face_type: f → wf
archi: liae
eyes_mouth_prio: on (phase 2+)
uniform_yaw: on (if yaw coverage skewed)
random_warp: on → off (phase 3)
gan_power: 0.1 (phase 4 only)
true_face_power: 0 (enable in phase 3 if identity drifts)
ct_mode: rct (start) / mkl / idt / sot (choose at merge)
clipgrad: on (stability)
pretrain: on when bootstrapping a new pair (huge time saver)
```

### 9.3 AMP (DeepFaceLive export / morphing)

```text
resolution: 224   face_type: wf
shared Encoder/Decoder, dual Inter
morph_factor: 0.1–0.5 (identity blend strength)
optimizer: AdaBelief lr=5e-5
eyes+mouth priority weight: 300×
export: .dfm via tf2onnx
```

### 9.4 XSeg (masker)

- U-Net backbone; FRNorm2D (no BatchNorm), TLU activation, BlurPool downsampling.
- Train on 100–200 labeled frames/side; iterate on failure frames until merge-ready.
- Export to ONNX for the real-time mask stage.

---

## 10. Blending, Masking & Enhancement Stack

| Concern | Default | Alternatives |
|---|---|---|
| Face mask | XSeg (trained per pair) | inswapper occluder mask; landmark hull + feather |
| Boundary blend | Laplacian pyramid + gaussian-weighted feather | Poisson seamless clone |
| Color transfer | `rct` Reinhard | `mkl`, `idt`, `sot` |
| Restoration | GFPGAN v1.4 (Split→Slice CoreML build) | CodeFormer (w≈0.7), GPEN |
| Lip sync (optional feature) | Wav2Lip | LivePortrait, SadTalker |
| Temporal smoothing | color-match clamp + optical-flow consistency check at merge | per-frame histogram lock |

---

## 11. Accuracy Evaluation & Metrics

This is the "matching accuracy" recipe — a measurable definition of success, not vibes. Metrics per the canonical face-swap survey stack (arXiv 2605.00883) and FF++/CASIA FaceSwapping protocols.

### 11.1 Identity accuracy (primary)

| Metric | Definition | Target (offline) | Target (real-time) |
|---|---|---|---|
| **ID cosine similarity** | cos(ArcFace(output), ArcFace(source identity)) — note: use a *different* recognition model than the one inside the swapper to avoid circularity | ≥ 0.60 | ≥ 0.50 |
| **ID Retrieval** | % of outputs where nearest identity in a reference gallery is the source | ≥ 95% (DiffSwap-class SOTA is 98.5% on FF++) | ≥ 90% |
| **Cross-ethnicity/cross-attribute protocol** | CASIA FaceSwapping 4K splits | report, no floor | report |

### 11.2 Attribute preservation (target likeness retained where intended)

| Metric | Definition | Tool |
|---|---|---|
| Pose error | ℓ2 on head-pose vectors vs. target frame | HopeNet ONNX |
| Expression error | ℓ2 on 3DMM expression coefficients vs. target | Deng et al. 2019-style reconstruction |
| Mouth/eye keypoint error | landmark ℓ2 on lip/eye regions (lip-sync quality) | MediaPipe landmarks |

### 11.3 Perceptual quality

| Metric | Notes |
|---|---|
| FID (Inception-v3) | vs. target video face crops; SOTA reference 2.16 (DiffSwap) |
| LPIPS (VGG16) | perceptual distance for ablations |
| L1/SSIM on same-id pairs | reconstruction sanity |

### 11.4 Video consistency

| Metric | Notes |
|---|---|
| Temporal consistency | frame-to-frame cosine of DINO subject features / CLIP background features (VBench-style) |
| FVD | overall video realism distribution |
| Optical-flow warping error | warp frame t by flow, diff vs. t+1 — catches flicker the others miss |

### 11.5 Evaluation harness

- `eval/run_eval.py --model <ckpt.onnx> --set casia-fs-4k --out reports/<date>.json`
- Runs every metric above against a fixed benchmark pack; stores per-config history so accuracy regressions surface in CI.
- Gate for release: ID retrieval and temporal consistency must meet targets on the 4K benchmark **and** watermarks must survive the detector self-check (Section 2.4).

---

## 12. Tech Stack & Dependencies

| Layer | Choice | Notes |
|---|---|---|
| Language (core) | Python 3.11 | pipeline & training |
| Inference | ONNX Runtime ≥1.18, `onnxruntime-coreml` | CoreML EP flags: `ModelFormat=MLProgram`, `AllowLowPrecision=1` |
| Training | PyTorch 2.x (MPS/CUDA) | DeepFaceLab fork for offline mode |
| CV/Image | OpenCV, scikit-image, mediapipe | detection, landmarks, color ops |
| Models (ONNX) | inswapper_128, SCRFD det_10g, GFPGAN v1.4, CodeFormer, XSeg custom | apply CoreML graph rewrites before shipping |
| App shell | Swift 6 / SwiftUI, AVFoundation, Metal | capture, preview, VideoToolbox encode |
| Provenance | c2pa-rs (Rust) via FFI | C2PA content credentials |
| Build/package | pyenv/uv; Xcode project; models as downloadable assets | models not checked into repo |
| Observability | simple JSONL run logs + eval report store | no cloud dependency |

---

## 13. Project Structure

```text
deepfake-app/
├── consent/                  # signed consent records (gate for all runs)
├── app/                      # Swift/SwiftUI macOS shell
├── core/
│   ├── pipeline/             # detect, align, swap, mask, blend, color, encode stages
│   ├── models/               # ONNX model wrappers + CoreML rewrite scripts
│   ├── realtime/             # threaded capture/swap loop, detection cache
│   └── watermark/            # c2pa + frequency watermark embed/verify
├── training/
│   ├── datasets/             # frame extraction, filtering, XSeg labels
│   ├── deepfacelab/          # vendored/forked trainer (Quick96, SAEHD, AMP)
│   ├── configs/              # per-pair training configs
│   └── export/               # ONNX / .dfm export tools
├── eval/
│   ├── run_eval.py           # metric harness (Section 11)
│   ├── benchmark_packs/      # CASIA-FS, FF++ subsets, internal actors pack
│   └── reports/
├── assets/models/            # downloaded ONNX weights (gitignored)
└── docs/
    ├── DEEPFAKE_BUILD_PLAN.md  # this document
    ├── consent-policy.md
    └── compliance.md
```

---

## 14. Milestones & Deliverables

| # | Milestone | Deliverable | Exit criteria |
|---|---|---|---|
| M1 | Scaffold + consent gate | Repo, app shell skeleton, consent ledger, provenance embed/verify | swap refuses to run without consent record; watermark verified on sample output |
| M2 | Real-time MVP | inswapper_128 + SCRFD + GFPGAN wired on CoreML EP, threaded detection | ≥30 FPS at 1080p on M2; eval harness reports ID sim/retrieval baseline |
| M3 | CoreML optimization | Graph rewrites; unit routing; zero-copy display | ≥45 FPS on M4-class; 1-partition swap model verified |
| M4 | Offline trainer | Quick96 → SAEHD pipeline with dataset tooling + XSeg loop | trained pair meets offline ID targets on benchmark |
| M5 | Accuracy program | Full metric stack in CI; accuracy tuning (embedding averaging, res scaling, restore tuning) | ID retrieval ≥95% offline / ≥90% RT; temporal targets met |
| M6 | Polish & packaging | Merge quality, ct_mode selection UX, trained-model live export (.dfm→ONNX), virtual camera | VFX-grade sample reel; signed/notarized build |
| M7 | Compliance release | Legal checklist sign-off, watermark durability report, docs | ship/no-ship review passed |

---

## 15. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| License violations (GPL-3.0 DeepFaceLive, DeepFaceLab terms, model weights licenses e.g. inswapper non-commercial) | Legal exposure | Track per-component license table; prefer reimplementation over vendoring GPL for a commercial app; legal review before release |
| Consent data mismanagement | Serious legal/ethical breach | Consent gate is a hard dependency in code, not a checklist; revocation = model deletion |
| Accuracy plateaus at RT resolutions | Feature underdelivers | Planned fallbacks: SimSwap++ export, embedding averaging, trained-model RT path via AMP/dfm |
| ANE/GPU contention kills frame rate | RT mode unusable | Budget table (7.2) enforced via profiling CI; detection caching; optional restore |
| GAN phase instability in training | Artifacts, wasted runs | Phase gates with metric thresholds before enabling gan_power |
| Watermark stripped by transcoding | Provenance failure | Detector self-check in eval gate; dual watermarks (C2PA + frequency domain) |
| Model weight distribution (some ONNX weights are site-gated) | Build breakage | Models = downloadable assets with integrity hashes; never committed |

---

## 16. References

**Swap model research**
- SimSwap (ACM MM 2020) & SimSwap++ (TPAMI 2024) — ID injection via AdaIN; weak feature matching loss.
- FaceShifter (CVPR 2020) — AEI-Net with AAD layers; HEAR-Net occlusion handling; ~412M params / ~97 GFLOPs.
- DiffSwap (CVPR 2023) — masked latent-diffusion inpainting; FF++ ID retrieval 98.54%, FID 2.16; midpoint identity loss; 512×512.
- Also evaluated: HifiFace (3DMM shape-aware), GHOST, InfoSwap/StyleSwap, FaceDancer, RAFSwap, DiffFace, DynamicFace (ICCV 2025 video-diffusion).

**Codebases**
- iperov/DeepFaceLab — Quick96/SAEHD/AMP/XSeg internals (branch e4b7543f).
- iperov/DeepFaceLive (GPL-3.0) — real-time DFM pipeline, Insight single-photo swap, Face Animator.
- hacksider/Deep-Live-Cam — Apple Silicon runtime playbook; `modules/onnx_optimize.py` CoreML graph rewrites (commit f65aeae).
- aniruddhha/swap-lense — minimal M-series reference with OBS Virtual Camera.

**Evaluation**
- Face-swap metrics survey (arXiv 2605.00883); CASIA FaceSwapping benchmark; FF++; VBench temporal metrics.

**Responsible use**
- deepfakesweb.com — disclosure-by-default and invisible watermarking practices.
- charityexcellence.co.uk — detection-signs guide and protective tooling (Fawkes/Glaze) informing Section 2.
- UK Online Safety Act / Sexual Offences Act; CA AB 602; TX SB 751; EU AI Act transparency obligations.

---

*End of v1.0. Next actions after approval: M1 scaffold (consent gate + provenance) and M2 real-time MVP.*
