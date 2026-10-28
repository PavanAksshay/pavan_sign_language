# Phase 18 — Indian Sign Language (ISL) Multi-Signer Data Collection Guide

## 1. Overview & Objectives

Phase 18 introduces a dedicated, local, privacy-preserving webcam data collection studio designed to capture genuine ISL alphabet recordings from two independent volunteer signers:
* **`signer_02`**: Adult Volunteer Signer A
* **`signer_03`**: Adult Volunteer Signer B

### Nominal Collection Target Matrix
$$\text{2 Signers} \times \text{2 Conditions} \times \text{26 Letters (A–Z)} \times \approx 30 \text{ Usable Frames} = 3,120 \text{ Frames}$$

Every collected sample is audited in real-time, checked for motion blur, illumination boundaries, duplicate frame hashes, and MediaPipe landmark visibility. Accepted samples are quarantined strictly into the **external test holdout partition** (`test_multisigner_manifest.json`) and **never leaked into training or validation splits**.

---

## 2. Volunteer Privacy & Informed Consent Protocol

1. **Local Workstation Processing**: All video captures, frame extractions, and landmark calculations take place **100% locally** on the host machine. No audio is recorded. Zero images or videos are transmitted to cloud services or external endpoints.
2. **Explicit Consent**: Prior to recording any session, the operator must inform the volunteer participant and enable the **"Volunteer Consent Confirmed"** toggle in the studio interface. Recording controls remain strictly locked until consent is confirmed.
3. **Right to Deletion**: Volunteers have the absolute right to request immediate deletion of their data at any time. The operator can purge any individual recording or an entire signer session via the studio deletion API (`DELETE /api/collect/recording/{recording_id}`).

---

## 3. Physical Recording Setup & Standardized Conditions

To measure true cross-condition generalization, data from each signer must be recorded under two standardized conditions:

### Condition 1: Desk Neutral (Frontal)
* **Lighting**: Uniform overhead white light (indoor office/fluorescent, ~400–600 lux). Avoid harsh single-side shadows.
* **Background**: Plain wall, solid neutral backdrop, or tidy desk background.
* **Camera Angle**: Frontal eye-to-chest level (0° horizontal tilt, 0° azimuth).
* **Distance**: $75 \text{ cm} \pm 5 \text{ cm}$ from camera lens to signer's chest.
* **Hand Placement**: Inside the center framing reticle at mid-chest height.

### Condition 2: Lateral Warm (15° Offset)
* **Lighting**: Ambient warm desk lamp or soft directional side window lighting (~250–400 lux).
* **Background**: Textured office or home living environment with soft natural shadowing.
* **Camera Angle**: Slight lateral diagonal offset (15° azimuth to signer's left or right).
* **Distance**: $85 \text{ cm} \pm 5 \text{ cm}$ from camera lens.
* **Hand Placement**: Mid-chest height, ensuring full finger contour visibility against the textured background.

---

## 4. Hardware & Technical Checklist

| Parameter | Specification | Quality Enforcement Gate |
| :--- | :--- | :--- |
| **Webcam Resolution** | 1280 × 720 (720p) or 1920 × 1080 (1080p) | Checked at browser feed initialization |
| **Target Frame Rate** | 30 FPS constant | Video property check during staging |
| **Illumination / Brightness** | Mean grayscale $\in [40.0, 220.0]$ | Automated rejection if $< 40$ (underexposed) or $> 220$ (overexposed) |
| **Motion Blur Threshold** | Laplacian variance $\ge 50.0$ | Automated rejection if $< 50.0$ |
| **Duplicate Prevention** | SHA-256 cryptographic hash & movement diff $\ge 0.4$ | Rejection of exact static freezes or duplicates |
| **Landmark Viability** | MediaPipe HandLandmarker $\ge 1$ active hand | Automated rejection if 0 hands detected |

---

## 5. Letter-by-Letter ISL Signing Instructions (A–Z)

Below are the exact anatomical hand shapes and critical distinction notes for all 26 ISL alphabet signs. Reference illustrations are available directly in the interface from `web/static/manual/{letter}.jpg`.

### Single-Hand Signs (Dominant Hand)

* **A**: Make a solid fist with the thumb held upright vertically against the side of the index finger. Palm faces forward.
* **B**: Extend all four fingers straight upward pressed closely together. Fold the thumb across the lower palm.
* **C**: Curve the four fingers and thumb into a smooth circular arc forming the letter "C". Keep thumb well-separated from the index finger.
* **D**: Extend the index finger straight up. Touch the tips of the middle finger, ring finger, and pinky to the tip of the thumb, forming an upright "D" loop.
* **E**: Curl all four fingertips tightly inward toward the palm. Fold the thumb horizontally beneath the curled fingers.
* **F**: Touch the tip of the index finger to the tip of the thumb forming a circle. Extend the middle, ring, and pinky fingers upward spread apart.
* **G**: Point the index finger and thumb horizontally forward in parallel, separated by 2–3 cm. Keep middle, ring, and pinky curled into palm.
* **H**: Extend the index and middle fingers horizontally forward together. Keep ring, pinky, and thumb folded in.
* **I**: Extend the little pinky finger straight up. Keep all other three fingers and thumb folded into a compact fist.
* **J**: Extend the little pinky finger upright, then execute a small fluid "J" hook arc in the air.
* **K**: Extend index finger upright; point middle finger horizontally forward at a 90° angle. Tuck the thumb against the knuckle joint between them.
* **L**: Form a crisp 90° right angle between the fully extended index finger and extended thumb. Palm facing forward.
* **M**: Make a fist where three fingers (index, middle, ring) are folded over the thumb.
* **N**: **CRITICAL FOCUS**: Make a fist where exactly **two fingers (index and middle)** are folded over the thumb. The ring finger and pinky must remain tightly tucked into the palm. *Do not let the ring finger slip over the thumb (avoids confusion with M).*
* **O**: Curve all five fingertips together to meet the thumb tip, forming an "O" circular ring.
* **R**: Cross the extended middle finger behind the extended index finger. Other fingers folded into fist.
* **S**: Make a tight fist with the thumb crossed over the front of the curled index and middle fingers.
* **T**: Tuck the thumb tip between the curled index finger and middle finger of a closed fist.
* **U**: Extend the index and middle fingers straight upward pressed closely together.
* **V**: Extend the index and middle fingers straight upward spread apart into a "V".
* **W**: Extend the index, middle, and ring fingers straight upward spread apart. Pinky held down by thumb.
* **X**: **CRITICAL FOCUS**: Bend the index finger into a hooked curve. Keep the middle, ring, and pinky fingers closed into the palm. *Avoid extending or spreading the middle finger (avoids confusion with V).*
* **Y**: Extend the thumb outward and the little pinky finger outward (shaka sign). Keep the middle three fingers closed into the palm.

### Two-Hand Signs
* **P**: Non-dominant hand holds an open palm facing upward; dominant hand extends the index finger and touches its tip to the center of the non-dominant palm.
* **Q**: Non-dominant hand forms an open upward loop or palm; dominant hand points downward through or clasps across it.
* **Z**: Extend dominant index finger forward and trace a crisp "Z" stroke in the air in front of the chest.

---

## 6. Step-by-Step Operator Capture Workflow

1. **Launch Server**:
   ```bash
   /Users/pavanaksshay/venv/bin/python3 server.py
   ```
2. **Open Capture Studio**:
   Navigate to `http://localhost:8050/collect` in Google Chrome or Safari on macOS.
3. **Confirm Consent**:
   Check the **"Volunteer Consent Confirmed"** toggle after informing the participant.
4. **Configure Session**:
   * Select Signer ID (`signer_02` or `signer_03`).
   * Select Condition (`condition_1` or `condition_2`).
   * Verify Session ID (e.g. `session_20261015_01`).
   * Ensure Sampling Interval is set to `Every 3 frames (~10 fps) [Recommended]`.
5. **Frame Signer**:
   Ensure the volunteer's hand is visible inside the framing reticle at chest height with clear lighting.
6. **Record Sample**:
   * Click the red **"Start Recording (3.0s Auto-Stop)"** button.
   * A 3-second countdown (3, 2, 1, SIGN!) will play.
   * The signer holds the sign steady for 3.0 seconds until auto-stop triggers.
7. **Interactive Review**:
   * The **Review Staged Recording** modal opens immediately.
   * Watch video playback to confirm sign correctness.
   * Inspect the quality audit breakdown (usable vs rejected frames).
   * Browse the thumbnail gallery of sampled frames.
   * Click **"Accept Recording & Ingest Frames"** to commit to the dataset, or **"Discard & Re-Record"** if the sign was malformed, blurred, or obscured.
8. **Track Progress**:
   The progress matrix updates in real time, advancing automatically to the next letter.

---

## 7. Storage Architecture & Directory Layout

```
data/
├── raw_recordings/
│   ├── signer_02/
│   │   ├── condition_1/
│   │   │   └── rec_signer_02_condition_1_A_1791980000.mp4
│   │   └── condition_2/
│   └── signer_03/
├── extracted_frames/
│   ├── signer_02/
│   │   ├── condition_1/
│   │   │   └── A/
│   │   │       ├── signer_02_session_01_A_0000.jpg
│   │   │       └── ...
├── collection_manifests/
│   ├── collection_ledger.json           # Complete audit transaction log
│   └── test_multisigner_manifest.json   # External test holdout partition
└── staging/                             # Temporary transient files pending review
```

---

## 8. Compliance & Split Isolation Guarantee

* **Holdout Quarantine**: Newly collected samples are **never** placed into `train_manifest.json` or `val_manifest.json`.
* **Zero Leakage**: All frames from a single recording session remain strictly within the external test partition (`external_test_multisigner`).
* **Audit Command**: Run the following at any time to verify zero partition leakage:
  ```bash
  /Users/pavanaksshay/venv/bin/python3 -c "from phase18_collector import collection_engine; print(collection_engine.audit_split_isolation())"
  ```
