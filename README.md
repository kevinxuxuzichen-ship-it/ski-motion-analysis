# Alpine Ski Motion Analyzer 🎿

An offline AI-powered technique analysis system for alpine (two-plank) skiing.
Upload a skiing video → on-device pose estimation (24 keypoints incl. skis & poles)
→ biomechanical metrics → turn-phase segmentation → coaching feedback via an expert rule system.

---

## ✨ Features

- **Pose estimation**: YOLO-based 24-keypoint detection (including skis and poles), exported to TFLite for on-device inference.
- **Biomechanical metrics** (`ski_physics.py`): joint angles, body inclination, angulation (upper/lower body separation), and center of mass — each with a self-assessed reliability flag.
- **Temporal analysis**: turn-phase segmentation (initiation / apex / completion / transition) derived from the inclination signal.
- **Expert system** (`ski_expert.py`): 16 coaching rules grounded in PSIA/CSIA instructional principles, with a **data-quality gate** and **confidence propagation** — the system honestly downgrades or withholds advice when the data (e.g. camera angle) is unreliable.

---

## 🧩 Architecture
Video frames
↓  YOLO pose model (TFLite)      ── on-device inference
24 keypoints [x, y, visibility]
↓  ski_physics.py                ── measurement layer
Biomechanical metrics (+ reliability flags)
↓  temporal analysis             ── turn-phase segmentation
Per-frame phase labels
↓  ski_expert.py                 ── judgement layer
Data-quality gate + 16 rules + confidence propagation
↓
Honest, graded technique report
**`ski_physics.py`** is the *measurement layer* — a "ruler" that computes metrics.
**`ski_expert.py`** is the *judgement layer* — a "coach" that interprets metrics and gives advice.

---

## 📁 Repository Structure

```
.
├── ski_physics.py       # Biomechanical metric computation
├── ski_expert.py        # Expert system: 16 rules + gate + confidence
├── notebooks/           # Development notebooks (training, analysis)
├── weights/             # Trained model weights (see weights/README.md)
├── requirements.txt
└── README.md
```
---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
```

```python
from ski_physics import extract_all
from ski_expert import assess_data_quality, run_expert_system

# extract_all(annotation) -> biomechanical metrics for one frame
# run_expert_system(...)   -> full technique report
```

See `notebooks/` for the end-to-end pipeline.

---

## 🔬 Design Philosophy

This project deliberately favors **honesty over false precision**:

- Every metric carries a **reliability flag**; the system stays silent when data is untrustworthy.
- 2D pose estimation has inherent **perspective distortion**; the system detects problematic camera angles and **lowers confidence accordingly** instead of pretending to be certain.
- Thresholds are **empirical placeholders**, pending calibration with real data and coach feedback.
- The goal is a tool that says *"I'm not sure"* when appropriate — leaving final judgement to the user or coach.

---

## 📊 Dataset & Attribution

The pose model was trained on the **Ski2DPose dataset** from EPFL:

> R. Bachmann, J. Spörri, P. Fua, H. Rhodin.
> *Motion Capture from Pan-Tilt Cameras with Unknown Orientation.*
> International Conference on 3D Vision (3DV), 2019.
> Dataset: https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/

**The dataset itself is NOT redistributed in this repository.** To reproduce training,
please obtain the dataset directly from EPFL under their terms.

---

## ⚖️ License & Disclaimer

1. All original code (metric computation, temporal analysis, the expert rule system,
   and visualization scripts) is authored by me and released under the **MIT License**.

2. The Ski2DPose dataset used for training is a third-party public resource that carries
   **no explicit open-source license**; it is not redistributed here.

3. **Disclaimer**: This project is intended solely for **student academic research and
   non-commercial study of skiing technique**. Model weights are provided for academic
   demonstration only and must **not** be used for commercial deployment. If the dataset's
   rights holders have any objection, please contact me and I will promptly remove the
   relevant files.

---

## 👤 Author

Zichen Xu · GitHub: kevinxuxuzichen-ship-it

*This project was developed independently as a high-school research project.*
