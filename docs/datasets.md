# Datasets Reference

Public datasets for binary classifier training and pseudo-label bootstrapping.
See `docs/binary_classifier_bootstrap.md` for how these feed into the pipeline.

**Note on Head/Torso:** Head and Torso behavioral features are extracted from YOLO11n-pose keypoints (pitch/yaw/roll, slouch ratio), not from CNN classifiers. The head pose datasets below are reference material for understanding geometric threshold calibration, not CNN training data.

---

## Eyes ROI

### MRL Eye Dataset
- **URL:** https://mrl.cs.vsb.cz/eyedataset.html
- **Kaggle mirror:** https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset
- **Classes:** `open`, `closed` (binary)
- **Size:** 84,898 infrared images from 37 subjects
- **License:** Research use (contact required from original site; widely mirrored on Kaggle)
- **Notes:** Gold standard for drowsiness eye-state research. Infrared images from three sensors (Intel RealSense, IDS, Aptina). Near-perfect class balance (~42k open, ~41k closed).

### MRL + CEW Composite
- **URL:** https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset
- **Classes:** `Open_Eyes`, `Closed_Eyes`
- **Size:** ~10,000 images (5k per class in V1; 4k in V4)
- **License:** CC0 Public Domain
- **Notes:** Merged from MRL + Closed Eyes in the Wild (CEW) + custom data. Pre-split, clean, most permissive license available. **Primary dataset for Eyes binary classifier.**

### Closed Eyes in the Wild (CEW)
- **URL:** http://parnec.nuaa.edu.cn/xtan/ClosedEyeDatabases.html
- **Classes:** `both_eyes_closed`, `eyes_open` (full-face, not pre-cropped)
- **Size:** 2,423 subjects
- **License:** Free non-commercial research (Nanjing University of Aeronautics and Astronautics)
- **Notes:** Unconstrained in-the-wild conditions. Full-face images — run eye detector before using as CNN input. Good for generalization testing.

### Eye Open/Close — Drowsiness Prediction
- **URL:** https://www.kaggle.com/datasets/dhirdevansh/eye-dataset-openclose-for-drowsiness-prediction
- **Classes:** `Open`, `Closed`
- **Size:** 4,000 pre-cropped images (2,000 per class), 93×93px greyscale PNG
- **License:** MIT
- **Notes:** Small, clean, pre-cropped single-eye images. Good for quick iteration and sanity checks.

---

## Mouth ROI

### Yawn Dataset (David Vazquez)
- **URL:** https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset
- **Classes:** `yawn`, `no_yawn`
- **Size:** ~5,119 images (~2,528 yawn, ~2,591 no-yawn)
- **License:** CC BY-NC-SA 4.0
- **Notes:** Sourced from Google Images, diverse subjects. `no_yawn` covers both closed and slightly open mouths — treated as single negative class.

### 4-Class Drowsiness Dataset
- **URL:** https://www.kaggle.com/datasets/hoangtung719/drowsiness-dataset
- **Classes:** `Open_Eyes`, `Closed_Eyes`, `Yawn`, `No_yawn`
- **Size:** 11,566 images (train 8,548 / val 1,554 / test 1,464), evenly distributed
- **License:** CC BY-NC-SA 4.0
- **Notes:** Merged from Yawn Dataset + MRL+CEW + cropped-mouth data. Pre-split. Mouth images pre-cropped. **Primary dataset for Mouth binary classifier** — use `Yawn` (positive) and `No_yawn` (negative); discard eye-class rows.

### YawDD — Yawning Detection Dataset
- **URL:** https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
- **Classes:** `normal`, `talking/singing`, `yawning` (video-level)
- **Size:** 351 videos (~124,000 extractable frames)
- **License:** Free non-commercial and research use (University of Ottawa)
- **Notes:** Real in-car recordings. Original labels are video-level only. YawDD+ paper (arXiv:2512.11446) provides frame-level annotation pipeline. Diverse ethnicities, genders, glasses/no-glasses.

---

## Head Pose (Reference / Threshold Calibration)

Head behavioral features are derived from YOLO11n-pose keypoints using geometric computation, not a CNN. These datasets inform pitch/yaw/roll threshold calibration for fatigue indicators (e.g., head_down when pitch < −15°).

### BIWI Kinect Head Pose Database
- **URL:** https://vision.ee.ethz.ch/datsets.html
- **Hugging Face:** https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose
- **Labels:** Continuous 6-DoF (pitch, yaw, roll + 3D translation)
- **Size:** ~15,000 frames from 20 subjects, RGB + depth at 640×480
- **License:** Academic (Creative Commons-style, ETH Zurich)
- **Notes:** Ground truth from 3D Kinect depth — most accurate head pose labels available. Useful for validating PnP solver output and calibrating angle thresholds.

### 300W-LP + AFLW2000-3D
- **URL:** http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
- **Labels:** Continuous pitch, yaw, roll
- **Size:** 300W-LP: 122,450 synthesized large-pose images. AFLW2000-3D: 2,000 real-world evaluation images.
- **License:** Academic research
- **Notes:** Most widely used head pose benchmark. Useful for verifying extractor behavior at extreme angles.

### DD-Pose (TU Delft Driver Pose)
- **URL:** https://dd-pose-dataset.tudelft.nl/
- **Labels:** Continuous 6-DoF per frame (pitch −69°/+57°, yaw −138°/+126°, roll −63°/+60°)
- **Size:** ~330,000 measurements from 27 subjects, multiple camera angles
- **License:** Academic, free registration required
- **Notes:** Real in-car driving scenarios. Most directly applicable to FatigueSense for threshold validation.

### DriveAHead
- **URL:** https://cvhci.anthropomatik.kit.edu/~mschroeder/publications/CVPRW2017.pdf
- **Labels:** Continuous 6-DoF, frame-by-frame from motion capture
- **Size:** ~1,000,000 frames from 20 subjects
- **License:** Research use
- **Notes:** Largest driver-specific head pose dataset. Per-frame occlusion, glasses, sunglasses annotations.

---

## Torso Posture (Reference)

Torso behavioral features (slouch ratio, lean angle) are extracted from shoulder/hip keypoints. These datasets provide posture label reference for threshold calibration and manual annotation verification.

### Sitting Posture Classification (Roboflow)
- **URL:** https://universe.roboflow.com/project-design-20242025/sitting-posture-classification-6vwq1
- **Classes:** Multiple sitting posture classes
- **Size:** 4,813 images
- **License:** Check Roboflow project page
- **Notes:** Most accessible posture dataset. Useful for validating slouch ratio thresholds against labeled posture images.

### Sitting Posture Detection (Roboflow — Project Medsafe AI)
- **URL:** https://universe.roboflow.com/project-medsafe-ai/sitting-posture-detection-3933f-7troi
- **Classes:** Good/bad posture variants (bounding box format)
- **Size:** 490 images
- **License:** Check Roboflow project page
- **Notes:** Small — use as supplemental reference only.

### Hierarchical Sitting Posture Dataset (PMC Study)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8022631/
- **Classes:** `sitting_straight`, `lightly_hunched`, `hunched_over`, `extremely_hunched`, `partially_lying`, `lying_down`
- **Size:** ~13,000+ labeled frames
- **License:** Research use (contact authors for data access)
- **Notes:** 6-class taxonomy maps to FatigueSense posture states (straight→upright, hunched→heavy_slouch). Side-view camera — perspective differs from driver-facing setup.

---

## Summary

| ROI   | Purpose | Primary Dataset | Gap |
|-------|---------|----------------|-----|
| Eyes  | CNN binary classifier | MRL+CEW Composite (CC0) | Dataset noise ceiling ~0.90 F1; supplement with MRL full or manual annotation |
| Mouth | CNN binary classifier | 4-class Drowsiness (CC BY-NC-SA) | `no_yawn` class mixes closed + slight-open — treated as single negative |
| Head  | Threshold calibration only (keypoint geometry) | DD-Pose (driver-specific) | No CNN training needed |
| Torso | Threshold calibration only (keypoint geometry) | Sitting Posture Roboflow | No CNN training needed; lean classes absent from most datasets |
