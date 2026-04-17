# Datasets Reference

Public datasets for binary classifier training and pseudo-label bootstrapping.
See `docs/binary_classifier_bootstrap.md` for how these feed into the pipeline.

---

## Eyes ROI

### MRL Eye Dataset
- **URL:** https://mrl.cs.vsb.cz/eyedataset.html
- **Kaggle mirror:** https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset
- **Classes:** `open`, `closed` (binary)
- **Size:** 84,898 infrared images from 37 subjects
- **License:** Research use (contact required from original site; widely mirrored on Kaggle)
- **Notes:** Gold standard for drowsiness eye-state research. Infrared images from three sensors (Intel RealSense, IDS, Aptina). Metadata includes per-image openness annotations that can be thresholded to create a `partially_closed` class. Near-perfect class balance (~42k open, ~41k closed).

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
- **Notes:** Small, clean, pre-cropped single-eye images. MIT license. Good for quick iteration and sanity checks.

---

## Mouth ROI

### Yawn Dataset (David Vazquez)
- **URL:** https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset
- **Classes:** `yawn`, `no_yawn`
- **Size:** ~5,119 images (~2,528 yawn, ~2,591 no-yawn)
- **License:** CC BY-NC-SA 4.0
- **Notes:** Sourced from Google Images, diverse subjects. No `slight_open` class — the `no_yawn` class covers both closed and slightly open mouths and can be sub-annotated.

### 4-Class Drowsiness Dataset
- **URL:** https://www.kaggle.com/datasets/hoangtung719/drowsiness-dataset
- **Classes:** `Open_Eyes`, `Closed_Eyes`, `Yawn`, `No_yawn`
- **Size:** 11,566 images (train 8,548 / val 1,554 / test 1,464), evenly distributed
- **License:** CC BY-NC-SA 4.0
- **Notes:** Merged from Yawn Dataset + MRL+CEW + a cropped-mouth Kaggle dataset. Pre-split into train/val/test. Mouth images are pre-cropped. Covers both Eyes and Mouth ROIs in one download. **Primary dataset for Mouth binary classifier** (discard eye-class rows).

### YawDD — Yawning Detection Dataset
- **URL:** https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
- **Mirror / info:** https://qualinet.github.io/databases/video/yawdd_a_yawning_detection_dataset/
- **Classes:** `normal`, `talking/singing`, `yawning` (video-level; YawDD+ adds frame-level)
- **Size:** 351 videos (~124,000 extractable frames). YawDD+ estimates ~24,840 yawn frames / ~99,361 no-yawn frames.
- **License:** Free non-commercial and research use (University of Ottawa)
- **Notes:** Real in-car recordings. Original labels are video-level only. The companion YawDD+ paper (arXiv:2512.11446) provides a semi-automated pipeline for frame-level annotation. Diverse ethnicities, genders, glasses/no-glasses.
  - **YawDD+ paper:** https://arxiv.org/abs/2512.11446

---

## Head ROI

All head pose datasets below use continuous pitch/yaw/roll angles. Bucket to discrete binary classes before use — see `docs/binary_classifier_bootstrap.md` for bucketing thresholds.

### BIWI Kinect Head Pose Database
- **URL:** https://vision.ee.ethz.ch/datsets.html
- **Hugging Face:** https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose
- **Classes:** Continuous 6-DoF (pitch, yaw, roll + 3D translation)
- **Size:** ~15,000 frames from 20 subjects, RGB + depth at 640×480
- **License:** Academic (Creative Commons-style, ETH Zurich)
- **Notes:** Ground truth from 3D Kinect depth — most accurate head pose labels of any public dataset. Available directly via the Hugging Face datasets API.

### 300W-LP + AFLW2000-3D
- **URL:** http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
- **Classes:** Continuous pitch, yaw, roll
- **Size:** 300W-LP: 122,450 synthesized large-pose images. AFLW2000-3D: 2,000 real-world evaluation images.
- **License:** Academic research
- **Notes:** 300W-LP is synthetically augmented from real photos rotated to many yaw angles. Most widely used benchmark for head pose regression training. Requires discretization for binary classification use.

### DD-Pose (TU Delft Driver Pose)
- **URL:** https://dd-pose-dataset.tudelft.nl/
- **Classes:** Continuous 6-DoF per frame (pitch −69°/+57°, yaw −138°/+126°, roll −63°/+60°)
- **Size:** ~330,000 measurements from 27 subjects, multiple camera angles
- **License:** Academic, free registration required
- **Notes:** Real in-car driving scenarios (inner-city, tunnel, lane merge, parking). Most directly applicable to FatigueSense. **Primary dataset for Head binary classifier.**

### DriveAHead
- **URL:** https://cvhci.anthropomatik.kit.edu/~mschroeder/publications/CVPRW2017.pdf
- **CVPR 2017 paper:** https://openaccess.thecvf.com/content_cvpr_2017_workshops/w13/papers/Schwarz_DriveAHead_-_A_CVPR_2017_paper.pdf
- **Classes:** Continuous 6-DoF, frame-by-frame from motion capture (pitch ±45°, roll ±40°, yaw ±90°)
- **Size:** ~1,000,000 frames from 20 subjects. IR + depth (Kinect v2), 512×424px.
- **License:** Research use
- **Notes:** Largest driver-specific head pose dataset. Per-frame occlusion, glasses, and sunglasses annotations included. Use for pretraining before fine-tuning on FatigueSense domain crops.

---

## Torso ROI

Torso is the hardest ROI. No public dataset directly covers lean_left/lean_right. Plan to supplement with manually annotated crops from FatigueSense YOLO output.

### Sitting Posture Classification (Roboflow)
- **URL:** https://universe.roboflow.com/project-design-20242025/sitting-posture-classification-6vwq1
- **Classes:** Multiple sitting posture classes (exact names visible in dataset browser)
- **Size:** 4,813 images
- **License:** Check Roboflow project page
- **Notes:** Most accessible posture dataset. Export in YOLO, COCO, or CSV. Class names need remapping to FatigueSense torso classes. Lean classes may not be present — verify before use. **Primary starting dataset for Torso binary classifier.**

### Sitting Posture Detection (Roboflow — Project Medsafe AI)
- **URL:** https://universe.roboflow.com/project-medsafe-ai/sitting-posture-detection-3933f-7troi
- **Classes:** Good/bad posture variants (bounding box format)
- **Size:** 490 images
- **License:** Check Roboflow project page
- **Notes:** Detection format (bounding boxes, not classification crops). Small — use as augmentation rather than primary training data.

### Hierarchical Sitting Posture Dataset (PMC Study)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8022631/
- **Classes:** `sitting_straight`, `lightly_hunched`, `hunched_over`, `extremely_hunched`, `partially_lying`, `lying_down`
- **Size:** ~13,000+ labeled frames
- **License:** Research use (contact authors via the PMC paper for data access)
- **Notes:** 6-class taxonomy maps well to FatigueSense torso classes (straight→upright, lightly_hunched→slight_slouch, hunched/extremely_hunched→heavy_slouch). Side-view camera — perspective differs from driver-facing setup. No lean_left/lean_right.

---

## Summary

| ROI   | Primary Dataset | Backup / Supplement | Gap |
|-------|----------------|---------------------|-----|
| Eyes  | MRL+CEW Composite (CC0) | MRL full (84k), CEW, Eye Open/Close (MIT) | `partially_closed` — threshold MRL metadata |
| Mouth | 4-class Drowsiness (CC BY-NC-SA) | Yawn Dataset, YawDD+YawDD+ | `slight_open` — sub-annotate `no_yawn` class |
| Head  | DD-Pose (academic, driver-specific) | BIWI, 300W-LP, DriveAHead | Continuous → discrete bucketing required |
| Torso | Sitting Posture Roboflow | PMC Hierarchical Posture | Lean classes missing everywhere — manual annotation needed |
