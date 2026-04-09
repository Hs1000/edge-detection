## Conveyor Belt Damage Detection (Hybrid AI + CV Pipeline)

---

# 📌 1. Overview

This project detects **damage on conveyor belts** using a hybrid approach:

* 🧠 **YOLOv8 (Deep Learning)** → Detect conveyor belt region
* 🔍 **Computer Vision (CV)** → Detect defects inside the belt

### 🎯 Defects Targeted

* Scratch damage (surface)
* Edge damage (belt boundary)

---

# 🧠 2. Approach

Since the dataset provides:

* ✅ Belt annotations
* ❌ No defect annotations

👉 We use a **Hybrid Pipeline**:

```text
Input Image
     ↓
YOLOv8 → Detect Belt
     ↓
Crop Belt Region
     ↓
Edge Detection + Contours
     ↓
Defect Bounding Boxes
     ↓
Annotated Image + JSON Output
```

---

# 🏗️ 3. Final Project Structure

```bash
project/
│
├── pipeline.py
├── train.py
├── split_data.py
├── requirements.txt
├── README.md
│
├── model_weights/
│   └── best.pt            # trained model (or link.txt)
│
├── train_data/
│   ├── images/
│   ├── labels/
│
├── val_data/
│   ├── images/
│   ├── labels/
│
└── outputs/               # generated after inference (sample results only)
```

---

# ⚙️ 4. Installation

```bash
pip install -r requirements.txt
```

---

# 🧾 5. Dataset Setup

Original dataset (not included in repo) should be placed as:

```bash
train/
├── images/
├── labels/
```

---

# 🔀 6. Split Dataset (90% Train / 10% Validation)

Run:

```bash
python split_data.py
```

This creates:

```bash
train_data/
├── images/
├── labels/

val_data/
├── images/
├── labels/
```

---

# 🧾 7. data.yaml (FINAL)

```yaml
path: .

train: train_data/images
val: val_data/images

nc: 1
names: ["belt"]
```

---

# 🏋️ 8. Train the Model

```bash
python train.py
```

OR:

```bash
yolo detect train data=train/data.yaml model=yolov8n.pt epochs=30 imgsz=640
```

---

# 📦 9. Training Output

After training:

```bash
runs/detect/train*/weights/best.pt
```

---

# 📥 10. Move Model Weights

```bash
mkdir -p model_weights
cp runs/detect/train*/weights/best.pt model_weights/
```

---

# ⚠️ 11. Update Model Path

Ensure `pipeline.py` contains:

```python
model = YOLO("model_weights/best.pt")
```

---

# 🚀 12. Run Inference

### Recommended (Validation Data)

```bash
python pipeline.py --image_dir val_data/images --output_dir outputs/
```

### Optional (Training Data)

```bash
python pipeline.py --image_dir train_data/images --output_dir outputs/
```

---

# 📍 13. Output Results

```bash
outputs/
├── image1.jpg    # annotated image
├── image1.json   # bounding boxes
```

---

# 📤 14. Output Format

```json
{
  "1": {
    "bbox_coordinates": [x_min, y_min, x_max, y_max]
  }
}
```

---

# 🔍 15. Defect Detection Logic

```text
Grayscale → Blur → Edge Detection → Contours → Filter → Bounding Boxes
```

---

# 📈 16. Evaluation Metric

**mF1@0.5–0.95**

* IoU thresholds: 0.50 → 0.95

---

# 🧠 17. Key Design Decisions

### ✅ Transfer Learning

* YOLOv8 pretrained on COCO dataset

### ✅ Proper Train/Validation Split

* 90% train / 10% validation for unbiased evaluation

### ✅ ROI-Based Detection

* Detect belt first → reduces noise

### ✅ Hybrid Approach

* Deep learning + classical CV

---

# ⚠️ 18. Limitations

* No defect labels available
* Cannot explicitly classify defect types
* Possible false positives

---

# 🚀 19. Future Improvements

* Add defect classification (scratch vs edge)
* Use anomaly detection (PatchCore / AutoEncoder)
* Improve filtering for better mF1 score
* Use segmentation instead of bounding boxes


# 🧩 20. End-to-End Workflow

```text
Split Data → Train Model → Copy best.pt → Run Pipeline → Generate Outputs
```

---

# 🏆 21. Best Practices Followed

✔ No hardcoded paths
✔ Portable project structure
✔ Proper dataset split
✔ Clean modular pipeline
✔ Reproducible results


