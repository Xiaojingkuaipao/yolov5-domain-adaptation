# Road Damage Detection with Domain Adaptation Project

[中文 🇨🇳](README-CN.md)


This project addresses two critical challenges in road damage detection: **high annotation costs** and **significant cross-domain differences**. We propose an improved YOLOv5-based domain adaptation solution that incorporates MK-MMD adversarial loss and an adaptive photo enhancement (APAGE) module to achieve efficient transfer learning on unlabeled target domain data.

---

## Data & Methodology Visualizations

### Source vs Target Domain Comparison
![Source vs Target](./assets/source_target_comparison.jpg)
*Left: Samples from source domain (RDD2020 dataset) | Right: Original target domain images (low-light/blurred)*

### APAGE Enhancement Results
![APAGE Result](./assets/apage_comparison.jpg)
*Left: Original underexposed image | Right: After APAGE enhancement*

### Automatic Labeling Demonstration
![Labeling Result](./assets/labeling_demo.jpg)
*Visualization of auto-annotation results after domain adaptation*

---

## 📦 Open-Source Resources

1. **Pre-trained Weights** (YOLOv5s-RDDC2020):  
   [yolov5s_rddc.pt Download (Quark)](https://pan.quark.cn/s/47adafea23e0)

2. **Domain-Adapted Weights** (Post DA training):  
   [yolov5s-da-best.pt Download (Quark)](https://pan.quark.cn/s/10f7aa927de5)

3. **Source Domain Dataset**:  
   [rddc_yolo.tar Download (Quark)](https://pan.quark.cn/s/91f94da37a1c)

4. **Target Domain Dataset**:  
   [my_road_rgb.tar Download (Quark)](https://pan.quark.cn/s/0968c299d25b)

5. **Official Pre-trained Weights** (Optional):  
   [yolov5s.pt Download (Quark)](https://pan.quark.cn/s/f22df190c7ca)

---

## 📁 Dataset Directory Structure

### After decompression:
```bash
# Source domain dataset (after extracting rddc_yolo.tar)
rddc_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
class_mapping.json  # Class mapping file

# Target domain dataset (after extracting my_road_rgb.tar)
rgb_images/
├── *.jpg
└── ...  # 5000+ unlabeled images
```

---

## 🔧 Quick Start Guide

### 1. Environment Setup
```bash
# Create Conda environment
conda create -n yoloda python=3.8
conda activate yoloda

# Install dependencies
git clone https://github.com/Xiaojingkuaipao/yolov5-domain-adaptation.git
cd yolov5-domain-adaptation
git checkout da
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Extract datasets
tar -xvf rddc_yolo.tar
tar -xvf my_road_rgb.tar
```

### 3. Branch Switching
- **Base Training** (YOLOv5-RDDC2020):  
  ```bash
  git checkout master  # Main branch for base training
  ```
- **Domain Adaptation** (DA module development):  
  ```bash
  git checkout da       # da branch contains core DA code
  ```

### 4. Weight Configuration
Place downloaded weight files in project root:
```bash
ls -l *.pt
# Should show yolov5s_rddc.pt and yolov5s-da-best.pt
```

### 5. Configuration Files
- Modify `my_cfg/target_data.yaml` for target domain path:
  ```yaml
  train: path/to/your_rgb_images   # ✅ Point to target domain images
  ```
- Modify `my_cfg/rddc_2020.yaml` for source domain path:
  ```yaml
   train: path/to/your_rddc2020_train/*.jpg
   val: path/to/your_rddc_val/*.jpg    # ✅ Point to source domain data
  ```

---

## 🚀 Training Strategy

### Two-Phase Training
The training strategy adopts a two-phase approach. First, the model learns to recognize road damage patterns, then adapts to the target domain through distribution alignment. You may either train from scratch or use our provided pre-trained weights for phase two.

1. **Phase 1: Source Domain Pre-training**
   ```bash
   # Execute on master branch (usage follows YOLOv5 v5.0 official repo)
   python train.py \
     --weights yolov5s.pt \          # Official pre-trained weights
     --cfg my_cfg/yolov5s.yaml \     # Network config
     --data my_cfg/rddc_2020.yaml \  # Source domain config
   ```

2. **Phase 2: Domain Adaptation**
   ```bash
   # Switch to da branch and execute
   python train.py \
     --weights yolov5s_rddc.pt \     # Phase 1 weights
     --cfg my_cfg/yolov5s.yaml \     # Network config
     --data my_cfg/rddc_2020.yaml \  # Source domain config
     --target-data my_cfg/target_data.yaml \  # Target domain config
     --da-layers [4, 6, 10] \        # MK-MMD layers
     --da-weights [0.33, 0.33, 0.33] # Loss weights
     --name domain_adaptation       # Output directory
   ```

## Detection Application
   ```bash
   # On da branch
    python detect.py --weights yolov5s-da-best.pt --img 640 --source path/to/your/images
   ```
---

## 📦 Key Features
- **Innovative Method**: MK-MMD based domain adaptation integrated with YOLOv5 for feature space alignment
- **Illumination Optimization**: Proprietary APAGE module for low-light image enhancement

---

Contributions via Issues and Pull Requests are welcome!