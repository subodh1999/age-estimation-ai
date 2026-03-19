# Setup Guide: Running the Consolidated Project Notebook

---

## 1. Data Sources

Download the following three datasets before running the notebook:

### 1.1 FairFace Race Dataset (Week 1 only)
- **Link:** https://www.kaggle.com/datasets/abdulwasay551/fairface-race
  *(Note: The original FairFace dataset was used for the initial audit. Since Week 1 analysis is already executed with outputs saved in the notebook, re-running this section is optional.)*

### 1.2 UTKFace Dataset
- **Link:** https://www.kaggle.com/datasets/jangedoo/utkface-new
- **What you get:** A folder of `.jpg` images named in the format `age_gender_race_timestamp.jpg` (e.g., `24_0_1_20170116220224650.jpg`)
- **Used in:** Week 2 (CNN training) and Week 3 (all three models)

### 1.3 MORPH-2 Dataset
- **Link:** https://www.kaggle.com/datasets/chiragsaipanuganti/morph
- **What you get:** A folder structure containing `Images/Train`, `Images/Validation`, `Images/Test`, and `Index/Train.csv`
- **Used in:** Week 2 (CNN + VGG16 training) and Week 3 (all three models)

---

## 2. Folder Structure

After downloading and extracting, your project directory should look like this:

```
your_project_folder/
│
├── Consolidated_Project_Notebook.ipynb
│
├── extracted_data/
│   ├── utk_face/
│   │   └── UTKFace/
│   │       ├── 1_0_0_20161219203650035.jpg
│   │       ├── 24_0_1_20170116220224650.jpg
│   │       └── ... (all UTKFace .jpg images here)
│   │
│   └── Morph-2/
│       └── Dataset/
│           ├── Images/
│           │   ├── Train/
│           │   │   ├── 00001_03M52.JPG
│           │   │   └── ...
│           │   ├── Validation/
│           │   │   └── ...
│           │   └── Test/
│           │       └── ...
│           └── Index/
│               └── Train.csv
│
└── Dataset/                    (Required for VGG16 section in Week 2)
    └── Images/
        ├── Train/
        │   └── ... (same MORPH-2 Train images)
        ├── Validation/
        │   └── ... (same MORPH-2 Validation images)
        └── Test/
            └── ... (same MORPH-2 Test images)
```

**Important:** The VGG16 code (Week 2) uses a separate `Dataset/` path while the CNN and Week 3 code uses `extracted_data/`. You can either:
- Copy the MORPH-2 images into both locations, OR
- Create a symbolic link: `ln -s extracted_data/Morph-2/Dataset Dataset`

---

## 3. Path Configuration

If your folder structure differs from the one above, update the following paths in the notebook:

### Week 1 (Cell 8)
```python
root_path = r"extracted_data/FairFace Race"
```
*(Only needed if you want to re-run the FairFace audit. Outputs are already saved in the notebook.)*

### Week 2 — CNN Section (Cell 31)
```python
UTK_DIR = r"extracted_data/utk_face/UTKFace"
MORPH_DIR = r"extracted_data/Morph-2/Dataset/Images/Train"
MORPH_CSV = r"extracted_data/Morph-2/Dataset/Index/Train.csv"
```

### Week 2 — CNN Test Evaluation (Cell 38)
```python
TEST_DIR = r"extracted_data/Morph-2/Dataset/Images/Test"
```

### Week 2 — VGG16 Section (Cells 54-55)
```python
base_path = 'Dataset/'
train_img_dir = os.path.join(base_path, 'Images/Train')
val_img_dir = os.path.join(base_path, 'Images/Validation')
```

### Week 2 — VGG16 Test Evaluation (Cells 58-59)
```python
test_img_dir = os.path.join(base_path, 'Images/Test')
```

### Week 3 — All Models (Cell 72)
```python
morph_dir  = 'extracted_data/Morph-2/Dataset/Images/Train'
utk_dir    = 'extracted_data/utk_face'
MORPH_CSV  = r"extracted_data/Morph-2/Dataset/Index/Train.csv"
TEST_DIR   = r"extracted_data/Morph-2/Dataset/Images/Test"
```

---

## 4. Saved Model Weights (.pth files)

During training, the notebook saves the best model weights as `.pth` files in the working directory. These are required for the evaluation, XAI, and fairness audit sections that come after training.

| Model | Filename | Saved During |
|---|---|---|
| CNN (Baseline) | `final_age_model.pth` | Week 2 — Cell 37 |
| VGG16 | `best_vgg16_age_model.pth` | Week 2 — Cell 57 |
| ResNet-50 (Unfrozen) | `best_resnet50_unfrozen.pth` | Week 3 — Cell 89 |
| ViT-B/16 (Unfrozen) | `best_vit_unfrozen.pth` | Week 3 — Cell 90 |
| MobileNetV2 | `best_mobilenetv2.pth` | Week 3 — Cell 91 |

If you already have these `.pth` files (e.g., from a previous run), place them in the same directory as the notebook and you can skip the training cells and jump directly to the model loading cell (Cell 92).

---

## 5. Python Dependencies

Make sure the following packages are installed:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install tqdm pillow
pip install pytest pytest-cov     # for Section 9 (Automated Testing)
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 6. Running the Notebook

### Option A: Run Everything from Scratch
1. Place datasets in the folder structure described in Section 2.
2. Open the notebook and run all cells sequentially from top to bottom.
3. Training will take significant time (especially ViT-B/16). A GPU is strongly recommended.

### Option B: Skip Training (Use Pre-trained Weights)
1. Place the 5 `.pth` files listed in Section 4 in the notebook directory.
2. Run Week 1 and Week 2 cells normally (these have cached outputs).
3. In Week 3, run the setup, data preparation, and model architecture cells.
4. Skip the training cells (Cells 89, 90, 91) and go directly to the model loading cell (Cell 92).
5. Continue running all evaluation, fairness, XAI, and testing cells normally.

---

## 7. Common Issues

| Problem | Solution |
|---|---|
| `FileNotFoundError` on image paths | Verify your folder structure matches Section 2. Check for extra nested folders after extraction. |
| `CUDA out of memory` | Reduce `batch_size` in the DataLoader cells (try 64 or 32 instead of 128). |
| VGG16 can't find `Dataset/` folder | Create a symlink: `ln -s extracted_data/Morph-2/Dataset Dataset` |
| `.pth` file not found during model loading | Either run the training cells first, or place pre-trained `.pth` files in the notebook directory. |
| `ModuleNotFoundError` for pytest | Run `pip install pytest pytest-cov` before the testing section. |
