🧠 COMSYS 2025 Hackathon Submission
===================================

🔍 Task A: Gender Classification (Binary) & Task B: Robust Face Matching under Adverse Visual Conditions
-----------------------------------------------------------------------------------------------

### 👨‍💻 Team Info:

- **Team Name:** **_`God Level Devs`_**
- **Members:** `Arnab Ghosh, Ayon Paul`
- **Institute:** `Dr. B. C. Roy Engineering College, Durgapur`

🧠 Problem Summary
------------------

The COMSYS 2025 challenge involved two computer vision tasks under visually adverse conditions:

*   **Task A: Gender Classification** — Classify face images as male or female with fairness and robustness.
*   **Task B: Face Verification** — Verify the identity of distorted face images by comparing them with clean reference faces.

    
Steps to run the solution
-------------

### ✅ **Task A: Gender Classification**

1.  **Open the notebook**: `Comsys_TaskA.ipynb`
    
2.  **Mount Google Drive** and unzip dataset:
    
    *   Ensure `Comys_Hackathon5.zip` is present in your Drive
        
3.  **Run all cells** to:
    
    *   Load data with transforms
    *   Train ResNet50 with weighted loss
    *   Evaluate metrics (Accuracy, Precision, Recall, F1)
        
4.  **To test manually**:
    
    *   Upload any face image and the model will predict gender
        

### ✅ Task B: Face Matching (Verification under Distortion)

1.  **Open the notebook**: Comsys\_TaskB.ipynb
2.  **Mount Google Drive** and unzip dataset
3.  **Load InsightFace** `(buffalo_l)` for face embeddings
4.  **Build identity gallery** from `train/` (clean images)
5.  **Run matching shell**:
    *   Compares distorted `val/` images to gallery using cosine similarity
    *   Outputs predictions and computes **Accuracy, Precision, Recall, F1 Score**
6.  **Optionally test any image pair manually** by setting paths
    

### ⚠️ Notes

*   Recommended to run in **Google Colab (with GPU)**
*   Use paths like `/content/drive/MyDrive/...` throughout



✅ Our Approach
==================

## Task A – Gender Classification

*   **Model:** ResNet-50 pretrained on ImageNet
*   **Input:** Clean face images (with augmentation to simulate distortions)
*   **Loss:** Weighted CrossEntropyLoss (to handle class imbalance)
*   **Optimizer:** Adam (learning rate = 1e-4)
*   **Augmentations:** Horizontal flip, color jitter, Gaussian blur
*   **Class Balancing:** WeightedRandomSampler to address ~5:1 male-to-female ratio
*   **Training:** On Google Colab with GPU (5 epochs)
    
### Pretrained model- RestNet50:
**Architecture:**

![image](https://github.com/user-attachments/assets/1580d267-0dfe-46cd-a123-53ffd1b9a61d)

### **Weight**: [Click Here](https://drive.google.com/file/d/1HoK3uXRvtS9YiedMPxooO8p-rT7YcTIP/view?usp=drive_link)

📊 Training Metrics:

```
✅ Accuracy  : 95.26%  
🎯 Precision : 96.70%  
📈 Recall    : 94.29%  
🏆 F1 Score  : 89.80%
```

📊 Final Validation Results:

```
✅ Accuracy  : 96.21%  
🎯 Precision : 98.90%  
📈 Recall    : 85.71%  
🏆 F1 Score  : 91.84%
```

## Task B – Robust Face Verification

*   **Model:** `InsightFace (buffalo\_l)`, pretrained 512-D ArcFace embeddings
*   **Gallery:** Built from clean images in `train/` + optionally `val/`
*   **Distortion Types:** Blurred, foggy, noisy, low-light, sunny, resized
*   **Matching Strategy:** Cosine similarity between test image and gallery embeddings
*   **Positive Match Rule:** `Match = 1` if predicted identity == actual identity **and** similarity ≥ thresholdElse → `Match = 0`
*   **Threshold Used:** `SIM_THRESHOLD = 0.65`
*   **Error Handling:** Skips images with undetectable faces and logs cleanly
    

### Pretrained model:
**Architecture:**

![image](https://github.com/user-attachments/assets/59c8c611-b17e-4b0c-bc2c-5e0aa7061892)

### **Weight**: We used the buffalo_l pretrained model from InsightFace, which loads weights internally via the FaceAnalysis API and does not require manual downloading or storage.

📊 Evaluation Results:

```
📊 Final Evaluation — Task B (Distorted Face Matching):
✅ Accuracy  : ~91.46%
🎯 Precision : 100%
📈 Recall    : ~91.46%
🏆 F1 Score  : ~95.54%
```

⚙️ Innovations & Strengths
--------------------------

*   🧠 Used powerful **InsightFace embeddings** for generalization without custom training
*   🧪 Simulated real-world distortions in Task A to improve robustness
*   🧼 Class-balanced training pipeline to overcome dataset imbalance
*   🛡️ Fail-safe design: Skips `unreadable/distorted/no-face` images without crashing
*   📈 Clear metric printouts & detailed logs for evaluation and debugging
    

📁 Repo Contents
----------------

```
├── Task A
|   └── Comsys_TaskA.ipynb
│   └── TaskA_testing
│       ├── male/
│       │   └── test_male_images.jpg / .webp ...
│       └── female/
│           ├── kalpana_chawla.webp
│           ├── Lata-Mangeshkar-Biography.webp
│           ├── test_f_1.webp
│           ├── test_f_3.jpeg
├── Task B
|   └── Comsys_TaskA.ipynb
│   └── TaskB_testing
│       ├── train/
│       └── val/
│           ├── pic7/
│           │   ├── pic7.jpg
│           │   └── distortion/
│           │       ├── pic7_foggy.jpg
│           │       └── pic7_light.jpg
│           ├── pic8/
│           └── pic10/
├── README.md
```

📝 Submission Checklist
-----------------------
```
✔️ Training + validation metrics for both tasks
✔️ Well-documented source code with comments
✔️ Pretrained model weights (.pth)
✔️ Test scripts accepting folder paths
✔️ 100% reproducible
```
📌 Notes & Limitations
----------------------

*   Some extreme distortions (e.g., over-exposed or heavily noisy images) may occasionally fail detection
*   Manual visual validation is also provided for reliability demo
    

🚀 Tools & Libraries
--------------------
**Task A:**
* Python, PyTorch, TorchVision
* Pre-trained model: **ResNet50**
* PIL, tqdm
* Google Colab (runtime + Drive)

**Task B**
* OpenCV
* Numpy
* Scikit-learn, tqdm
* Pre-trained model: **InsightFace**, ONNX Runtime
* Google Colab (runtime + Drive)
