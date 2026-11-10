# Digital Image Processing (DIP) - Real-Time Number Plate Recognition System (ANPR)

An end-to-end, classical **Digital Image Processing (DIP)** system for automatic license plate detection, perspective deskewing, character segmentation, and recognition from **live CCTV camera feeds, webcams, and recorded traffic videos**.

Built specifically to demonstrate fundamental image processing algorithms without reliance on opaque deep-learning models, while also supporting an optional hybrid YOLOv8 commercial mode.

---

## 📸 Key Features

* **Real-Time Video & CCTV Stream Processing:** Supports live Webcams (`--source 0`), video files (`.mp4`, `.avi`), and IP Camera CCTV streams (`rtsp://...`).
* **Dual-Mode Detection Engine:** Switch on the fly between **Classical DIP Mode** (Sobel-X, HSV color slicing, morphology, Otsu) and **YOLOv8 AI Mode** via the `[M]` key.
* **Live "DIP Pipeline Inspector" (2x3 Dashboard):** Press `[D]` to open an interactive 6-stage window visualizing every intermediate mathematical transform in real time.
* **Dual-Polarity Character Segmentation:** Self-selecting polarity analysis handles both dark-on-light and light-on-dark plates with zero manual tuning.
* **Overhead CCTV Perspective Rectification:** Straightens angled and skewed plates via 4-point homography (`warpPerspective`).
* **Adaptive Illumination & CLAHE:** Combines dynamic scene gamma correction with Contrast Limited Adaptive Histogram Equalization to handle headlights, shadows, nighttime grain, and glare.
* **Temporal Positional Voting Buffer:** Multi-frame character-level majority voting eliminates OCR jitter and single-frame misreads.
* **Automated Audit Logging:** Saves timestamps, recognized license plate strings, confidence levels, and vehicle plate crops into `data/output_logs/entry_logs.csv`.
* **Automated Verification Test Suite:** Comprehensive unit tests and benchmark suite (`evaluate.py`) validating all 7 DIP stages.

---

## 🔬 The 7-Stage DIP Pipeline & Mathematical Foundations

```
[ Raw Video Frame ]
        │
        ▼ (Stage 1: Preprocessing)
 ┌────────────────────────────────────────────────────────┐
 │ • Grayscale Conversion (Luminance weighting)           │
 │ • Adaptive Gamma Correction: γ = ln(0.5) / ln(μ_norm)  │
 │ • Bilateral Filter (Edge-preserving noise suppression) │
 │ • CLAHE (Local adaptive histogram equalization)        │
 │ • Top-Hat & Black-Hat Morphological enhancement        │
 └────────────────────────────────────────────────────────┘
        │
        ▼ (Stage 2: Feature Extraction & Gradient Filtering)
 ┌────────────────────────────────────────────────────────┐
 │ • Sobel-X Gradient Operator (Vertical edge emphasis)   │
 │ • Lightness-Normalized HSV Color Slicing (White/Yellow)│
 │ • Morphological Closing with Rectangular Kernel (21x5) │
 └────────────────────────────────────────────────────────┘
        │
        ▼ (Stage 3: Image Segmentation & Morphology)
 ┌────────────────────────────────────────────────────────┐
 │ • Otsu's Optimum Global Thresholding                   │
 │ • Morphological Opening & Erosion (noise removal)      │
 │ • Contour Detection & Geometric Prior Filtering        │
 │ • Non-Maximum Suppression (NMS) candidate filtering    │
 └────────────────────────────────────────────────────────┘
        │
        ▼ (Stage 4: Geometric Transformation / Deskewing)
 ┌────────────────────────────────────────────────────────┐
 │ • Minimum Area Rotated Bounding Box                    │
 │ • 4-Point Homography & Perspective Transformation      │
 └────────────────────────────────────────────────────────┘
        │
        ▼ (Stage 5: Character Segmentation)
 ┌────────────────────────────────────────────────────────┐
 │ • Dual-Polarity Self-Selection (White/Black background)│
 │ • Vertical Disconnection Morphological Kernel (1x3)    │
 │ • Margin Clearance (Mounting screw & border removal)   │
 │ • Connected Component Geometric Filtering              │
 │ • Spatial Left-to-Right Sorting of Glyphs              │
 └────────────────────────────────────────────────────────┘
        │
        ▼ (Stage 6: OCR & Positional Disambiguation)
 ┌────────────────────────────────────────────────────────┐
 │ • Super-Resolution Pre-OCR Upscaling (2.2x + Unsharp)  │
 │ • Multi-Engine OCR (EasyOCR / Tesseract / Template)    │
 │ • Multi-Font, Multi-Weight Template Matching           │
 │ • Positional Context Correction ('O' vs '0', 'I' vs '1')│
 └────────────────────────────────────────────────────────┘
        │
        ▼ (Stage 7: Video Tracking & Logging)
 ┌────────────────────────────────────────────────────────┐
 │ • Positional Majority Voting Buffer across frames      │
 │ • Temporal Consensus Gate (≥ 70% character agreement)  │
 │ • Persistent CSV Logging & Cropped Plate Archival      │
 └────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Directory Structure

```text
DIP/
├── data/
│   ├── sample_videos/        # Synthetic demo traffic video and test assets
│   └── output_logs/          # Generated CSV logs, benchmark reports, and cropped plate snapshots
├── src/
│   ├── __init__.py           # Package initializer
│   ├── preprocessor.py       # Grayscale, Adaptive Gamma, Bilateral Filter, CLAHE, Top-Hat/Black-Hat
│   ├── plate_detector.py     # Sobel-X, Color slicing, Morphology, Otsu, NMS, Homography
│   ├── segmenter.py          # Dual-polarity character binarization, vertical split, CC labeling
│   ├── recognizer.py         # Multi-engine OCR (EasyOCR, Tesseract, Multi-Font Template Matching)
│   ├── tracker.py            # Positional majority voting tracker & CSV logger
│   └── ui_visualizer.py      # HUD overlay & 2x3 DIP Pipeline Inspector dashboard
├── create_sample_data.py     # Synthetic HD traffic video generator
├── download_samples.py       # Benchmark real vehicle dataset downloader
├── evaluate.py               # Automated unit test suite & dataset benchmark audit
├── main.py                   # Main CLI execution orchestrator
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation & viva guide
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Open PowerShell or Terminal in the project root:
```bash
pip install -r requirements.txt
```

*(Optional for Tesseract OCR)*: If you wish to use Tesseract OCR, install the [Tesseract-OCR Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your PATH. **Note:** The system includes both EasyOCR and a built-in classical DIP multi-font template matching engine, so it runs completely out of the box even without Tesseract installed!

### 3. Run the Automated Evaluation Suite
Validate all 7 stages and run dataset accuracy audits:
```bash
python evaluate.py
```

### 4. Run the ANPR System

#### A. Real Highway Traffic Video
```bash
python main.py --source LP_Video.mp4
```

#### B. Synthetic Demo Traffic Video (Default)
If you don't have a camera or car video yet, simply run:
```bash
python main.py
```
*The system will automatically generate a realistic synthetic traffic video and start detecting in real time!*

#### C. Live Laptop Webcam / USB Camera
```bash
python main.py --source 0
```

#### D. Benchmark Samples & Static Image Testing
Download high-quality real vehicle benchmark images:
```bash
python download_samples.py
python main.py --source data/sample_videos/us_car_1.jpg
python main.py --source data/sample_videos/eu_car_2.jpg
```

#### E. Pure Classical DIP Mode (No Deep Learning)
```bash
python main.py --source data/sample_videos/sample_car.jpg --detector dip
```

#### F. Custom OCR Engine Selection
```bash
python main.py --source LP_Video.mp4 --ocr easyocr
python main.py --source sample_car.jpg --ocr template
```

#### G. Live CCTV IP Camera (RTSP Stream)
```bash
python main.py --source "rtsp://admin:password@192.168.1.100:554/stream1"
```

---

## 🎮 Interactive Keyboard Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`[D]`** | **Toggle DIP Inspector** | Opens/closes the 2x3 intermediate stage visualizer window. |
| **`[M]`** | **Toggle Detector Mode** | Cycles through `AUTO` $\rightarrow$ `YOLO` $\rightarrow$ `CLASSICAL DIP` live on the fly. |
| **`[C]`** | **Toggle CLAHE** | Turns adaptive histogram equalization ON or OFF to observe shadow impact. |
| **`[P]`** | **Pause / Play** | Freezes the video frame to inspect detections in detail. |
| **`[S]`** | **Save Snapshot** | Manually saves the current frame and detection to `data/output_logs/`. |
| **`[Q]`** or **`[ESC]`** | **Exit** | Gracefully releases camera and terminates the program. |

---

## 🎓 Viva & Exam Preparation Guide (Questions & Answers)

Here are key questions professors and external examiners typically ask during project defense:

### Q1: Why did you use Sobel-X ($G_x$) instead of Sobel-Y ($G_y$) or Laplacian for plate detection?
**Answer:** Alphanumeric characters on license plates consist predominantly of vertical line strokes (e.g., the vertical stems in '1', 'H', 'E', 'B', 'M', 'N', 'D'). Computing the horizontal gradient ($\frac{\partial f}{\partial x}$) with Sobel-X highlights this dense cluster of vertical edges. Sobel-Y would highlight horizontal edges such as car bumpers, road lane markers, and vehicle roof lines, introducing unwanted clutter.

### Q2: Why use a Bilateral Filter instead of a standard Gaussian Filter?
**Answer:** A standard Gaussian filter computes weights based solely on spatial distance, which smooths across edges and blurs sharp boundaries. The Bilateral Filter introduces a photometric range component:
$$I^{\text{filtered}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i)-I(x)\|) g_s(\|x_i-x\|)$$
Pixels with large intensity differences (like high-contrast character edges) receive near-zero weight in $f_r$, preserving sharp borders while smoothing sensor noise in flat regions.

### Q3: Why is Morphological Closing performed with a wide rectangular kernel (e.g., 21x5)?
**Answer:** License plate characters are separated by small horizontal gaps. Morphological closing (dilation followed by erosion) using a structuring element wider than the character spacing merges the separate vertical character strokes into one continuous, solid horizontal block. This enables contour detection to extract the entire license plate as a single candidate object.

### Q4: How does Otsu's Thresholding find the optimal threshold?
**Answer:** Otsu's method evaluates all possible gray-level thresholds $T \in [0, 255]$ and selects the threshold $T^*$ that maximizes the inter-class variance $\sigma_B^2(T)$:
$$\sigma_B^2(T) = \omega_0(T)\omega_1(T)[\mu_0(T) - \mu_1(T)]^2$$
Where $\omega_0, \omega_1$ are the probabilities of background and foreground classes, and $\mu_0, \mu_1$ are their respective mean intensities. This is completely non-parametric and self-adaptive.

### Q5: How does Perspective Transformation (Homography) help in CCTV feeds?
**Answer:** CCTV cameras are usually mounted at an elevated position and angled downward, causing projective distortion (plates appear as trapezoids rather than rectangles). By identifying the 4 corner points of the minimum area bounding box, we calculate a 3x3 homography matrix $\mathbf{H}$ using `cv2.getPerspectiveTransform`:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
Applying `cv2.warpPerspective` rectifies the plate into an upright, flat $280 \times 84$ pixel image, making character segmentation significantly more accurate.

### Q6: How does Dual-Polarity Self-Selection work in Character Segmentation?
**Answer:** Real-world license plates vary in polarity: standard civilian plates are dark characters on a light background, whereas certain commercial, military, or diplomatic plates are light characters on a dark background. Instead of hardcoding an inversion assumption, the system binarizes the plate in both direct and inverted forms, runs connected component labeling on both, and selects the polarity that yields candidate glyphs matching the expected physical aspect ratio ($0.12 \le w/h \le 1.35$), height ratio, and horizontal baseline alignment.

### Q7: Why use a vertical morphological opening kernel (1x3) during character segmentation?
**Answer:** Due to shadow smearing or slight over-binarization, adjacent characters often touch at the bottom baseline, causing multiple letters to merge into a single wide component. A vertical $(1 \times 3)$ opening kernel erodes thin horizontal baseline connections while preserving the vertical character stems, cleanly separating merged glyphs.

### Q8: Why is Positional Majority Voting superior to simple string equality in video streams?
**Answer:** In video feeds, single-frame OCR is vulnerable to camera shake, frame compression artifacts, motion blur, and glare. If 4 consecutive frames produce `MH12DE1432`, `MH12DE143Z`, `MH12DE1432`, `MH12DE1432`, naive string equality treats the second reading as a distinct plate. Positional majority voting constructs a column-by-column frequency histogram for each character index:
$$\text{Char}_i^* = \arg\max_{c \in \Sigma} \sum_{t=1}^N \mathbf{1}(R_t[i] == c)$$
At index 9, the vote between `'2'` (3 votes) and `'Z'` (1 vote) reliably resolves to `'2'`, completely eliminating single-character flicker.

### Q9: What is the purpose of Adaptive Gamma Correction before CLAHE?
**Answer:** While CLAHE equalizes local contrast, severely underexposed night footage has a narrow histogram compressed near zero, whereas high-beam headlights compress values near 255. Adaptive gamma dynamically shifts the scene's mean intensity $\mu$ toward mid-gray ($0.5$):
$$\gamma = \frac{\ln(0.5)}{\ln(\mu / 255.0)}$$
Applying non-linear lookup table mapping before bilateral filtering and CLAHE prevents noise amplification in dark regions and highlight clipping under glare.
