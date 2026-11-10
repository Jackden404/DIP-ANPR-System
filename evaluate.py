"""
evaluate.py - Automated DIP Pipeline Evaluation & Benchmark Suite

Runs end-to-end quantitative evaluation and unit tests across all 7 DIP pipeline stages:
1. Stage 1: Preprocessing & Adaptive Gamma
2. Stage 2 & 3: Plate Localization (YOLO vs Classical DIP) & Perspective Homography
3. Stage 4 & 5: Dual-Polarity Character Segmentation & Glyph Filtering
4. Stage 6: OCR Engines (EasyOCR, Tesseract, Template Matching)
5. Stage 7: Temporal Positional Majority Voting Tracker

Generates a formatted summary report and exports metrics to data/output_logs/benchmark_report.md.
"""

import os
import sys
import time
import glob
import cv2
import numpy as np

from src.preprocessor import Preprocessor
from src.plate_detector import PlateDetector
from src.segmenter import CharacterSegmenter
from src.recognizer import CharacterRecognizer
from src.tracker import PlateTracker


def run_unit_tests():
    print("=" * 70)
    print(" RUNNING STAGE-BY-STAGE UNIT TESTS")
    print("=" * 70)

    prep = Preprocessor()
    det = PlateDetector()
    seg = CharacterSegmenter()
    rec = CharacterRecognizer()
    tracker = PlateTracker(buffer_size=5, min_detections=3)

    # 1. Preprocessor Test
    dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    prep_res = prep.process(dummy, use_clahe=True)
    assert 'light_enhanced' in prep_res, "Preprocessor missing light_enhanced key"
    assert prep_res['light_enhanced'].shape == (480, 640), "Preprocessor shape mismatch"
    print("  [PASS] Stage 1: Preprocessor & Gamma Normalization")

    # 2. Perspective Homography Test
    pts = np.array([[10, 10], [100, 15], [95, 45], [12, 40]], dtype=np.float32)
    warped = det.rectify_perspective(dummy, pts, 280, 84)
    assert warped.shape == (84, 280, 3), "Homography rectification shape mismatch"
    print("  [PASS] Stage 2 & 3: 4-Point Homography Perspective Deskewing")

    # 3. Dual-Polarity Character Segmentation Test
    plate_canvas = np.ones((84, 280, 3), dtype=np.uint8) * 240
    # Draw sample glyphs
    cv2.putText(plate_canvas, "ABC123", (30, 58), cv2.FONT_HERSHEY_DUPLEX, 1.3, (20, 20, 20), 3)
    chars, binary, annotated = seg.segment_characters(plate_canvas)
    assert len(chars) >= 4, f"Segmentation extracted too few characters: {len(chars)}"
    print(f"  [PASS] Stage 4 & 5: Dual-Polarity Character Segmentation ({len(chars)} glyphs isolated)")

    # 4. OCR Engine Test (Template Matching)
    rec_template = CharacterRecognizer(preferred_engine="template")
    txt, conf, eng = rec_template.recognize_plate(plate_canvas, characters=chars)
    assert len(txt) >= 4, f"Template matching output too short: '{txt}'"
    print(f"  [PASS] Stage 6: Multi-Font Template Matching (Recognized: '{txt}', Conf: {conf:.2f})")

    # 5. Temporal Positional Majority Voting Test
    noisy_sequence = ["MH12DE1432", "MH12DE143Z", "MH12DE1432", "MH12DE1432"]
    confirmed = None
    for item in noisy_sequence:
        confirmed, _ = tracker.update(item, 0.75, plate_canvas, engine="test")
    assert confirmed == "MH12DE1432", f"Positional voting failed to converge: expected 'MH12DE1432', got '{confirmed}'"
    print(f"  [PASS] Stage 7: Positional Majority Voting (Resolved flicker 'Z' -> '2': '{confirmed}')")
    print("\nAll unit tests passed successfully!\n")


def run_benchmark():
    print("=" * 70)
    print(" RUNNING DATASET BENCHMARK & PERFORMANCE AUDIT")
    print("=" * 70)

    prep = Preprocessor()
    det = PlateDetector()
    seg = CharacterSegmenter()
    rec = CharacterRecognizer()

    sample_dir = "data/sample_videos"
    image_paths = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))

    if not image_paths:
        print(f"[Warning] No sample images found in {sample_dir}. Run 'python download_samples.py' first.")
        return

    results = []
    total_time = 0.0

    print(f"{'Filename':18s} | {'DIP Plate':12s} | {'Conf':6s} | {'Engine':12s} | {'Chars':5s} | {'Time (ms)':9s}")
    print("-" * 72)

    for p in image_paths:
        fname = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            continue

        t0 = time.time()

        prep_data = prep.process(img, use_clahe=True)
        cands, _ = det.detect_candidates(img, prep_data['light_enhanced'])

        plate_text = "N/A"
        conf = 0.0
        engine = "none"
        char_count = 0

        if cands:
            roi = cands[0]['warped']
            chars, _, _ = seg.segment_characters(roi)
            char_count = len(chars)
            t_txt, t_conf, t_eng = rec.recognize_plate(roi, characters=chars)
            if t_txt:
                plate_text = t_txt
                conf = t_conf
                engine = t_eng

        elapsed_ms = (time.time() - t0) * 1000.0
        total_time += elapsed_ms

        results.append({
            'filename': fname,
            'plate': plate_text,
            'conf': conf,
            'engine': engine,
            'chars': char_count,
            'time_ms': elapsed_ms
        })

        print(f"{fname:18s} | {plate_text:12s} | {conf:5.2f}  | {engine:12s} | {char_count:5d} | {elapsed_ms:8.1f}")

    avg_time = total_time / len(results) if results else 0
    effective_fps = 1000.0 / avg_time if avg_time > 0 else 0
    detected_count = sum(1 for r in results if r['plate'] != 'N/A')
    accuracy_pct = (detected_count / float(len(results))) * 100.0 if results else 0

    print("-" * 72)
    print(f"Overall Detection & Recognition Rate: {detected_count}/{len(results)} ({accuracy_pct:.1f}%)")
    print(f"Average Pipeline Latency: {avg_time:.1f} ms per frame ({effective_fps:.1f} FPS)")

    # Export report to markdown
    report_path = "data/output_logs/benchmark_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ANPR System Benchmark & Verification Report\n\n")
        f.write(f"- **Total Test Samples:** {len(results)}\n")
        f.write(f"- **Overall Recognition Rate:** {accuracy_pct:.1f}%\n")
        f.write(f"- **Average Latency:** {avg_time:.1f} ms ({effective_fps:.1f} FPS)\n\n")
        f.write("| Sample Filename | Recognized Plate | Confidence | OCR Engine | Segmented Characters | Latency (ms) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(f"| `{r['filename']}` | **`{r['plate']}`** | {r['conf']:.2f} | `{r['engine']}` | {r['chars']} | {r['time_ms']:.1f} |\n")

    print(f"\n[Export] Detailed benchmark report saved to: {report_path}")


if __name__ == "__main__":
    run_unit_tests()
    run_benchmark()

