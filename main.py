"""
main.py - Digital Image Processing License Plate Recognition System

Main entry point for running the real-time ANPR system on:
- Live Webcam / USB camera
- Video files (MP4, AVI)
- CCTV IP Camera (RTSP stream)

Usage Examples:
    # 1. Run on synthetic demo traffic video (auto-generated if needed)
    python main.py

    # 2. Run on live webcam
    python main.py --source 0

    # 3. Run on a specific video file
    python main.py --source data/sample_videos/my_car.mp4

    # 4. Run on a live CCTV RTSP stream
    python main.py --source "rtsp://admin:password@192.168.1.100:554/stream1"
"""

import os
import sys
import time
import argparse
import cv2

from src.preprocessor import Preprocessor
from src.plate_detector import PlateDetector
from src.segmenter import CharacterSegmenter
from src.recognizer import CharacterRecognizer
from src.tracker import PlateTracker
from src.ui_visualizer import UIVisualizer
from create_sample_data import generate_sample_traffic_video


def parse_args():
    parser = argparse.ArgumentParser(description="DIP Number Plate Recognition System (Video & Live CCTV)")
    default_source = "LP_Video.mp4" if os.path.exists("LP_Video.mp4") else "data/sample_videos/traffic_demo.mp4"
    parser.add_argument(
        "--source", 
        type=str, 
        default=default_source,
        help="Input source: '0' for webcam, file path for video, or 'rtsp://...' for CCTV stream."
    )
    parser.add_argument(
        "--clahe", 
        action="store_true", 
        default=True,
        help="Enable Contrast Limited Adaptive Histogram Equalization (default: True)"
    )
    parser.add_argument(
        "--inspector", 
        action="store_true", 
        default=True,
        help="Start with DIP Pipeline Inspector window enabled (default: True)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["auto", "yolo", "dip"],
        default="auto",
        help="Plate detector mode: 'auto' (YOLO with DIP fallback), 'yolo' (YOLO only), or 'dip' (pure classical DIP)."
    )
    parser.add_argument(
        "--ocr",
        type=str,
        choices=["auto", "easyocr", "tesseract", "template"],
        default="auto",
        help="OCR engine: 'auto' (EasyOCR -> Tesseract -> Template), 'easyocr', 'tesseract', or 'template'."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output_logs",
        help="Output directory for CSV logs and snapshot crops"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print(" DIGITAL IMAGE PROCESSING (DIP) - ANPR SYSTEM")
    print("=" * 65)

    source = args.source
    # Convert integer strings like '0' to int for webcam
    if source.isdigit():
        source = int(source)
        print(f"[Info] Opening Webcam device: {source}")
    else:
        # Check if requested file exists
        if not os.path.exists(str(source)) and not str(source).startswith("rtsp://") and not str(source).startswith("http://"):
            if os.path.exists("LP_Video.mp4"):
                print(f"[Info] Target '{source}' not found. Found local real traffic video 'LP_Video.mp4' - using it!")
                source = "LP_Video.mp4"
            else:
                print(f"[Info] Video '{source}' not found. Generating synthetic demo video...")
                source = "data/sample_videos/traffic_demo.mp4"
                generate_sample_traffic_video(source)

        print(f"[Info] Processing stream/video: {source}")

    is_image = False
    if isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        is_image = True
        static_img = cv2.imread(source)
        if static_img is None:
            print(f"[Error] Failed to read image file: {source}")
            sys.exit(1)
        cap = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Error] Failed to open video source: {source}")
            sys.exit(1)

    # Initialize DIP Pipeline Components
    # Configured with adaptive bounds for real-world traffic videos
    preprocessor = Preprocessor(clip_limit=3.0, tile_grid_size=(8, 8))
    detector = PlateDetector(min_aspect_ratio=1.6, max_aspect_ratio=7.2, min_area=350, mode=args.detector)
    segmenter = CharacterSegmenter()
    recognizer = CharacterRecognizer(preferred_engine=args.ocr)
    tracker = PlateTracker(buffer_size=10, min_detections=3, cooldown_seconds=5.0, output_dir=args.output)
    visualizer = UIVisualizer()

    use_clahe = args.clahe
    inspector_mode = args.inspector
    detector_mode = args.detector
    paused = False

    print("\n[Controls]")
    print("  [D] Toggle 6-Stage DIP Pipeline Inspector Window")
    print("  [C] Toggle CLAHE Contrast Enhancement On/Off")
    print("  [M] Toggle Detector Mode (Auto -> YOLO -> Classical DIP)")
    print("  [P] Pause / Play Video")
    print("  [S] Save Frame Snapshot & Log")
    print("  [Q] or [ESC] Exit\n")

    prev_time = time.time()
    fps = 0.0

    while True:
        if not paused:
            if is_image:
                frame = static_img.copy()
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    # Loop video if playing a video file
                    if not isinstance(source, int) and not str(source).startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("[Info] Stream ended or disconnected.")
                        break

            # Normalize large video resolutions (e.g. 1080p/4K) to maintain smooth 30 FPS
            fh, fw = frame.shape[:2]
            if fw > 1280:
                scale_factor = 1280.0 / fw
                frame = cv2.resize(frame, (1280, int(fh * scale_factor)))

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # -------------------------------------------------------------
            # STAGE 1: DIP Preprocessing (Grayscale, Bilateral, CLAHE)
            # -------------------------------------------------------------
            prep_data = preprocessor.process(frame, use_clahe=use_clahe)
            enhanced_gray = prep_data['light_enhanced']

            # -------------------------------------------------------------
            # STAGE 2 & 3: Sobel-X, Morphology, Thresholding & Contours
            # -------------------------------------------------------------
            candidates, debug_imgs = detector.detect_candidates(frame, enhanced_gray)

            # -------------------------------------------------------------
            # STAGE 4 & 5 & 6: Plate Rectification, Character Segmentation & OCR
            # -------------------------------------------------------------
            current_raw_text = ""
            current_confidence = 0.0
            best_plate_roi = None
            best_char_mask = None
            engine_used = "none"
            verified_candidates = []

            # Evaluate top candidate ROIs (capped at 3 to preserve high video FPS)
            for cand in candidates[:3]:
                roi = cand['warped']
                chars, char_binary, annotated_plate = segmenter.segment_characters(roi)
                txt, conf, eng = recognizer.recognize_plate(roi, characters=chars)

                if txt and len(txt) >= 4 and conf >= 0.28:
                    cand['text'] = txt
                    cand['conf'] = conf
                    cand['engine'] = eng
                    cand['char_mask'] = annotated_plate
                    verified_candidates.append(cand)

            if verified_candidates:
                top_cand = verified_candidates[0]
                best_plate_roi = top_cand['warped']
                best_char_mask = top_cand.get('char_mask')
                current_raw_text = top_cand['text']
                current_confidence = top_cand['conf']
                engine_used = top_cand['engine']
            elif candidates:
                best_plate_roi = candidates[0]['warped']
                _, _, annotated_plate = segmenter.segment_characters(best_plate_roi)
                best_char_mask = annotated_plate

            # -------------------------------------------------------------
            # STAGE 7: Temporal Stabilization & Logging
            # -------------------------------------------------------------
            confirmed_plate, is_new = tracker.update(
                current_raw_text, current_confidence, best_plate_roi, engine=engine_used
            )

            if is_new:
                print(f"[EVENT] Logged Confirmed Plate: {confirmed_plate} (Conf: {current_confidence:.2f}, Engine: {engine_used})")

        # -------------------------------------------------------------
        # DISPLAY & VISUALIZATION
        # -------------------------------------------------------------
        main_display = visualizer.draw_hud(
            frame, verified_candidates, confirmed_plate, fps, use_clahe, inspector_mode, detector_mode=detector_mode
        )
        cv2.imshow("DIP ANPR - Live CCTV Feed", main_display)

        # Show DIP Pipeline Inspector if enabled
        if inspector_mode:
            inspector_grid = visualizer.build_dip_inspector(
                enhanced_gray=enhanced_gray,
                grad_x=debug_imgs.get('grad_x'),
                closed_mask=debug_imgs.get('closed'),
                thresh_mask=debug_imgs.get('thresh'),
                plate_roi=best_plate_roi,
                char_mask=best_char_mask,
                plate_text=confirmed_plate or current_raw_text
            )
            cv2.imshow("DIP Pipeline Inspector (6-Stages)", inspector_grid)
        else:
            if cv2.getWindowProperty("DIP Pipeline Inspector (6-Stages)", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("DIP Pipeline Inspector (6-Stages)")

        # Keyboard interaction
        key = cv2.waitKey(20 if not paused else 100) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # ESC
            print("[Info] Exiting application.")
            break
        elif key == ord('d') or key == ord('D'):
            inspector_mode = not inspector_mode
            print(f"[Toggle] DIP Pipeline Inspector: {'ON' if inspector_mode else 'OFF'}")
        elif key == ord('c') or key == ord('C'):
            use_clahe = not use_clahe
            print(f"[Toggle] CLAHE Contrast Enhancement: {'ON' if use_clahe else 'OFF'}")
        elif key == ord('m') or key == ord('M'):
            modes = ["auto", "yolo", "dip"]
            curr_idx = modes.index(detector_mode) if detector_mode in modes else 0
            detector_mode = modes[(curr_idx + 1) % len(modes)]
            detector.mode = detector_mode
            print(f"[Toggle] Detector Mode: {detector_mode.upper()}")
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print(f"[Toggle] Playback {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('s') or key == ord('S'):
            snapshot_file = os.path.join(args.output, f"manual_snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snapshot_file, main_display)
            print(f"[Saved] Snapshot saved to: {snapshot_file}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
