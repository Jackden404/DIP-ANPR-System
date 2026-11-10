"""
plate_detector.py - Digital Image Processing License Plate Candidate Detection

This module implements:
1. Commercial Deep Learning Localization (YOLOv8)
2. Foundational Classical DIP Algorithm:
   - Sobel-X Gradient Computation (vertical edge emphasis)
   - Lightness-normalized Plate Color Slicing (HSV white/yellow)
   - Morphological Closing (fuses character strokes into unified plate region)
   - Otsu's Optimum Global Thresholding & Morphological Cleaning
   - Contour Geometric Prior Filtering (Aspect ratio, Area, Solidity, Vertical Alignment)
   - Non-Maximum Suppression (NMS) Candidate Deduplication
   - Perspective Rectification (4-point Homography Deskewing)
"""

import os
import urllib.request
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class PlateDetector:
    def __init__(self, 
                 min_aspect_ratio: float = 1.6, 
                 max_aspect_ratio: float = 7.2,
                 min_area: int = 350, 
                 max_area_ratio: float = 0.25,
                 mode: str = "auto"):
        """
        Initializes the Plate Detector.
        
        Args:
            min_aspect_ratio: Minimum allowable width/height ratio.
            max_aspect_ratio: Maximum allowable width/height ratio (supports wide European/Indian plates).
            min_area: Minimum bounding box area in pixels.
            max_area_ratio: Maximum allowable area relative to frame size.
            mode: 'auto' (YOLO with DIP fallback), 'yolo' (YOLO only), or 'dip' (pure classical DIP).
        """
        self.min_ar = min_aspect_ratio
        self.max_ar = max_aspect_ratio
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.mode = mode.lower()

        self.yolo_model = None
        if YOLO_AVAILABLE:
            model_path = "license_plate_detector.pt"
            if not os.path.exists(model_path):
                try:
                    from huggingface_hub import hf_hub_download
                    print("[Model] Downloading verified YOLOv8 License Plate Model from Hugging Face...")
                    downloaded_file = hf_hub_download(repo_id="Koushim/yolov8-license-plate-detection", filename="best.pt")
                    import shutil
                    shutil.copy(downloaded_file, model_path)
                    print(f"[Model] Download complete: {model_path}")
                except Exception as e1:
                    print(f"[Model] HuggingFace Hub download attempt: {e1}")
                    urls = [
                        "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
                        "https://huggingface.co/AZIIIIIIIIZ/License-plate-detection/resolve/main/best.pt"
                    ]
                    for url in urls:
                        try:
                            print(f"[Model] Downloading from {url}...")
                            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                                out_file.write(response.read())
                            print("[Model] Download successful!")
                            break
                        except Exception as e2:
                            print(f"[Model] URL failed: {e2}")

            if os.path.exists(model_path):
                try:
                    self.yolo_model = YOLO(model_path)
                    print("[Model] YOLO License Plate Detector loaded successfully!")
                except Exception as e:
                    print(f"[Model] Could not load YOLO model: {e}")

    def compute_sobel_gradient(self, gray: np.ndarray) -> np.ndarray:
        """
        Computes Sobel gradient along X-direction (vertical edges).
        Highlights vertical character strokes while suppressing horizontal road/bumper noise.
        """
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        if max_val - min_val > 0:
            grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        return grad_x.astype(np.uint8)

    def apply_morphological_closing(self, grad_x: np.ndarray, kernel_size: tuple = (21, 5)) -> np.ndarray:
        """
        Performs Morphological Closing with a horizontal rectangular kernel to connect character gradients.
        """
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)

    def threshold_and_clean(self, closed: np.ndarray) -> np.ndarray:
        """
        Binarizes the closed gradient image using Otsu's thresholding, followed by
        morphological opening and vertical consolidation to eliminate background noise.
        """
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, square_kernel)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, vert_kernel)
        return cleaned

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders 4 coordinates: [top-left, top-right, bottom-right, bottom-left]."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def rectify_perspective(self, frame: np.ndarray, pts: np.ndarray, target_w: int = 280, target_h: int = 84) -> np.ndarray:
        """
        Warps a rotated, skewed plate candidate into an upright standardized rectangular image.
        Uses 3x3 Homography Matrix (cv2.warpPerspective).
        """
        rect = self.order_points(pts)
        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, matrix, (target_w, target_h))

    def apply_nms(self, candidates: list, iou_thresh: float = 0.35) -> list:
        """Suppresses overlapping candidate boxes to prevent duplicate OCR evaluations."""
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda c: c['score'], reverse=True)
        keep = []
        while candidates:
            best = candidates.pop(0)
            keep.append(best)
            bx, by, bw, bh = best['straight_box']
            b_area = bw * bh
            remaining = []
            for c in candidates:
                cx, cy, cw, ch = c['straight_box']
                c_area = cw * ch
                ix1 = max(bx, cx)
                iy1 = max(by, cy)
                ix2 = min(bx + bw, cx + cw)
                iy2 = min(by + bh, cy + ch)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih
                union_area = b_area + c_area - inter_area
                iou = inter_area / float(union_area) if union_area > 0 else 0
                if iou < iou_thresh:
                    remaining.append(c)
            candidates = remaining
        return keep

    def detect_candidates(self, frame: np.ndarray, preprocessed_gray: np.ndarray) -> tuple:
        """
        Finds all candidate license plate regions in the frame.

        Returns:
            candidates: List of candidate dictionaries sorted by detection score
            debug_images: dict of intermediate DIP visualization frames
        """
        frame_h, frame_w = frame.shape[:2]
        total_frame_area = frame_h * frame_w
        max_area = total_frame_area * self.max_area_ratio

        # -------------------------------------------------------------
        # 1. High-Precision YOLO License Plate Detector (Commercial Mode)
        # -------------------------------------------------------------
        if self.mode in ['auto', 'yolo'] and self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, verbose=False, conf=0.25)[0]
                yolo_candidates = []
                for box in results.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = map(int, coords)

                    # Add slight border padding so characters don't touch crop boundary
                    sw = x2 - x1
                    sh = y2 - y1
                    if sw <= 18 or sh <= 8:
                        continue

                    pad_w = max(2, int(sw * 0.05))
                    pad_h = max(2, int(sh * 0.08))
                    px1 = max(0, x1 - pad_w)
                    py1 = max(0, y1 - pad_h)
                    px2 = min(frame_w, x2 + pad_w)
                    py2 = min(frame_h, y2 + pad_h)

                    straight_box = (x1, y1, sw, sh)
                    box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    warped = frame[py1:py2, px1:px2]
                    if warped.size == 0:
                        continue
                    warped = cv2.resize(warped, (280, 84))
                    aspect_ratio = float(sw) / float(sh) if sh > 0 else 3.5

                    yolo_candidates.append({
                        'contour': box_pts,
                        'box': box_pts,
                        'straight_box': straight_box,
                        'warped': warped,
                        'aspect_ratio': aspect_ratio,
                        'edge_density': 0.45,
                        'strokes': 7,
                        'score': conf * 10.0,
                        'area': sw * sh
                    })

                if yolo_candidates or self.mode == 'yolo':
                    yolo_candidates = self.apply_nms(yolo_candidates)
                    grad_x = self.compute_sobel_gradient(preprocessed_gray)
                    closed = self.apply_morphological_closing(grad_x)
                    thresh = self.threshold_and_clean(closed)
                    return yolo_candidates, {
                        'grad_x': grad_x,
                        'closed': closed,
                        'thresh': thresh,
                        'contour_debug': frame.copy()
                    }
            except Exception as e:
                print(f"[Detector] YOLO exception, fallback to DIP: {e}")

        # -------------------------------------------------------------
        # 2. Classical Digital Image Processing (DIP) Detection Fallback
        # -------------------------------------------------------------
        # 1. Sobel-X Vertical Gradient
        grad_x = self.compute_sobel_gradient(preprocessed_gray)

        # 2. Lightness-Normalized Color Segmentation (White & Yellow Plates)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 115]), np.array([180, 85, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([12, 50, 90]), np.array([45, 255, 255]))
        plate_color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        plate_color_mask = cv2.dilate(plate_color_mask, dilate_kernel, iterations=1)
        grad_x_masked = cv2.bitwise_and(grad_x, grad_x, mask=plate_color_mask)

        # 3. Morphological Closing
        closed = self.apply_morphological_closing(grad_x_masked)

        # 4. Otsu Threshold & Clean
        thresh = self.threshold_and_clean(closed)

        # 5. Contour Filtering
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

        candidates = []
        contour_debug = frame.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > max_area:
                continue

            x, y, sw, sh = cv2.boundingRect(c)

            # Exclude sky / top 25% of frame
            if y < int(frame_h * 0.25):
                continue

            # Bounding box dimension bounds
            if sh < 12 or sh > 110 or sw < 35 or sw > 360:
                continue

            # Aspect ratio bounds
            aspect_ratio = float(sw) / float(sh)
            if not (self.min_ar <= aspect_ratio <= self.max_ar):
                continue

            # Minimum area rotated bounding box for deskewing
            min_rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = min_rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90.0
            while angle > 45: angle -= 90
            while angle < -45: angle += 90
            if abs(angle) > 30.0:
                continue

            # Edge density verification inside candidate region
            roi_grad = grad_x[y:y+sh, x:x+sw]
            if roi_grad.size == 0:
                continue
            edge_density = np.count_nonzero(roi_grad > 25) / float(sw * sh)
            if edge_density < 0.06:
                continue

            # Rectify perspective to get upright plate ROI
            box = cv2.boxPoints(min_rect)
            box = np.intp(box)
            try:
                warped = self.rectify_perspective(frame, box.astype("float32"), target_w=280, target_h=84)
            except Exception:
                warped = frame[y:y+sh, x:x+sw]
                if warped.size == 0:
                    continue
                warped = cv2.resize(warped, (280, 84))

            # Vertical Projection Stroke Verification
            w_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
            _, w_bin = cv2.threshold(w_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            if np.mean(w_bin[:3, :]) > 127:
                w_bin = cv2.bitwise_not(w_bin)

            col_sums = np.sum(w_bin > 0, axis=0)
            threshold_lvl = np.mean(col_sums) * 0.60
            above_thresh = col_sums > threshold_lvl
            transitions = np.count_nonzero(np.diff(above_thresh.astype(int)) != 0)
            estimated_strokes = transitions // 2

            if estimated_strokes < 3:
                continue

            # DIP Quality Score
            bbox_area = sw * sh
            solidity = area / float(bbox_area) if bbox_area > 0 else 0
            ar_diff = abs(aspect_ratio - 4.2)
            score = (edge_density * 4.5) + (solidity * 2.0) + (estimated_strokes * 0.4) - (ar_diff * 0.6)

            candidates.append({
                'contour': c,
                'box': box,
                'straight_box': (x, y, sw, sh),
                'warped': warped,
                'aspect_ratio': aspect_ratio,
                'edge_density': edge_density,
                'strokes': estimated_strokes,
                'score': score,
                'area': area
            })
            cv2.drawContours(contour_debug, [box], 0, (0, 255, 0), 2)

        candidates = self.apply_nms(candidates)
        candidates = candidates[:5]

        debug_images = {
            'grad_x': grad_x_masked,
            'closed': closed,
            'thresh': thresh,
            'contour_debug': contour_debug
        }
        return candidates, debug_images
