"""
segmenter.py - Digital Image Processing Character Segmentation

This module isolates individual alphanumeric character glyphs from a cropped,
rectified license plate region.

Key DIP Operations:
1. Local CLAHE Contrast Normalization
2. Adaptive Gaussian Binarization with Vertical Disconnection Kernel (1x3)
3. Dual-Polarity Evaluation (Self-selecting White-on-Black vs Black-on-White)
4. Border Frame & Mounting Screw Margin Clearance
5. Connected Component Analysis with Geometric Prior Filtering:
   - Height Ratio, Width Ratio, Aspect Ratio, Solidity, Area Bounds
   - Horizontal Baseline & Vertical Centering Alignment Verification
6. Non-Maximum Overlap Character Suppression & Left-to-Right Sorting
"""

import cv2
import numpy as np


class CharacterSegmenter:
    def __init__(self, 
                 min_char_h_ratio: float = 0.22, 
                 max_char_h_ratio: float = 0.88,
                 min_char_w_ratio: float = 0.02, 
                 max_char_w_ratio: float = 0.30):
        """
        Args:
            min_char_h_ratio: Minimum character height relative to plate height.
            max_char_h_ratio: Maximum character height relative to plate height.
            min_char_w_ratio: Minimum character width relative to plate width.
            max_char_w_ratio: Maximum character width relative to plate width.
        """
        self.min_h_ratio = min_char_h_ratio
        self.max_h_ratio = max_char_h_ratio
        self.min_w_ratio = min_char_w_ratio
        self.max_w_ratio = max_char_w_ratio

    def binarize_plate(self, plate_roi: np.ndarray) -> np.ndarray:
        """
        Binarizes the plate and returns the optimal polarity mask where characters
        are white (foreground) and plate background is black.
        """
        if plate_roi is None or plate_roi.size == 0:
            return np.zeros((84, 280), dtype=np.uint8)

        if len(plate_roi.shape) == 3:
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_roi.copy()

        h, w = gray.shape

        # CLAHE local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 8)

        # Vertical opening kernel to disconnect horizontal baseline smudges/bridges between adjacent letters
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, v_kernel)

        # Margin clearance (zero out 4% borders to remove frame and screw artifacts)
        margin_y = max(2, int(h * 0.04))
        margin_x = max(2, int(w * 0.03))

        b1 = opened.copy()
        b1[:margin_y, :] = 0
        b1[h - margin_y:, :] = 0
        b1[:, :margin_x] = 0
        b1[:, w - margin_x:] = 0

        b2 = cv2.bitwise_not(b1)
        b2[:margin_y, :] = 0
        b2[h - margin_y:, :] = 0
        b2[:, :margin_x] = 0
        b2[:, w - margin_x:] = 0

        # Evaluate candidate counts for both polarities
        c1 = self._extract_candidate_boxes(b1, h, w)
        c2 = self._extract_candidate_boxes(b2, h, w)

        return b2 if len(c2) > len(c1) else b1

    def _extract_candidate_boxes(self, bin_img: np.ndarray, plate_h: int, plate_w: int) -> list:
        """Finds valid character component bounding boxes within a binary mask."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        candidates = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            hr = h / float(plate_h)
            wr = w / float(plate_w)
            ar = w / float(h) if h > 0 else 0
            sol = area / float(w * h) if (w * h) > 0 else 0

            if (self.min_h_ratio <= hr <= self.max_h_ratio and
                self.min_w_ratio <= wr <= self.max_w_ratio and
                0.12 <= ar <= 1.35 and
                sol >= 0.16 and
                area >= 20):

                # Ensure component is reasonably vertically centered
                mid_y = y + h / 2.0
                if abs(mid_y - plate_h / 2.0) < plate_h * 0.40:
                    candidates.append((x, y, w, h))

        return sorted(candidates, key=lambda b: b[0])

    def segment_characters(self, plate_roi: np.ndarray) -> tuple:
        """
        Extracts segmented character bounding boxes from the rectified plate ROI.

        Returns:
            characters: List of dicts with 'char_crop', 'bbox', and 'index'
            binary: Binarized plate image
            annotated: Plate ROI with color bounding boxes and character index badges
        """
        binary = self.binarize_plate(plate_roi)
        plate_h, plate_w = binary.shape
        annotated = plate_roi.copy() if len(plate_roi.shape) == 3 else cv2.cvtColor(plate_roi, cv2.COLOR_GRAY2BGR)

        candidates = self._extract_candidate_boxes(binary, plate_h, plate_w)

        # Filter overlapping or nested components
        filtered = []
        for box in candidates:
            if not filtered:
                filtered.append(box)
                continue
            prev_x, prev_y, prev_w, prev_h = filtered[-1]
            if box[0] < prev_x + prev_w * 0.50:
                if box[2] * box[3] > prev_w * prev_h:
                    filtered[-1] = box
            else:
                filtered.append(box)

        characters = []
        for idx, (x, y, w, h) in enumerate(filtered):
            char_crop = binary[y:y+h, x:x+w]
            characters.append({
                'char_crop': char_crop,
                'bbox': (x, y, w, h),
                'index': idx
            })
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, str(idx + 1), (x, max(12, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        return characters, binary, annotated
