"""
ui_visualizer.py - HUD Overlay & Interactive DIP Pipeline Inspector

This module creates the visual interface:
1. Live Video Feed with styled HUD:
   - Target bounding boxes with corner accents
   - Real-time plate tags with confidence percentages
   - High-visibility Confirmed Vehicle Banner
   - Top Telemetry Bar (FPS, Detector Mode, CLAHE status)
   - Bottom Interactive Controls Legend
2. Multi-Panel "DIP Pipeline Inspector" (2x3 composite dashboard):
   - Panel 1: Illumination & CLAHE Enhancement
   - Panel 2: Sobel-X Vertical Gradient Filtering
   - Panel 3: Morphological Closing (Character Stroke Fusion)
   - Panel 4: Otsu Binarization & Cleaned Plate Mask
   - Panel 5: Perspective-Rectified Plate ROI (Homography)
   - Panel 6: Segmented Character Glyphs with Index Tags & OCR Result
"""

import cv2
import numpy as np


class UIVisualizer:
    def __init__(self, window_name: str = "DIP Number Plate Recognition System"):
        self.window_name = window_name

    def draw_hud(self, 
                 frame: np.ndarray, 
                 candidates: list, 
                 confirmed_plate: str, 
                 fps: float, 
                 use_clahe: bool,
                 inspector_mode: bool,
                 detector_mode: str = "auto") -> np.ndarray:
        """
        Draws bounding boxes, heads-up display banners, and telemetry info on the raw video frame.
        """
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw verified candidate bounding boxes
        for cand in candidates:
            plate_text = cand.get('text', '')
            conf = cand.get('conf', 0.0)

            if not plate_text or len(plate_text) < 4:
                continue

            box = cand['box']
            # Neon-green bounding box
            cv2.polylines(display, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Corner accents
            for pt in box:
                cv2.circle(display, tuple(pt), 4, (0, 220, 255), -1)

            # Draw tag banner above the plate
            min_x = int(np.min(box[:, 0]))
            min_y = int(np.min(box[:, 1]))
            banner_y = max(30, min_y - 10)
            
            tag_text = f"PLATE: {plate_text} ({int(conf * 100)}%)" if conf > 0 else f"PLATE: {plate_text}"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            cv2.rectangle(display, (min_x, banner_y - th - 6), (min_x + tw + 10, banner_y + 4), (0, 160, 0), -1)
            cv2.putText(display, tag_text, (min_x + 5, banner_y), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        # Confirmed Plate Banner (if temporally confirmed)
        if confirmed_plate:
            conf_str = f"CONFIRMED VEHICLE: {confirmed_plate}"
            (cw, ch), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)
            cv2.rectangle(display, (15, 68), (15 + cw + 16, 68 + ch + 14), (0, 130, 255), -1)
            cv2.rectangle(display, (15, 68), (15 + cw + 16, 68 + ch + 14), (255, 255, 255), 1)
            cv2.putText(display, conf_str, (23, 68 + ch + 6), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2)

        # Top Bar HUD Overlay (Semi-transparent black banner)
        top_bar = display[0:55, 0:w].copy()
        cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.addWeighted(display[0:55, 0:w], 0.75, top_bar, 0.25, 0, display[0:55, 0:w])

        # HUD Titles
        title_text = "DIP ANPR System | Video & CCTV"
        cv2.putText(display, title_text, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 220), 2)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display, fps_text, (w - 140, 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)

        # Status indicators (CLAHE, Detector Mode)
        mode_text = f"DETECTOR: {detector_mode.upper()}"
        cv2.putText(display, mode_text, (w - 530, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 2)

        clahe_status = "CLAHE: ON" if use_clahe else "CLAHE: OFF"
        clahe_color = (0, 255, 0) if use_clahe else (100, 100, 255)
        cv2.putText(display, clahe_status, (w - 320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clahe_color, 2)

        # Bottom Controls Legend Bar
        bottom_y = h - 35
        bottom_bar = display[bottom_y:h, 0:w].copy()
        cv2.rectangle(display, (0, bottom_y), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(display[bottom_y:h, 0:w], 0.8, bottom_bar, 0.2, 0, display[bottom_y:h, 0:w])

        controls_text = "[D] Inspector | [C] CLAHE | [M] Detector Mode | [P] Pause | [S] Snapshot | [Q] Exit"
        cv2.putText(display, controls_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        return display

    def build_dip_inspector(self, 
                            enhanced_gray: np.ndarray,
                            grad_x: np.ndarray,
                            closed_mask: np.ndarray,
                            thresh_mask: np.ndarray,
                            plate_roi: np.ndarray,
                            char_mask: np.ndarray,
                            plate_text: str) -> np.ndarray:
        """
        Builds a 2x3 composite dashboard displaying the 6 foundational DIP pipeline stages.
        """
        panel_w, panel_h = 360, 220

        def prepare_panel(img: np.ndarray, label: str) -> np.ndarray:
            """Converts image to 3-channel, resizes, and attaches a label header."""
            if img is None or img.size == 0:
                p = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(p, "No ROI Detected", (50, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            else:
                if len(img.shape) == 2:
                    p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    p = img.copy()
                p = cv2.resize(p, (panel_w, panel_h))

            # Header banner for the panel
            cv2.rectangle(p, (0, 0), (panel_w, 30), (30, 30, 30), -1)
            cv2.putText(p, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
            # Border around panel
            cv2.rectangle(p, (0, 0), (panel_w - 1, panel_h - 1), (70, 70, 70), 1)
            return p

        # Row 1 panels
        p1 = prepare_panel(enhanced_gray, "1. Contrast & Gamma Enhanced")
        p2 = prepare_panel(grad_x, "2. Sobel-X Vertical Gradient")
        p3 = prepare_panel(closed_mask, "3. Morphological Closing (21x5)")

        # Row 2 panels
        p4 = prepare_panel(thresh_mask, "4. Otsu Binarization & Cleaned")
        p5 = prepare_panel(plate_roi, "5. Rectified Plate ROI (Homography)")
        
        # Panel 6: Segmented characters + result text
        read_label = f"6. Characters | Read: {plate_text if plate_text else '...'}"
        p6 = prepare_panel(char_mask, read_label)

        # Assemble into 2x3 grid
        row1 = np.hstack([p1, p2, p3])
        row2 = np.hstack([p4, p5, p6])
        grid = np.vstack([row1, row2])

        return grid
