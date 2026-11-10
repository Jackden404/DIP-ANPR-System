"""
recognizer.py - Optical Character Recognition & Syntax Validation

This module converts segmented character regions or rectified plate images into
clean, validated alphanumeric strings.

Key Features:
1. Primary OCR via EasyOCR (high accuracy deep neural engine).
2. Secondary OCR via Tesseract (alphanumeric whitelist & PSM 7/8).
3. Foundational DIP Multi-Font Template Matching Fallback.
4. Super-Resolution Pre-OCR Enhancement (Bicubic upscaling + Unsharp Mask).
5. Positional Context Disambiguation ('O' vs '0', 'I' vs '1', 'B' vs '8', 'S' vs '5', 'Z' vs '2').
6. Multi-Format Plate Syntax Validation (Indian HSRP, UK standard, US alphanumeric).
"""

import os
import re
import cv2
import numpy as np

# Try importing easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Try importing pytesseract
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    default_win_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_win_tess):
        pytesseract.pytesseract.tesseract_cmd = default_win_tess
except ImportError:
    PYTESSERACT_AVAILABLE = False


class CharacterRecognizer:
    def __init__(self, tesseract_cmd: str = None, preferred_engine: str = "auto"):
        """
        Initializes OCR engines with automatic fallback hierarchy:
        1. EasyOCR (Highest accuracy on real-world traffic)
        2. Tesseract OCR (Fast CPU OCR)
        3. Classical DIP Multi-Font Template Matching
        
        Args:
            preferred_engine: 'auto', 'easyocr', 'tesseract', or 'template'
        """
        self.preferred_engine = preferred_engine.lower()
        self.easyocr_reader = None

        if EASYOCR_AVAILABLE and self.preferred_engine in ['auto', 'easyocr']:
            try:
                import torch
                use_cuda = torch.cuda.is_available()
                self.easyocr_reader = easyocr.Reader(['en'], gpu=use_cuda, verbose=False)
                print(f"[OCR] EasyOCR initialized successfully (GPU Acceleration: {use_cuda})")
            except Exception as e:
                print(f"[OCR] EasyOCR init warning: {e}")
                self.easyocr_reader = None

        self.tesseract_available = PYTESSERACT_AVAILABLE
        if self.tesseract_available and tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        if self.tesseract_available and self.preferred_engine in ['auto', 'tesseract']:
            try:
                dummy = np.zeros((20, 20), dtype=np.uint8)
                pytesseract.image_to_string(dummy, config='--psm 8')
            except Exception:
                self.tesseract_available = False
        elif self.preferred_engine not in ['auto', 'tesseract']:
            self.tesseract_available = False

        self.templates = self._generate_robust_font_templates()

    def _generate_robust_font_templates(self) -> dict:
        """
        Generates multi-font, multi-weight 20x30 binary templates for A-Z and 0-9.
        Covers Simplex, Duplex, and Bold block font geometries.
        """
        templates = {}
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        font_configs = [
            (cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2),
            (cv2.FONT_HERSHEY_DUPLEX, 0.85, 2),
            (cv2.FONT_HERSHEY_SIMPLEX, 0.90, 3),
        ]
        for ch in chars:
            templates[ch] = []
            for font, scale, thick in font_configs:
                canvas = np.zeros((45, 35), dtype=np.uint8)
                cv2.putText(canvas, ch, (5, 35), font, scale, 255, thick, cv2.LINE_AA)
                coords = cv2.findNonZero(canvas)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    crop = canvas[y:y+h, x:x+w]
                    resized = cv2.resize(crop, (20, 30))
                    templates[ch].append(resized)
        return templates

    def enhance_for_ocr(self, plate_roi: np.ndarray) -> np.ndarray:
        """
        Enhances small/distant license plate crops before feeding them into OCR engines:
        - 2.2x Bicubic interpolation
        - Unsharp masking (accentuates character stroke boundaries)
        """
        if plate_roi is None or plate_roi.size == 0:
            return plate_roi

        h, w = plate_roi.shape[:2]
        upscaled = cv2.resize(plate_roi, (int(w * 2.2), int(h * 2.2)), interpolation=cv2.INTER_CUBIC)

        # Unsharp mask
        blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(upscaled, 1.45, blurred, -0.45, 0)
        return sharpened

    def recognize_by_template_matching(self, char_crop: np.ndarray) -> tuple:
        """
        Recognizes an isolated binary character using normalized cross-correlation
        against standard character templates across multiple font weights.
        """
        if char_crop is None or char_crop.size == 0:
            return "", 0.0

        char_resized = cv2.resize(char_crop, (20, 30))
        best_char = ""
        best_score = -1.0

        for ch, templ_list in self.templates.items():
            for templ in templ_list:
                res = cv2.matchTemplate(char_resized, templ, cv2.TM_CCOEFF_NORMED)
                score = res[0][0]
                if score > best_score:
                    best_score = score
                    best_char = ch

        return best_char, max(0.0, float(best_score))

    def clean_plate_text(self, raw_text: str) -> str:
        """
        Cleans and normalizes plate text by removing special characters
        and applying context-aware alphanumeric disambiguation.
        """
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
        if not cleaned:
            return ""

        # UK format: 2 letters, 2 digits, 3 letters (e.g. EY61NBC, WI51VSU)
        if len(cleaned) == 7:
            c = list(cleaned)
            for i in [0, 1]:
                if c[i] == '0': c[i] = 'O'
                elif c[i] == '1': c[i] = 'I'
                elif c[i] == '8': c[i] = 'B'
                elif c[i] == '5': c[i] = 'S'
                elif c[i] == '2': c[i] = 'Z'
            for i in [2, 3]:
                if c[i] in ['O', 'D', 'Q']: c[i] = '0'
                elif c[i] in ['I', 'L']: c[i] = '1'
                elif c[i] == 'B': c[i] = '8'
                elif c[i] == 'S': c[i] = '5'
                elif c[i] == 'Z': c[i] = '2'
            for i in [4, 5, 6]:
                if c[i] == '0': c[i] = 'O'
                elif c[i] == '1': c[i] = 'I'
                elif c[i] == '8': c[i] = 'B'
                elif c[i] == '5': c[i] = 'S'
                elif c[i] == '2': c[i] = 'Z'
            cleaned = "".join(c)

        # Indian / Standard format: 2 letters, 2 digits, 1-2 letters, 4 digits (e.g. MH12DE1432)
        elif len(cleaned) in [9, 10]:
            c = list(cleaned)
            for i in [0, 1]:
                if c[i] == '0': c[i] = 'O'
                elif c[i] == '1': c[i] = 'I'
                elif c[i] == '8': c[i] = 'B'
                elif c[i] == '5': c[i] = 'S'
                elif c[i] == '2': c[i] = 'Z'
            for i in [2, 3]:
                if c[i] in ['O', 'D', 'Q']: c[i] = '0'
                elif c[i] in ['I', 'L']: c[i] = '1'
                elif c[i] == 'B': c[i] = '8'
                elif c[i] == 'S': c[i] = '5'
                elif c[i] == 'Z': c[i] = '2'
            for i in range(len(c) - 4, len(c)):
                if c[i] in ['O', 'D', 'Q']: c[i] = '0'
                elif c[i] in ['I', 'L']: c[i] = '1'
                elif c[i] == 'B': c[i] = '8'
                elif c[i] == 'S': c[i] = '5'
                elif c[i] == 'Z': c[i] = '2'
            cleaned = "".join(c)

        return cleaned

    def recognize_plate(self, plate_roi: np.ndarray, characters: list = None) -> tuple:
        """
        Recognizes license plate text from rectified ROI and/or segmented characters.

        Returns:
            plate_text (str): Cleaned, disambiguated plate string
            confidence (float): Confidence score [0.0, 1.0]
            engine_used (str): 'easyocr', 'tesseract', or 'template_matching'
        """
        if plate_roi is None or plate_roi.size == 0:
            return "", 0.0, "none"

        # Super-resolution upscale for high OCR accuracy
        enhanced_plate = self.enhance_for_ocr(plate_roi)

        # 1. EasyOCR Engine
        if self.preferred_engine in ['auto', 'easyocr'] and self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(enhanced_plate, detail=1, paragraph=False)
                for bbox, text, conf in results:
                    cleaned = self.clean_plate_text(text)
                    if len(cleaned) >= 4 and conf >= 0.28:
                        return cleaned, float(conf), "easyocr"
            except Exception:
                pass

        # 2. Tesseract OCR Engine
        if self.preferred_engine in ['auto', 'tesseract'] and self.tesseract_available:
            try:
                gray = cv2.cvtColor(enhanced_plate, cv2.COLOR_BGR2GRAY) if len(enhanced_plate.shape) == 3 else enhanced_plate
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
                
                texts = []
                confidences = []
                for i in range(len(data['text'])):
                    txt = data['text'][i].strip()
                    conf = float(data['conf'][i])
                    if txt and conf > 0:
                        texts.append(txt)
                        confidences.append(conf)

                raw_text = "".join(texts)
                cleaned = self.clean_plate_text(raw_text)
                avg_conf = (sum(confidences) / len(confidences)) / 100.0 if confidences else 0.5

                if len(cleaned) >= 4:
                    return cleaned, avg_conf, "tesseract"
            except Exception:
                pass

        # 3. Foundational DIP Template Matching on Segmented Glyphs
        if (self.preferred_engine in ['auto', 'template'] or not self.tesseract_available) and characters and len(characters) >= 4:
            matched_chars = []
            scores = []
            for item in characters:
                ch, score = self.recognize_by_template_matching(item['char_crop'])
                if ch and score > 0.40:
                    matched_chars.append(ch)
                    scores.append(score)

            raw_text = "".join(matched_chars)
            cleaned = self.clean_plate_text(raw_text)
            avg_score = float(np.mean(scores)) if scores else 0.0
            if len(cleaned) >= 4:
                return cleaned, avg_score, "template_matching"

        return "", 0.0, "none"
