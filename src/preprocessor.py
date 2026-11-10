"""
preprocessor.py - Digital Image Processing (DIP) Preprocessing Pipeline

This module handles initial enhancement, noise reduction, gamma normalization,
and contrast equalization of raw video/CCTV frames.

Key DIP Operations:
1. Grayscale Conversion (Luminance weighting: Y = 0.299R + 0.587G + 0.114B)
2. Adaptive Gamma Correction (Dynamic range normalization for night/glare)
3. Bilateral Filtering (Edge-preserving noise suppression)
4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Morphological Top-Hat & Black-Hat Transforms (Character stroke enhancement)
"""

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)):
        """
        Initializes the preprocessor with CLAHE parameters.
        
        Args:
            clip_limit (float): Threshold for contrast limiting. Prevents noise amplification.
            tile_grid_size (tuple): Grid size for dividing the image into localized contextual tiles.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts a 3-channel BGR frame into a single-channel grayscale image.
        Uses standard luminance weighting: Y = 0.114*B + 0.587*G + 0.299*R
        """
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def apply_auto_gamma(self, gray: np.ndarray) -> np.ndarray:
        """
        Applies non-linear adaptive gamma correction based on mean scene illumination.
        - Dark scenes (mean < 100): gamma < 1.0 brightens shadows without blowing highlights.
        - High-glare scenes (mean > 165): gamma > 1.0 compresses bright spots.
        """
        mean_intensity = np.mean(gray)
        if mean_intensity <= 5 or mean_intensity >= 250:
            return gray

        # Compute optimal gamma to pull mean intensity toward target of 128
        gamma = np.log(128.0 / 255.0) / np.log(mean_intensity / 255.0)
        gamma = float(np.clip(gamma, 0.65, 1.85))

        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(gray, lut)

    def apply_bilateral_filter(self, gray: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Applies Bilateral Filtering to remove sensor noise while preserving sharp character edges.
        Combines spatial domain distance with range domain photometric similarity.
        """
        return cv2.bilateralFilter(gray, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization.
        Solves uneven lighting, shadows, and headlight glare in CCTV traffic videos.
        """
        return self.clahe.apply(gray)

    def extract_tophat_blackhat(self, gray: np.ndarray, kernel_size: tuple = (13, 5)) -> tuple:
        """
        Computes Top-Hat and Black-Hat morphological operations.
        - Top-Hat: Image - Opening(Image). Isolates bright features on dark background.
        - Black-Hat: Closing(Image) - Image. Isolates dark glyphs on bright plate background.
        """
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        return tophat, blackhat

    def process(self, frame: np.ndarray, use_clahe: bool = True) -> dict:
        """
        Runs the full preprocessing pipeline on a single frame.

        Returns:
            dict containing intermediate images for visual inspection:
            - 'original': raw BGR frame
            - 'gray': grayscale image
            - 'gamma': illumination normalized image
            - 'denoised': bilateral filtered image
            - 'enhanced': CLAHE contrast enhanced image
            - 'light_enhanced': combined top-hat / black-hat feature frame
        """
        gray = self.to_grayscale(frame)
        gamma_corrected = self.apply_auto_gamma(gray)
        denoised = self.apply_bilateral_filter(gamma_corrected)
        
        if use_clahe:
            enhanced = self.apply_clahe(denoised)
        else:
            enhanced = denoised.copy()

        # Enhance plate character contrast using morphological Top-Hat & Black-Hat
        tophat, blackhat = self.extract_tophat_blackhat(enhanced)
        # Add tophat (bright regions), subtract blackhat (dark regions)
        light_enhanced = cv2.add(enhanced, tophat)
        light_enhanced = cv2.subtract(light_enhanced, blackhat)

        return {
            'original': frame,
            'gray': gray,
            'gamma': gamma_corrected,
            'denoised': denoised,
            'enhanced': enhanced,
            'light_enhanced': light_enhanced
        }
