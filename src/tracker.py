"""
tracker.py - Temporal Stabilization & Traffic Event Logger

This module provides multi-frame temporal voting to stabilize OCR readings
from continuous video/CCTV streams and logs recognized plates into a persistent CSV.

Key Features:
1. Positional Character Majority Voting (eliminates single-character OCR flicker).
2. Sliding window buffer to stabilize continuous video frames.
3. Duplicate event suppression with configurable time cooldown.
4. Automated snapshot archival of cropped license plate ROIs.
5. Persistent CSV audit logging with timestamps and confidence metrics.
"""

import os
import csv
import time
from collections import deque, Counter
import cv2
import numpy as np


class PlateTracker:
    def __init__(self, 
                 buffer_size: int = 10, 
                 min_detections: int = 3, 
                 cooldown_seconds: float = 5.0,
                 output_dir: str = "data/output_logs"):
        """
        Args:
            buffer_size: Number of recent frames to retain in the sliding window.
            min_detections: Minimum frequency required in buffer to confirm plate reading.
            cooldown_seconds: Cooldown time before logging the same plate number again.
            output_dir: Directory where CSV logs and snapshot crops are stored.
        """
        self.buffer_size = buffer_size
        self.min_detections = min_detections
        self.cooldown_seconds = cooldown_seconds
        self.output_dir = output_dir

        self.history = deque(maxlen=buffer_size)
        self.logged_plates = {}  # {plate_text: last_logged_timestamp}
        self.current_confirmed_plate = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "entry_logs.csv")
        self._init_csv()

    def _init_csv(self):
        """Creates the CSV file with headers if it does not already exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Plate_Number", "Confidence", "Snapshot_File", "Engine"])

    def _positional_majority_voting(self, readings: list) -> str:
        """
        Performs character-by-character majority voting across readings of matching length.
        Eliminates single-frame OCR flicker and optical jitter.
        """
        if not readings:
            return ""

        lengths = [len(r) for r in readings]
        most_common_len, len_count = Counter(lengths).most_common(1)[0]
        # Require that at least min_detections share the same string length
        if len_count < self.min_detections:
            return ""

        subset = [r for r in readings if len(r) == most_common_len]

        voted_chars = []
        for i in range(most_common_len):
            col_chars = [r[i] for r in subset]
            most_common_char, _ = Counter(col_chars).most_common(1)[0]
            voted_chars.append(most_common_char)

        voted_str = "".join(voted_chars)

        # Consensus Gate: Require at least min_detections to closely match the voted string (>=70% chars)
        close_matches = 0
        for r in subset:
            matches = sum(1 for c1, c2 in zip(voted_str, r) if c1 == c2)
            if matches >= max(3, int(most_common_len * 0.70)):
                close_matches += 1

        if close_matches >= self.min_detections:
            return voted_str
        return ""

    def update(self, detected_plate: str, confidence: float, plate_roi: np.ndarray, engine: str = "tesseract") -> tuple:
        """
        Updates the temporal voting buffer with the current frame's reading.

        Returns:
            confirmed_text: The stabilized license plate string (or None if not yet stabilized)
            is_newly_logged: Boolean indicating if this was logged as a new event
        """
        current_time = time.time()
        is_newly_logged = False

        if detected_plate and len(detected_plate) >= 4:
            self.history.append((detected_plate, confidence, plate_roi, engine))
        else:
            self.history.append(("", 0.0, None, "none"))

        valid_entries = [entry for entry in self.history if entry[0]]
        valid_readings = [entry[0] for entry in valid_entries]

        if len(valid_readings) >= self.min_detections:
            # Perform character-level positional majority voting
            voted_plate = self._positional_majority_voting(valid_readings)

            if voted_plate:
                self.current_confirmed_plate = voted_plate

                # Check cooldown to prevent duplicate logging
                last_time = self.logged_plates.get(voted_plate, 0)
                if current_time - last_time > self.cooldown_seconds:
                    self.logged_plates[voted_plate] = current_time
                    is_newly_logged = True

                    # Find best representative ROI and average confidence
                    matching_entries = [e for e in valid_entries if e[0] == voted_plate]
                    if not matching_entries:
                        matching_entries = valid_entries
                    best_entry = max(matching_entries, key=lambda e: e[1])
                    avg_conf = float(np.mean([e[1] for e in matching_entries]))

                    self._log_event(voted_plate, avg_conf, best_entry[2], best_entry[3])

        return self.current_confirmed_plate, is_newly_logged

    def _log_event(self, plate_text: str, confidence: float, plate_roi: np.ndarray, engine: str):
        """Saves a snapshot of the plate ROI and records an entry in the CSV file."""
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"plate_{plate_text}_{file_timestamp}.jpg"
        snapshot_path = os.path.join(self.output_dir, snapshot_filename)

        if plate_roi is not None and plate_roi.size > 0:
            cv2.imwrite(snapshot_path, plate_roi)
        else:
            snapshot_filename = "N/A"

        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str,
                    plate_text,
                    f"{confidence:.2f}",
                    snapshot_filename,
                    engine
                ])
        except Exception as e:
            print(f"[Warning] Failed to write to log CSV: {e}")
