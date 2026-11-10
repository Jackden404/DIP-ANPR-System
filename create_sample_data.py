"""
create_sample_data.py - High-Definition (HD) Synthetic Traffic Video & Test Asset Generator

Generates realistic 1280x720 HD traffic video footage with accurate vehicle geometry,
metallic shading, headlights, road perspective, and a crisp license plate.
"""

import os
import cv2
import numpy as np


def draw_crisp_license_plate(plate_text: str = "MH12DE1432", width: int = 320, height: int = 90) -> np.ndarray:
    """
    Renders an ultra-crisp, high-definition vehicle license plate with realistic HSRP styling.
    """
    plate = np.ones((height, width, 3), dtype=np.uint8) * 248
    
    # Outer black embossed border
    cv2.rectangle(plate, (2, 2), (width - 3, height - 3), (15, 15, 15), 4)
    cv2.rectangle(plate, (5, 5), (width - 6, height - 6), (220, 220, 220), 1)

    # Blue Country / IND Identification Band (Left strip)
    strip_w = int(width * 0.13)
    cv2.rectangle(plate, (2, 2), (strip_w, height - 3), (170, 40, 10), -1)
    
    # Yellow Chakra / Ashoka emblem circle
    cv2.circle(plate, (strip_w // 2, int(height * 0.35)), 7, (0, 200, 230), -1)
    cv2.putText(plate, "IND", (strip_w // 2 - 14, int(height * 0.75)), 
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # High-Contrast Alphanumeric License Plate Typography
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.35
    thickness = 3
    (tw, th), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    start_x = strip_w + (width - strip_w - tw) // 2
    start_y = (height + th) // 2

    # Subtle drop shadow for 3D embossed look
    cv2.putText(plate, plate_text, (start_x + 1, start_y + 1), font, font_scale, (100, 100, 100), thickness, cv2.LINE_AA)
    # Main crisp dark lettering
    cv2.putText(plate, plate_text, (start_x, start_y), font, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)

    # Mounting bolt/screw fixtures
    cv2.circle(plate, (strip_w + 14, height // 2), 4, (90, 90, 90), -1)
    cv2.circle(plate, (strip_w + 14, height // 2), 2, (180, 180, 180), -1)
    cv2.circle(plate, (width - 16, height // 2), 4, (90, 90, 90), -1)
    cv2.circle(plate, (width - 16, height // 2), 2, (180, 180, 180), -1)

    return plate


def render_hd_traffic_frame(w: int, h: int, progress: float, plate_img: np.ndarray) -> np.ndarray:
    """
    Renders a single 1280x720 HD traffic camera frame with perspective road and approaching vehicle.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Sky and Horizon
    horizon_y = int(h * 0.30)
    # Sky gradient
    for y in range(horizon_y):
        ratio = y / float(horizon_y)
        frame[y, :] = (int(140 - 40 * ratio), int(120 - 30 * ratio), int(100 - 20 * ratio))

    # Distant roadside trees/buildings
    cv2.rectangle(frame, (0, horizon_y - 25), (w, horizon_y), (45, 55, 45), -1)

    # 2. Road Surface (Asphalt with textured noise)
    road_gray = 48
    frame[horizon_y:h, :] = (road_gray, road_gray, road_gray)

    # Left & Right Road Shoulders / Curbs
    road_top_left = (int(w * 0.36), horizon_y)
    road_top_right = (int(w * 0.64), horizon_y)
    road_bottom_left = (-100, h)
    road_bottom_right = (w + 100, h)

    # Left Shoulder Grass
    grass_pts_l = np.array([[0, horizon_y], road_top_left, road_bottom_left, [0, h]], np.int32)
    cv2.fillPoly(frame, [grass_pts_l], (35, 65, 35))

    # Right Shoulder Grass
    grass_pts_r = np.array([road_top_right, [w, horizon_y], [w, h], road_bottom_right], np.int32)
    cv2.fillPoly(frame, [grass_pts_r], (35, 65, 35))

    # Solid White Edge Lane Markings
    cv2.line(frame, road_top_left, road_bottom_left, (210, 210, 210), 5, cv2.LINE_AA)
    cv2.line(frame, road_top_right, road_bottom_right, (210, 210, 210), 5, cv2.LINE_AA)

    # Dashed Center Lane Marking
    center_top = (w // 2, horizon_y)
    center_bot = (w // 2, h)
    dash_count = 14
    for d in range(dash_count):
        t1 = d / float(dash_count)
        t2 = (d + 0.55) / float(dash_count)
        y1 = int(horizon_y + t1 * (h - horizon_y))
        y2 = int(horizon_y + t2 * (h - horizon_y))
        lw = max(2, int(1 + 7 * t2))
        cv2.line(frame, (w // 2, y1), (w // 2, y2), (230, 230, 230), lw, cv2.LINE_AA)

    # 3. Vehicle Animation (Perspective scaling as vehicle moves toward CCTV camera)
    scale = 0.40 + 0.65 * progress
    car_w = int(460 * scale)
    car_h = int(280 * scale)

    # Vehicle trajectory down the lane
    car_cx = int(w * 0.48 + np.sin(progress * 2.5) * 35)
    car_cy = int(horizon_y + 40 + progress * (h - horizon_y - 120))

    car_x1 = max(0, car_cx - car_w // 2)
    car_y1 = max(0, car_cy - car_h // 2)
    car_x2 = min(w, car_x1 + car_w)
    car_y2 = min(h, car_y1 + car_h)

    # Car Ground Shadow
    shadow_w = int(car_w * 1.08)
    shadow_h = int(car_h * 0.22)
    cv2.ellipse(frame, (car_cx, car_y2), (shadow_w // 2, shadow_h // 2), 0, 0, 360, (20, 20, 20), -1)

    # Car Windshield & Cabin Roof
    cabin_w = int(car_w * 0.78)
    cabin_h = int(car_h * 0.45)
    cabin_top = car_y1
    cabin_pts = np.array([
        [car_cx - cabin_w // 2 + int(cabin_w * 0.12), cabin_top],
        [car_cx + cabin_w // 2 - int(cabin_w * 0.12), cabin_top],
        [car_cx + cabin_w // 2, cabin_top + cabin_h],
        [car_cx - cabin_w // 2, cabin_top + cabin_h]
    ], np.int32)
    cv2.fillPoly(frame, [cabin_pts], (30, 30, 35))
    # Windshield tint & reflection line
    cv2.line(frame, (car_cx - int(cabin_w * 0.35), cabin_top + int(cabin_h * 0.3)),
                    (car_cx + int(cabin_w * 0.25), cabin_top + int(cabin_h * 0.8)), (80, 100, 110), 2, cv2.LINE_AA)

    # Car Hood / Lower Body (Deep Metallic Navy Blue)
    hood_top = cabin_top + int(cabin_h * 0.85)
    hood_pts = np.array([
        [car_x1 + int(car_w * 0.05), hood_top],
        [car_x2 - int(car_w * 0.05), hood_top],
        [car_x2, car_y2 - int(car_h * 0.25)],
        [car_x1, car_y2 - int(car_h * 0.25)]
    ], np.int32)
    cv2.fillPoly(frame, [hood_pts], (110, 50, 25))

    # Front Bumper
    bumper_h = int(car_h * 0.30)
    bumper_y1 = car_y2 - bumper_h
    cv2.rectangle(frame, (car_x1, bumper_y1), (car_x2, car_y2), (40, 40, 45), -1)
    cv2.rectangle(frame, (car_x1, bumper_y1), (car_x2, car_y2), (80, 80, 85), 2)

    # Front Radiator Grille (Textured dark mesh)
    grille_w = int(car_w * 0.44)
    grille_h = int(bumper_h * 0.45)
    grille_x = car_cx - grille_w // 2
    grille_y = bumper_y1 + 4
    cv2.rectangle(frame, (grille_x, grille_y), (grille_x + grille_w, grille_y + grille_h), (20, 20, 20), -1)
    # Grille horizontal chrome slats
    for gy in range(grille_y + 4, grille_y + grille_h, 5):
        cv2.line(frame, (grille_x + 3, gy), (grille_x + grille_w - 3, gy), (140, 140, 140), 1)

    # Modern Headlights (Dual angled projectors with LED DRL glare)
    hl_w = int(car_w * 0.17)
    hl_h = int(bumper_h * 0.42)
    # Left Headlight
    cv2.ellipse(frame, (car_x1 + int(hl_w * 0.85), bumper_y1 + hl_h // 2), (hl_w // 2, hl_h // 2), -5, 0, 360, (240, 255, 255), -1)
    cv2.circle(frame, (car_x1 + int(hl_w * 0.85), bumper_y1 + hl_h // 2), int(hl_h * 0.25), (200, 240, 255), -1)
    # Right Headlight
    cv2.ellipse(frame, (car_x2 - int(hl_w * 0.85), bumper_y1 + hl_h // 2), (hl_w // 2, hl_h // 2), 5, 0, 360, (240, 255, 255), -1)
    cv2.circle(frame, (car_x2 - int(hl_w * 0.85), bumper_y1 + hl_h // 2), int(hl_h * 0.25), (200, 240, 255), -1)

    # License Plate Attachment (Centered on the bumper)
    pl_w = int(185 * scale)
    pl_h = int(55 * scale)
    pl_x = car_cx - pl_w // 2
    pl_y = car_y2 - int(bumper_h * 0.65)

    if pl_w > 20 and pl_h > 10 and pl_x > 0 and pl_y > 0 and pl_x + pl_w < w and pl_y + pl_h < h:
        resized_plate = cv2.resize(plate_img, (pl_w, pl_h), interpolation=cv2.INTER_CUBIC)
        frame[pl_y:pl_y+pl_h, pl_x:pl_x+pl_w] = resized_plate
        # Thin mounting bracket shadow
        cv2.rectangle(frame, (pl_x - 2, pl_y - 2), (pl_x + pl_w + 2, pl_y + pl_h + 2), (10, 10, 10), 1)

    # CCTV Telemetry Banner Overlay
    cctv_time = f"CCTV HIGHWAY CAM-02 | 2026-10-20 18:35:12.{(int(progress * 100) % 99):02d}"
    cv2.putText(frame, cctv_time, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)

    return frame


def generate_sample_traffic_video(output_path: str = "data/sample_videos/traffic_demo.mp4", 
                                  num_frames: int = 120, 
                                  fps: int = 25):
    """
    Generates a 1280x720 HD realistic traffic camera video file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_w, frame_h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    plate_img = draw_crisp_license_plate("MH12DE1432", 340, 95)

    print(f"[Generator] Creating 1280x720 HD Traffic Video: {output_path} ({num_frames} frames)...")

    for i in range(num_frames):
        progress = i / float(num_frames)
        frame = render_hd_traffic_frame(frame_w, frame_h, progress, plate_img)
        writer.write(frame)

    writer.release()
    print(f"[Generator] HD Video successfully saved to: {output_path}")

    # Also save a high-res sample image for static testing
    sample_img_path = "data/sample_videos/sample_car.jpg"
    final_frame = render_hd_traffic_frame(frame_w, frame_h, 0.85, plate_img)
    cv2.imwrite(sample_img_path, final_frame)
    print(f"[Generator] HD Sample Image saved to: {sample_img_path}")


if __name__ == "__main__":
    generate_sample_traffic_video()
