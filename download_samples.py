"""
download_samples.py - Real-World Traffic Test Footage & Image Downloader

Downloads real vehicle and license plate test images from open-source
benchmarks (OpenALPR benchmark suite) for realistic real-world evaluation.
"""

import os
import urllib.request

SAMPLES = {
    # High-quality real vehicle photos with clear license plates
    "us_car_1.jpg": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/us/us1.jpg",
    "us_car_2.jpg": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/us/us2.jpg",
    "us_car_3.jpg": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/us/us3.jpg",
    "eu_car_1.jpg": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/eu/eu1.jpg",
    "eu_car_2.jpg": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/eu/eu2.jpg",
}


def download_all_samples(target_dir: str = "data/sample_videos"):
    os.makedirs(target_dir, exist_ok=True)
    print("=" * 60)
    print(" Downloading Real-World License Plate Test Samples")
    print("=" * 60)

    for filename, url in SAMPLES.items():
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path):
            print(f"[Exists] {filename}")
            continue

        try:
            print(f"[Downloading] {filename} from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as f:
                f.write(resp.read())
            print(f"[Success] Saved to {dest_path}")
        except Exception as e:
            print(f"[Skip] Could not download {filename}: {e}")

    print("\nDownload finished. You can run:")
    print("  python main.py --source data/sample_videos/us_car_1.jpg")


if __name__ == "__main__":
    download_all_samples()
