"""
siyi_capture_calibration.py

Takes a series of photos via the SIYI SDK (CMD_ID 0x0C) for checkerboard
camera calibration, without needing a physical transmitter or RC controller.

Workflow:
  1. Press ENTER to capture a photo (gimbal saves it to SD card)
  2. Gimbal confirms success via CMD_ID 0x0B Function Feedback
  3. After all captures, downloads every image from the SD card over HTTP
     to a local folder ready for OpenCV calibration

Prerequisites:
  - SD card inserted into the ZR10 (required — photos cannot be taken without it)
  - Gimbal reachable at 192.168.144.25 (ping first to confirm)
  - Checkerboard pattern printed and flat

Manual reference:
  - CMD_ID 0x0C (Photo and Record): func_type=0 takes a picture, no direct ACK
  - CMD_ID 0x0B (Function Feedback): info_type=0 success, 1=fail (no SD card)
  - Web server API /api/v1/getmedialist: lists photos stored on SD card
  - Hex example from manual: 556601010000000c0034ce  (Take a Picture)
"""

import socket
import struct
import time
import os
import requests

GIMBAL_IP   = "192.168.144.25"
GIMBAL_PORT = 37260
HTTP_BASE   = f"http://{GIMBAL_IP}"

# ── Verified against manual example table (page 63) ───────────────────────────
TAKE_PHOTO_PACKET = bytes.fromhex("556601010000000c0034ce")


def crc16_cal(data: bytes, crc_init: int = 0) -> int:
    """CRC16/CCITT — matches manual's verified example packets."""
    poly = 0x1021
    crc  = crc_init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def send_take_photo(sock: socket.socket) -> None:
    """Send CMD_ID 0x0C with func_type=0 (Take a Picture)."""
    sock.sendto(TAKE_PHOTO_PACKET, (GIMBAL_IP, GIMBAL_PORT))


def read_function_feedback(sock: socket.socket, timeout: float = 3.0) -> dict | None:
    """
    Listen for CMD_ID 0x0B (Function Feedback) ACK after a photo command.

    info_type values (from manual):
      0 = Success
      1 = Fail to take photo (check SD card)
      2 = HDR ON
      3 = HDR OFF
      4 = Fail to record video (check SD card)
    """
    sock.settimeout(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data, _ = sock.recvfrom(64)
        except socket.timeout:
            break

        if len(data) < 10:
            continue

        cmd_id = data[7]
        if cmd_id != 0x0B:
            continue                        # different message — keep waiting

        data_len  = struct.unpack_from("<H", data, 3)[0]
        info_type = data[8] if data_len >= 1 else -1
        return {"cmd_id": cmd_id, "info_type": info_type}

    return None                             # timeout — no feedback received


def list_photos_on_sdcard() -> list[dict]:
    """
    Use the gimbal's HTTP web server API to list all JPG images on the SD card.
    Returns a list of {"name": "...", "url": "..."} dicts.
    """
    url = f"{HTTP_BASE}/api/v1/getmedialist"
    payload = {"media_type": 0, "path": "", "start": 0, "count": 9999}
    try:
        resp = requests.get(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data.get("data", {}).get("list", [])
    except Exception as e:
        print(f"  [HTTP] Could not list media: {e}")
    return []


def download_photo(url: str, dest_path: str) -> bool:
    """Download a single photo from the gimbal's HTTP server."""
    try:
        resp = requests.get(url, timeout=10, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  [HTTP] Download failed for {url}: {e}")
        return False


def capture_session(
    output_dir: str = "calibration_images",
    min_photos:  int = 20,  
):
    """
    Interactive capture loop.

    Press ENTER to take a photo, 'q' + ENTER to quit early.
    Recommended: aim for 20-30 photos from diverse angles for a good calibration.
    """
    os.makedirs(output_dir, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(3.0)

    print("=" * 60)
    print("SIYI ZR10 — Checkerboard Calibration Photo Capture")
    print("=" * 60)
    print(f"  Gimbal IP  : {GIMBAL_IP}:{GIMBAL_PORT}")
    print(f"  Save folder: {output_dir}/")
    print(f"  Target     : {min_photos}+ photos recommended")
    print()
    print("TIPS for good calibration:")
    print("  • Hold the checkerboard at many different angles and distances")
    print("  • Cover all corners of the camera frame across shots")
    print("  • Keep the board fully visible — no cropping at edges")
    print("  • Avoid motion blur — hold still when capturing")
    print()
    print("Controls:")
    print("  ENTER → take photo")
    print("  q     → quit and download all photos")
    print("=" * 60)

    # Confirm SD card is present before starting
    print("\nChecking SD card (attempting a test capture)...")
    send_take_photo(sock)
    fb = read_function_feedback(sock)
    if fb is None:
        print("  WARNING: No feedback received — confirm gimbal is powered and")
        print("  reachable at 192.168.144.25 (run: ping 192.168.144.25)")
        sock.close()
        return
    if fb["info_type"] == 1:
        print("  ERROR: Gimbal reports 'Fail to take photo — SD card not found'.")
        print("  Insert an SD card and restart the script.")
        sock.close()
        return
    if fb["info_type"] == 0:
        print("  SD card confirmed — photo 1 captured.\n")
        photo_count = 1
    else:
        print(f"  Unexpected feedback info_type={fb['info_type']} — proceeding anyway.")
        photo_count = 0

    # Main capture loop
    try:
        while True:
            remaining = max(0, min_photos - photo_count)
            prompt = (
                f"[{photo_count} captured"
                + (f", {remaining} more suggested" if remaining > 0 else ", target reached!")
                + "] ENTER=capture  q=quit: "
            )
            user = input(prompt).strip().lower()

            if user == "q":
                break

            send_take_photo(sock)
            fb = read_function_feedback(sock)

            if fb is None:
                print("  No feedback — gimbal may have been slow; try again.")
                continue
            if fb["info_type"] == 1:
                print("  FAILED: SD card missing or full.")
                continue
            if fb["info_type"] == 0:
                photo_count += 1
                print(f"  Captured! Total: {photo_count}")
            else:
                print(f"  Unexpected feedback: info_type={fb['info_type']}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        sock.close()

    print(f"\nCapture complete — {photo_count} photos on SD card.")

    # Download photos
    if photo_count == 0:
        print("No photos to download.")
        return

    print("\nDownloading photos from SD card via HTTP...")
    photos = list_photos_on_sdcard()

    if not photos:
        print("  Could not retrieve file list from gimbal.")
        print("  Photos are saved on the SD card — remove it and copy manually,")
        print(f"  then place JPG files into: {output_dir}/")
        return

    # Download only JPG files, sort by name (newest last)
    jpg_files = [p for p in photos if p.get("name", "").lower().endswith(".jpg")]
    jpg_files.sort(key=lambda x: x.get("name", ""))

    print(f"  Found {len(jpg_files)} images on SD card.")
    downloaded = 0
    for item in jpg_files:
        name     = item["name"]
        url      = item["url"]
        dest     = os.path.join(output_dir, name)
        if os.path.exists(dest):
            print(f"  Skipping (already exists): {name}")
            continue
        success = download_photo(url, dest)
        if success:
            downloaded += 1
            print(f"  Downloaded: {name}  ({downloaded}/{len(jpg_files)})")

    print(f"\nDone. {downloaded} images saved to '{output_dir}/'")
    print("Next step: run OpenCV checkerboard calibration on these images.")


if __name__ == "__main__":
    capture_session(
        output_dir = "calibration_images",
        min_photos = 30,        # recommended minimum for good calibration
    )
