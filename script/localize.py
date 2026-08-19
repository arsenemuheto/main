"""
Ground-test localization pipeline for UAV target localization.

Flow per frame:
  1. Read gimbal attitude via SIYI SDK (CMD 0x0D)
  2. Build rotation matrix R from yaw/pitch/roll
  3. Build translation vector T = -R @ C  (C = gimbal position in world)
  4. Form full projection matrix P = K @ [R | T]
  5. Run YOLO on current frame → bounding box centre (u, v)
  6. Undistort (u, v) using distortion coefficients from calibration → (u', v')
  7. Back-project (u', v') with Zw=0 constraint → world coords (Xw, Yw)
  8. Print result; optionally log against ground-truth for validation

Physical setup:
  - World origin Ow : point on ground directly below the gimbal lens
  
  - Zw axis         : straight up
  - Xw axis         : direction gimbal faces when yaw = 0 (compass North)
  - Gimbal height   : measured from ground to lens centre
  - Gimbal is mounted but stationary on ground
"""

import socket
import struct
import time
import math
import csv
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

# ── Network ────────────────────────────────────────────────────────────────────
GIMBAL_IP   = "192.168.144.25"
GIMBAL_PORT = 37260

# ── Camera intrinsics (from checkerboard calibration) ─────────────────────────
# Image resolution: 2560 × 1440
FX, FY = 2684.0, 2656.0
CX, CY = 1343.8,  612.3

K = np.array([
    [FX,  0,  CX],
    [ 0, FY,  CY],
    [ 0,  0,   1],
], dtype=np.float64)

K_INV = np.linalg.inv(K)

# ── Distortion coefficients (from checkerboard calibration) ───────────────────
# Order: [k1, k2, p1, p2, k3]
# k1 = -0.303 indicates notable barrel distortion — must not skip undistortion.
DIST_COEFFS = np.array(
    [[-3.02789089e-01,  1.05554899e-01,
       1.16311271e-03,  2.54655348e-05,
       1.17429031e+00]],
    dtype=np.float64
)

# ── World geometry ─────────────────────────────────────────────────────────────
# Gimbal optical centre in world coordinates C = (0, 0, h)
# Measure h from the ground to the lens centre with a tape measure.
GIMBAL_HEIGHT = 0.12          # metres  ← UPDATE if your setup differs

C = np.array([[0.0],
              [0.0],
              [GIMBAL_HEIGHT]], dtype=np.float64)

# ── Known object heights (metres) ────────────────────────────────────────────
# Used to set Zw = object_height / 2 for the back-projection instead of Zw=0.
# The bounding box centre is at roughly mid-height of the object, so the
# correct plane to intersect is Zw = height/2, not the ground plane Zw=0.
# Add more classes as needed.  Use 0.0 for flat/ground-level objects.
OBJECT_HEIGHTS = {
    "person": 1.73,   # metres  — measured
    "chair":  1.02,   # metres  — measured
}
DEFAULT_HEIGHT = 0.0  # fallback for unknown classes → back-projects to Zw=0

# ── Camera-in-gimbal fixed offset ─────────────────────────────────────────────
# The ZR10 IMU reports the orientation of the gimbal platform body, not the
# camera optical axis directly.  The camera is mounted inside the gimbal with
# its optical axis pointing along the gimbal's +X body axis (not +Z).
# Diagnostic confirmed: at pitch=-90°,yaw=0°,roll=0° the raw gimbal R gives
# optical axis [+1,0,0] (pointing North) instead of [0,0,-1] (pointing down).
# Fix: apply a fixed -90° rotation around the gimbal Y axis after R_gimbal,
# which rotates the camera's Z axis to align with the gimbal's X axis.
# Verified: R_gimbal @ R_CAM_IN_GIMBAL gives [0,0,-1] at pitch=-90°. ✓
R_CAM_IN_GIMBAL = Rotation.from_euler('Y', math.radians(-90)).as_matrix()

# ── SIYI SDK helpers ───────────────────────────────────────────────────────────
REQUEST_ATTITUDE = bytes.fromhex("556601000000000de805")   # CMD 0x0D


def crc16_cal(data: bytes, crc_init: int = 0) -> int:
    """CRC16/CCITT — verified against SIYI manual example packet."""
    poly = 0x1021
    crc  = crc_init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def read_gimbal_attitude(sock: socket.socket, timeout: float = 1.0) -> dict:
    """
    Send CMD 0x0D and parse ACK.
    Returns yaw, pitch, roll in degrees (world-frame, direct from ZR10 IMU).
    """
    sock.settimeout(timeout)
    sock.sendto(REQUEST_ATTITUDE, (GIMBAL_IP, GIMBAL_PORT))
    data, _ = sock.recvfrom(64)

    data_len = struct.unpack_from("<H", data, 3)[0]
    cmd_id   = data[7]
    if cmd_id != 0x0D:
        raise ValueError(f"Unexpected CMD_ID in ACK: {cmd_id:#x}")

    yaw, pitch, roll, *_ = struct.unpack("<hhhhhh", data[8:8 + data_len])
    return {
        "yaw_deg":   yaw   / 10.0,
        "pitch_deg": pitch / 10.0,
        "roll_deg":  roll  / 10.0,
    }


# ── Undistortion ───────────────────────────────────────────────────────────────

def undistort_point(u: float, v: float) -> tuple[float, float]:
    """
    Remove lens distortion from a single image point using the calibrated
    distortion model.

    cv2.undistortPoints with P=K returns corrected pixel coordinates in the
    same image coordinate system — ready to pass straight into back-projection.

    Without P=K, OpenCV would return normalised (divided by focal length)
    coordinates, which would silently break the back-projection math.
    """
    pt = np.array([[[u, v]]], dtype=np.float32)
    corrected = cv2.undistortPoints(pt, K, DIST_COEFFS, P=K)
    return float(corrected[0, 0, 0]), float(corrected[0, 0, 1])


# ── Rotation & extrinsic matrix ────────────────────────────────────────────────

def build_rotation_matrix(yaw_deg: float,
                          pitch_deg: float,
                          roll_deg: float) -> np.ndarray:
    """
    Build 3×3 rotation matrix from ZR10 world-frame Euler angles.

    Two-step construction:
      1. R_gimbal  — rotates world frame to gimbal body frame using the
                     ZYX Euler angles reported by the ZR10 IMU.
      2. R_full = R_gimbal @ R_CAM_IN_GIMBAL  — applies the fixed offset
                     between the gimbal body frame and the camera optical
                     frame (-90° around Y).  This is required because the
                     ZR10 camera optical axis points along the gimbal +X
                     body axis, not +Z.

    The final R rotates vectors FROM world frame TO camera frame,
    which is what the extrinsic matrix [R | T] requires.
    """
    R_gimbal = Rotation.from_euler(
        'ZYX',
        [math.radians(yaw_deg),
         math.radians(pitch_deg),
         math.radians(roll_deg)]
    ).as_matrix()

    R_full = R_gimbal @ R_CAM_IN_GIMBAL
    return R_full


def build_extrinsic(R: np.ndarray, C: np.ndarray):
    """
    T = -R @ C   (camera position C in world → translation vector T)
    Returns (RT [3×4], T [3×1])
    """
    T  = -R @ C
    RT = np.hstack([R, T])
    return RT, T


# ── Back-projection ────────────────────────────────────────────────────────────

def backproject_to_ground(u: float, v: float,
                          R: np.ndarray, T: np.ndarray,
                          Zw: float = 0.0) -> tuple[float, float, float]:
    """
    Back-project undistorted pixel (u, v) onto the plane Zw = const.

    Steps:
      1. Xc = K⁻¹ · [u, v, 1]ᵀ          normalised camera ray
      2. s · Xc = R · W + T              projection equation
      3. Use Zw row of W to solve for scalar s
      4. W = R⁻¹ · (s·Xc − T)           recover world point

    Zw should be set to half the object's known real-world height
    (object_height / 2), since the bounding box centre sits at mid-body,
    not at ground level.  Pass Zw=0 only for flat/ground-level objects.

    Zw_check (third element of W) should equal the Zw you passed in.
    A value very close to Zw confirms the math is consistent.
    """
    uv1   = np.array([[u], [v], [1.0]], dtype=np.float64)
    Xc    = K_INV @ uv1          # normalised ray, 3×1

    R_inv = np.linalg.inv(R)     # == Rᵀ for rotation matrices
    r2    = R_inv[2, :]          # third row, shape (3,)

    s_num = Zw + float(r2 @ T)
    s_den = float(r2 @ Xc)

    if abs(s_den) < 1e-9:
        raise ValueError(
            "Back-projection degenerate: ray is parallel to the ground plane. "
            "Is the gimbal pointing horizontally?"
        )

    s = s_num / s_den
    W = R_inv @ (s * Xc - T)    # 3×1

    Xw, Yw, Zw_check = float(W[0]), float(W[1]), float(W[2])

    # Sanity check — Zw_check should ≈ Zw (numerical noise only)
    if abs(Zw_check - Zw) > 0.01:
        print(f"  [WARN] Zw_check={Zw_check:.4f} deviates from Zw={Zw:.4f} "
              "— check R and T.")

    return Xw, Yw, Zw_check


# ── Ground-truth validation helpers ───────────────────────────────────────────

def prompt_ground_truth() -> tuple[float, float] | None:
    """
    Ask the operator to enter the tape-measured position of the current
    test object.  Returns (Xw_true, Yw_true) or None if skipped.
    """
    raw = input(
        "\n  Enter ground-truth position as  Xw,Yw  (metres, e.g. 1.5,0.3)\n"
        "  or press ENTER to skip: "
    ).strip()
    if not raw:
        return None
    try:
        parts = raw.split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        print("  Could not parse — skipping.")
        return None


def euclidean_error(Xw_est, Yw_est, Xw_true, Yw_true) -> float:
    return math.sqrt((Xw_est - Xw_true) ** 2 + (Yw_est - Yw_true) ** 2)


def save_validation_log(records: list[dict], path: str = "validation_log.csv"):
    """Write all validation records to a CSV for later analysis."""
    if not records:
        return
    fieldnames = ["test_pt", "class_name", "obj_height_m", "Zw",
                  "Xw_true", "Yw_true",
                  "Xw_est", "Yw_est", "error_m",
                  "yaw_deg", "pitch_deg", "roll_deg",
                  "u_raw", "v_raw", "u_undist", "v_undist"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nValidation log saved → {path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    rtsp_url:     str  = "rtsp://192.168.144.25:8554/main.264",
    yolo_model:   str  = "yolo26n.pt",
    display:      bool = True,
    validate:     bool = False,   # set True to enter ground-truth comparison mode
):
    """
    Main loop.

    validate=True adds an interactive ground-truth entry step each time
    you press 'c' (capture) — logs estimated vs measured positions to CSV.
    """
    print("Loading YOLO model...")
    model = YOLO(yolo_model)

    print("Opening RTSP stream...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"\nGimbal : {GIMBAL_IP}:{GIMBAL_PORT}")
    print(f"Origin : Ow directly below gimbal on ground")
    print(f"Camera : C = (0, 0, {GIMBAL_HEIGHT}) m")
    print(f"Dist   : k1={DIST_COEFFS[0,0]:.4f}  k2={DIST_COEFFS[0,1]:.4f}  "
          f"k3={DIST_COEFFS[0,4]:.4f}")
    if validate:
        print("\nValidation mode ON  —  press 'c' to capture + enter ground truth")
    print("Press 'q' to quit.\n")

    validation_records = []
    test_point_idx     = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream read failed — retrying...")
                time.sleep(0.1)
                continue

            # ── Step 1: Gimbal attitude ────────────────────────────────────
            try:
                att = read_gimbal_attitude(sock)
            except socket.timeout:
                print("Attitude timeout — skipping frame")
                continue

            yaw_deg   = att["yaw_deg"]
            pitch_deg = att["pitch_deg"]
            roll_deg  = att["roll_deg"]

            # ── Step 2–3: R and T ──────────────────────────────────────────
            R      = build_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
            
            RT, T  = build_extrinsic(R, C)

            # ── Step 4: YOLO detection ─────────────────────────────────────
            results = model(frame, verbose=False)
            boxes   = results[0].boxes

            if boxes is None or len(boxes) == 0:
                annotated = frame.copy()
                cv2.putText(annotated, "No detection", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if display:
                    cv2.imshow("Localization", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                continue

            # Pick highest-confidence detection
            confs      = boxes.conf.cpu().numpy()
            best_idx   = int(np.argmax(confs))
            box        = boxes.xyxy[best_idx].cpu().numpy()
            class_id   = int(boxes.cls[best_idx].cpu().numpy())
            class_name = model.names[class_id]

            u_raw = float((box[0] + box[2]) / 2)
            v_raw = float((box[1] + box[3]) / 2)

            # ── Zw from known object height ────────────────────────────────
            obj_height = OBJECT_HEIGHTS.get(class_name, DEFAULT_HEIGHT)
            Zw         = obj_height / 2.0

            # ── Step 5: Undistort ──────────────────────────────────────────
            u_und, v_und = undistort_point(u_raw, v_raw)

            # ── Step 6: Back-project ───────────────────────────────────────
            try:
                Xw, Yw, _ = backproject_to_ground(u_und, v_und, R, T, Zw=Zw)
            except ValueError as e:
                print(f"  Back-projection skipped: {e}")
                continue

            # ── Step 7: Print ──────────────────────────────────────────────
            undist_shift = math.sqrt((u_und - u_raw)**2 + (v_und - v_raw)**2)
            print(
                f"att  yaw={yaw_deg:+7.1f}°  pitch={pitch_deg:+6.1f}°  "
                f"roll={roll_deg:+5.1f}°  |  "
                f"px_raw=({u_raw:.0f},{v_raw:.0f})  "
                f"px_und=({u_und:.0f},{v_und:.0f})  "
                f"[shift={undist_shift:.1f}px]  |  "
                f"{class_name}(h={obj_height:.2f}m,Zw={Zw:.3f}m)  |  "
                f"Xw={Xw:+6.3f}m  Yw={Yw:+6.3f}m"
            )

            # ── Step 8: Optional display ───────────────────────────────────
            if display:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # raw centre = blue,  undistorted centre = red
                cv2.circle(frame, (int(u_raw), int(v_raw)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(u_und), int(v_und)), 5, (0, 0, 255), -1)
                lbl = f"{class_name}  Xw={Xw:+.2f}m  Yw={Yw:+.2f}m  (Zw={Zw:.2f}m)"
                cv2.putText(frame, lbl, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if validate:
                    cv2.putText(frame, "c=capture  q=quit", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Localization", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                if validate and key == ord('c'):
                    # Freeze current estimates, ask operator for ground truth
                    test_point_idx += 1
                    print(f"\n── Test point {test_point_idx} ──")
                    print(f"   Estimated:  Xw={Xw:+.4f} m   Yw={Yw:+.4f} m")
                    gt = prompt_ground_truth()
                    if gt is not None:
                        err = euclidean_error(Xw, Yw, gt[0], gt[1])
                        print(f"   True:       Xw={gt[0]:+.4f} m   Yw={gt[1]:+.4f} m")
                        print(f"   Error:      {err*100:.1f} cm  ({err:.4f} m)")
                        validation_records.append({
                            "test_pt":      test_point_idx,
                            "class_name":   class_name,
                            "obj_height_m": obj_height,
                            "Zw":           round(Zw, 4),
                            "Xw_true":      gt[0],
                            "Yw_true":      gt[1],
                            "Xw_est":       round(Xw, 4),
                            "Yw_est":       round(Yw, 4),
                            "error_m":      round(err, 4),
                            "yaw_deg":      yaw_deg,
                            "pitch_deg":    pitch_deg,
                            "roll_deg":     roll_deg,
                            "u_raw":        round(u_raw, 1),
                            "v_raw":        round(v_raw, 1),
                            "u_undist":     round(u_und, 1),
                            "v_undist":     round(v_und, 1),
                        })

    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()

        if validation_records:
            errors = [r["error_m"] for r in validation_records]
            print(f"\n{'─'*50}")
            print(f"Validation summary  ({len(errors)} test points)")
            print(f"  Mean error : {sum(errors)/len(errors)*100:.1f} cm")
            print(f"  Min  error : {min(errors)*100:.1f} cm")
            print(f"  Max  error : {max(errors)*100:.1f} cm")
            print(f"{'─'*50}")
            save_validation_log(validation_records)

        print("Pipeline stopped.")


if __name__ == "__main__":
    run_pipeline(
        rtsp_url     = "rtsp://192.168.144.25:8554/main.264",
        yolo_model   = "yolo26n.pt",
        display      = True,
        validate     = True,    # ← set False for normal run without GT entry
    )