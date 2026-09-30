#!/usr/bin/env python3
"""
Project - pipeline script (report sections 1 to 6)

  1. Live video acquisition      ZR10 gimbal camera, RTSP stream
  2. Object detection            Ultralytics YOLO (yolo26n.pt, class: person)
  3. Live detections + selection OpenCV window, click inside a box to select the target
  4. Tracking algorithm          ByteTrack via model.track(tracker="bytetrack.yaml")
  5. Real angle measurement      gimbal yaw/pitch (SIYI SDK, CMD_ID 0x0D)
                                 + pixel offset of the tracked box centre (camera intrinsics)
  6. Gimbal angle control        absolute yaw/pitch commands (SIYI SDK, CMD_ID 0x0E)
Keys in the video window
  left click : select the target (click anywhere inside its bounding box)
  c          : clear the selected target
  f          : toggle gimbal follow mode
  q / ESC    : quit

Requirements: pip install ultralytics opencv-python numpy
"""

import argparse
import csv
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SIYI_IP = "192.168.144.25"                                 # ZR10 default IP
SIYI_UDP_PORT = 37260                                      # SIYI SDK UDP port
DEFAULT_RTSP = "rtsp://192.168.144.25:8554/main.264"       # ZR10 default RTSP stream

# Phone IP-webcam apps (phone and laptop must be on the same Wi-Fi)
PHONE_APPS = {"ipwebcam": ("8080", "/video"),     # Android app "IP Webcam"
              "droidcam": ("4747", "/video")}     # app "DroidCam"

MODEL_PATH = "yolo26n.pt"        # off-the-shelf model (for now)
PERSON_CLASS = 0                 # target: person (for now)
TRACKER_CFG = "Bytetrack.yaml"   # ByteTrack, bundled with ultralytics
conf = 0.5
CALIB_FILE = "calibration.npz"   # from the checkerboard calibration (keys: camera_matrix, dist_coeffs)
FALLBACK_HFOV_DEG = 62.0         # ONLY used if no calibration file is found (rough placeholder)

# Image v grows downward, while a positive gimbal pitch looks up. So a target BELOW the
# image centre lies at a more negative pitch. If your gimbal convention differs, flip this.
PITCH_SIGN = -1.0

YAW_LIMITS = (-135.0, 135.0)     # ZR10 mechanical range (deg)
PITCH_LIMITS = (-90.0, 25.0)


# ----------------------------------------------------------------------------
# SIYI SDK over UDP (minimal implementation)
# ----------------------------------------------------------------------------
def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_packet(cmd_id: int, data: bytes = b"", seq: int = 0, need_ack: bool = True) -> bytes:
    """STX(0x5566) | CTRL | DATA_LEN(2, LE) | SEQ(2, LE) | CMD_ID | DATA | CRC16(2, LE)"""
    body = (b"\x55\x66" + bytes([0x01 if need_ack else 0x00]) + struct.pack("<H", len(data))
            + struct.pack("<H", seq & 0xFFFF) + bytes([cmd_id]) + data)
    return body + struct.pack("<H", crc16_xmodem(body))


def parse_packet(raw: bytes):
    """Return (cmd_id, data) or None if the packet is invalid."""
    if len(raw) < 10 or raw[:2] != b"\x55\x66":
        return None
    data_len = struct.unpack("<H", raw[3:5])[0]
    if len(raw) < 10 + data_len:
        return None
    body, crc = raw[:8 + data_len], struct.unpack("<H", raw[8 + data_len:10 + data_len])[0]
    if crc16_xmodem(body) != crc:
        return None
    return raw[7], raw[8:8 + data_len]


class SiyiGimbal:
    CMD_GET_ATTITUDE = 0x0D   # acquire gimbal attitude
    CMD_SET_ANGLES = 0x0E     # set gimbal absolute yaw / pitch

    def __init__(self, ip=SIYI_IP, port=SIYI_UDP_PORT, poll_hz=20.0):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self._seq = 0
        self._lock = threading.Lock()
        self._att = None                 # (yaw, pitch, roll, timestamp)
        self._running = True
        self._poll_dt = 1.0 / poll_hz
        threading.Thread(target=self._rx_loop, daemon=True).start()
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _send(self, cmd_id, data=b""):
        with self._lock:
            self._seq += 1
            pkt = build_packet(cmd_id, data, self._seq)
        self.sock.sendto(pkt, self.addr)

    def _poll_loop(self):
        while self._running:
            self._send(self.CMD_GET_ATTITUDE)
            time.sleep(self._poll_dt)

    def _rx_loop(self):
        while self._running:
            try:
                raw, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed = parse_packet(raw)
            if not parsed:
                continue
            cmd_id, data = parsed
            # both 0x0D and 0x0E replies start with yaw, pitch, roll (int16, 0.1 deg)
            if cmd_id in (self.CMD_GET_ATTITUDE, self.CMD_SET_ANGLES) and len(data) >= 6:
                yaw, pitch, roll = (v / 10.0 for v in struct.unpack("<hhh", data[:6]))
                with self._lock:
                    self._att = (yaw, pitch, roll, time.time())

    def get_attitude(self, max_age=0.5):
        """Latest (yaw, pitch, roll) in degrees, or None if missing / stale."""
        with self._lock:
            att = self._att
        if att is None or time.time() - att[3] > max_age:
            return None
        return att[:3]

    def set_angles(self, yaw_deg, pitch_deg):
        """Command absolute gimbal angles (deg), clamped to the ZR10 range."""
        yaw = min(max(yaw_deg, YAW_LIMITS[0]), YAW_LIMITS[1])
        pitch = min(max(pitch_deg, PITCH_LIMITS[0]), PITCH_LIMITS[1])
        self._send(self.CMD_SET_ANGLES, struct.pack("<hh", int(round(yaw * 10)), int(round(pitch * 10))))
        return yaw, pitch

    def close(self):
        self._running = False
        self.sock.close()


# ----------------------------------------------------------------------------
# 1. Live video acquisition (latest-frame reader, avoids RTSP buffer lag)
# ----------------------------------------------------------------------------
def resolve_source(args):
    """Return the video source. --phone-ip builds the phone webcam URL; otherwise --source is used."""
    if not args.phone_ip:
        return args.source
    port, path = PHONE_APPS[args.phone_app]
    url = f"http://{args.phone_ip}:{args.phone_port or port}{path}"
    print(f"[source] phone webcam: {url}")
    if not args.no_gimbal:
        print("[source] phone camera has no SIYI gimbal -> running with --no-gimbal")
        args.no_gimbal = True
    return url


def add_source_args(ap):
    ap.add_argument("--source", default=DEFAULT_RTSP, help="RTSP URL, video file or camera index")
    ap.add_argument("--phone-ip", help="phone IP-webcam address, e.g. 192.168.1.5 (replaces --source)")
    ap.add_argument("--phone-port", help="phone app port (default 8080 for ipwebcam, 4747 for droidcam)")
    ap.add_argument("--phone-app", choices=list(PHONE_APPS), default="ipwebcam")


class VideoStream:
    """Live sources (RTSP / camera): a background thread keeps only the newest frame.
    Video files: frames are read one by one in order, so the video plays normally
    (and loops when it reaches the end)."""

    def __init__(self, source):
        src = str(source)
        self.is_live = src.isdigit() or src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))
        self.src = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # keep the delay small
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        if self.is_live:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last_ok = time.time()
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                last_ok = time.time()
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)
                if time.time() - last_ok > 3.0:            # Wi-Fi drop / stream stopped: reconnect
                    print("[video] no frames for 3 s - reconnecting ...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.src)
                    last_ok = time.time()

    def read(self):
        if self.is_live:
            with self._lock:
                return None if self._frame is None else self._frame.copy()
        ok, frame = self.cap.read()
        if not ok:                                   # end of file: start again
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self._running = False
        time.sleep(0.05)
        self.cap.release()


# ----------------------------------------------------------------------------
# 5. Camera model: pixel -> angular offset from the optical axis
# ----------------------------------------------------------------------------
class CameraModel:
    def __init__(self, K, dist):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]

    @classmethod
    def load(cls, path, frame_w, frame_h):
        try:
            d = np.load(path)
            K, dist = d["camera_matrix"].copy(), d["dist_coeffs"]
            if "image_size" in d:                       # rescale if streamed at another resolution
                w0, h0 = d["image_size"]
                K[0, :] *= frame_w / w0
                K[1, :] *= frame_h / h0
            print(f"[camera] intrinsics loaded from {path}")
            return cls(K, dist)
        except (OSError, KeyError):
            fx = (frame_w / 2) / math.tan(math.radians(FALLBACK_HFOV_DEG / 2))
            K = [[fx, 0, frame_w / 2], [0, fx, frame_h / 2], [0, 0, 1]]
            print(f"[camera] WARNING: {path} not found - using a rough placeholder "
                  f"(HFOV {FALLBACK_HFOV_DEG} deg, no distortion). Angles will be less accurate.")
            return cls(K, np.zeros(5))

    def offsets_deg(self, u, v):
        """Angular offsets (d_yaw, d_pitch) in degrees of pixel (u, v) from the image centre.
        d_yaw   = atan((u - cx) / fx)   (positive: target to the right)
        d_pitch = atan((v - cy) / fy)   (positive: target below centre, image v grows downward)
        The point is undistorted first, so lens distortion does not bias the angle."""
        pt = np.array([[[u, v]]], dtype=np.float64)
        x, y = cv2.undistortPoints(pt, self.K, self.dist, P=self.K)[0, 0]
        return (math.degrees(math.atan2(x - self.cx, self.fx)),
                math.degrees(math.atan2(y - self.cy, self.fy)))


def put_label(frame, text, org, color=(255, 255, 255), scale=1.0):
    """Readable text: size follows the frame height and a dark box sits behind the letters."""
    k = max(0.6, frame.shape[0] / 720.0) * scale
    thick = max(1, int(round(2 * k)))
    (tw, th), base_ln = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, thick)
    x = max(0, min(int(org[0]), frame.shape[1] - tw - 6))
    y = max(th + 4, min(int(org[1]), frame.shape[0] - base_ln - 2))
    cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + base_ln), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, color, thick, cv2.LINE_AA)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def measure_target_angles(camera, gimbal_att, u, v):
    """Real yaw/pitch (deg) of the target = gimbal angles + angular offset of the box centre."""
    d_yaw, d_pitch = camera.offsets_deg(u, v)
    g_yaw, g_pitch, _ = gimbal_att
    return wrap180(g_yaw + d_yaw), g_pitch + PITCH_SIGN * d_pitch, d_yaw, d_pitch


# ----------------------------------------------------------------------------
# 3. Target selection (mouse click inside a bounding box)
# ----------------------------------------------------------------------------
class TargetSelector:
    def __init__(self, window):
        self.track_id = None
        self._click = None
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    def resolve_click(self, tracks):
        """tracks: list of (id, x1, y1, x2, y2, conf). Select the smallest box containing the click."""
        if self._click is None:
            return
        cx, cy = self._click
        self._click = None
        hits = [t for t in tracks if t[1] <= cx <= t[3] and t[2] <= cy <= t[4]]
        if hits:
            t = min(hits, key=lambda t: (t[3] - t[1]) * (t[4] - t[2]))
            self.track_id = t[0]
            print(f"[select] target ID {t[0]}, box centre pixels = "
                  f"({(t[1] + t[3]) / 2:.0f}, {(t[2] + t[4]) / 2:.0f})")

    def clear(self):
        self.track_id = None


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def run_goto(args):
    """Section 6 test mode: command absolute yaw/pitch, print the resulting attitude, exit."""
    gimbal = SiyiGimbal(args.gimbal_ip)
    yaw, pitch = gimbal.set_angles(args.goto[0], args.goto[1])
    print(f"[gimbal] commanded yaw={yaw:.1f}, pitch={pitch:.1f}")
    time.sleep(2.0)
    print(f"[gimbal] attitude now (yaw, pitch, roll) = {gimbal.get_attitude(max_age=2.0)}")
    gimbal.close()


def main():
    ap = argparse.ArgumentParser(description="Mini-Umwambi: detect -> select -> track -> measure angles -> gimbal control")
    add_source_args(ap)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    ap.add_argument("--calib", default=CALIB_FILE)
    ap.add_argument("--gimbal-ip", default=SIYI_IP)
    ap.add_argument("--no-gimbal", action="store_true", help="run without the SIYI gimbal link (angle offsets only)")
    ap.add_argument("--follow", action="store_true", help="re-point the gimbal at the tracked target")
    ap.add_argument("--goto", nargs=2, type=float, metavar=("YAW", "PITCH"),
                    help="command absolute gimbal angles (deg) and exit")
    ap.add_argument("--log", help="CSV file to log measured angles (for ground-truth validation)")
    args = ap.parse_args()

    if args.goto:
        run_goto(args)
        return

    source = resolve_source(args)
    from ultralytics import YOLO   # imported here so --goto works without ultralytics installed

    model = YOLO(args.model)
    gimbal = None if args.no_gimbal else SiyiGimbal(args.gimbal_ip)
    stream = VideoStream(source)

    window = "Mini-Umwambi - click a target"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    selector = TargetSelector(window)

    log_file, log_writer = None, None
    if args.log:
        log_file = open(args.log, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["time", "track_id", "u", "v", "gimbal_yaw", "gimbal_pitch",
                             "d_yaw", "d_pitch", "target_yaw", "target_pitch"])

    camera, follow, last_cmd = None, args.follow, 0.0

    try:
        while True:
            frame = stream.read()
            if frame is None:
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue
            h, w = frame.shape[:2]
            if camera is None:
                camera = CameraModel.load(args.calib, w, h)

            # ---- 2 + 4. detection and ByteTrack tracking
            res = model.track(frame, persist=True, tracker=TRACKER_CFG, classes=[PERSON_CLASS],
                              conf=args.conf, verbose=False)[0]
            tracks = []
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.int().cpu().tolist()
                confs = res.boxes.conf.cpu().tolist()
                tracks = [(i, *b, c) for i, b, c in zip(ids, xyxy, confs)]

            # ---- 3. operator target selection
            selector.resolve_click(tracks)

            status = "click a person to select a target"
            for (tid, x1, y1, x2, y2, conf) in tracks:
                selected = tid == selector.track_id
                color = (0, 0, 255) if selected else (160, 160, 160)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3 if selected else 1)
                put_label(frame, f"person ID {tid} {conf:.2f}", (x1, y1 - 6), color)
                if not selected:
                    continue

                # ---- 5. real angle measurement of the tracked target
                u, v = (x1 + x2) / 2, (y1 + y2) / 2
                cv2.drawMarker(frame, (int(u), int(v)), color, cv2.MARKER_CROSS, 18, 2)
                att = gimbal.get_attitude() if gimbal else (0.0, 0.0, 0.0)
                if att is None:
                    status = f"ID {tid}: waiting for gimbal attitude..."
                    continue
                t_yaw, t_pitch, d_yaw, d_pitch = measure_target_angles(camera, att, u, v)
                status = (f"ID {tid} | target yaw {t_yaw:+.1f}  pitch {t_pitch:+.1f} deg | "
                          f"offset {d_yaw:+.1f}/{PITCH_SIGN * d_pitch:+.1f}")
                if log_writer:
                    log_writer.writerow([f"{time.time():.3f}", tid, f"{u:.1f}", f"{v:.1f}",
                                         att[0], att[1], f"{d_yaw:.3f}", f"{d_pitch:.3f}",
                                         f"{t_yaw:.3f}", f"{t_pitch:.3f}"])

                # ---- 6. gimbal angle control: absolute command puts the target on the optical axis
                if follow and gimbal and time.time() - last_cmd > 0.2 and max(abs(d_yaw), abs(d_pitch)) > 0.5:
                    gimbal.set_angles(t_yaw, t_pitch)
                    last_cmd = time.time()

            if selector.track_id is not None and selector.track_id not in [t[0] for t in tracks]:
                status = f"ID {selector.track_id} LOST - press 'c' to clear or click a new target"

            cv2.circle(frame, (w // 2, h // 2), 6, (0, 255, 0), 1)     # image centre (optical axis)
            put_label(frame, status, (10, 30), (0, 255, 255))
            put_label(frame, f"follow: {'ON' if follow else 'OFF'}", (10, h - 10),
                      (0, 255, 0) if follow else (200, 200, 200))
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                selector.clear()
            if key == ord("f"):
                follow = not follow
    finally:
        stream.release()
        if gimbal:
            gimbal.close()
        if log_file:
            log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
