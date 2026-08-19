"""
Command the SIYI ZR10 gimbal to a specific absolute pitch/yaw angle,
using CMD_ID 0x0E (Send Control Angle to Gimbal).

This lets you physically reposition the camera for ground testing without
an RC transmitter or joystick - just specify the target angle in degrees.

ZR10 angle limits (from manual):
    yaw:   -135.0 to 135.0 degrees
    pitch: -90.0 to 25.0 degrees
"""

import socket
import struct
import time

GIMBAL_IP = "192.168.144.25"
GIMBAL_PORT = 37260

YAW_MIN, YAW_MAX = -135.0, 135.0
PITCH_MIN, PITCH_MAX = -90.0, 25.0


def crc16_cal(data: bytes, crc_init: int = 0) -> int:
    """CRC16/CCITT, polynomial G(X) = X^16 + X^12 + X^5 + 1, per SIYI SDK spec."""
    poly = 0x1021
    crc = crc_init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_set_angle_packet(yaw_deg: float, pitch_deg: float, seq: int = 0) -> bytes:
    """
    Build a CMD_ID 0x0E packet to command the gimbal to an absolute yaw/pitch.
    Angles are sent as int16, scaled by 10 (one decimal place of precision).
    """
    if not (YAW_MIN <= yaw_deg <= YAW_MAX):
        raise ValueError(f"yaw_deg {yaw_deg} outside ZR10 range [{YAW_MIN}, {YAW_MAX}]")
    if not (PITCH_MIN <= pitch_deg <= PITCH_MAX):
        raise ValueError(f"pitch_deg {pitch_deg} outside ZR10 range [{PITCH_MIN}, {PITCH_MAX}]")

    yaw_val = int(round(yaw_deg * 10))
    pitch_val = int(round(pitch_deg * 10))

    data = struct.pack("<hh", yaw_val, pitch_val)  # little-endian int16, int16
    data_len = len(data)

    header = (
        bytes([0x55, 0x66, 0x01])       # STX, STX, CTRL (need_ack)
        + struct.pack("<H", data_len)    # Data_len
        + struct.pack("<H", seq)         # SEQ
        + bytes([0x0E])                  # CMD_ID
    )
    frame_no_crc = header + data
    crc = crc16_cal(frame_no_crc)
    return frame_no_crc + struct.pack("<H", crc)


def parse_set_angle_ack(data: bytes):
    """
    ACK format for CMD_ID 0x0E: int16 yaw, int16 pitch, int16 roll (current angles, x10).
    """
    data_len = struct.unpack_from("<H", data, 3)[0]
    cmd_id = data[7]
    if cmd_id != 0x0E:
        raise ValueError(f"Unexpected CMD_ID in ACK: {cmd_id:#x}")

    payload = data[8:8 + data_len]
    yaw, pitch, roll = struct.unpack("<hhh", payload)
    return {
        "yaw_deg": yaw / 10.0,
        "pitch_deg": pitch / 10.0,
        "roll_deg": roll / 10.0,
    }


def set_gimbal_angle(sock: socket.socket, yaw_deg: float, pitch_deg: float, timeout: float = 1.0):
    packet = build_set_angle_packet(yaw_deg, pitch_deg)
    sock.settimeout(timeout)
    sock.sendto(packet, (GIMBAL_IP, GIMBAL_PORT))

    data, _addr = sock.recvfrom(64)
    return parse_set_angle_ack(data)


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        target_yaw = -20
        target_pitch = 15  # negative = looking downward

        print(f"Commanding gimbal to yaw={target_yaw} deg, pitch={target_pitch} deg ...")
        ack = set_gimbal_angle(sock, target_yaw, target_pitch)

        print("Gimbal acknowledged - current angles immediately before new target:")
        for k, v in ack.items():
            print(f"  {k}: {v}")

        print("\nWaiting 2s for physical movement to settle...")
        time.sleep(2.0)
        print("Now read attitude again (e.g. with siyi_attitude_test.py) to confirm final position.")

    except socket.timeout:
        print("No response - check network connectivity to the gimbal (try: ping 192.168.144.25)")
    except ValueError as e:
        print(f"Invalid input: {e}")
    finally:
        sock.close()