"""
Minimal proof-of-concept: request gimbal attitude from a SIYI ZR10
over UDP and parse the response.

Run this on the computer once it's on the same network as the gimbal.
"""

import socket
import struct

GIMBAL_IP = "192.168.144.25"
GIMBAL_PORT = 37260

# Exact hex packet from the SIYI manual for CMD_ID 0x0D (Request Gimbal Attitude)
# 55 66 01 00 00 00 00 0d e8 05
REQUEST_ATTITUDE = bytes.fromhex("556601000000000de805")


def request_attitude(sock, timeout=1.0):
    sock.settimeout(timeout)
    sock.sendto(REQUEST_ATTITUDE, (GIMBAL_IP, GIMBAL_PORT))

    data, addr = sock.recvfrom(64)
    return data


def parse_attitude(data: bytes):
    """
    Frame format (from SDK Protocol Format section):
    STX(2) CTRL(1) Data_len(2) SEQ(2) CMD_ID(1) DATA(Data_len) CRC16(2)

    For CMD_ID 0x0D ACK, DATA is 6x int16_t, little-endian:
    yaw, pitch, roll, yaw_velocity, pitch_velocity, roll_velocity
    Each value / 10 = actual degrees (or deg/s for velocity).
    """
    # Header is 8 bytes before DATA: STX(2)+CTRL(1)+Data_len(2)+SEQ(2)+CMD_ID(1)
    data_len = struct.unpack_from("<H", data, 3)[0]
    cmd_id = data[7]

    if cmd_id != 0x0D:
        raise ValueError(f"Unexpected CMD_ID: {cmd_id:#x}")

    payload = data[8:8 + data_len]
    yaw, pitch, roll, yaw_vel, pitch_vel, roll_vel = struct.unpack("<hhhhhh", payload)

    return {
        "yaw_deg": yaw / 10.0,
        "pitch_deg": pitch / 10.0,
        "roll_deg": roll / 10.0,
        "yaw_velocity_deg_s": yaw_vel / 10.0,
        "pitch_velocity_deg_s": pitch_vel / 10.0,
        "roll_velocity_deg_s": roll_vel / 10.0,
    }


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        raw = request_attitude(sock)
        print("Raw response (hex):", raw.hex())

        attitude = parse_attitude(raw)
        print("Parsed attitude:")
        for k, v in attitude.items():
            print(f"  {k}: {v}")
    except socket.timeout:
        print("No response - check network connectivity to the gimbal (try: ping 192.168.144.25)")
    finally:
        sock.close()
