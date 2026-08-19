# Detect classes of interest
# Select target by clicking anywhere inside a YOLO bounding box
# Selected target is tracked across frames; its center coordinates are printed each frame
# All other detections remain visible until target is selected; clearing the selection restores normal view


import cv2
from ultralytics import YOLO

# ZR10 gimbal camera RTSP stream (default SIYI IP/port).
# main.264 = full-res main stream; use sub.264 for the lower-res substream if bandwidth/latency is an issue.
SOURCE = "rtsp://192.168.144.25:8554/main.264"

model = YOLO("yolo26n.pt")

selected_id = None
latest_boxes = None  # updated each frame so the mouse callback can read current detections

def on_mouse(action, x, y, *_):
    global selected_id, latest_boxes
    if action == cv2.EVENT_LBUTTONDOWN and latest_boxes is not None:
        ids = latest_boxes.id
        for i, box in enumerate(latest_boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Assign selected_id only if tracking has assigned an ID to this box
                selected_id = int(ids[i]) if ids is not None else None
                break

cv2.namedWindow("YOLO Detection")
cv2.setMouseCallback("YOLO Detection", on_mouse)

printed_shape = False

for result in model.track(source=0, show=False, stream=True, conf=0.5, persist=True, tracker="bytetrack.yaml"):
    latest_boxes = result.boxes
    # Show all YOLO detections only when no target is selected
    frame = result.orig_img.copy() if selected_id is not None else result.plot()

    if not printed_shape:
        print("Captured frame shape (H, W, C):", result.orig_img.shape)
        printed_shape = True

    # Highlight the selected target and print its center
    if selected_id is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().tolist()
        if selected_id in ids:
            idx = ids.index(selected_id)
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx])
            Cx = (x1 + x2) // 2
            Cy = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, (Cx, Cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Target  center=({Cx},{Cy})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(Cx, Cy)

    cv2.putText(frame, "Click object to select | C: clear | Q: quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLO Detection", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):  # Quit
        break
    elif k == ord('c'):  # Clear selected target
        selected_id = None

cv2.destroyAllWindows()
