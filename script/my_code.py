import cv2
from ultralytics import YOLO

# the ip_camera or ZR10 gimbal camera RTSP stream
SOURCE = "https://10.0.13.185:8080/video"
model= "yolo26n.pt"
CONF = 0.5
CLASSES_OF_INTEREST = [0, 2, 3, 5, 7]

# Tracker configuration
TRACKER = "bytetrack.yaml"
WINDOW_NAME = "YOLO Detection"
model = YOLO(model)

selected_id = None   # Selected ByteTrack object ID

# Current detections.
# This is updated every frame and accessed by the mouse callback.
latest_boxes = None

def on_mouse(event, x, y, flags, param):
    global selected_id, latest_boxes

    # Only react to left mouse button
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # No detections available yet
    if latest_boxes is None:
        return

    # Tracker IDs are required to select a persistent target
    if latest_boxes.id is None:
        print("No tracking IDs available yet.")
        return
    ids = latest_boxes.id.int().tolist()

    # Check every bounding box
    for i, box in enumerate(latest_boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Click anywhere inside the bounding box
        if x1 <= x <= x2 and y1 <= y <= y2:

            selected_id = ids[i]

            # Get class information
            cls_id = int(latest_boxes.cls[i])
            class_name = model.names[cls_id]

            print(f"Target selected: "f"ID={selected_id}, " f"class={class_name}")     
            break
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

printed_shape = False

for result in model.track(source=SOURCE,show=False,  stream=True,conf=CONF, classes=CLASSES_OF_INTEREST, persist=True,tracker=TRACKER):

    original_frame = result.orig_img

    if not printed_shape:
        print("Captured frame shape (H, W, C):",original_frame.shape)
        printed_shape = True
        latest_boxes = result.boxes

    # Start with a clean original frame.
    # We will draw detections ourselves so we have full
    # control over selected/unselected targets.
    frame = original_frame.copy()

    # Get tracking IDs
    ids = None

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().tolist()
  
     # No target selected:
     # Show ALL detections belonging to the classes of interest.
     # Clicking one of these boxes selects its tracking ID.
    if selected_id is None:

        if result.boxes is not None:

            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)

                # Class
                cls_id = int(result.boxes.cls[i])
                class_name = model.names[cls_id]
                confidence = float(result.boxes.conf[i])         # Confidence

                # Tracking ID
                track_id = None
                if ids is not None:
                    track_id = ids[i]

                cv2.rectangle(frame,(x1, y1),(x2, y2), (255, 0, 0),  2)   # Draw bounding box
                
                # Label
                if track_id is not None:
                    label = (f"{class_name} "f"ID:{track_id} "f"{confidence:.2f}")
                else:
                    label = (f"{class_name} "
                             f"{confidence:.2f}")
            
                cv2.putText(frame,label,(x1, max(y1 - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0), 2)
                
         # Only the selected target is displayed.
         # The target is identified using its persistent ByteTrack ID.
    else:
        target_found = False

        if (result.boxes is not None and result.boxes.id is not None):
            ids = result.boxes.id.int().tolist()

            # Look for the selected tracking ID
            if selected_id in ids:
                idx = ids.index(selected_id)
                target_found = True

                x1, y1, x2, y2 = map(int,result.boxes.xyxy[idx])   # Bounding box

                # Class
                cls_id = int(result.boxes.cls[idx])
                class_name = model.names[cls_id]
                confidence = float(result.boxes.conf[idx])  # Confidence

                # Calculate center coordinates
                Cx = (x1 + x2) // 2
                Cy = (y1 + y2) // 2
                cv2.rectangle(frame,(x1, y1),(x2, y2), (0, 255, 0),3)       # Draw selected target
                cv2.circle(frame,(Cx, Cy),6,(0, 255, 0),-1)                 # Center point
                cv2.line(frame,(Cx - 15, Cy),(Cx + 15, Cy),(0, 255, 0),2)   # Horizontal center line
                cv2.line( frame,(Cx, Cy - 15),(Cx, Cy + 15),(0, 255, 0),2)  # Vertical center line
                label = (f"TARGET | "f"{class_name} | "f"ID:{selected_id} | "f"center=({Cx},{Cy})") # Target information

                cv2.putText(frame,label,(x1, max(y1 - 10, 25)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0, 255, 0),2)

                print(f"Target ID={selected_id} "f"center=({Cx}, {Cy})")      # Print center coordinates EVERY FRAME

        if not target_found:
            cv2.putText(frame,f"TARGET ID {selected_id} - NOT DETECTED",(10, 55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)
    if selected_id is None:
        cv2.putText(frame,"Click inside a bounding box to select target",(10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2)

    else:
        cv2.putText(frame,"TARGET SELECTED | C: clear selection | Q: quit",(10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0),2)
        
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):   #quit
        break
    elif key == ord("c"): #clear selection

        if selected_id is not None:
            print(f"Target ID={selected_id} cleared.")
        selected_id = None
cv2.destroyAllWindows()