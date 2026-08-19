# Detect classes of interest
# Select target, by drawning manual box around it via mouse drag. Useful to choose part of the target to aim for
# Only see target bounding box, and its center coordinates 
# When target is cleared, see other detections 


import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Target bounding box state
top_left = None
bottom_right = None
drawing = False
target_box = None  # finalized box as (top_left, bottom_right)

def on_mouse(action, x, y, *_):
    global top_left, bottom_right, drawing, target_box
    if action == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left = (x, y)
        bottom_right = (x, y)
    elif action == cv2.EVENT_MOUSEMOVE and drawing:
        bottom_right = (x, y)
    elif action == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        bottom_right = (x, y)
        # Only keep the box if the user actually dragged (not just a click)
        if abs(bottom_right[0] - top_left[0]) > 5 and abs(bottom_right[1] - top_left[1]) > 5:
            target_box = (top_left, bottom_right)

cv2.namedWindow("YOLO Detection")
cv2.setMouseCallback("YOLO Detection", on_mouse)

for result in model(source=0, show=False, stream=True, conf=0.5):
    # Show YOLO detections only when no target box has been drawn yet
    if target_box is None and not drawing:
        frame = result.plot()
    else:
        frame = result.orig_img.copy()

    # Draw in-progress box (while mouse button is held)
    if drawing and top_left and bottom_right:
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Draw finalized target box with center
    if target_box and not drawing:
        cv2.rectangle(frame, target_box[0], target_box[1], (0, 255, 0), 2)
        Cx = (target_box[0][0] + target_box[1][0]) // 2
        Cy = (target_box[0][1] + target_box[1][1]) // 2
        cv2.circle(frame, (Cx, Cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, "Target", target_box[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(Cx, Cy)

    cv2.putText(frame, "Draw box: click+drag | C: clear | Q: quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLO Detection", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):  #Quit when Q key is clicked
        break
    elif k == ord('c'):  #Delete target when C key is clicked
        target_box = None
        top_left = None
        bottom_right = None
        drawing = False

cv2.destroyAllWindows()
