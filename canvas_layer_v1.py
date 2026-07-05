import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- VARIABLES ----------------
canvas = None
prev_x, prev_y = 0, 0

smoothening = 0.85
mode = "DRAW"

BRUSH_COLOR = (0, 0, 255)   # Red
BRUSH_SIZE = 5
ERASER_SIZE = 60


# ---------------- FINGER COUNT ----------------
def count_fingers(handLms):
    tips = [8, 12, 16, 20]
    count = 0

    for tip in tips:
        if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
            count += 1

    return count


# ---------------- MAIN LOOP ----------------
while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            finger_count = count_fingers(handLms)

            # ---------------- GESTURES ----------------
            if finger_count == 1:
                mode = "DRAW"

            elif finger_count == 2:
                mode = "ERASE"

            elif finger_count == 3:
                canvas[:] = 0

            # ---------------- INDEX FINGER ----------------
            index_tip = handLms.landmark[8]
            index_pip = handLms.landmark[6]

            target_x = int(index_tip.x * w)
            target_y = int(index_tip.y * h)

            index_up = index_tip.y < index_pip.y

            if index_up:

                if prev_x == 0 and prev_y == 0:
                    prev_x = target_x
                    prev_y = target_y

                # Smooth movement
                x = int(prev_x + (target_x - prev_x) * smoothening)
                y = int(prev_y + (target_y - prev_y) * smoothening)

                # Ignore tiny camera jitter
                distance = np.hypot(x - prev_x, y - prev_y)

                if distance < 4:
                    prev_x, prev_y = x, y
                    continue

                # Draw
                if mode == "DRAW":

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        BRUSH_COLOR,
                        BRUSH_SIZE,
                        cv2.LINE_AA
                    )

                # Erase
                elif mode == "ERASE":

                    cv2.circle(
                        canvas,
                        (x, y),
                        ERASER_SIZE,
                        (0, 0, 0),
                        -1
                    )

                prev_x = x
                prev_y = y

            else:
                # Finger lifted
                prev_x = 0
                prev_y = 0

    else:
        # Hand disappeared
        prev_x = 0
        prev_y = 0

    # ---------------- DISPLAY ----------------
    combined = cv2.add(frame, canvas)

    cv2.putText(
        combined,
        f"Mode : {mode}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Smart Gesture Board", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        filename = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, canvas)
        print("Saved:", filename)

cap.release()
cv2.destroyAllWindows()
