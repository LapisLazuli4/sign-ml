"""
COLLECT.PY FOR DATA COLLECTION
"""
import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np

from features import FEATURE_LENGTH, FrameWindow, frame_to_features

DATA_DIR = "data"
SPACE_KEY = 32
QUIT_KEY = 27  # Esc key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="your name, used in the file name")
    parser.add_argument("--camera", type=int, default=0, help="webcam number (0, 1, ...)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{args.name}.csv")
    is_new_file = not os.path.exists(out_path)

    csv_file = open(out_path, "a", newline="")
    writer = csv.writer(csv_file)
    if is_new_file:
        header = ["label", "contributor"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
        writer.writerow(header)

    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1.")

    window = FrameWindow()
    selected_label = None    # the sign chosen with its letter key
    recording = False        # True between a SPACE start and SPACE stop
    session_counts = {}

    # WINDOW_NORMAL lets the window be resized and maximised.
    window_name = "Collect ISL samples (Esc to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Stop if the window's close button was clicked.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break

            # Mirror the image so it feels like a mirror. run.py does the same.
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand, mp.solutions.hands.HAND_CONNECTIONS
                    )

            features = frame_to_features(results)
            window.add(features)

            if recording and selected_label is not None and features is not None:
                row = np.round(window.vector(), 5).tolist()
                writer.writerow([selected_label, args.name] + row)
                session_counts[selected_label] = session_counts.get(selected_label, 0) + 1

            csv_file.flush()

            key = cv2.waitKey(1) & 0xFF
            if key == QUIT_KEY:
                break
            elif key == SPACE_KEY:
                # SPACE starts recording the selected sign, and stops it again.
                if selected_label is not None:
                    recording = not recording
            elif key != 255 and chr(key).isalnum():
                # Choosing a new sign stops any recording in progress.
                selected_label = chr(key).upper()
                recording = False

            if recording:
                status = f"RECORDING {selected_label} - press SPACE to stop"
                colour = (0, 0, 255)
            elif selected_label is not None:
                status = f"Selected {selected_label} - press SPACE to start"
                colour = (0, 255, 255)
            else:
                status = "Press a letter or digit to choose a sign"
                colour = (255, 255, 255)

            hand_text = "Hand: yes" if features is not None else "Hand: no"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(frame, hand_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(session_counts.items()))
            cv2.putText(frame, summary[:90], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
    finally:
        cap.release()
        csv_file.close()
        hands.close()
        cv2.destroyAllWindows()
        print(f"Saved to {out_path}")
        print("Samples this session:", session_counts)


if __name__ == "__main__":
    main()
