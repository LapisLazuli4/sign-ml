"""
RUN.PY LIVE TRANSLATOR

Keys (click the video window first):
    C : clear the transcript                        Q : quit
"""
import pickle
import subprocess
import time
from collections import deque

import cv2
import mediapipe as mp
import sklearn

from features import FrameWindow, WINDOW_FRAMES, FEATURE_LENGTH, frame_to_features

MODEL_PATH = "isl_model.pkl"
CONFIDENCE = 0.7     # raise if you see wrong words, lower if signs are missed
STABLE_FRAMES = 4    # agreeing predictions needed (4 x 3 frames = about 0.4 s)
PREDICT_EVERY = 3    # run the model on every 3rd frame

# A PowerShell process that stays open and speaks each line as it arrives.
SPEAK_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "while (($line = [Console]::In.ReadLine()) -ne $null) { $s.SpeakAsync($line) | Out-Null }"
)


def start_speaker():
    return subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", SPEAK_SCRIPT],
        stdin=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,  # no flashing window
    )


def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("window_frames") != WINDOW_FRAMES or bundle.get("feature_length") != FEATURE_LENGTH:
        raise SystemExit(
            "isl_model.pkl was trained with different settings. Run train.py again "
            "with the current features.py."
        )
    if bundle["sklearn_version"] != sklearn.__version__:
        print(f"Warning: model was trained with sklearn {bundle['sklearn_version']}, "
              f"but this computer has {sklearn.__version__}.")
    model = bundle["model"]
    model.n_jobs = 1
    return model


def predict_sign(model, vector):
    """Return the predicted sign, or None if the model is unsure."""
    probabilities = model.predict_proba([vector])[0]
    best = probabilities.argmax()
    if probabilities[best] < CONFIDENCE:
        return None
    return model.classes_[best]


def main():
    model = load_model()
    speaker = start_speaker()
    time.sleep(1.5)  # give PowerShell time to load the voice

    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    window = FrameWindow()
    recent = deque(maxlen=STABLE_FRAMES)
    transcript = []
    last_accepted = None
    current = None
    frame_count = 0
    fps_start, fps_count = time.perf_counter(), 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # same mirror as collect.py
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            features = frame_to_features(results)
            window.add(features)

            predicted_now = False
            if features is None:
                current = None
                predicted_now = True
            else:
                frame_count += 1
                if frame_count % PREDICT_EVERY == 0:
                    current = predict_sign(model, window.vector())
                    predicted_now = True

            if predicted_now:
                recent.append(current)
                if current is None:
                    last_accepted = None
                else:
                    stable = len(recent) == STABLE_FRAMES and all(x == current for x in recent)
                    if stable and current != last_accepted:
                        transcript.append(current)
                        speaker.stdin.write(f"{current}\n")  # speak now
                        speaker.stdin.flush()
                        last_accepted = current

            shown_text = " ".join(transcript)[-60:]
            cv2.rectangle(frame, (0, frame.shape[0] - 90), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, shown_text, (15, frame.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Now: {current or '-'}", (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Sign ML (C clears, Q quits)", frame)

            fps_count += 1
            if time.perf_counter() - fps_start >= 2:
                print(f"FPS: {fps_count / (time.perf_counter() - fps_start):.1f}")
                fps_start, fps_count = time.perf_counter(), 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                transcript.clear()
                last_accepted = None
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        speaker.stdin.close()
        speaker.terminate()


if __name__ == "__main__":
    main()
