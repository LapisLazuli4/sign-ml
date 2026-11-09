"""
FEATURES.PY
"""
from collections import deque

import numpy as np

NUM_LANDMARKS = 21
VALUES_PER_HAND = NUM_LANDMARKS * 3       # 63
WRIST = 0
MIDDLE_KNUCKLE = 9                        # used to measure hand size

FRAME_LENGTH = VALUES_PER_HAND * 2 + 3 * 2  # 132 numbers per frame
WINDOW_FRAMES = 30                          # about 1 second at 30 frames per second
FEATURE_LENGTH = FRAME_LENGTH * WINDOW_FRAMES  # 3960 numbers per sample


def normalize_hand(points):
    """Make one hand's points independent of position and size.

    points: numpy array of shape (21, 3). Returns an array of the same shape.
    """
    points = points - points[WRIST]  # wrist to the origin
    scale = np.linalg.norm(points[MIDDLE_KNUCKLE])
    if scale < 1e-6:
        return np.zeros((NUM_LANDMARKS, 3))
    return points / scale


def hand_to_array(hand_landmarks):
    """Convert MediaPipe's landmark object into a (21, 3) numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])


def frame_to_features(results):
    """Build the 132 numbers for one frame.

    results: the object returned by mediapipe Hands.process(rgb_frame).
    Returns a numpy array of length 132, or None if no hand was found.
    """
    if not results.multi_hand_landmarks:
        return None

    right = np.zeros(VALUES_PER_HAND)
    left = np.zeros(VALUES_PER_HAND)
    right_wrist = np.zeros(3)
    left_wrist = np.zeros(3)

    for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label  # "Left" or "Right"
        points = hand_to_array(hand)
        normalized = normalize_hand(points).ravel()
        if label == "Right":
            right, right_wrist = normalized, points[WRIST]
        else:
            left, left_wrist = normalized, points[WRIST]

    return np.concatenate([right, left, right_wrist, left_wrist])


class FrameWindow:
    """Keeps the last WINDOW_FRAMES frames.

    A frame with no hand is stored as zeros. Before the first frames arrive,
    the window is filled with zeros.
    """

    def __init__(self):
        self.frames = deque(
            [np.zeros(FRAME_LENGTH) for _ in range(WINDOW_FRAMES)],
            maxlen=WINDOW_FRAMES,
        )

    def add(self, frame_features):
        if frame_features is None:
            self.frames.append(np.zeros(FRAME_LENGTH))
        else:
            self.frames.append(frame_features)

    def vector(self):
        """All frames joined end to end: FEATURE_LENGTH numbers."""
        return np.concatenate(list(self.frames))
