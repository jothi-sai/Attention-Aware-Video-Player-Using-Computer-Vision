"""
attention_detector.py
---------------------
Core computer vision module.

Uses MediaPipe Face Mesh (468 landmarks) to detect:
  1. Eye Aspect Ratio (EAR)  -> eyes closed / drowsiness
  2. Head Pose (yaw/pitch)   -> looking away
  3. Face presence           -> user absent

Attention States (in order of severity):
  ATTENTIVE      - all good, video plays normally
  DROWSY         - EAR slightly low, warning shown
  EYES_CLOSED    - EAR below threshold for N frames, video paused
  LOOKING_AWAY   - head yaw/pitch exceed limits, video paused
  FACE_ABSENT    - no face detected, video paused
"""

import cv2
import mediapipe as mp
import numpy as np
from enum import Enum


# ─────────────────────────────────────────────
#  Attention State Enum
# ─────────────────────────────────────────────

class AttentionState(Enum):
    ATTENTIVE    = "attentive"
    DROWSY       = "drowsy"
    EYES_CLOSED  = "eyes_closed"
    LOOKING_AWAY = "looking_away"
    FACE_ABSENT  = "face_absent"


# ─────────────────────────────────────────────
#  Main Detector Class
# ─────────────────────────────────────────────

class AttentionDetector:
    """
    Detects user attention using facial landmarks.

    EAR Formula (Soukupová & Čech, 2016):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        p1, p4 = horizontal eye corners
        p2, p3, p5, p6 = vertical eye landmarks
        EAR ≈ 0.3 when open, drops toward 0 when closed
    """

    # ── MediaPipe landmark indices for each eye ──────────────────────
    # These are the 6 landmarks forming the eye ellipse
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    # ── 3D face model points for solvePnP head pose ──────────────────
    # Standard 3D coordinates of key facial points (in mm, face-centered)
    MODEL_POINTS_3D = np.array([
        (0.0,    0.0,    0.0),     # Nose tip         (landmark 1)
        (0.0,   -330.0, -65.0),    # Chin             (landmark 152)
        (-225.0, 170.0, -135.0),   # Left eye corner  (landmark 263)
        (225.0,  170.0, -135.0),   # Right eye corner (landmark 33)
        (-150.0,-150.0, -125.0),   # Left mouth corner(landmark 287)
        (150.0, -150.0, -125.0),   # Right mouth corner(landmark 57)
    ], dtype=np.float64)

    # Corresponding MediaPipe landmark indices
    POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]

    # ── Thresholds ────────────────────────────────────────────────────
    EAR_CLOSED_THRESHOLD = 0.22   # below this → eyes closed
    EAR_DROWSY_THRESHOLD = 0.28   # below this → drowsy
    CLOSED_FRAMES_LIMIT  = 4      # consecutive frames before EYES_CLOSED state
    YAW_LIMIT            = 30     # degrees left/right before LOOKING_AWAY
    PITCH_LIMIT          = 25     # degrees up/down before LOOKING_AWAY

    def __init__(self, webcam_index: int = 0):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._closed_frame_count = 0
        self.state = AttentionState.ATTENTIVE

    # ── Public API ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray):
        """
        Process a single BGR webcam frame.

        Returns:
            state   (AttentionState) : current attention state
            frame   (np.ndarray)     : annotated frame
            metrics (dict)           : ear, yaw, pitch values for logging
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # ── No face detected ─────────────────────────────────────────
        if not results.multi_face_landmarks:
            self.state = AttentionState.FACE_ABSENT
            self._closed_frame_count = 0
            return self.state, frame, {}

        landmarks = results.multi_face_landmarks[0].landmark

        # ── Compute EAR ───────────────────────────────────────────────
        left_ear  = self._compute_ear(landmarks, self.LEFT_EYE,  w, h)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        # ── Compute Head Pose ─────────────────────────────────────────
        pitch, yaw, roll = self._compute_head_pose(landmarks, w, h)

        metrics = {
            "ear"  : round(avg_ear, 3),
            "yaw"  : round(float(yaw), 1),
            "pitch": round(float(pitch), 1),
        }

        # ── Classify Attention State ──────────────────────────────────
        if avg_ear < self.EAR_CLOSED_THRESHOLD:
            self._closed_frame_count += 1
            if self._closed_frame_count >= self.CLOSED_FRAMES_LIMIT:
                self.state = AttentionState.EYES_CLOSED
            else:
                self.state = AttentionState.DROWSY

        elif avg_ear < self.EAR_DROWSY_THRESHOLD:
            self._closed_frame_count = 0
            self.state = AttentionState.DROWSY

        elif abs(yaw) > self.YAW_LIMIT or abs(pitch) > self.PITCH_LIMIT:
            self._closed_frame_count = 0
            self.state = AttentionState.LOOKING_AWAY

        else:
            self._closed_frame_count = 0
            self.state = AttentionState.ATTENTIVE

        # ── Draw Annotations ──────────────────────────────────────────
        frame = self._draw_landmarks(frame, landmarks, w, h)
        frame = self._draw_metrics(frame, avg_ear, yaw, pitch)

        return self.state, frame, metrics

    def close(self):
        self.face_mesh.close()

    # ── Private Helpers ───────────────────────────────────────────────

    def _compute_ear(self, landmarks, eye_indices, w, h) -> float:
        """
        Eye Aspect Ratio — measures how open the eye is.
        Value ~0.3 for open eye, ~0.0 for closed eye.
        """
        pts = [
            np.array([landmarks[i].x * w, landmarks[i].y * h])
            for i in eye_indices
        ]
        # Vertical distances
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C)

    def _compute_head_pose(self, landmarks, w, h):
        """
        Estimates head rotation using solvePnP.
        Returns pitch (up/down), yaw (left/right), roll.
        """
        image_points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in self.POSE_LANDMARKS
        ], dtype=np.float64)

        focal_length  = w
        camera_matrix = np.array([
            [focal_length, 0,            w / 2],
            [0,            focal_length, h / 2],
            [0,            0,            1    ]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        rot_mat, _ = cv2.Rodrigues(rvec)
        pose_mat   = cv2.hconcat([rot_mat, tvec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = euler.flatten()
        return pitch, yaw, roll

    def _draw_landmarks(self, frame, landmarks, w, h):
        """Draw green dots on eye landmarks."""
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 100), -1)
        return frame

    def _draw_metrics(self, frame, ear, yaw, pitch):
        """Draw metric readouts on bottom-left of webcam frame."""
        h = frame.shape[0]
        ear_color = (0, 255, 0) if ear > self.EAR_DROWSY_THRESHOLD else (0, 100, 255)
        cv2.putText(frame, f"EAR  : {ear:.3f}", (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 2)
        cv2.putText(frame, f"Yaw  : {yaw:+.1f}deg", (10, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 50), 2)
        cv2.putText(frame, f"Pitch: {pitch:+.1f}deg", (10, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 50), 2)
        return frame
