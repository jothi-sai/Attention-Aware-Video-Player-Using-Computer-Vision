"""
main.py  —  Attention-Aware Video Player
-----------------------------------------
Plays a local video file while monitoring the viewer's attention via webcam.

Behaviour:
  - ATTENTIVE    → video plays normally
  - DROWSY       → orange warning banner on video
  - EYES_CLOSED  → video paused + red alert overlay
  - LOOKING_AWAY → video paused + red alert overlay
  - FACE_ABSENT  → video paused + purple alert overlay

Controls:
  SPACE  → manual pause / resume
  Q / ESC → quit

Usage:
  python main.py                    # opens file dialog to pick video
  python main.py path/to/video.mp4  # direct path
"""

import sys
import os
import time
import tempfile
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import pygame

from attention_detector import AttentionDetector, AttentionState
from logger import AttentionLogger


# ─────────────────────────────────────────────────────────────────────
#  Display constants
# ─────────────────────────────────────────────────────────────────────

PANEL_W, PANEL_H = 640, 480   # size of each half of the combined window
WINDOW_TITLE = "Attention-Aware Video Player  |  SPACE=pause  Q=quit"

# BGR colours per state
STATE_COLORS = {
    AttentionState.ATTENTIVE   : (80,  200, 80),
    AttentionState.DROWSY      : (30,  165, 255),
    AttentionState.EYES_CLOSED : (40,  40,  220),
    AttentionState.LOOKING_AWAY: (40,  40,  220),
    AttentionState.FACE_ABSENT : (180, 60,  180),
}

# Alert text shown over the video when paused
ALERT_MESSAGES = {
    AttentionState.DROWSY      : "DROWSY DETECTED  —  Wake up!",
    AttentionState.EYES_CLOSED : "EYES CLOSED  —  Video Paused",
    AttentionState.LOOKING_AWAY: "LOOK AT THE SCREEN  —  Video Paused",
    AttentionState.FACE_ABSENT : "FACE NOT DETECTED  —  Video Paused",
}

# Whether this state should pause the video
PAUSE_STATES = {
    AttentionState.EYES_CLOSED,
    AttentionState.LOOKING_AWAY,
    AttentionState.FACE_ABSENT,
}


# ─────────────────────────────────────────────────────────────────────
#  Audio Manager
# ─────────────────────────────────────────────────────────────────────

class AudioManager:
    """Extracts audio from video and plays it via pygame, in sync with video."""

    def __init__(self, video_path: str):
        self._temp_audio = None
        self._has_audio   = False

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        # Extract audio to a temporary WAV file using moviepy
        try:
            print("  Extracting audio from video...")
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                print("  [Audio] No audio track found in video.")
                clip.close()
                return

            self._temp_audio = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            self._temp_audio.close()

            clip.audio.write_audiofile(
                self._temp_audio.name,
                fps=44100,
                logger=None    # suppress moviepy progress bar
            )
            clip.close()

            pygame.mixer.music.load(self._temp_audio.name)
            self._has_audio = True
            print("  [Audio] Ready.")

        except Exception as e:
            print(f"  [Audio] Could not extract audio: {e}")

    def play(self):
        if self._has_audio:
            pygame.mixer.music.play()

    def pause(self):
        if self._has_audio and pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()

    def unpause(self):
        if self._has_audio:
            pygame.mixer.music.unpause()

    def stop(self):
        if self._has_audio:
            pygame.mixer.music.stop()

    def cleanup(self):
        self.stop()
        pygame.mixer.quit()
        if self._temp_audio and os.path.exists(self._temp_audio.name):
            try:
                os.remove(self._temp_audio.name)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────
#  Helper: pick video via file dialog if no argument given
# ─────────────────────────────────────────────────────────────────────

def pick_video() -> str:
    """Open a file-picker dialog and return the chosen path."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All files", "*.*")]
    )
    root.destroy()
    return path


# ─────────────────────────────────────────────────────────────────────
#  Drawing utilities
# ─────────────────────────────────────────────────────────────────────

def draw_status_bar(frame: np.ndarray, state: AttentionState, elapsed: float,
                    paused_manual: bool) -> np.ndarray:
    """Top status bar with state label and elapsed time."""
    color  = STATE_COLORS[state]
    h, w   = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (25, 25, 25), -1)

    pause_tag = "  [MANUAL PAUSE]" if paused_manual else ""
    label = f"  {state.value.upper()}{pause_tag}"
    mm, ss = int(elapsed // 60), int(elapsed % 60)
    time_str = f"{mm:02d}:{ss:02d}"

    cv2.putText(frame, label,    (8,  26), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2)
    cv2.putText(frame, time_str, (w - 70, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 180, 180), 1)
    return frame


def draw_alert_overlay(frame: np.ndarray, state: AttentionState) -> np.ndarray:
    """Semi-transparent dark banner in the centre of the video with alert text."""
    message = ALERT_MESSAGES.get(state)
    if message is None:
        return frame

    h, w   = frame.shape[:2]
    color  = STATE_COLORS[state]

    # dark band
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 55), (w, h // 2 + 55), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # pulsing border (thin rectangle around frame)
    cv2.rectangle(frame, (4, 4), (w - 4, h - 4), color, 3)

    # centred message
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.80
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(message, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = h // 2 + th // 2
    cv2.putText(frame, message, (tx, ty), font, font_scale, color, thickness)
    return frame


def draw_panel_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Small label at bottom of each panel."""
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    return frame


def draw_attention_indicator(frame: np.ndarray, state: AttentionState) -> np.ndarray:
    """Coloured circle indicator in top-right of webcam panel."""
    color = STATE_COLORS[state]
    w = frame.shape[1]
    cv2.circle(frame, (w - 20, 20), 12, color, -1)
    cv2.circle(frame, (w - 20, 20), 12, (255, 255, 255), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────
#  Main Player Class
# ─────────────────────────────────────────────────────────────────────

class AttentionAwarePlayer:

    def __init__(self, video_path: str, webcam_index: int = 0):
        # Video source
        self.cap_video = cv2.VideoCapture(video_path)
        if not self.cap_video.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # Webcam
        self.cap_cam = cv2.VideoCapture(webcam_index)
        if not self.cap_cam.isOpened():
            raise RuntimeError("Cannot open webcam (index 0). Check your camera.")

        # Modules
        self.detector = AttentionDetector()
        self.logger   = AttentionLogger()
        self.audio    = AudioManager(video_path)

        # Playback state
        self.manual_paused    = False
        self.attention_paused = False
        self.last_vid_frame   = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        self.start_time       = time.time()

        # FPS from video (fallback 25)
        fps = self.cap_video.get(cv2.CAP_PROP_FPS)
        self.frame_delay_ms = max(1, int(1000 / (fps if fps > 0 else 25)))

        print(f"\n  Video   : {os.path.basename(video_path)}")
        print(f"  FPS     : {fps:.1f}")
        print(f"  Controls: SPACE = pause/resume | Q or ESC = quit\n")

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, PANEL_W * 2, PANEL_H)

        self.audio.play()   # ← start audio

        while True:
            elapsed = time.time() - self.start_time

            # ── Webcam frame ──────────────────────────────────────────
            ret_cam, cam_frame = self.cap_cam.read()
            if not ret_cam:
                print("[Warning] Webcam read failed.")
                break
            cam_frame = cv2.flip(cam_frame, 1)  # mirror

            # ── Attention detection ───────────────────────────────────
            state, cam_frame, metrics = self.detector.detect(cam_frame)
            self.logger.log(elapsed, state, metrics)

            # ── Decide whether attention forces a pause ───────────────
            self.attention_paused = state in PAUSE_STATES

            # ── Sync audio with playback state ────────────────────────
            currently_paused = self.manual_paused or self.attention_paused
            if currently_paused:
                self.audio.pause()
            else:
                self.audio.unpause()

            # ── Video frame ───────────────────────────────────────────
            playing = not self.manual_paused and not self.attention_paused
            if playing:
                ret_vid, vid_frame = self.cap_video.read()
                if not ret_vid:
                    print("\n  Video finished.")
                    break
                self.last_vid_frame = vid_frame
            else:
                vid_frame = self.last_vid_frame.copy()

            # ── Resize panels ─────────────────────────────────────────
            cam_panel = cv2.resize(cam_frame, (PANEL_W, PANEL_H))
            vid_panel = cv2.resize(vid_frame, (PANEL_W, PANEL_H))

            # ── Overlays ──────────────────────────────────────────────
            if self.attention_paused:
                vid_panel = draw_alert_overlay(vid_panel, state)

            if self.manual_paused and not self.attention_paused:
                # simple "PAUSED" text
                h, w = vid_panel.shape[:2]
                cv2.putText(vid_panel, "PAUSED", (w // 2 - 60, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (200, 200, 200), 3)

            # Status bars
            cam_panel = draw_status_bar(cam_panel, state, elapsed, self.manual_paused)
            vid_panel = draw_status_bar(vid_panel, state, elapsed, self.manual_paused)

            # Extra decorations
            cam_panel = draw_attention_indicator(cam_panel, state)
            cam_panel = draw_panel_label(cam_panel, "WEBCAM  |  Face & Eye Tracking")
            vid_panel = draw_panel_label(vid_panel, "VIDEO PLAYER")

            # ── Combine side-by-side ──────────────────────────────────
            divider   = np.full((PANEL_H, 3, 3), 80, dtype=np.uint8)
            combined  = np.hstack([cam_panel, divider, vid_panel])

            cv2.imshow(WINDOW_TITLE, combined)

            # ── Key handling ──────────────────────────────────────────
            key = cv2.waitKey(self.frame_delay_ms) & 0xFF
            if key in (ord('q'), 27):          # Q or ESC
                break
            elif key == ord(' '):              # SPACE = manual pause toggle
                self.manual_paused = not self.manual_paused

        self._cleanup()

    # ── Cleanup ───────────────────────────────────────────────────────

    def _cleanup(self):
        print("\n  Shutting down...")
        self.cap_video.release()
        self.cap_cam.release()
        self.detector.close()
        self.audio.cleanup()
        self.logger.save()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Get video path from CLI argument or file dialog
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("No video path given — opening file picker...")
        video_path = pick_video()

    if not video_path:
        print("No video selected. Exiting.")
        sys.exit(0)

    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    player = AttentionAwarePlayer(video_path)
    player.run()
