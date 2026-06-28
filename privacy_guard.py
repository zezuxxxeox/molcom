from __future__ import annotations

import ctypes
import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from ctypes import wintypes

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk


APP_NAME = "AI Privacy Guard"
APP_DIR = Path.home() / ".ai_privacy_guard"
CONFIG_PATH = APP_DIR / "config.json"
PROFILE_PATH = APP_DIR / "owner_profile.json"


@dataclass
class AppConfig:
    camera_index: int = 0
    danger_seconds: float = 0.2
    cooldown_seconds: float = 5.0
    match_threshold: float = 0.74
    protection_mode: str = "overlay"
    target_window_title: str = ""
    overlay_text: str = "보호 화면"
    overlay_min_seconds: float = 2.5
    safe_seconds_to_hide: float = 1.0
    show_debug_camera: bool = False


def ensure_app_dir() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    ensure_app_dir()
    if not CONFIG_PATH.exists():
        config = AppConfig()
        save_config(config)
        return config

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    defaults = asdict(AppConfig())
    defaults.update({k: v for k, v in raw.items() if k in defaults})
    return AppConfig(**defaults)


def save_config(config: AppConfig) -> None:
    ensure_app_dir()
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)


def load_profile() -> Optional[dict]:
    if not PROFILE_PATH.exists():
        return None
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(embeddings: list[np.ndarray], config: AppConfig) -> None:
    ensure_app_dir()
    if not embeddings:
        raise ValueError("No embeddings to save")

    matrix = np.vstack(embeddings).astype(np.float32)
    payload = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": "mediapipe_face_mesh_geometry_v1",
        "sample_count": int(matrix.shape[0]),
        "embedding_size": int(matrix.shape[1]),
        "mean_embedding": matrix.mean(axis=0).round(6).tolist(),
        "sample_embeddings": matrix.round(6).tolist(),
        "match_threshold": config.match_threshold,
    }
    with PROFILE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def profile_embeddings(profile: dict) -> np.ndarray:
    if "sample_embeddings" in profile:
        return np.array(profile["sample_embeddings"], dtype=np.float32)
    return np.array([profile["mean_embedding"]], dtype=np.float32)


class FaceEmbedder:
    """MediaPipe landmarks -> normalized numeric face signature."""

    KEYPOINTS = [
        1,
        4,
        6,
        10,
        33,
        46,
        52,
        55,
        61,
        70,
        78,
        80,
        81,
        82,
        84,
        88,
        91,
        93,
        105,
        107,
        127,
        132,
        133,
        136,
        144,
        145,
        146,
        148,
        149,
        150,
        152,
        153,
        154,
        155,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        172,
        173,
        176,
        178,
        181,
        185,
        191,
        234,
        246,
        249,
        251,
        263,
        276,
        282,
        285,
        291,
        300,
        308,
        310,
        311,
        312,
        314,
        318,
        321,
        323,
        334,
        336,
        356,
        361,
        362,
        365,
        373,
        374,
        375,
        377,
        378,
        379,
        380,
        381,
        382,
        384,
        385,
        386,
        387,
        388,
        389,
        390,
        397,
        398,
        400,
        402,
        405,
        409,
        415,
        454,
        466,
    ]

    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 152
    FOREHEAD = 10

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 4) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def close(self) -> None:
        self.face_mesh.close()

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        faces: list[dict] = []
        for face_landmarks in result.multi_face_landmarks:
            points = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
            bbox = self._bbox(points, w, h)
            embedding = self._embedding(points)
            if embedding is not None:
                faces.append({"embedding": embedding, "bbox": bbox})

        faces.sort(key=lambda face: _bbox_area(face["bbox"]), reverse=True)
        return faces

    def _embedding(self, points: np.ndarray) -> Optional[np.ndarray]:
        left_eye = points[self.LEFT_EYE]
        right_eye = points[self.RIGHT_EYE]
        center = (left_eye + right_eye) / 2.0
        eye_vec = right_eye - left_eye
        scale = float(np.linalg.norm(eye_vec[:2]))
        if scale < 8:
            return None

        angle = math.atan2(float(eye_vec[1]), float(eye_vec[0]))
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        selected = points[self.KEYPOINTS].copy()
        selected[:, :2] = (selected[:, :2] - center[:2]) @ rot.T / scale
        selected[:, 2] = (selected[:, 2] - center[2]) / scale

        ratios = self._distance_ratios(points, scale)
        embedding = np.concatenate(
            [
                selected[:, :2].reshape(-1),
                selected[:, 2] * 0.35,
                ratios,
            ]
        ).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-6:
            return None
        return embedding / norm

    def _distance_ratios(self, points: np.ndarray, scale: float) -> np.ndarray:
        pairs = [
            (self.NOSE, self.CHIN),
            (self.FOREHEAD, self.CHIN),
            (self.MOUTH_LEFT, self.MOUTH_RIGHT),
            (self.LEFT_EYE, self.NOSE),
            (self.RIGHT_EYE, self.NOSE),
            (self.LEFT_EYE, self.MOUTH_LEFT),
            (self.RIGHT_EYE, self.MOUTH_RIGHT),
            (234, 454),
            (132, 361),
            (10, 152),
        ]
        values = []
        for a, b in pairs:
            values.append(float(np.linalg.norm(points[a, :2] - points[b, :2]) / scale))
        return np.array(values, dtype=np.float32)

    def _bbox(self, points: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
        x1 = max(0, int(np.min(points[:, 0])))
        y1 = max(0, int(np.min(points[:, 1])))
        x2 = min(w - 1, int(np.max(points[:, 0])))
        y2 = min(h - 1, int(np.max(points[:, 1])))
        return x1, y1, x2, y2


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def best_owner_score(embedding: np.ndarray, owner_embeddings: np.ndarray) -> float:
    scores = owner_embeddings @ embedding
    return float(np.max(scores))


def draw_faces(frame: np.ndarray, faces: list[dict], labels: Iterable[str]) -> np.ndarray:
    for face, label in zip(faces, labels):
        x1, y1, x2, y2 = face["bbox"]
        color = (40, 210, 80) if label.startswith("OWNER") else (40, 40, 230)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


class WindowActivator:
    user32 = ctypes.windll.user32

    @classmethod
    def visible_windows(cls) -> list[tuple[int, str]]:
        windows: list[tuple[int, str]] = []

        def callback(hwnd: int, _lparam: int) -> bool:
            if not cls.user32.IsWindowVisible(hwnd):
                return True
            length = cls.user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            cls.user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if title:
                windows.append((hwnd, title))
            return True

        enum_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )(callback)
        cls.user32.EnumWindows(enum_proc, 0)
        return windows

    @classmethod
    def activate_by_title(cls, title_part: str) -> bool:
        needle = title_part.strip().lower()
        if not needle:
            return False

        for hwnd, title in cls.visible_windows():
            if needle in title.lower():
                cls.user32.ShowWindow(hwnd, 9)
                cls.user32.BringWindowToTop(hwnd)
                cls.user32.SetForegroundWindow(hwnd)
                return True
        return False


class ProtectionOverlay:
    def __init__(self, root: tk.Tk, config_getter: Callable[[], AppConfig]) -> None:
        self.root = root
        self.config_getter = config_getter
        self.window: Optional[tk.Toplevel] = None
        self.shown_at = 0.0

    def show(self) -> None:
        config = self.config_getter()
        if self.window and self.window.winfo_exists():
            self.window.lift()
            self.window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title(APP_NAME)
        win.configure(bg="black")
        win.attributes("-fullscreen", True)
        win.attributes("-topmost", True)
        win.protocol("WM_DELETE_WINDOW", self.hide)
        win.bind("<Escape>", lambda _event: self.hide())

        frame = tk.Frame(win, bg="black")
        frame.place(relx=0.5, rely=0.5, anchor="center")

        label = tk.Label(
            frame,
            text=config.overlay_text or "보호 화면",
            fg="white",
            bg="black",
            font=("Malgun Gothic", 52, "bold"),
        )
        label.pack(padx=48, pady=(0, 16))

        sub = tk.Label(
            frame,
            text="ESC 키로 닫기",
            fg="#bdbdbd",
            bg="black",
            font=("Malgun Gothic", 16),
        )
        sub.pack()

        self.window = win
        self.shown_at = time.monotonic()
        win.focus_force()

    def hide(self) -> None:
        if self.window and self.window.winfo_exists():
            self.window.destroy()
        self.window = None

    def maybe_hide_when_safe(self, safe_for_seconds: float) -> None:
        if not self.window or not self.window.winfo_exists():
            return
        config = self.config_getter()
        visible_for = time.monotonic() - self.shown_at
        if (
            visible_for >= config.overlay_min_seconds
            and safe_for_seconds >= config.safe_seconds_to_hide
        ):
            self.hide()


class PrivacyMonitor(threading.Thread):
    def __init__(
        self,
        config_getter: Callable[[], AppConfig],
        event_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.config_getter = config_getter
        self.event_queue = event_queue
        self.stop_event = stop_event
        try:
            profile = load_profile()
            self.owner_embeddings = (
                profile_embeddings(profile)
                if profile
                else np.empty((0, 0), dtype=np.float32)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.owner_embeddings = np.empty((0, 0), dtype=np.float32)

    def emit(self, event: str, **payload: object) -> None:
        self.event_queue.put({"event": event, **payload})

    def run(self) -> None:
        config = self.config_getter()
        if self.owner_embeddings.size == 0:
            self.emit("error", message="등록된 얼굴 프로필이 없습니다.")
            return

        cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.emit("error", message=f"카메라 {config.camera_index}번을 열 수 없습니다.")
            return

        embedder = FaceEmbedder(static_image_mode=False, max_num_faces=4)
        danger_started_at: Optional[float] = None
        safe_started_at = time.monotonic()
        last_trigger_at = 0.0
        last_status_at = 0.0

        self.emit("status", message="감시 시작")
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.emit("status", message="카메라 프레임을 읽지 못했습니다.")
                    time.sleep(0.1)
                    continue

                config = self.config_getter()
                faces = embedder.detect(frame)
                unknown_count = 0
                owner_count = 0
                labels = []

                for face in faces:
                    score = best_owner_score(face["embedding"], self.owner_embeddings)
                    if score >= config.match_threshold:
                        owner_count += 1
                        labels.append(f"OWNER {score:.2f}")
                    else:
                        unknown_count += 1
                        labels.append(f"UNKNOWN {score:.2f}")

                now = time.monotonic()
                is_danger = unknown_count >= 1

                if is_danger:
                    if danger_started_at is None:
                        danger_started_at = now
                    danger_for = now - danger_started_at
                    safe_started_at = now
                else:
                    danger_started_at = None
                    danger_for = 0.0
                    if now - safe_started_at >= config.safe_seconds_to_hide:
                        self.emit(
                            "safe",
                            safe_for_seconds=now - safe_started_at,
                            faces=len(faces),
                        )

                can_trigger = now - last_trigger_at >= config.cooldown_seconds
                if is_danger and danger_for >= config.danger_seconds and can_trigger:
                    last_trigger_at = now
                    self.emit(
                        "danger",
                        unknown_count=unknown_count,
                        owner_count=owner_count,
                        faces=len(faces),
                    )

                if now - last_status_at > 0.4:
                    status = (
                        f"얼굴 {len(faces)}명 | 본인 {owner_count}명 | "
                        f"타인 {unknown_count}명"
                    )
                    if is_danger:
                        status += f" | 위험 {danger_for:.1f}s"
                    self.emit("status", message=status)
                    last_status_at = now

                if config.show_debug_camera:
                    debug = frame.copy()
                    draw_faces(debug, faces, labels)
                    cv2.imshow(APP_NAME, debug)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.stop_event.set()
                else:
                    cv2.waitKey(1)

                time.sleep(0.02)
        finally:
            embedder.close()
            cap.release()
            cv2.destroyAllWindows()
            self.emit("stopped", message="감시 중지")


def register_owner(config: AppConfig, status_callback: Callable[[str], None]) -> bool:
    cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        messagebox.showerror(APP_NAME, f"카메라 {config.camera_index}번을 열 수 없습니다.")
        return False

    embedder = FaceEmbedder(static_image_mode=False, max_num_faces=1)
    embeddings: list[np.ndarray] = []
    needed_samples = 24
    started = time.monotonic()
    last_capture = 0.0

    status_callback("얼굴 등록 중: 카메라 창을 보고 정면/좌우로 조금씩 움직여 주세요.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                status_callback("카메라 프레임을 읽지 못했습니다.")
                time.sleep(0.1)
                continue

            faces = embedder.detect(frame)
            label = "얼굴을 화면 중앙에 맞춰 주세요"
            color = (40, 40, 230)

            now = time.monotonic()
            if faces:
                face = faces[0]
                x1, y1, x2, y2 = face["bbox"]
                area_ratio = _bbox_area(face["bbox"]) / float(frame.shape[0] * frame.shape[1])
                if area_ratio >= 0.04 and now - last_capture >= 0.12:
                    embeddings.append(face["embedding"])
                    last_capture = now
                label = f"등록 샘플 {len(embeddings)}/{needed_samples}"
                color = (40, 210, 80)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.2,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "ESC: cancel",
                (24, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(f"{APP_NAME} - owner registration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                status_callback("얼굴 등록이 취소되었습니다.")
                return False
            if len(embeddings) >= needed_samples:
                save_profile(embeddings, config)
                status_callback(f"얼굴 등록 완료: 특징값 {len(embeddings)}개 저장")
                return True
            if now - started > 45:
                status_callback("얼굴 등록 시간이 초과되었습니다.")
                return False
    finally:
        embedder.close()
        cap.release()
        cv2.destroyAllWindows()


class PrivacyGuardApp:
    def __init__(self) -> None:
        self.config = load_config()
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.geometry("760x520")
        self.root.minsize(720, 480)

        self.event_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.monitor: Optional[PrivacyMonitor] = None
        self.overlay = ProtectionOverlay(self.root, self.read_config_from_ui)

        self.vars: dict[str, tk.Variable] = {}
        self.status_var = tk.StringVar(value="대기 중")
        self.profile_var = tk.StringVar(value="")

        self.build_ui()
        self.refresh_profile_status()
        self.root.after(100, self.process_events)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Malgun Gothic", 16, "bold"))
        style.configure("Status.TLabel", font=("Malgun Gothic", 10))

        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        ttk.Label(main, text="AI Privacy Guard", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            main,
            textvariable=self.status_var,
            style="Status.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 16))

        settings = ttk.LabelFrame(main, text="설정", padding=14)
        settings.grid(row=2, column=0, sticky="nsew")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        self.vars["camera_index"] = tk.StringVar(value=str(self.config.camera_index))
        self.vars["danger_seconds"] = tk.StringVar(value=str(self.config.danger_seconds))
        self.vars["cooldown_seconds"] = tk.StringVar(value=str(self.config.cooldown_seconds))
        self.vars["match_threshold"] = tk.StringVar(value=str(self.config.match_threshold))
        self.vars["protection_mode"] = tk.StringVar(value=self.config.protection_mode)
        self.vars["target_window_title"] = tk.StringVar(value=self.config.target_window_title)
        self.vars["overlay_text"] = tk.StringVar(value=self.config.overlay_text)
        self.vars["show_debug_camera"] = tk.BooleanVar(value=self.config.show_debug_camera)

        self.add_entry(settings, 0, "카메라 번호", "camera_index")
        self.add_entry(settings, 1, "위험 지속 시간(초)", "danger_seconds")
        self.add_entry(settings, 2, "쿨다운(초)", "cooldown_seconds")
        self.add_entry(settings, 3, "본인 판정 임계값", "match_threshold")

        ttk.Label(settings, text="보호 방식").grid(row=4, column=0, sticky="w", pady=8)
        mode = ttk.Combobox(
            settings,
            textvariable=self.vars["protection_mode"],
            values=("overlay", "window"),
            state="readonly",
        )
        mode.grid(row=4, column=1, sticky="ew", padx=(8, 20), pady=8)

        ttk.Label(settings, text="가져올 창 제목").grid(row=4, column=2, sticky="w", pady=8)
        ttk.Entry(settings, textvariable=self.vars["target_window_title"]).grid(
            row=4, column=3, sticky="ew", padx=(8, 0), pady=8
        )

        ttk.Label(settings, text="오버레이 문구").grid(row=5, column=0, sticky="w", pady=8)
        ttk.Entry(settings, textvariable=self.vars["overlay_text"]).grid(
            row=5, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=8
        )

        ttk.Checkbutton(
            settings,
            text="디버그 카메라 창 표시",
            variable=self.vars["show_debug_camera"],
        ).grid(row=6, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=8)

        ttk.Label(settings, textvariable=self.profile_var).grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(16, 0)
        )

        actions = ttk.Frame(main)
        actions.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        actions.columnconfigure(5, weight=1)

        ttk.Button(actions, text="얼굴 등록", command=self.on_register).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(actions, text="설정 저장", command=self.on_save_config).grid(
            row=0, column=1, padx=(0, 8)
        )
        ttk.Button(actions, text="감시 시작", command=self.start_monitor).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(actions, text="감시 중지", command=self.stop_monitor).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Button(actions, text="보호화면 테스트", command=self.test_protection).grid(
            row=0, column=4, padx=(0, 8)
        )
        ttk.Button(actions, text="종료", command=self.on_close).grid(row=0, column=6)

    def add_entry(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        col = 0 if row % 2 == 0 else 2
        actual_row = row // 2
        ttk.Label(parent, text=label).grid(row=actual_row, column=col, sticky="w", pady=8)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(
            row=actual_row,
            column=col + 1,
            sticky="ew",
            padx=(8, 20 if col == 0 else 0),
            pady=8,
        )

    def read_config_from_ui(self) -> AppConfig:
        def f(key: str, default: float) -> float:
            try:
                return float(str(self.vars[key].get()).strip())
            except ValueError:
                return default

        def i(key: str, default: int) -> int:
            try:
                return int(str(self.vars[key].get()).strip())
            except ValueError:
                return default

        return AppConfig(
            camera_index=i("camera_index", self.config.camera_index),
            danger_seconds=max(0.2, f("danger_seconds", self.config.danger_seconds)),
            cooldown_seconds=max(0.5, f("cooldown_seconds", self.config.cooldown_seconds)),
            match_threshold=min(0.99, max(0.4, f("match_threshold", self.config.match_threshold))),
            protection_mode=str(self.vars["protection_mode"].get()),
            target_window_title=str(self.vars["target_window_title"].get()).strip(),
            overlay_text=str(self.vars["overlay_text"].get()).strip() or "보호 화면",
            show_debug_camera=bool(self.vars["show_debug_camera"].get()),
        )

    def on_save_config(self) -> None:
        self.config = self.read_config_from_ui()
        save_config(self.config)
        self.set_status("설정 저장 완료")

    def on_register(self) -> None:
        if self.monitor and self.monitor.is_alive():
            messagebox.showinfo(APP_NAME, "감시를 중지한 뒤 얼굴을 등록해 주세요.")
            return
        self.config = self.read_config_from_ui()
        save_config(self.config)
        ok = register_owner(self.config, self.set_status)
        if ok:
            self.refresh_profile_status()

    def refresh_profile_status(self) -> None:
        profile = load_profile()
        if profile:
            self.profile_var.set(
                f"등록된 얼굴: 샘플 {profile.get('sample_count', '?')}개 | "
                f"생성 {profile.get('created_at', '?')} | 저장 위치 {PROFILE_PATH}"
            )
        else:
            self.profile_var.set("등록된 얼굴 없음: 먼저 얼굴 등록을 실행하세요.")

    def start_monitor(self) -> None:
        if self.monitor and self.monitor.is_alive():
            self.set_status("이미 감시 중입니다.")
            return
        if not load_profile():
            messagebox.showinfo(APP_NAME, "먼저 본인 얼굴을 등록해 주세요.")
            return

        self.config = self.read_config_from_ui()
        save_config(self.config)
        self.stop_event = threading.Event()
        self.monitor = PrivacyMonitor(self.read_config_from_ui, self.event_queue, self.stop_event)
        self.monitor.start()

    def stop_monitor(self) -> None:
        if self.monitor and self.monitor.is_alive():
            self.stop_event.set()
            self.set_status("감시 중지 요청")
        else:
            self.set_status("감시가 실행 중이 아닙니다.")

    def test_protection(self) -> None:
        self.handle_protection()

    def process_events(self) -> None:
        try:
            while True:
                item = self.event_queue.get_nowait()
                event = item.get("event")
                if event == "status":
                    self.set_status(str(item.get("message", "")))
                elif event == "danger":
                    self.set_status(
                        f"타인 감지: 얼굴 {item.get('faces')}명, "
                        f"타인 {item.get('unknown_count')}명"
                    )
                    self.handle_protection()
                elif event == "safe":
                    self.overlay.maybe_hide_when_safe(float(item.get("safe_for_seconds", 0.0)))
                elif event == "error":
                    self.set_status(str(item.get("message", "")))
                    messagebox.showerror(APP_NAME, str(item.get("message", "")))
                elif event == "stopped":
                    self.set_status(str(item.get("message", "감시 중지")))
        except queue.Empty:
            pass
        self.root.after(100, self.process_events)

    def handle_protection(self) -> None:
        config = self.read_config_from_ui()
        if config.protection_mode == "window":
            if WindowActivator.activate_by_title(config.target_window_title):
                return
        self.overlay.show()

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.root.update_idletasks()

    def on_close(self) -> None:
        self.stop_event.set()
        self.overlay.hide()
        self.root.after(150, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ensure_app_dir()
    app = PrivacyGuardApp()
    app.run()


if __name__ == "__main__":
    main()
