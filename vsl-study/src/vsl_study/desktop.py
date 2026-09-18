"""Windows desktop app with native file and folder pickers. No browser server."""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import traceback
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import tkinter as tk

from vsl_study.capture_meta import apply_desktop_capture_event
from vsl_study.models import ProcessSettings
from vsl_study.transcribe import SettingsError, validate_model_language

BG = "#F3F0EA"
SURFACE = "#FFFCF7"
INK = "#1A1714"
MUTED = "#6B6560"
LINE = "#D9D1C7"
ACCENT = "#1E3A2F"
ACCENT_HOVER = "#2C5344"
ACCENT_OFF = "#8A968F"
ON_ACCENT = "#F6F3EC"
LOG_BG = "#EBE6DD"

SPEECH_QUALITY_OPTIONS = [
    ("Recommended — best balance for most videos", "recommended"),
    ("Faster — finishes sooner, may miss a few words", "faster"),
    ("More accurate — catches more, takes longer", "accurate"),
]
SPEECH_OUTPUT_OPTIONS = [
    ("Write the spoken words as they are", "transcribe"),
    ("Translate the speech into English", "translate"),
]
LANGUAGE_OPTIONS = [
    ("English", "en"),
    ("Spanish", "es"),
    ("Portuguese", "pt"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Russian", "ru"),
    ("Ukrainian", "uk"),
    ("Turkish", "tr"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Indonesian", "id"),
    ("Vietnamese", "vi"),
    ("Thai", "th"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese", "zh"),
    ("Greek", "el"),
    ("Hebrew", "he"),
    ("Swedish", "sv"),
    ("Norwegian", "no"),
    ("Danish", "da"),
    ("Finnish", "fi"),
    ("Czech", "cs"),
    ("Hungarian", "hu"),
    ("Romanian", "ro"),
    ("Detect automatically", "auto"),
]
STAGE_TITLES = {
    "doctor": "",
    "run": "Starting",
    "inspect": "Checking the video",
    "audio": "Preparing the soundtrack",
    "transcribe": "Writing down the spoken words",
    "scenes": "Finding when the picture changes",
    "frames": "Taking screenshots",
    "ocr": "Reading text on screen",
    "export": "Saving your folder",
    "done": "Finished",
    "error": "Something went wrong",
    "recording": "Recording",
    "saving": "Saving",
    "validate": "Validating",
}
MESSAGE_REWRITES = (
    ("Ready.", "Choose a video and a folder, then create a study folder."),
    ("Loading transcription libraries", "Loading speech recognition. The first run can take a minute."),
    ("Validating input and inspecting streams", "Opening the video and checking that it can be read."),
    ("Using cached transcript", "This video was already transcribed. Using that saved transcript."),
    ("Using cached scene list", "Using the scene list already saved for this video."),
    ("Using cached screenshots", "Using screenshots already saved for this video."),
    ("Using cached on-screen text", "Using on-screen text already saved for this video."),
    (
        "Writing transcripts, reports, and evidence folders",
        "Writing the transcript, screenshots, and report into your folder.",
    ),
    ("Loading OCR models", "Loading the on-screen text reader. The first time can take a minute."),
    ("Reading on-screen text from screenshots", "Reading prices, headlines, and other text in the pictures."),
    ("Detecting scenes with PySceneDetect", "Looking for moments when the picture changes."),
)
VIDEO_TYPES = [
    ("Video files", "*.mp4 *.mov *.mkv *.webm *.MP4 *.MOV *.MKV *.WEBM"),
    ("MP4", "*.mp4 *.MP4"),
    ("MOV", "*.mov *.MOV"),
    ("MKV", "*.mkv *.MKV"),
    ("WebM", "*.webm *.WEBM"),
    ("All files", "*.*"),
]


def _labels(options: list[tuple[str, str]]) -> list[str]:
    return [label for label, _value in options]


def _label_for(options: list[tuple[str, str]], value: str, fallback: str) -> str:
    for label, stored in options:
        if stored == value:
            return label
    return fallback


def _value_for(options: list[tuple[str, str]], label: str, fallback: str) -> str:
    for text, stored in options:
        if text == label:
            return stored
    return fallback


def _quality_from_model(model: str) -> str:
    name = (model or "").strip().lower()
    if name.startswith(("tiny", "base")) or name == "turbo":
        return "faster"
    if name.startswith(("medium", "large")):
        return "accurate"
    return "recommended"


def _quality_from_prefs(prefs: dict) -> str:
    raw = str(prefs.get("speech_quality") or "").strip().lower()
    if raw in {"recommended", "faster", "accurate"}:
        return raw
    return _quality_from_model(str(prefs.get("model") or "small.en"))


def _whisper_model(quality: str, language: str, task: str = "transcribe") -> str:
    """Map UI quality to a Whisper model.

    Explicit English uses .en models. Automatic detection and other languages
    use multilingual models so Whisper can identify the spoken language.
    Translation always needs a multilingual model.
    """
    use_english_only = language == "en" and task != "translate"
    if quality == "faster":
        return "base.en" if use_english_only else "base"
    if quality == "accurate":
        return "medium.en" if use_english_only else "medium"
    return "small.en" if use_english_only else "small"


def format_job_completion(result: dict) -> tuple[str, str]:
    """Status line and log line after process_video. Visual outputs may exist without speech."""
    job = str(result.get("job") or "")
    status = str(result.get("transcript_status") or "complete")
    if status == "complete":
        return "Finished. Your folder is ready.", f"Wrote evidence package to {job}"
    if status == "failed":
        return (
            "Study folder saved, but the spoken words could not be written down.",
            f"Visual outputs are in {job}. Speech recognition failed.",
        )
    if status == "skipped":
        return (
            "Study folder saved. There was no speech track to write down.",
            f"Visual outputs are in {job}. Transcription was skipped.",
        )
    return (
        "Study folder saved, with a partial transcript.",
        f"Wrote evidence package to {job} (transcript: {status}).",
    )


def unique_job_dir(dest: Path, suffix: str) -> Path:
    dest = dest.expanduser()
    if not (dest / "job.json").exists():
        return dest
    parent = dest.parent
    candidate = parent / f"{dest.name}-{suffix}"
    n = 2
    while (candidate / "job.json").exists():
        candidate = parent / f"{dest.name}-{suffix}-{n}"
        n += 1
    return candidate


def _language_label(raw: str) -> str:
    text = (raw or "en").strip()
    for label, code in LANGUAGE_OPTIONS:
        if text.casefold() in {label.casefold(), code.casefold()}:
            return label
    return "English"


def _friendly_text(stage: str, message: str) -> str:
    text = message.strip()
    for prefix, rewrite in MESSAGE_REWRITES:
        if text == prefix or text.startswith(prefix):
            text = rewrite
            break
    else:
        if stage == "ocr" and "/" in text.split(" ", 1)[0]:
            count = text.split(" ", 1)[0]
            text = f"Reading on-screen text ({count} pictures)."
        elif text.startswith("Wrote evidence package to "):
            text = "Finished. Your folder is ready:\n" + text.removeprefix("Wrote evidence package to ")
        elif "Not found on PATH" in text:
            text = (
                "This computer is missing FFmpeg, which VSL Study needs to open videos. "
                "Install FFmpeg, then close this app and open it again."
            )
    title = STAGE_TITLES.get(stage, "")
    if not title or text.lower().startswith(title.lower()):
        return text
    if stage in {"done", "error", "doctor", "run"}:
        return text
    return f"{title} — {text}"


def _config_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    folder = base / "VSL Study"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "desktop.json"


def _load_prefs() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_prefs(data: dict) -> None:
    try:
        _config_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def enable_windows_dpi_awareness() -> None:
    """Per-monitor DPI so the window tracks this display's scale, not a fixed 96 DPI."""
    if os.name != "nt":
        return
    import ctypes

    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


@dataclass
class MonitorLayout:
    work_x: int
    work_y: int
    work_w: int
    work_h: int
    dpi: int

    @property
    def factor(self) -> float:
        return max(0.75, self.dpi / 96.0)

    def px(self, logical: int) -> int:
        return max(1, int(round(logical * self.factor)))


def clamp_window_to_work_area(
    x: int,
    y: int,
    width: int,
    height: int,
    layout: MonitorLayout,
) -> tuple[int, int]:
    """Keep a window fully inside the monitor work area (taskbar excluded)."""
    max_x = layout.work_x + max(0, layout.work_w - width)
    max_y = layout.work_y + max(0, layout.work_h - height)
    return min(max(int(x), layout.work_x), max_x), min(max(int(y), layout.work_y), max_y)


def read_monitor_layout(hwnd: int | None = None) -> MonitorLayout:
    if os.name == "nt":
        try:
            return _windows_monitor_layout(hwnd)
        except Exception:
            pass
    return MonitorLayout(0, 0, 1280, 720, 96)


def _windows_monitor_layout(hwnd: int | None) -> MonitorLayout:
    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
        ]

    user32 = ctypes.windll.user32
    MONITOR_DEFAULTTONEAREST = 2
    if hwnd:
        hmon = user32.MonitorFromWindow(wintypes.HWND(hwnd), MONITOR_DEFAULTTONEAREST)
    else:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        hmon = user32.MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST)

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    if not user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
        raise OSError("GetMonitorInfoW failed")
    work = info.rcWork
    dpi = 96
    try:
        dpi_x = ctypes.c_uint()
        dpi_y = ctypes.c_uint()
        ctypes.windll.shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
        dpi = int(dpi_x.value or 96)
    except Exception:
        try:
            dpi = int(user32.GetDpiForWindow(wintypes.HWND(hwnd))) if hwnd else dpi
        except Exception:
            dpi = 96
    return MonitorLayout(
        work_x=int(work.left),
        work_y=int(work.top),
        work_w=int(work.right - work.left),
        work_h=int(work.bottom - work.top),
        dpi=max(96, dpi) if dpi > 0 else 96,
    )


def apply_tk_scaling(root: Tk, dpi: int) -> None:
    try:
        root.tk.call("tk", "scaling", max(1.0, dpi / 72.0))
    except Exception:
        pass


def _apply_clam_theme(root: Tk) -> None:
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure(
        ".",
        background=BG,
        foreground=INK,
        fieldbackground=SURFACE,
        bordercolor=LINE,
        lightcolor=SURFACE,
        darkcolor=LINE,
        troughcolor=LOG_BG,
        font=("Segoe UI", 10),
    )
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
    style.configure("TCheckbutton", background=BG, foreground=INK, font=("Segoe UI", 10))
    style.configure(
        "TCombobox",
        fieldbackground=SURFACE,
        background=SURFACE,
        foreground=INK,
        arrowcolor=INK,
        bordercolor=LINE,
        lightcolor=LINE,
        darkcolor=LINE,
        padding=12,
        arrowsize=22,
        font=("Segoe UI", 11),
    )
    style.configure(
        "TSpinbox",
        fieldbackground=SURFACE,
        background=SURFACE,
        foreground=INK,
        arrowcolor=INK,
        bordercolor=LINE,
        padding=12,
        arrowsize=22,
        font=("Segoe UI", 11),
    )
    style.configure("TEntry", fieldbackground=SURFACE, foreground=INK, bordercolor=LINE, padding=12, font=("Segoe UI", 11))
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE)], foreground=[("readonly", INK)])


class BigDropdown(tk.Frame):
    """A select control with a large chevron. ttk's built-in arrow stays tiny on Windows."""

    def __init__(
        self,
        parent: tk.Misc,
        variable: StringVar,
        options: list,
        on_change=None,  # noqa: ANN001
    ) -> None:
        super().__init__(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor=ACCENT,
            cursor="hand2",
        )
        self.variable = variable
        self.choices = _labels(options)
        self.on_change = on_change

        self.value = tk.Label(
            self,
            textvariable=variable,
            bg=SURFACE,
            fg=INK,
            font=("Segoe UI", 11),
            anchor="w",
            justify="left",
            cursor="hand2",
        )
        self.value.pack(side="left", fill="both", expand=True, padx=(14, 8), pady=12)
        self.arrow = tk.Label(
            self,
            text="▾",
            bg=SURFACE,
            fg=INK,
            font=("Segoe UI", 20),
            width=3,
            cursor="hand2",
        )
        self.arrow.pack(side="right", fill="y", padx=(0, 4))
        for widget in (self, self.value, self.arrow):
            widget.bind("<Button-1>", self._open)
            widget.bind("<Enter>", self._hover)
            widget.bind("<Leave>", self._unhover)

    def _hover(self, _event=None) -> None:  # noqa: ANN001
        for widget in (self, self.value, self.arrow):
            widget.configure(bg=LOG_BG)
        self.configure(highlightbackground=ACCENT)

    def _unhover(self, _event=None) -> None:  # noqa: ANN001
        for widget in (self, self.value, self.arrow):
            widget.configure(bg=SURFACE)
        self.configure(highlightbackground=LINE)

    def _open(self, _event=None) -> None:  # noqa: ANN001
        menu = tk.Menu(
            self,
            tearoff=0,
            font=("Segoe UI", 12),
            bg=SURFACE,
            fg=INK,
            activebackground=ACCENT,
            activeforeground=ON_ACCENT,
            bd=0,
            relief="flat",
        )
        for label in self.choices:
            menu.add_command(label=label, command=lambda text=label: self._pick(text))
        self.update_idletasks()
        try:
            menu.tk_popup(self.winfo_rootx(), self.winfo_rooty() + self.winfo_height())
        finally:
            menu.grab_release()

    def _pick(self, label: str) -> None:
        if label == self.variable.get():
            return
        self.variable.set(label)
        if self.on_change:
            self.on_change()


class BigCheck(tk.Canvas):
    """A large square checkbox. Native ttk indicators stay tiny on Windows."""

    def __init__(
        self,
        parent: tk.Misc,
        variable: BooleanVar,
        size: int = 28,
        command=None,  # noqa: ANN001
    ) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=BG,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            takefocus=1,
        )
        self.variable = variable
        self.box_size = size
        self.command = command
        self.bind("<Button-1>", self._toggle)
        self.bind("<space>", self._toggle)
        self.bind("<Return>", self._toggle)
        self.variable.trace_add("write", lambda *_args: self._draw())
        self._draw()

    def set_size(self, size: int) -> None:
        self.box_size = size
        self.configure(width=size, height=size)
        self._draw()

    def _toggle(self, _event=None) -> str:  # noqa: ANN001
        self.variable.set(not bool(self.variable.get()))
        if self.command:
            self.command()
        return "break"

    def _draw(self) -> None:
        self.delete("all")
        s = float(self.box_size)
        m = max(1.5, s * 0.08)
        stroke = max(2, int(s / 12))
        on = bool(self.variable.get())
        if on:
            self.create_rectangle(m, m, s - m, s - m, fill=ACCENT, outline=ACCENT, width=stroke)
            self.create_line(
                s * 0.24,
                s * 0.50,
                s * 0.42,
                s * 0.70,
                s * 0.76,
                s * 0.30,
                fill=ON_ACCENT,
                width=max(2.5, s / 8),
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
        else:
            self.create_rectangle(m, m, s - m, s - m, fill=SURFACE, outline=INK, width=stroke)


class VSLStudyApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("VSL Study")
        self.root.configure(bg=BG)
        self.layout = read_monitor_layout()
        self._last_dpi = self.layout.dpi
        apply_tk_scaling(root, self.layout.dpi)
        _apply_clam_theme(root)
        self._place_on_monitor(self.layout)

        prefs = _load_prefs()
        self.input_var = StringVar(value=str(prefs.get("input") or ""))
        self.output_var = StringVar(value=str(prefs.get("output") or ""))
        quality = _quality_from_prefs(prefs)
        self.speech_var = StringVar(
            value=_label_for(SPEECH_QUALITY_OPTIONS, quality, SPEECH_QUALITY_OPTIONS[0][0])
        )
        self.language_var = StringVar(value=_language_label(str(prefs.get("language") or "en")))
        task = str(prefs.get("task") or "transcribe")
        self.speech_out_var = StringVar(
            value=_label_for(SPEECH_OUTPUT_OPTIONS, task, SPEECH_OUTPUT_OPTIONS[0][0])
        )
        self.ocr_var = BooleanVar(value=bool(prefs["ocr_on"]) if "ocr_on" in prefs else False)
        self.status_var = StringVar(value="Choose a video and a folder.")
        self.running = False
        self.capturing = False
        self._capture_server = None
        self._processed_captures: set[str] = set()
        self._capture_session_id: str | None = None
        self._capture_generation = 0
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.last_job: str | None = None
        self._log_placeholder = True
        self._options: tk.Toplevel | None = None

        self._build()
        self._style_for_dpi()
        self._note_missing_ffmpeg()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Configure>", self._on_configure)
        self.root.after(0, self._drain_queue)
        self.root.after(200, self._refresh_monitor)

    def _hwnd(self) -> int | None:
        try:
            return int(self.root.winfo_id())
        except Exception:
            return None

    def _place_on_monitor(self, layout: MonitorLayout) -> None:
        width = min(layout.px(840), int(layout.work_w * 0.58))
        height = min(layout.px(720), int(layout.work_h * 0.72))
        width = max(layout.px(640), width)
        height = max(layout.px(560), height)
        x = layout.work_x + max(0, (layout.work_w - width) // 2)
        y = layout.work_y + max(0, (layout.work_h - height) // 2)
        self.root.minsize(layout.px(600), layout.px(520))
        self.root.maxsize(layout.work_w, layout.work_h)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _style_for_dpi(self) -> None:
        _apply_clam_theme(self.root)
        if hasattr(self, "title_label"):
            self.title_label.configure(font=("Segoe UI", 26, "bold"))
        if hasattr(self, "subtitle"):
            self.subtitle.configure(font=("Segoe UI", 11))
        if hasattr(self, "status_label"):
            self.status_label.configure(font=("Segoe UI", 11))
        if hasattr(self, "log"):
            self.log.configure(font=("Segoe UI", 10))
        if hasattr(self, "run_btn"):
            self.run_btn.configure(font=("Segoe UI", 11, "bold"))
        if hasattr(self, "ocr_mark"):
            self.ocr_mark.set_size(self.layout.px(28))
        self._refresh_ocr()
        self._update_wrap()

    def _update_wrap(self) -> None:
        self.root.update_idletasks()
        inner = max(280, self.root.winfo_width() - self.layout.px(72))
        if hasattr(self, "subtitle"):
            self.subtitle.configure(wraplength=inner)
        if hasattr(self, "status_label"):
            self.status_label.configure(wraplength=inner)
        if hasattr(self, "ocr_caption"):
            self.ocr_caption.configure(wraplength=max(200, inner - self.layout.px(80)))

    def _refresh_monitor(self) -> None:
        layout = read_monitor_layout(self._hwnd())
        if layout.dpi != self._last_dpi or abs(layout.work_w - self.layout.work_w) > 32:
            self.layout = layout
            self._last_dpi = layout.dpi
            apply_tk_scaling(self.root, layout.dpi)
            self.root.minsize(layout.px(600), layout.px(520))
            self.root.maxsize(layout.work_w, layout.work_h)
            self._style_for_dpi()
        self._update_wrap()

    def _on_configure(self, event) -> None:  # noqa: ANN001
        if event.widget is not self.root:
            return
        self._update_wrap()

    def _build(self) -> None:
        pad = self.layout.px(36)
        main = tk.Frame(self.root, bg=BG, padx=pad, pady=pad)
        main.pack(fill="both", expand=True)
        self.main = main

        header = tk.Frame(main, bg=BG)
        header.pack(fill="x")
        self.title_label = tk.Label(
            header, text="VSL Study", bg=BG, fg=INK, font=("Segoe UI", 26, "bold"), anchor="w"
        )
        self.title_label.pack(side="left")
        options_btn = self._outline_button(header, "Options", self._open_options)
        options_btn.pack(side="right")

        self.subtitle = tk.Label(
            main,
            text="A transcript, screenshots, and a study folder from a video on this PC. Nothing is uploaded.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 11),
            anchor="w",
            justify="left",
        )
        self.subtitle.pack(fill="x", pady=(self.layout.px(6), self.layout.px(28)))

        card = tk.Frame(
            main,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor=LINE,
        )
        card.pack(fill="x")
        inner = tk.Frame(card, bg=SURFACE, padx=self.layout.px(22), pady=self.layout.px(20))
        inner.pack(fill="x")
        self._file_field(inner, "Video", self.input_var, "Choose file", self._browse_input).pack(fill="x")
        record_row = tk.Frame(inner, bg=SURFACE)
        record_row.pack(fill="x", pady=(self.layout.px(12), 0))
        self.record_btn = self._outline_button(record_row, "Record a browser tab", self._record_tab)
        self.record_btn.pack(side="left")
        self.cancel_record_btn = self._outline_button(record_row, "Cancel recording", self._cancel_recording)
        self.cancel_record_btn.pack(side="left", padx=(self.layout.px(8), 0))
        self.cancel_record_btn.configure(state="disabled")
        tk.Label(
            record_row,
            text="Opens Chrome or Edge. Keep this window open. Capture takes real playback time.",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=self.layout.px(220),
        ).pack(side="left", padx=(self.layout.px(12), 0))
        tk.Frame(inner, bg=LINE, height=1).pack(fill="x", pady=self.layout.px(16))
        self._file_field(inner, "Save folder", self.output_var, "Choose folder", self._browse_output).pack(fill="x")

        ocr_row = tk.Frame(main, bg=BG)
        ocr_row.pack(fill="x", pady=(self.layout.px(22), 0))
        self.ocr_mark = BigCheck(
            ocr_row,
            self.ocr_var,
            size=self.layout.px(28),
            command=self._refresh_ocr,
        )
        self.ocr_mark.pack(side="left", anchor="n", pady=(self.layout.px(1), 0))
        ocr_text = tk.Frame(ocr_row, bg=BG)
        ocr_text.pack(side="left", fill="x", expand=True, padx=(self.layout.px(12), 0))
        self.ocr_title = tk.Label(
            ocr_text, text="Copy on-screen text", bg=BG, fg=INK, font=("Segoe UI", 11), anchor="w", cursor="hand2"
        )
        self.ocr_title.pack(fill="x")
        self.ocr_caption = tk.Label(
            ocr_text,
            text="Turn this on to include prices, headlines, and captions from the pictures.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            cursor="hand2",
        )
        self.ocr_caption.pack(fill="x")
        for widget in (self.ocr_title, self.ocr_caption, ocr_text, ocr_row):
            widget.bind("<Button-1>", self._toggle_ocr)
        self._refresh_ocr()

        self.run_btn = tk.Button(
            main,
            text="Create study folder",
            command=self._run,
            bg=ACCENT,
            fg=ON_ACCENT,
            activebackground=ACCENT_HOVER,
            activeforeground=ON_ACCENT,
            disabledforeground=ON_ACCENT,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 11, "bold"),
            padx=self.layout.px(18),
            pady=self.layout.px(14),
        )
        self.run_btn.pack(fill="x", pady=(self.layout.px(26), self.layout.px(10)))
        self.run_btn.bind("<Enter>", lambda _e: self._hover_run(True))
        self.run_btn.bind("<Leave>", lambda _e: self._hover_run(False))

        links = tk.Frame(main, bg=BG)
        links.pack(fill="x")
        self._ghost_button(links, "Open save folder", self._open_output).pack(side="left")
        tk.Label(links, text="·", bg=BG, fg=LINE, font=("Segoe UI", 12)).pack(side="left", padx=self.layout.px(8))
        self._ghost_button(links, "Open report", self._open_report).pack(side="left")

        tk.Frame(main, bg=LINE, height=1).pack(fill="x", pady=(self.layout.px(24), self.layout.px(16)))

        self.status_label = tk.Label(
            main,
            textvariable=self.status_var,
            bg=BG,
            fg=INK,
            font=("Segoe UI", 11),
            anchor="w",
            justify="left",
        )
        self.status_label.pack(fill="x")

        self.log = ScrolledText(
            main,
            height=8,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 10),
            bg=LOG_BG,
            fg=MUTED,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=self.layout.px(12),
            pady=self.layout.px(10),
        )
        self.log.pack(fill="both", expand=True, pady=(self.layout.px(10), 0))
        self._set_log_placeholder("Progress will appear here after you create a study folder.")

    def _outline_button(self, parent: tk.Misc, text: str, command) -> tk.Button:  # noqa: ANN001
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=SURFACE,
            fg=INK,
            activebackground=LOG_BG,
            activeforeground=INK,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor=ACCENT,
            cursor="hand2",
            font=("Segoe UI", 11),
            padx=self.layout.px(16),
            pady=self.layout.px(8),
        )
        btn.bind("<Enter>", lambda _e, b=btn: b.configure(bg=LOG_BG))
        btn.bind("<Leave>", lambda _e, b=btn: b.configure(bg=SURFACE))
        return btn

    def _ghost_button(self, parent: tk.Misc, text: str, command) -> tk.Button:  # noqa: ANN001
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=parent.cget("bg"),
            fg=MUTED,
            activebackground=parent.cget("bg"),
            activeforeground=INK,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 10),
            padx=0,
            pady=2,
        )
        btn.bind("<Enter>", lambda _e, b=btn: b.configure(fg=INK))
        btn.bind("<Leave>", lambda _e, b=btn: b.configure(fg=MUTED))
        return btn

    def _file_field(
        self,
        parent: tk.Misc,
        label: str,
        variable: StringVar,
        action: str,
        command,
    ) -> tk.Frame:
        block = tk.Frame(parent, bg=SURFACE)
        tk.Label(
            block,
            text=label.upper(),
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x")
        row = tk.Frame(block, bg=SURFACE)
        row.pack(fill="x", pady=(self.layout.px(6), 0))
        entry = tk.Entry(
            row,
            textvariable=variable,
            font=("Segoe UI", 11),
            bg=SURFACE,
            fg=INK,
            insertbackground=INK,
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=self.layout.px(8))
        self._outline_button(row, action, command).pack(side="right", padx=(self.layout.px(12), 0))
        return block

    def _hover_run(self, inside: bool) -> None:
        if self.running:
            return
        self.run_btn.configure(bg=ACCENT_HOVER if inside else ACCENT)

    def _toggle_ocr(self, _event=None) -> None:  # noqa: ANN001
        self.ocr_var.set(not bool(self.ocr_var.get()))
        self._refresh_ocr()

    def _refresh_ocr(self) -> None:
        if not hasattr(self, "ocr_title"):
            return
        on = bool(self.ocr_var.get())
        self.ocr_title.configure(fg=INK if on else MUTED)
        if hasattr(self, "ocr_mark"):
            self.ocr_mark._draw()

    def _set_running(self, running: bool) -> None:
        self.running = running
        if running:
            self.run_btn.configure(state="disabled", text="Working…", bg=ACCENT_OFF, cursor="arrow")
            self._sync_capture_buttons()
        else:
            self.run_btn.configure(state="normal", text="Create study folder", bg=ACCENT, cursor="hand2")
            self._sync_capture_buttons()

    def _set_log_placeholder(self, text: str) -> None:
        self._log_placeholder = True
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", text)
        self.log.configure(state="disabled", fg=MUTED)

    def _place_on_visible_screen(
        self,
        win: tk.Misc,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        try:
            if not win.winfo_exists():
                return
        except tk.TclError:
            return
        win.update_idletasks()
        width = int(width if width is not None else max(win.winfo_width(), win.winfo_reqwidth()))
        height = int(height if height is not None else max(win.winfo_height(), win.winfo_reqheight()))
        layout = read_monitor_layout(self._hwnd())
        pref_x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
        pref_y = layout.work_y + max(0, (layout.work_h - height) // 2)
        x, y = clamp_window_to_work_area(pref_x, pref_y, width, height, layout)
        win.geometry(f"+{x}+{y}")
        win.update_idletasks()
        try:
            if not win.winfo_exists():
                return
        except tk.TclError:
            return
        overflow_x = win.winfo_rootx() + win.winfo_width() - (layout.work_x + layout.work_w)
        overflow_y = win.winfo_rooty() + win.winfo_height() - (layout.work_y + layout.work_h)
        dx = max(0, layout.work_x - win.winfo_rootx()) - max(0, overflow_x)
        dy = max(0, layout.work_y - win.winfo_rooty()) - max(0, overflow_y)
        if dx or dy:
            win.geometry(f"+{x + dx}+{y + dy}")

    def _open_options(self) -> None:
        if self._options is not None and self._options.winfo_exists():
            self._place_on_visible_screen(self._options)
            self._options.lift()
            self._options.focus_force()
            return
        win = tk.Toplevel(self.root)
        win.withdraw()
        win.title("Options")
        win.configure(bg=BG)
        win.transient(self.root)
        self._options = win
        pad = self.layout.px(28)
        body = tk.Frame(win, bg=BG, padx=pad, pady=pad)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Options", bg=BG, fg=INK, font=("Segoe UI", 18, "bold"), anchor="w").pack(fill="x")
        tk.Label(
            body,
            text="Language and how careful the transcript should be.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=self.layout.px(460),
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(self.layout.px(6), self.layout.px(20)))

        self._option_block(
            body,
            "Spoken language",
            "Pick the language people are speaking in the video.",
            lambda parent: self._dropdown(parent, self.language_var, LANGUAGE_OPTIONS),
        )
        self._option_block(
            body,
            "Speech output",
            "Keep the original language unless you want an English translation.",
            lambda parent: self._dropdown(parent, self.speech_out_var, SPEECH_OUTPUT_OPTIONS),
        )
        self._option_block(
            body,
            "Speech quality",
            "Recommended is the right choice unless a run is too slow or the transcript is sloppy.",
            lambda parent: self._dropdown(parent, self.speech_var, SPEECH_QUALITY_OPTIONS),
        )

        done = tk.Button(
            body,
            text="Done",
            command=win.destroy,
            bg=ACCENT,
            fg=ON_ACCENT,
            activebackground=ACCENT_HOVER,
            activeforeground=ON_ACCENT,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 12, "bold"),
            padx=self.layout.px(18),
            pady=self.layout.px(12),
        )
        done.pack(fill="x", pady=(self.layout.px(10), 0))
        win.update_idletasks()
        width = max(self.layout.px(520), win.winfo_reqwidth())
        height = win.winfo_reqheight() + self.layout.px(8)
        win.minsize(self.layout.px(480), min(height, self.layout.px(280)))
        win.geometry(f"{width}x{height}")
        win.deiconify()
        win.lift()
        win.focus_force()
        win.update_idletasks()
        self._place_on_visible_screen(win, width, height)
        win.after_idle(lambda: self._place_on_visible_screen(win, width, height))

    def _dropdown(
        self,
        parent: tk.Misc,
        variable: StringVar,
        options: list,
        on_change=None,  # noqa: ANN001
    ) -> BigDropdown:
        return BigDropdown(parent, variable, options, on_change=on_change)

    def _speech_settings(self) -> tuple[str, str, str]:
        language = _value_for(LANGUAGE_OPTIONS, self.language_var.get(), "en")
        task = _value_for(SPEECH_OUTPUT_OPTIONS, self.speech_out_var.get(), "transcribe")
        quality = _value_for(SPEECH_QUALITY_OPTIONS, self.speech_var.get(), "recommended")
        model = _whisper_model(quality, language, task)
        return model, language, task

    def _option_block(self, parent: tk.Misc, title: str, caption: str, factory) -> None:  # noqa: ANN001
        block = tk.Frame(parent, bg=BG)
        block.pack(fill="x", pady=(0, self.layout.px(14)))
        tk.Label(block, text=title, bg=BG, fg=INK, font=("Segoe UI", 10), anchor="w").pack(fill="x")
        widget = factory(block)
        widget.pack(fill="x", pady=(self.layout.px(4), self.layout.px(4)))
        tk.Label(
            block,
            text=caption,
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=self.layout.px(460),
            anchor="w",
            justify="left",
        ).pack(fill="x")

    def _browse_input(self) -> None:
        initial = self.input_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial and Path(initial).exists() else str(Path.home())
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose the video",
            initialdir=initial_dir,
            filetypes=VIDEO_TYPES,
        )
        if not path:
            return
        self.input_var.set(path)
        current_out = self.output_var.get().strip()
        if not current_out:
            stem = Path(path).stem
            self.output_var.set(str(Path(path).with_name(f"{stem}-vsl-study")))

    def _browse_output(self) -> None:
        initial = self.output_var.get().strip()
        if initial:
            start = initial if Path(initial).is_dir() else str(Path(initial).parent)
        elif self.input_var.get().strip():
            start = str(Path(self.input_var.get()).parent)
        else:
            start = str(Path.home() / "Documents")
        kwargs = {"parent": self.root, "title": "Choose a save folder", "initialdir": start}
        try:
            path = filedialog.askdirectory(mustexist=False, **kwargs)
        except TypeError:
            path = filedialog.askdirectory(**kwargs)
        if path:
            self.output_var.set(path)

    def _note_missing_ffmpeg(self) -> None:
        missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
        if not missing:
            return
        text = (
            "This computer is missing FFmpeg, which VSL Study needs to open videos. "
            "Install FFmpeg, then close this app and open it again."
        )
        self.status_var.set(text)
        self._append_log("error", text)

    def _append_log(self, stage: str, message: str) -> None:
        line = _friendly_text(stage, message)
        self.log.configure(state="normal", fg=INK)
        if self._log_placeholder:
            self.log.delete("1.0", "end")
            self._log_placeholder = False
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_queue(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                stage, message = payload.split("\t", 1)
                self.status_var.set(_friendly_text(stage, message))
                self._append_log(stage, message)
            elif kind == "done":
                self._set_running(False)
                data = json.loads(payload) if isinstance(payload, str) and payload.startswith("{") else {"job": payload, "transcript_status": "complete"}
                self.last_job = str(data.get("job") or "")
                status_line, log_line = format_job_completion(data)
                self.status_var.set(status_line)
                self._append_log("done", log_line)
            elif kind in {"capture_created", "capture_ready", "capture_incomplete", "capture_abandoned"}:
                data = json.loads(payload) if isinstance(payload, str) else payload
                self._on_capture_event(kind, data if isinstance(data, dict) else {})
            elif kind == "error":
                self._set_running(False)
                self.capturing = False
                self._sync_capture_buttons()
                self.status_var.set("Something went wrong.")
                self._append_log("error", payload)
                messagebox.showerror("VSL Study", payload, parent=self.root)
        self.root.after(150, self._drain_queue)

    def _run(self) -> None:
        if self.running:
            return
        if self.capturing:
            messagebox.showwarning(
                "VSL Study",
                "Finish the browser recording first, or click Cancel recording.",
                parent=self.root,
            )
            return
        source = self.input_var.get().strip().strip('"')
        dest = self.output_var.get().strip().strip('"')
        if not source:
            messagebox.showwarning("VSL Study", "Choose a video file first.", parent=self.root)
            return
        if not Path(source).is_file():
            messagebox.showwarning("VSL Study", f"Video not found:\n{source}", parent=self.root)
            return
        if not dest:
            messagebox.showwarning("VSL Study", "Choose a folder to save the results.", parent=self.root)
            return
        settings = self._snapshot_settings()
        if settings is None:
            return
        _save_prefs(
            {
                "input": source,
                "output": dest,
                "speech_quality": _value_for(SPEECH_QUALITY_OPTIONS, self.speech_var.get(), "recommended"),
                "language": settings.language,
                "task": settings.task,
                "ocr_on": settings.ocr,
            }
        )
        self._start_process(source, dest, settings)

    def _snapshot_settings(self, capture: dict | None = None) -> ProcessSettings | None:
        try:
            model, language, task = self._speech_settings()
            validate_model_language(model, language, task)
        except SettingsError as exc:
            messagebox.showerror("VSL Study", str(exc), parent=self.root)
            return None
        return ProcessSettings(
            model=model,
            language=language,
            task=task,
            detector="adaptive",
            interval=5.0,
            ocr=bool(self.ocr_var.get()),
            device="auto",
            capture=dict(capture) if capture else None,
        )

    def _start_process(self, source: str, dest: str, settings: ProcessSettings) -> None:
        if self.running:
            return
        settings = replace(settings, capture=dict(settings.capture) if settings.capture else None)
        dest_path = Path(dest)
        rec_id = (settings.capture or {}).get("recording_id") or ""
        if rec_id:
            dest_path = unique_job_dir(dest_path, rec_id[:8])
            dest = str(dest_path)
        self._set_running(True)
        self.status_var.set("Starting…")
        self._append_log("run", f"Working on {source}\nSaving to {dest}")

        def worker() -> None:
            def progress(stage: str, message: str) -> None:
                self.messages.put(("progress", f"{stage}\t{message}"))

            try:
                progress("run", "Loading transcription libraries")
                from vsl_study.cache import JobConflictError
                from vsl_study.pipeline import process_video

                out = dest
                try:
                    result = process_video(source, out, settings=settings, progress=progress)
                except JobConflictError:
                    alt = unique_job_dir(Path(out), "rec")
                    progress("export", f"That folder already belongs to another video. Saving to {alt}")
                    result = process_video(source, alt, settings=settings, progress=progress)
                self.messages.put(("done", json.dumps({"job": result["job"], "transcript_status": result.get("transcript_status", "complete")})))
            except Exception as exc:  # noqa: BLE001
                self.messages.put(("error", str(exc) or traceback.format_exc()))

        threading.Thread(target=worker, daemon=True, name="vsl-study-job").start()

    def _captures_root(self) -> Path:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        path = base / "VSL Study" / "captures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _ensure_capture_server(self):
        if self._capture_server is not None and self._capture_server.alive():
            return self._capture_server
        from vsl_study.capture_server import CaptureServer

        def on_event(kind: str, payload: dict) -> None:
            self.messages.put((kind, json.dumps(payload)))

        self._capture_server = CaptureServer(self._captures_root(), on_event=on_event)
        self._capture_server.start()
        return self._capture_server

    def _record_tab(self) -> None:
        if self.running:
            messagebox.showinfo("VSL Study", "Wait until the current job finishes.", parent=self.root)
            return
        dest = self.output_var.get().strip().strip('"')
        if not dest:
            messagebox.showwarning("VSL Study", "Choose a folder to save the results first.", parent=self.root)
            return
        try:
            server = self._ensure_capture_server()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("VSL Study", f"Could not start the local recorder.\n{exc}", parent=self.root)
            return
        self._capture_generation = server.note_expected_client()
        self._capture_session_id = None
        self.capturing = True
        self._sync_capture_buttons()
        self.status_var.set("Recording page opened. Choose the tab with audio, then start.")
        self._append_log(
            "recording",
            "Opened the local recorder. After you stop, this window transcribes and takes screenshots. "
            "Keep the computer awake. This app cannot detect when the video ends. "
            "If you close the recorder tab, wait about three minutes or click Cancel recording.",
        )
        webbrowser.open(server.recorder_url)

    def _capture_gate(self) -> dict:
        return {
            "capturing": self.capturing,
            "running": self.running,
            "session_id": self._capture_session_id,
            "generation": self._capture_generation,
            "processed": set(self._processed_captures),
        }

    def _sync_capture_buttons(self) -> None:
        recording_busy = self.running or self.capturing
        if hasattr(self, "record_btn"):
            self.record_btn.configure(state="disabled" if recording_busy else "normal")
        if hasattr(self, "cancel_record_btn"):
            self.cancel_record_btn.configure(state="normal" if self.capturing and not self.running else "disabled")

    def _cancel_recording(self) -> None:
        if not self.capturing:
            return
        if self._capture_server is not None:
            try:
                self._capture_server.abandon_page("desktop_cancel")
            except Exception:
                pass
        self.capturing = False
        self._capture_session_id = None
        self._sync_capture_buttons()
        message = "Recording cancelled. Partial media already saved was kept. You can record again or choose a file."
        self.status_var.set(message)
        self._append_log("recording", message)

    def _on_capture_event(self, kind: str, data: dict) -> None:
        next_state = apply_desktop_capture_event(self._capture_gate(), kind, data)
        if next_state.get("ignored"):
            return
        self.capturing = bool(next_state.get("capturing"))
        self._capture_session_id = next_state.get("session_id")
        self._processed_captures = set(next_state.get("processed") or [])
        self._sync_capture_buttons()
        if kind == "capture_created":
            return
        if kind == "capture_abandoned":
            reason = str(data.get("reason") or "abandoned")
            if reason == "desktop_cancel":
                return
            message = (
                "The recorder page was closed or stopped sending heartbeats. "
                "Partial media already saved was kept. You can record again or choose a file."
            )
            self.status_var.set(message)
            self._append_log("recording", message)
            return
        if kind == "capture_incomplete":
            message = data.get("message") or "Recording stopped without a complete capture. Partial media was kept and not analyzed as the full video."
            self.status_var.set(message)
            self._append_log("recording", message)
            return
        if kind == "capture_ready" and next_state.get("process"):
            self._start_capture_job(data)

    def _start_capture_job(self, data: dict) -> None:
        if data.get("duplicate") and self.running:
            return
        path = str(data.get("path") or "")
        capture = data.get("capture") if isinstance(data.get("capture"), dict) else None
        dest = self.output_var.get().strip().strip('"')
        if not path or not dest:
            self.status_var.set("Recording saved, but the save folder is missing.")
            self._append_log("error", "Recording finished without a save folder.")
            return
        settings = self._snapshot_settings(capture)
        if settings is None:
            return
        self.input_var.set(path)
        self._append_log("validate", "Recording saved. Starting the study folder.")
        self._start_process(path, dest, settings)

    def _open_output(self) -> None:
        path = self.last_job or self.output_var.get().strip()
        if not path or not Path(path).exists():
            messagebox.showinfo("VSL Study", "No save folder yet.", parent=self.root)
            return
        os.startfile(path)  # noqa: S606

    def _open_report(self) -> None:
        job = self.last_job or self.output_var.get().strip()
        report = Path(job) / "report.html" if job else None
        if not report or not report.exists():
            messagebox.showinfo("VSL Study", "No report yet. Create a study folder first.", parent=self.root)
            return
        os.startfile(report)  # noqa: S606

    def _on_close(self) -> None:
        if self.running or self.capturing:
            if not messagebox.askyesno(
                "VSL Study",
                "A recording or job is still running. Close anyway? Partial media already saved on disk is kept. Capture in the browser will stop.",
                parent=self.root,
            ):
                return
        if self._capture_server is not None:
            try:
                self._capture_server.stop()
            except Exception:
                pass
            self._capture_server = None
        self.root.destroy()


def main() -> int:
    enable_windows_dpi_awareness()
    root = Tk()
    root.withdraw()
    root.configure(bg=BG)
    VSLStudyApp(root)
    root.deiconify()
    root.update_idletasks()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
