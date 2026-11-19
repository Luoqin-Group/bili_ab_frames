# -*- coding: utf-8 -*-
"""
B站 120fps AB 交替帧合成工具（tkinter GUI）

原理：
    输出时间轴按 t = i / out_fps 采样。
    - 输出第 1,3,5... 帧（0-based 偶数位）来自 A，默认保留 A 的第 1,3,5... 原始帧（0-based 索引 0,2,4...）
    - 输出第 2,4,6... 帧（0-based 奇数位）来自 B，默认保留 B 的第 2,4,6... 原始帧（0-based 索引 1,3,5...）
    - B 站 60fps 取输出第 1,3,5... 帧 -> 全 A
    - B 站 30fps 取输出第 2,6,10... 帧 -> 全 B

关键点：
    * 用 ffprobe 读取真实 duration / width / height / r_frame_rate / nb_frames。
    * 不是“删帧后连续写”，而是按输出时间轴做最近帧采样，因此不会变速。
    * 自动时长取较长源 duration + 0.5s；短源结束后按循环/保持末帧/黑屏补齐。
    * 直接通过 rawvideo 管道交给 ffmpeg/libx264，不生成中间视频。
    * 支持拉伸 / 居中裁剪 / 黑边三种画面适配；高级选项默认折叠。
    * 调试模式会显示 ffprobe 原始 JSON、ffmpeg 命令及英文原始输出。
    * 所有编解码在后台线程执行，GUI 只通过 queue + after 更新。

依赖：
    pip install opencv-python numpy
    FFmpeg（需要 ffmpeg 与 ffprobe 在 PATH 中）
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif",
}

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".m4v", ".ts",
    ".mts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".3gp", ".ogv",
}

FILE_DIALOG_TYPES = [
    (
        "视频 / 图片",
        "*.mp4 *.mov *.mkv *.avi *.flv *.webm *.m4v *.ts *.mts *.m2ts "
        "*.wmv *.mpg *.mpeg *.3gp *.ogv "
        "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.gif",
    ),
    ("所有文件", "*.*"),
]

# 修复 #6：选项文案改为“第N帧”表述，消除“奇数帧/偶数帧”的 0-based/1-based 歧义。
#   括号里的 0-based 索引仅作技术说明。
A_KEEP_OPTIONS: List[Tuple[str, Optional[int]]] = [
    ("保留第1,3,5…帧（删第2,4,6…帧）", 0),
    ("保留第2,4,6…帧（删第1,3,5…帧）", 1),
    ("不删帧", None),
]
B_KEEP_OPTIONS: List[Tuple[str, Optional[int]]] = [
    ("保留第2,4,6…帧（删第1,3,5…帧）", 1),
    ("保留第1,3,5…帧（删第2,4,6…帧）", 0),
    ("不删帧", None),
]

PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"]

# 源画面比例与输出比例不一致时的适配方式。
FIT_MODE_OPTIONS: List[Tuple[str, str]] = [
    ("拉伸铺满（可能变形）", "stretch"),
    ("裁剪铺满（保持比例，切掉边缘）", "crop"),
    ("黑边（保持比例，完整画面）", "pad"),
]
FIT_MODE_LABELS = {mode: label for label, mode in FIT_MODE_OPTIONS}

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    is_image: bool
    # 修复 #1/#2：记录 nb_frames 是否为估算值，供采样越界检测参考。
    nb_frames_estimated: bool = False
    # 日志/调试用的补充信息。
    codec: str = ""
    pix_fmt: str = ""
    bit_rate: int = 0
    format_duration: float = 0.0


@dataclass
class ComposeConfig:
    path_a: str
    path_b: str
    output: str

    auto_resolution: bool
    width: int
    height: int

    out_fps: int

    auto_duration: bool
    duration: float

    keep_a: Optional[int]
    keep_b: Optional[int]

    end_mode: str          # "hold" / "black"
    loop_short: bool

    crf: int
    preset: str

    # 新版本参数；给默认值以兼容旧调用。
    fit_mode: str = "stretch"   # stretch / crop / pad
    debug: bool = False


class ComposeCancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def is_image_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def parse_rate(value) -> float:
    """解析 ffprobe 的 '30000/1001' 或 '30' 形式帧率。"""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text in ("N/A", "0/0"):
        return 0.0
    try:
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            num = float(num_s)
            den = float(den_s)
            return num / den if den else 0.0
        return float(text)
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def even_up(value: int) -> int:
    """向上对齐到偶数（H.264 yuv420p 要求）。"""
    value = int(value)
    if value < 0:
        raise ValueError("even_up 只接受非负整数。")
    return value if value % 2 == 0 else value + 1


def find_required_tools() -> Tuple[str, str]:
    """返回 (ffmpeg, ffprobe)；缺失时给出明确错误，不静默降级。"""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    missing = []
    if not ffmpeg:
        missing.append("ffmpeg")
    if not ffprobe:
        missing.append("ffprobe")
    if missing:
        raise RuntimeError(
            "未找到 " + "、".join(missing) + "。\n\n"
            "本工具必须依赖 FFmpeg 的 ffmpeg 与 ffprobe，"
            "请先安装 FFmpeg，并把其 bin 目录加入系统 PATH，"
            "然后重新启动本程序。\n\n"
            "可在命令行运行 `ffmpeg -version` 和 `ffprobe -version` 检查。"
        )
    return ffmpeg, ffprobe


def _run_ffprobe(
    ffprobe: str,
    path: str,
    debug_log=None,
) -> dict:
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    if debug_log is not None:
        debug_log("ffprobe 命令: " + subprocess.list2cmdline(cmd))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "无法启动 ffprobe，请确认 FFmpeg 已安装并加入 PATH。"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe 读取超时：\n{path}") from exc

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if debug_log is not None:
        if stderr:
            debug_log("ffprobe stderr:\n" + stderr)
        # 原始 JSON 便于核对 r_frame_rate / duration / nb_frames 等字段。
        debug_log("ffprobe 原始 JSON:\n" + stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe 读取失败（退出码 {proc.returncode}）：\n{path}\n\n{stderr}"
        )

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe 返回了无法解析的 JSON：\n{path}") from exc


def load_image_bgr(path: str) -> np.ndarray:
    """使用 imdecode 读取图片，兼容 Windows 中文路径。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"无法读取图片文件：\n{path}\n\n{exc}") from exc

    if data.size == 0:
        raise RuntimeError(f"图片文件为空：\n{path}")

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法解码图片：\n{path}")
    return image


def probe_media(
    path: str,
    ffprobe: str,
    debug_log=None,
) -> MediaInfo:
    """
    只信任 ffprobe 的媒体信息读取。
    对视频严格要求读到 width/height/fps/duration；
    对图片允许 duration=0（静态图没有时长概念）。
    """
    image_mode = is_image_path(path)
    data = _run_ffprobe(ffprobe, path, debug_log=debug_log)

    streams = data.get("streams") or []
    video_streams = []
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        disposition = stream.get("disposition") or {}
        if disposition.get("attached_pic"):
            continue
        video_streams.append(stream)

    if not video_streams:
        if image_mode:
            image = load_image_bgr(path)
            h, w = image.shape[:2]
            return MediaInfo(path, w, h, 1.0, 0.0, 1, True, False)
        raise RuntimeError(f"文件中没有可用的视频流：\n{path}")

    stream = video_streams[0]
    fmt = data.get("format") or {}

    codec = str(stream.get("codec_name") or "")
    pix_fmt = str(stream.get("pix_fmt") or "")
    bit_rate = int(parse_float(stream.get("bit_rate")) or 0)
    format_duration = parse_float(fmt.get("duration"))

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    if image_mode and (width <= 0 or height <= 0):
        image = load_image_bgr(path)
        h, w = image.shape[:2]
        width, height = w, h

    if width <= 0 or height <= 0:
        raise RuntimeError(f"ffprobe 无法读取有效分辨率：\n{path}")

    fps = parse_rate(stream.get("r_frame_rate"))
    if fps <= 0:
        fps = parse_rate(stream.get("avg_frame_rate"))

    duration = parse_float(stream.get("duration"))
    if duration <= 0:
        duration = parse_float(fmt.get("duration"))

    nb_frames = 0
    raw_nb = stream.get("nb_frames")
    try:
        if raw_nb is not None and str(raw_nb).strip() not in ("", "N/A"):
            nb_frames = int(float(raw_nb))
    except (TypeError, ValueError):
        nb_frames = 0

    if image_mode:
        if fps <= 0:
            fps = 1.0
        return MediaInfo(
            path, width, height, fps, 0.0, 1, True, False,
            codec=codec,
            pix_fmt=pix_fmt,
            bit_rate=bit_rate,
            format_duration=format_duration,
        )

    if fps <= 0:
        raise RuntimeError(
            "ffprobe 无法读取帧率（r_frame_rate / avg_frame_rate）。\n"
            f"文件：{path}"
        )

    if duration <= 0 and nb_frames > 0:
        duration = nb_frames / fps
    if duration <= 0:
        raise RuntimeError(f"ffprobe 无法读取有效时长：\n{path}")

    nb_frames_estimated = False
    if nb_frames <= 0:
        # 极少数容器不写 nb_frames，用真实 duration × fps 估算；
        # 这只影响日志中的帧数显示和末帧边界，时间轴仍按 duration 计算。
        nb_frames = max(1, int(round(duration * fps)))
        nb_frames_estimated = True

    return MediaInfo(
        path, width, height, fps, duration, nb_frames, False,
        nb_frames_estimated,
        codec=codec,
        pix_fmt=pix_fmt,
        bit_rate=bit_rate,
        format_duration=format_duration,
    )


def _media_tech_suffix(info: MediaInfo) -> str:
    parts = []
    if info.codec:
        parts.append(info.codec)
    if info.pix_fmt:
        parts.append(info.pix_fmt)
    if info.bit_rate > 0:
        parts.append(f"{info.bit_rate / 1_000_000:.2f} Mbps")
    return (" | " + " ".join(parts)) if parts else ""


def format_source_line(label: str, info: MediaInfo) -> str:
    if info.is_image:
        return (
            f"{label}: 1帧 图片 0.000s "
            f"{info.width}x{info.height}"
            + _media_tech_suffix(info)
        )
    suffix = "（估算）" if info.nb_frames_estimated else ""
    return (
        f"{label}: {info.nb_frames}帧{suffix} {info.fps:.3f}fps "
        f"{info.duration:.3f}s {info.width}x{info.height}"
        + _media_tech_suffix(info)
    )


def retained_frame_count(info: MediaInfo, keep_parity: Optional[int]) -> int:
    if info.is_image:
        return 1
    if keep_parity is None:
        return int(info.nb_frames)
    return max(0, (int(info.nb_frames) - int(keep_parity) + 1) // 2)


def _max_valid_original_index(info: MediaInfo, keep_parity: Optional[int]) -> int:
    """
    修复 #1/#2：返回“实际可能存在”的最大原始帧索引。
    用 nb_frames-1 与 duration*fps 的较小值做保守约束，
    避免 nb_frames 估算偏大时采到不存在的帧。
    """
    if info.is_image:
        return 0
    by_nb = int(info.nb_frames) - 1
    by_duration = int(math.floor(info.duration * info.fps + 1e-6)) - 1
    if by_nb <= 0:
        by_nb = by_duration
    if by_duration <= 0:
        by_duration = by_nb
    last = max(0, min(by_nb, by_duration))
    # 对齐到保留奇偶性
    if keep_parity is not None:
        last = last - ((last - keep_parity) % 2)
    return last


def select_source_index(
    info: MediaInfo,
    keep_parity: Optional[int],
    t: float,
) -> int:
    """
    在源时间 t（秒）处，选择“符合保留奇偶性”的最近原始帧索引。
    距离相同时选择更早的帧，保证采样单调。

    keep_parity:
        0 -> 原索引 0,2,4...
        1 -> 原索引 1,3,5...
        None -> 所有原始帧
    """
    count = retained_frame_count(info, keep_parity)
    if count <= 0:
        raise RuntimeError("源视频删帧后没有可用帧。")

    # 修复 #1/#2：把可用范围夹到“实际可能存在”的最大索引，
    # 防止 nb_frames 估算偏大时采样越界。
    max_orig = _max_valid_original_index(info, keep_parity)
    if keep_parity is None:
        max_k = max_orig
    else:
        max_k = (max_orig - keep_parity) // 2
    max_k = max(0, min(count - 1, max_k))

    x = max(0.0, float(t)) * float(info.fps)

    if keep_parity is None:
        base = int(math.floor(x))
        candidates = (base, base + 1)

        def original_index(k: int) -> int:
            return k
    else:
        base = int(math.floor((x - keep_parity) / 2.0))
        candidates = (base, base + 1)

        def original_index(k: int) -> int:
            return keep_parity + 2 * k

    best_idx = None
    best_dist = float("inf")
    for k in candidates:
        k = max(0, min(max_k, k))
        idx = original_index(k)
        dist = abs(idx / info.fps - t)
        if (
            dist < best_dist - 1e-12
            or (abs(dist - best_dist) <= 1e-12
                and (best_idx is None or idx < best_idx))
        ):
            best_idx = idx
            best_dist = dist

    if best_idx is None:
        raise RuntimeError("源视频采样失败：没有可用帧。")
    return int(best_idx)


def resize_to_output(
    frame: np.ndarray,
    out_w: int,
    out_h: int,
    fit_mode: str = "stretch",
) -> np.ndarray:
    """
    把源帧适配到输出画布。

    fit_mode:
        stretch -> 直接拉伸到 W×H（可能变形）
        crop    -> 等比放大到铺满，居中裁剪多余部分
        pad     -> 等比缩放到完整放入，四周补黑边
    """
    if frame is None:
        raise RuntimeError("读取到空帧。")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 2:
        # 修复 #18：灰度+alpha（Y + A），转灰度后按 BGR 处理。
        frame = cv2.cvtColor(frame, cv2.COLOR_YA2BGR)

    src_h, src_w = frame.shape[:2]
    if src_w == out_w and src_h == out_h:
        return np.ascontiguousarray(frame)

    if fit_mode not in ("stretch", "crop", "pad"):
        fit_mode = "stretch"

    if fit_mode == "stretch":
        interpolation = (
            cv2.INTER_AREA
            if (src_w > out_w or src_h > out_h)
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            frame, (out_w, out_h), interpolation=interpolation
        )
        return np.ascontiguousarray(resized)

    if fit_mode == "crop":
        scale = max(out_w / src_w, out_h / src_h)
        new_w = max(out_w, int(round(src_w * scale)))
        new_h = max(out_h, int(round(src_h * scale)))
        interpolation = (
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            frame, (new_w, new_h), interpolation=interpolation
        )
        x0 = max(0, (new_w - out_w) // 2)
        y0 = max(0, (new_h - out_h) // 2)
        cropped = resized[y0:y0 + out_h, x0:x0 + out_w]
        return np.ascontiguousarray(cropped)

    # pad：完整画面 + 黑边
    scale = min(out_w / src_w, out_h / src_h)
    new_w = min(out_w, max(1, int(round(src_w * scale))))
    new_h = min(out_h, max(1, int(round(src_h * scale))))
    interpolation = (
        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        frame, (new_w, new_h), interpolation=interpolation
    )
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# 顺序读取视频帧（避免把整段视频读进内存，也避免依赖不可靠的 seek）
# ---------------------------------------------------------------------------

class VideoFrameReader:
    """
    按原始帧序顺序解码视频，只保留需要的奇偶帧。
    调用 frame_at(target_original_index) 时，内部只向前推进；
    循环播放导致目标索引回退时，自动重新打开视频从头读取。
    """

    def __init__(
        self,
        path: str,
        keep_parity: Optional[int],
        out_w: int,
        out_h: int,
        fit_mode: str = "stretch",
    ) -> None:
        self.path = path
        self.keep_parity = keep_parity
        self.out_w = out_w
        self.out_h = out_h
        self.fit_mode = fit_mode

        self.cap: Optional[cv2.VideoCapture] = None
        self.raw_index = 0
        self.current_idx: Optional[int] = None
        self.current_frame: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None
        self.eof = False
        # 修复 #1/#2：记录最近一次 frame_at 是否因越界退化为末帧。
        self.last_was_eof_fallback = False

        self._open()

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                "OpenCV 无法打开视频文件，请检查文件是否损坏、"
                "编码是否受支持：\n"
                f"{self.path}"
            )
        self.cap = cap
        self.raw_index = 0
        self.current_idx = None
        self.current_frame = None
        # 修复 #15：restart 后不应保留上一轮的末帧。
        self.last_frame = None
        self.eof = False
        self.last_was_eof_fallback = False

    def _keep(self, index: int) -> bool:
        return (
            self.keep_parity is None
            or index % 2 == self.keep_parity
        )

    def _decode_one_retained(self) -> bool:
        if self.cap is None or self.eof:
            return False

        while True:
            ok, raw = self.cap.read()
            if not ok or raw is None:
                self.eof = True
                return False

            idx = self.raw_index
            self.raw_index += 1

            if not self._keep(idx):
                continue

            frame = resize_to_output(
                raw, self.out_w, self.out_h, self.fit_mode
            )
            self.current_idx = idx
            self.current_frame = frame
            self.last_frame = frame
            return True

    def restart(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self._open()

    def frame_at(self, target_original_index: int) -> np.ndarray:
        target = int(target_original_index)
        self.last_was_eof_fallback = False

        if self.current_idx is not None and target < self.current_idx:
            self.restart()

        while self.current_idx is None or self.current_idx < target:
            if not self._decode_one_retained():
                # 修复 #1/#2：目标索引无法到达 -> 退化为末帧，并打标记。
                self.last_was_eof_fallback = True
                break

        if self.current_frame is None:
            raise RuntimeError(
                "视频删帧后没有读取到任何可用帧：\n"
                f"{self.path}"
            )
        return self.current_frame

    def read_last(self) -> np.ndarray:
        while self._decode_one_retained():
            pass
        if self.last_frame is None:
            raise RuntimeError(
                "视频删帧后没有读取到任何可用帧：\n"
                f"{self.path}"
            )
        return self.last_frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# 后台合成线程
# ---------------------------------------------------------------------------

class ComposeWorker(threading.Thread):
    def __init__(self, cfg: ComposeConfig, msg_queue: "queue.Queue"):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = msg_queue
        self.cancel_event = threading.Event()
        # 修复 #10：每次任务一个 token，GUI 只接受当前 token 的 auto_* 消息。
        self.token = uuid.uuid4().hex

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines: Deque[str] = deque(maxlen=80)
        self._success = False
        # 修复 #7：只有 ffmpeg 真正启动后，失败/取消才允许删除输出文件。
        self._output_touched = False

    # ---- 与 GUI 的通信 ---------------------------------------------------

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def _log(self, text: str) -> None:
        self.q.put(("log", text))

    def _debug(self, text: str) -> None:
        """仅在调试模式写入；统一加 [调试] 前缀，便于区分。"""
        if self.cfg.debug:
            self.q.put(("log", "[调试] " + text))

    def _status(self, text: str) -> None:
        self.q.put(("status", text))

    def _progress(self, fraction: float) -> None:
        self.q.put(("progress", max(0.0, min(1.0, float(fraction)))))

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise ComposeCancelled()

    # ---- 主流程 ----------------------------------------------------------

    def run(self) -> None:
        try:
            self._compose()
        except ComposeCancelled:
            self._log("已取消，未保留半成品输出文件。")
            self.q.put(("cancelled", self.token))
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self._log("错误: " + message)
            if not isinstance(exc, RuntimeError):
                self._log(traceback.format_exc().rstrip())
            self.q.put(("error", message, self.token))
        finally:
            self._cleanup_encoder()

    def _debug_tool_versions(self, ffmpeg: str, ffprobe: str) -> None:
        if not self.cfg.debug:
            return
        self._debug(f"Python: {sys.version.splitlines()[0]}")
        try:
            self._debug(f"OpenCV: {cv2.__version__}  NumPy: {np.__version__}")
        except Exception:
            pass
        for name, exe in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
            try:
                proc = subprocess.run(
                    [exe, "-version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    creationflags=CREATE_NO_WINDOW,
                )
                lines = proc.stdout.decode(
                    "utf-8", errors="replace"
                ).strip().splitlines()
                self._debug(f"{name} 路径: {exe}")
                if lines:
                    self._debug(f"{name} 版本: {lines[0]}")
            except Exception as exc:
                self._debug(f"{name} 版本读取失败: {exc}")

    def _compose(self) -> None:
        cfg = self.cfg
        self._status("检查 FFmpeg / 读取源信息…")
        ffmpeg, ffprobe = find_required_tools()
        self._debug_tool_versions(ffmpeg, ffprobe)

        debug_probe = self._debug if cfg.debug else None
        info_a = probe_media(cfg.path_a, ffprobe, debug_log=debug_probe)
        info_b = probe_media(cfg.path_b, ffprobe, debug_log=debug_probe)

        self._log(format_source_line("A", info_a))
        self._log(format_source_line("B", info_b))
        self._log(f"A 文件: {os.path.abspath(cfg.path_a)}")
        self._log(f"B 文件: {os.path.abspath(cfg.path_b)}")
        for label, info in (("A", info_a), ("B", info_b)):
            if cfg.debug:
                try:
                    size_mb = os.path.getsize(info.path) / (1024 * 1024)
                    self._debug(f"{label} 文件大小: {size_mb:.2f} MB")
                except OSError:
                    pass
            if info.format_duration > 0 and abs(
                info.format_duration - info.duration
            ) > 0.05:
                self._log(
                    f"{label} 时长提示: 视频流 {info.duration:.3f}s，"
                    f"容器 {info.format_duration:.3f}s；"
                    "当前按视频流时长计算。"
                )
            if info.nb_frames_estimated:
                self._log(
                    f"{label} 提示: 容器未写 nb_frames，"
                    "帧数由 duration × fps 估算；实际解码范围会自动兜底。"
                )

        # 自动分辨率：只取 max(A宽,B宽) × max(A高,B高)，并向上对齐偶数。
        if cfg.auto_resolution:
            raw_w = max(info_a.width, info_b.width)
            raw_h = max(info_a.height, info_b.height)
            out_w = even_up(raw_w)
            out_h = even_up(raw_h)
            if (out_w, out_h) != (raw_w, raw_h):
                self._log(
                    f"自动分辨率 -> {out_w}x{out_h}"
                    f"（由 {raw_w}x{raw_h} 对齐偶数）"
                )
            else:
                self._log(f"自动分辨率 -> {out_w}x{out_h}")
            self.q.put(("auto_size", out_w, out_h, self.token))
        else:
            out_w = int(cfg.width)
            out_h = int(cfg.height)
            if out_w <= 0 or out_h <= 0:
                raise ValueError("手动分辨率必须是正整数。")
            if out_w % 2 or out_h % 2:
                raise ValueError("H.264 yuv420p 要求宽高为偶数，请修改手动分辨率。")
            self._log(f"手动分辨率 -> {out_w}x{out_h}")

        fit_mode = cfg.fit_mode if cfg.fit_mode in (
            "stretch", "crop", "pad"
        ) else "stretch"
        self._log(
            f"画面适配: {FIT_MODE_LABELS.get(fit_mode, fit_mode)}"
            f"（源 A {info_a.width}x{info_a.height}，"
            f"源 B {info_b.width}x{info_b.height}）"
        )

        # 自动时长：较长源的真实 duration + 0.5s 余量。
        if cfg.auto_duration:
            if info_a.is_image and info_b.is_image:
                raise ValueError(
                    "A、B 都是静态图片，无法自动推断时长。"
                    "请取消勾选“自动时长”，手动填写目标时长。"
                )
            base_duration = max(info_a.duration, info_b.duration)
            target_duration = base_duration + 0.5
            self._log(
                f"自动时长: 较长源 {base_duration:.3f}s"
                f" -> 目标 {target_duration:.3f}s"
            )
            longer_label = (
                "A" if info_a.duration >= info_b.duration else "B"
            )
            if not (info_a.is_image or info_b.is_image):
                self._log(
                    f"自动时长依据: {longer_label} 更长"
                    f"（A={info_a.duration:.3f}s，B={info_b.duration:.3f}s）"
                )
        else:
            target_duration = float(cfg.duration)
            if target_duration <= 0:
                raise ValueError("手动时长必须大于 0 秒。")
            self._log(f"手动时长: 目标 {target_duration:.3f}s")

        out_fps = int(cfg.out_fps)
        if out_fps <= 0 or out_fps % 2 != 0:
            raise ValueError("输出帧率必须是正偶数（例如 120 或 240）。")

        total_frames = max(
            1, int(math.floor(target_duration * out_fps + 0.5))
        )
        out_duration = total_frames / out_fps

        self._log(
            f"输出: {total_frames} 帧 = {out_duration:.3f}s @ {out_fps}fps"
        )
        if cfg.auto_duration:
            self.q.put(("auto_target", target_duration, self.token))

        keep_a = None if info_a.is_image else cfg.keep_a
        keep_b = None if info_b.is_image else cfg.keep_b

        count_a = retained_frame_count(info_a, keep_a)
        count_b = retained_frame_count(info_b, keep_b)
        if not info_a.is_image and count_a <= 0:
            raise ValueError(
                "A 按当前删帧选项处理后没有可用帧，"
                "请改选“不删帧”或另一种保留方式。"
            )
        if not info_b.is_image and count_b <= 0:
            raise ValueError(
                "B 按当前删帧选项处理后没有可用帧，"
                "请改选“不删帧”或另一种保留方式。"
            )

        self._log(self._keep_log_line("A", info_a, keep_a, count_a))
        self._log(self._keep_log_line("B", info_b, keep_b, count_b))

        for label, info, count in (
            ("A", info_a, count_a), ("B", info_b, count_b)
        ):
            if not info.is_image and info.duration > 0:
                self._log(
                    f"{label} 删帧后有效帧率: "
                    f"{count / info.duration:.3f} fps"
                    f"（{count} 帧 / {info.duration:.3f}s）"
                )

        self._log(
            "B站播放预期: 60fps 模式取 A 的帧；30fps 模式取 B 的帧。"
        )

        if cfg.loop_short:
            fill_text = "循环播放"
        elif cfg.end_mode == "hold":
            fill_text = "保持最后一帧"
        else:
            fill_text = "黑屏"

        ended_labels = []
        for label, info in (("A", info_a), ("B", info_b)):
            if info.is_image:
                continue
            if info.duration < out_duration - 1e-9:
                ended_labels.append(label)
                self._log(
                    f"{label} 实际在 {info.duration:.3f} s 结束，"
                    f"之后用「{fill_text}」补齐"
                )

        if ended_labels:
            self._log(
                "说明: 某个源结束后只补齐它对应的奇/偶帧位，"
                "另一路仍继续播放，输出总时长不会因此提前结束。"
            )

        self._log(
            f"采样: t=i/{out_fps}，按输出时间轴选择最近帧，"
            "不压缩时间轴、不变速。"
        )
        self._log(
            f"编码: libx264 preset={cfg.preset} crf={cfg.crf}, "
            "yuv420p, faststart"
        )
        self._log(f"输出文件: {os.path.abspath(cfg.output)}")
        if cfg.debug:
            self._debug("调试模式已开启：会显示 ffprobe/ffmpeg 的原始英文输出。")

        self._check_cancel()

        image_a = None
        image_b = None
        if info_a.is_image:
            image_a = resize_to_output(
                load_image_bgr(cfg.path_a), out_w, out_h, fit_mode
            )
        if info_b.is_image:
            image_b = resize_to_output(
                load_image_bgr(cfg.path_b), out_w, out_h, fit_mode
            )

        readers: List[VideoFrameReader] = []
        reader_a: Optional[VideoFrameReader] = None
        reader_b: Optional[VideoFrameReader] = None
        try:
            if not info_a.is_image:
                reader_a = VideoFrameReader(
                    cfg.path_a, keep_a, out_w, out_h, fit_mode
                )
                readers.append(reader_a)
            if not info_b.is_image:
                reader_b = VideoFrameReader(
                    cfg.path_b, keep_b, out_w, out_h, fit_mode
                )
                readers.append(reader_b)

            self._encode(
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                out_w=out_w,
                out_h=out_h,
                out_fps=out_fps,
                total_frames=total_frames,
                out_duration=out_duration,
                info_a=info_a,
                info_b=info_b,
                reader_a=reader_a,
                reader_b=reader_b,
                image_a=image_a,
                image_b=image_b,
                keep_a=keep_a,
                keep_b=keep_b,
            )
        finally:
            for reader in readers:
                reader.close()

    @staticmethod
    def _keep_log_line(
        label: str,
        info: MediaInfo,
        keep_parity: Optional[int],
        count: int,
    ) -> str:
        if info.is_image:
            return (
                f"{label} 为静态图片，忽略删帧设置，"
                "整段保持同一帧。"
            )

        if keep_parity is None:
            choice = "不删帧"
            indices = "全部原始帧"
        elif keep_parity == 0:
            choice = "保留第1,3,5…帧（删第2,4,6…帧）"
            indices = "原索引 0,2,4..."
        else:
            choice = "保留第2,4,6…帧（删第1,3,5…帧）"
            indices = "原索引 1,3,5..."

        return (
            f"{label} 删帧预处理: {choice}（{indices}）"
            f" -> 可用 {count} 帧"
        )

    def _source_frame(
        self,
        *,
        label: str,
        info: MediaInfo,
        reader: Optional[VideoFrameReader],
        image: Optional[np.ndarray],
        keep_parity: Optional[int],
        t: float,
        loop_cycles: dict,
    ) -> Optional[np.ndarray]:
        if info.is_image:
            return image

        if reader is None:
            raise RuntimeError(f"{label} 视频读取器未初始化。")

        if t < info.duration:
            target_idx = select_source_index(info, keep_parity, t)
            frame = reader.frame_at(target_idx)
            # 修复 #1/#2：采样越界时记一次日志（每路只记一次，避免刷屏）。
            if reader.last_was_eof_fallback and not loop_cycles.get(
                label + "_warned", False
            ):
                loop_cycles[label + "_warned"] = True
                self._log(
                    f"警告: {label} 在 t={t:.3f}s 的目标帧索引超出实际可解码范围，"
                    "已退化为末帧。可能是容器未写精确 nb_frames。"
                )
            return frame

        # 已经超过源视频末尾，按策略补齐。
        if self.cfg.loop_short and info.duration > 0:
            if not loop_cycles.get(label + "_fill_logged", False):
                loop_cycles[label + "_fill_logged"] = True
                self._debug(
                    f"{label} 在输出 t={t:.3f}s 超过源时长 "
                    f"{info.duration:.3f}s，开始循环播放"
                )
            wrapped_t = t % info.duration
            cycle = int(math.floor(t / info.duration))
            if cycle != loop_cycles.get(label, 0):
                # 修复 #3：显式 restart，处理回绕边界；cycle 跳变多次也只 restart 一次，
                # 后续 frame_at 内部的 target<current 检测会兜住剩余回退。
                reader.restart()
                loop_cycles[label] = cycle
            target_idx = select_source_index(info, keep_parity, wrapped_t)
            frame = reader.frame_at(target_idx)
            if reader.last_was_eof_fallback and not loop_cycles.get(
                label + "_warned", False
            ):
                loop_cycles[label + "_warned"] = True
                self._log(
                    f"警告: {label} 循环回绕后在 t={wrapped_t:.3f}s 的目标帧索引"
                    "超出实际可解码范围，已退化为末帧。"
                )
            return frame

        if self.cfg.end_mode == "hold":
            if not loop_cycles.get(label + "_fill_logged", False):
                loop_cycles[label + "_fill_logged"] = True
                self._debug(
                    f"{label} 在输出 t={t:.3f}s 超过源时长，保持最后一帧"
                )
            return reader.read_last()

        if not loop_cycles.get(label + "_fill_logged", False):
            loop_cycles[label + "_fill_logged"] = True
            self._debug(
                f"{label} 在输出 t={t:.3f}s 超过源时长，开始黑屏补齐"
            )
        return None  # 黑屏补齐

    def _encode(
        self,
        *,
        ffmpeg: str,
        ffprobe: str,
        out_w: int,
        out_h: int,
        out_fps: int,
        total_frames: int,
        out_duration: float,
        info_a: MediaInfo,
        info_b: MediaInfo,
        reader_a: Optional[VideoFrameReader],
        reader_b: Optional[VideoFrameReader],
        image_a: Optional[np.ndarray],
        image_b: Optional[np.ndarray],
        keep_a: Optional[int],
        keep_b: Optional[int],
    ) -> None:
        cfg = self.cfg

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-loglevel", "info" if cfg.debug else "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", str(out_fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", cfg.preset,
            "-crf", str(cfg.crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            cfg.output,
        ]

        self._status("启动 ffmpeg 编码器…")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "无法启动 ffmpeg，请确认 FFmpeg 已安装并加入 PATH。"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"启动 ffmpeg 失败：\n{exc}") from exc

        self._proc = proc
        # 修复 #7：ffmpeg 已成功启动，从现在起失败/取消才允许删除输出文件。
        self._output_touched = True
        self._stderr_lines.clear()
        self._debug("ffmpeg 命令: " + subprocess.list2cmdline(cmd))

        def drain_stderr() -> None:
            assert proc.stderr is not None
            try:
                for raw_line in iter(proc.stderr.readline, b""):
                    line = raw_line.decode(
                        "utf-8", errors="replace"
                    ).rstrip()
                    if line:
                        self._stderr_lines.append(line)
                        if cfg.debug:
                            self.q.put(("log", "[ffmpeg] " + line))
            except Exception:
                pass

        self._stderr_thread = threading.Thread(
            target=drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        black = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        loop_cycles = {"A": 0, "B": 0}
        started = time.time()
        next_log_percent = 10
        last_ui_update = 0.0

        try:
            for i in range(total_frames):
                self._check_cancel()
                t = i / float(out_fps)
                if cfg.debug and (i < 12 or i % 600 == 0):
                    self._debug(
                        f"输出第 {i + 1} 帧: t={t:.6f}s -> "
                        f"{'A' if i % 2 == 0 else 'B'}"
                    )

                if i % 2 == 0:
                    frame = self._source_frame(
                        label="A",
                        info=info_a,
                        reader=reader_a,
                        image=image_a,
                        keep_parity=keep_a,
                        t=t,
                        loop_cycles=loop_cycles,
                    )
                else:
                    frame = self._source_frame(
                        label="B",
                        info=info_b,
                        reader=reader_b,
                        image=image_b,
                        keep_parity=keep_b,
                        t=t,
                        loop_cycles=loop_cycles,
                    )

                if frame is None:
                    frame = black
                elif frame.shape[0] != out_h or frame.shape[1] != out_w:
                    frame = resize_to_output(
                        frame, out_w, out_h, cfg.fit_mode
                    )

                if proc.poll() is not None:
                    raise RuntimeError(
                        "ffmpeg 编码进程提前退出。\n"
                        + self._ffmpeg_error_text()
                    )

                try:
                    assert proc.stdin is not None
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    raise RuntimeError(
                        "写入 ffmpeg 管道失败，编码器可能已退出。\n"
                        + self._ffmpeg_error_text()
                    ) from exc

                # GUI 状态最多约 0.15s 更新一次，避免长视频产生海量队列消息。
                now = time.time()
                if now - last_ui_update >= 0.15 or i == total_frames - 1:
                    self._progress((i + 1) / total_frames)
                    self._status(
                        f"合成中 {i + 1}/{total_frames}"
                        f"（{t:.1f}s / {out_duration:.1f}s）"
                    )
                    last_ui_update = now

                percent = int((i + 1) * 100 / total_frames)
                if percent >= next_log_percent:
                    elapsed = max(0.001, time.time() - started)
                    speed = (i + 1) / elapsed
                    remain = (
                        (total_frames - i - 1) / speed
                        if speed > 0 else 0.0
                    )
                    self._log(
                        f"进度: {percent}%"
                        f"（{i + 1}/{total_frames} 帧，"
                        f"{speed:.1f} fps，预计剩余 {remain:.0f}s）"
                    )
                    while next_log_percent <= percent:
                        next_log_percent += 10

            self._check_cancel()

            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg 编码失败（退出码 {return_code}）。\n"
                    + self._ffmpeg_error_text()
                )

            if not os.path.isfile(cfg.output):
                raise RuntimeError(
                    "ffmpeg 未生成输出文件：\n" + cfg.output
                )
            size = os.path.getsize(cfg.output)
            if size <= 0:
                raise RuntimeError(
                    "ffmpeg 生成了空输出文件：\n" + cfg.output
                )

            self._success = True
            verify_summary = self._verify_output(
                ffprobe=ffprobe,
                out_w=out_w,
                out_h=out_h,
                out_fps=out_fps,
                total_frames=total_frames,
                out_duration=out_duration,
            )
            elapsed = time.time() - started
            self._progress(1.0)
            self._status("合成完成")
            self._log(
                f"完成: {cfg.output}"
                f"（{size / (1024 * 1024):.2f} MB，"
                f"用时 {elapsed:.1f}s）"
            )
            self.q.put(
                ("done", cfg.output, elapsed, verify_summary, self.token)
            )
        finally:
            # stdin 未关闭时，退出前必须收尾，避免 ffmpeg 挂着。
            if proc.poll() is None and not self._success:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

    def _verify_output(
        self,
        *,
        ffprobe: str,
        out_w: int,
        out_h: int,
        out_fps: int,
        total_frames: int,
        out_duration: float,
    ) -> str:
        """再让 ffprobe 看一眼成品，并在日志里给出可核对的真实数据。"""
        try:
            check = probe_media(
                self.cfg.output,
                ffprobe,
                debug_log=self._debug if self.cfg.debug else None,
            )
        except Exception as exc:
            self._log(f"校验: 无法读取成品信息（{exc}）")
            return "成品信息未校验"

        summary = (
            f"{check.duration:.3f}s / {check.nb_frames}帧 / "
            f"{check.width}x{check.height}"
        )
        self._log(
            f"校验: 输出 {check.nb_frames}帧 {check.duration:.3f}s "
            f"{check.width}x{check.height} @{check.fps:.3f}fps"
        )

        issues = []
        if (check.width, check.height) != (out_w, out_h):
            issues.append(
                f"分辨率应为 {out_w}x{out_h}，实际 {check.width}x{check.height}"
            )
        if abs(check.duration - out_duration) > 0.05:
            issues.append(
                f"时长应为 {out_duration:.3f}s，实际 {check.duration:.3f}s"
            )
        # 修复 #8：成品 nb_frames 可能是 0/估算值，不能直接与 total_frames 比。
        expected_frames = total_frames
        if check.nb_frames <= 0 or check.nb_frames_estimated:
            frames_ok = True  # 无法可靠校验，跳过
        else:
            frames_ok = (check.nb_frames == expected_frames)
        if not frames_ok:
            issues.append(
                f"帧数应为 {expected_frames}，实际 {check.nb_frames}"
            )
        if issues:
            self._log("警告: 成品校验不一致 -> " + "；".join(issues))
            self.q.put(("status", "完成（成品校验有警告）", self.token))
        else:
            self._log("校验通过: 成品帧数、时长、分辨率与计划一致。")
        return summary

    def _ffmpeg_error_text(self) -> str:
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=0.3)
        if not self._stderr_lines:
            return "（ffmpeg 没有输出错误信息）"
        return "\n".join(self._stderr_lines)

    def _cleanup_encoder(self) -> None:
        proc = self._proc
        if proc is None:
            return

        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=0.5)

        # 修复 #7：只有 ffmpeg 真正开始写过文件后，失败/取消时才删除半成品；
        # 若在启动 ffmpeg 前就报错，绝不碰用户原有的输出文件。
        if (not self._success) and self._output_touched:
            try:
                if os.path.isfile(self.cfg.output):
                    os.remove(self.cfg.output)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    # 日志只保留最近一部分，防止超长任务把 Tk Text 撑到卡死。
    MAX_LOG_LINES = 4000
    QUEUE_BATCH = 300

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("B站 120fps AB 交替帧合成工具")
        self.root.geometry("1000x780")
        self.root.minsize(880, 650)

        self.msg_queue: "queue.Queue" = queue.Queue()
        self.worker: Optional[ComposeWorker] = None
        self._closing = False

        self._build_ui()
        self._sync_resolution_state()
        self._sync_duration_state()
        self._sync_end_mode_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)

    # ---- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        main = ttk.Frame(root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(10, weight=1)

        self.a_var = tk.StringVar()
        self.b_var = tk.StringVar()
        self.out_var = tk.StringVar()

        self._file_row(
            main, 0,
            "A 视频/图片（会员60fps）：",
            self.a_var, self._browse_a,
        )
        self._file_row(
            main, 1,
            "B 视频/图片（非会员30fps）：",
            self.b_var, self._browse_b,
        )
        self._file_row(
            main, 2, "输出 mp4：", self.out_var, self._browse_output
        )

        options = ttk.LabelFrame(main, text="合成参数", padding=8)
        options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        options.columnconfigure(1, weight=0)
        options.columnconfigure(3, weight=0)
        options.columnconfigure(5, weight=1)

        self.auto_res_var = tk.BooleanVar(value=True)
        self.auto_res_check = ttk.Checkbutton(
            options,
            text="自动分辨率",
            variable=self.auto_res_var,
            command=self._sync_resolution_state,
        )
        self.auto_res_check.grid(row=0, column=0, sticky="w", padx=(0, 8))

        ttk.Label(options, text="宽").grid(row=0, column=1, sticky="e")
        self.width_var = tk.StringVar(value="1920")
        self.width_entry = ttk.Entry(
            options, textvariable=self.width_var, width=8
        )
        self.width_entry.grid(row=0, column=2, sticky="w", padx=4)

        ttk.Label(options, text="x").grid(row=0, column=3)
        ttk.Label(options, text="高").grid(row=0, column=4, sticky="e")
        self.height_var = tk.StringVar(value="1080")
        self.height_entry = ttk.Entry(
            options, textvariable=self.height_var, width=8
        )
        self.height_entry.grid(row=0, column=5, sticky="w", padx=4)

        ttk.Label(options, text="输出帧率").grid(
            row=0, column=6, sticky="e", padx=(18, 4)
        )
        self.fps_var = tk.StringVar(value="120")
        self.fps_combo = ttk.Combobox(
            options,
            textvariable=self.fps_var,
            values=["120", "240"],
            state="readonly",
            width=7,
        )
        self.fps_combo.grid(row=0, column=7, sticky="w")

        self.auto_dur_var = tk.BooleanVar(value=True)
        self.auto_dur_check = ttk.Checkbutton(
            options,
            text="自动时长",
            variable=self.auto_dur_var,
            command=self._sync_duration_state,
        )
        self.auto_dur_check.grid(row=1, column=0, sticky="w", pady=(8, 0))

        ttk.Label(options, text="手动时长（秒）").grid(
            row=1, column=1, columnspan=2, sticky="e", pady=(8, 0)
        )
        self.duration_var = tk.StringVar(value="10.0")
        self.duration_entry = ttk.Entry(
            options, textvariable=self.duration_var, width=10
        )
        self.duration_entry.grid(
            row=1, column=3, sticky="w", padx=4, pady=(8, 0)
        )

        self.show_advanced_var = tk.BooleanVar(value=False)
        self.advanced_toggle = ttk.Checkbutton(
            main,
            text="高级选项（画面适配 / 删帧 / 结束策略 / 编码 / 调试）",
            variable=self.show_advanced_var,
            command=self._toggle_advanced,
        )
        self.advanced_toggle.grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )

        advanced = ttk.LabelFrame(main, text="高级选项", padding=8)
        advanced.grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(0, 6)
        )
        self.advanced_frame = advanced

        ttk.Label(advanced, text="画面适配").grid(
            row=0, column=0, sticky="w"
        )
        self.fit_mode_var = tk.StringVar(value=FIT_MODE_OPTIONS[0][0])
        self.fit_mode_combo = ttk.Combobox(
            advanced,
            textvariable=self.fit_mode_var,
            values=[x[0] for x in FIT_MODE_OPTIONS],
            state="readonly",
            width=32,
        )
        self.fit_mode_combo.grid(
            row=0, column=1, sticky="w", padx=(4, 16)
        )

        self.debug_var = tk.BooleanVar(value=False)
        self.debug_check = ttk.Checkbutton(
            advanced,
            text="调试模式（显示 ffprobe / ffmpeg 英文原文）",
            variable=self.debug_var,
        )
        self.debug_check.grid(
            row=0, column=2, columnspan=2, sticky="w"
        )

        ttk.Label(advanced, text="A 删帧").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.a_keep_var = tk.StringVar(value=A_KEEP_OPTIONS[0][0])
        self.a_keep_combo = ttk.Combobox(
            advanced,
            textvariable=self.a_keep_var,
            values=[x[0] for x in A_KEEP_OPTIONS],
            state="readonly",
            width=30,
        )
        self.a_keep_combo.grid(
            row=1, column=1, sticky="w", padx=(4, 16), pady=(8, 0)
        )

        ttk.Label(advanced, text="B 删帧").grid(
            row=1, column=2, sticky="e", padx=(0, 4), pady=(8, 0)
        )
        self.b_keep_var = tk.StringVar(value=B_KEEP_OPTIONS[0][0])
        self.b_keep_combo = ttk.Combobox(
            advanced,
            textvariable=self.b_keep_var,
            values=[x[0] for x in B_KEEP_OPTIONS],
            state="readonly",
            width=30,
        )
        self.b_keep_combo.grid(
            row=1, column=3, sticky="w", pady=(8, 0)
        )

        ttk.Label(advanced, text="结束策略").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self.end_mode_var = tk.StringVar(value="hold")
        self.hold_radio = ttk.Radiobutton(
            advanced,
            text="保持最后一帧",
            variable=self.end_mode_var,
            value="hold",
        )
        self.hold_radio.grid(
            row=2, column=1, sticky="w", padx=(4, 16), pady=(8, 0)
        )

        self.black_radio = ttk.Radiobutton(
            advanced,
            text="黑屏",
            variable=self.end_mode_var,
            value="black",
        )
        self.black_radio.grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )

        self.loop_var = tk.BooleanVar(value=True)
        self.loop_check = ttk.Checkbutton(
            advanced,
            text="短的一方循环播放（优先于结束策略）",
            variable=self.loop_var,
            command=self._sync_end_mode_state,
        )
        self.loop_check.grid(
            row=2, column=3, sticky="w", pady=(8, 0)
        )

        ttk.Label(advanced, text="CRF").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self.crf_var = tk.StringVar(value="18")
        self.crf_spin = ttk.Spinbox(
            advanced,
            from_=0,
            to=51,
            textvariable=self.crf_var,
            width=7,
        )
        self.crf_spin.grid(
            row=3, column=1, sticky="w", padx=(4, 16), pady=(8, 0)
        )

        ttk.Label(advanced, text="preset").grid(
            row=3, column=2, sticky="e", padx=(0, 4), pady=(8, 0)
        )
        self.preset_var = tk.StringVar(value="fast")
        self.preset_combo = ttk.Combobox(
            advanced,
            textvariable=self.preset_var,
            values=PRESETS,
            state="readonly",
            width=12,
        )
        self.preset_combo.grid(
            row=3, column=3, sticky="w", pady=(8, 0)
        )

        # 默认折叠高级选项。
        self.advanced_frame.grid_remove()

        buttons = ttk.Frame(main)
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)
        buttons.columnconfigure(0, weight=1)

        self.start_btn = ttk.Button(
            buttons, text="开始合成", command=self._start
        )
        self.start_btn.grid(row=0, column=1, padx=(0, 8))

        self.cancel_btn = ttk.Button(
            buttons,
            text="取消",
            command=self._cancel,
            state="disabled",
        )
        self.cancel_btn.grid(row=0, column=2)

        self.progress = ttk.Progressbar(
            main, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress.grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(2, 4)
        )

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(
            main, textvariable=self.status_var, anchor="w"
        ).grid(row=8, column=0, columnspan=3, sticky="ew")

        ttk.Label(main, text="日志").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(6, 2)
        )
        log_frame = ttk.Frame(main)
        log_frame.grid(
            row=10, column=0, columnspan=3, sticky="nsew"
        )
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            height=14,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            background="#111111",
            foreground="#e6e6e6",
            insertbackground="#e6e6e6",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _file_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", pady=3
        )
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=(4, 6), pady=3)
        ttk.Button(parent, text="浏览…", command=command).grid(
            row=row, column=2, sticky="e", pady=3
        )

    # ---- 状态联动 --------------------------------------------------------

    def _sync_resolution_state(self) -> None:
        state = "disabled" if self.auto_res_var.get() else "normal"
        self.width_entry.configure(state=state)
        self.height_entry.configure(state=state)

    def _sync_duration_state(self) -> None:
        state = "disabled" if self.auto_dur_var.get() else "normal"
        self.duration_entry.configure(state=state)

    def _sync_end_mode_state(self) -> None:
        state = "disabled" if self.loop_var.get() else "normal"
        self.hold_radio.configure(state=state)
        self.black_radio.configure(state=state)

    def _toggle_advanced(self) -> None:
        if self.show_advanced_var.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    # ---- 文件选择 --------------------------------------------------------

    def _browse_a(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 A 视频或图片（会员60fps）",
            filetypes=FILE_DIALOG_TYPES,
        )
        if path:
            self.a_var.set(path)

    def _browse_b(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 B 视频或图片（非会员30fps）",
            filetypes=FILE_DIALOG_TYPES,
        )
        if path:
            self.b_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择输出 mp4",
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4")],
        )
        if path:
            self.out_var.set(path)

    # ---- 日志 / 队列 -----------------------------------------------------

    def _append_logs(self, lines: List[str]) -> None:
        """把一批日志一次性插入 Text，避免每条都 configure/see 造成卡顿。"""
        if not lines:
            return
        # 用户正往上翻看历史时不强行拉到底部。
        at_bottom = self.log_text.yview()[1] >= 0.999
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "\n".join(lines) + "\n")

        # 超过上限时批量裁掉最旧的部分，限制 Tk 文本规模。
        try:
            line_count = int(
                self.log_text.index("end-1c").split(".", 1)[0]
            )
            if line_count > self.MAX_LOG_LINES + 500:
                remove = line_count - self.MAX_LOG_LINES
                self.log_text.delete("1.0", f"{remove + 1}.0")
        except tk.TclError:
            pass

        if at_bottom:
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _current_token(self) -> Optional[str]:
        worker = self.worker
        if worker is not None and worker.is_alive():
            return worker.token
        return None

    def _handle_terminal(self, message) -> None:
        kind = message[0]
        if kind == "done":
            self._set_running(False)
            self.progress["value"] = 100
            detail = (
                message[3] if len(message) > 3
                else "成品信息未校验"
            )
            self.status_var.set("合成完成 · " + detail)
            if not self._closing:
                messagebox.showinfo(
                    "完成",
                    "视频已生成：\n"
                    + message[1]
                    + "\n\n成品信息："
                    + detail,
                )
        elif kind == "error":
            self._set_running(False)
            self.status_var.set("出错")
            if not self._closing:
                messagebox.showerror("合成失败", message[1])
        elif kind == "cancelled":
            self._set_running(False)
            self.status_var.set("已取消")
            if not self._closing:
                messagebox.showinfo("已取消", "合成已取消。")

    def _poll_queue(self) -> None:
        """
        每轮只处理有限条消息，进度/状态只取最后一条，日志合并成一次插入。
        这样即使后台瞬间产生大量消息，Tk 主线程也不会长时间卡住。
        """
        logs: List[str] = []
        last_progress = None
        last_status = None
        terminal = None
        processed = 0

        try:
            while processed < self.QUEUE_BATCH:
                message = self.msg_queue.get_nowait()
                processed += 1
                kind = message[0]
                token = message[-1] if isinstance(message[-1], str) else None

                # 丢弃上一任务的 auto_* / done / error / cancelled 消息。
                if kind in (
                    "auto_size", "auto_target", "done", "error", "cancelled",
                ):
                    cur = self._current_token()
                    if cur is not None and token is not None and token != cur:
                        continue

                if kind == "log":
                    logs.append(message[1])
                elif kind == "progress":
                    last_progress = float(message[1])
                elif kind == "status":
                    last_status = message[1]
                elif kind == "auto_size":
                    if self.auto_res_var.get():
                        self.width_var.set(str(message[1]))
                        self.height_var.set(str(message[2]))
                elif kind == "auto_target":
                    if self.auto_dur_var.get():
                        self.duration_var.set(f"{float(message[1]):.3f}")
                else:
                    terminal = message
                    break
        except queue.Empty:
            pass

        # 先落日志，再处理进度/状态/结束提示，保证最后几行不会丢。
        if logs:
            self._append_logs(logs)
        if last_progress is not None:
            self.progress["value"] = last_progress * 100.0
        if last_status is not None:
            self.status_var.set(last_status)
        if terminal is not None:
            self._handle_terminal(terminal)
            more = False
        else:
            # 到达本批上限说明队列可能还有积压，尽快继续；否则正常轮询。
            more = (processed >= self.QUEUE_BATCH)

        if not self._closing:
            self.root.after(
                10 if more else 100, self._poll_queue
            )

    # ---- 启动 / 取消 -----------------------------------------------------

    def _collect_config(self) -> ComposeConfig:
        path_a = self.a_var.get().strip()
        path_b = self.b_var.get().strip()
        output = self.out_var.get().strip()

        if not path_a:
            raise ValueError("请先选择 A 文件。")
        if not path_b:
            raise ValueError("请先选择 B 文件。")
        if not output:
            raise ValueError("请先选择输出 mp4 路径。")

        for label, path in (("A", path_a), ("B", path_b)):
            if not os.path.isfile(path):
                raise ValueError(f"{label} 文件不存在：\n{path}")

        if not output.lower().endswith(".mp4"):
            output += ".mp4"
            self.out_var.set(output)

        abs_out = os.path.abspath(output)
        abs_a = os.path.abspath(path_a)
        abs_b = os.path.abspath(path_b)
        if os.path.normcase(abs_out) in (
            os.path.normcase(abs_a),
            os.path.normcase(abs_b),
        ):
            raise ValueError("输出文件不能覆盖 A 或 B 源文件。")

        parent = os.path.dirname(abs_out)
        if parent and not os.path.isdir(parent):
            raise ValueError(f"输出目录不存在：\n{parent}")

        auto_resolution = bool(self.auto_res_var.get())
        if auto_resolution:
            width, height = 0, 0
        else:
            try:
                width = int(self.width_var.get().strip())
                height = int(self.height_var.get().strip())
            except ValueError as exc:
                raise ValueError("手动分辨率必须填写整数宽高。") from exc
            if width <= 0 or height <= 0:
                raise ValueError("手动分辨率必须大于 0。")
            if width % 2 or height % 2:
                raise ValueError("H.264 yuv420p 要求宽高为偶数。")

        try:
            out_fps = int(self.fps_var.get().strip())
        except ValueError as exc:
            raise ValueError("输出帧率无效。") from exc
        if out_fps <= 0 or out_fps % 2 != 0:
            raise ValueError("输出帧率必须是正偶数。")

        auto_duration = bool(self.auto_dur_var.get())
        if auto_duration:
            duration = 0.0
        else:
            try:
                duration = float(self.duration_var.get().strip())
            except ValueError as exc:
                raise ValueError("手动时长必须是数字（秒）。") from exc
            if duration <= 0:
                raise ValueError("手动时长必须大于 0 秒。")

        a_index = self.a_keep_combo.current()
        if a_index < 0:
            a_index = 0
        b_index = self.b_keep_combo.current()
        if b_index < 0:
            b_index = 0
        keep_a = A_KEEP_OPTIONS[a_index][1]
        keep_b = B_KEEP_OPTIONS[b_index][1]

        try:
            crf = int(self.crf_var.get().strip())
        except ValueError as exc:
            raise ValueError("CRF 必须是 0～51 的整数。") from exc
        if crf < 0 or crf > 51:
            raise ValueError("CRF 必须在 0～51 之间。")

        preset = self.preset_var.get().strip()
        if preset not in PRESETS:
            raise ValueError("请选择有效的 preset。")

        end_mode = self.end_mode_var.get()
        if end_mode not in ("hold", "black"):
            end_mode = "hold"

        fit_index = self.fit_mode_combo.current()
        if fit_index < 0:
            fit_index = 0
        fit_mode = FIT_MODE_OPTIONS[fit_index][1]

        return ComposeConfig(
            path_a=path_a,
            path_b=path_b,
            output=os.path.abspath(output),
            auto_resolution=auto_resolution,
            width=width,
            height=height,
            out_fps=out_fps,
            auto_duration=auto_duration,
            duration=duration,
            keep_a=keep_a,
            keep_b=keep_b,
            end_mode=end_mode,
            loop_short=bool(self.loop_var.get()),
            crf=crf,
            preset=preset,
            fit_mode=fit_mode,
            debug=bool(self.debug_var.get()),
        )

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        try:
            cfg = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if os.path.isfile(cfg.output):
            if not messagebox.askyesno(
                "覆盖确认",
                "输出文件已存在，是否覆盖？\n\n" + cfg.output,
            ):
                return

        self._clear_log()
        self.progress["value"] = 0
        self.status_var.set("准备中…")
        self._set_running(True)

        self.worker = ComposeWorker(cfg, self.msg_queue)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.request_cancel()
            self.cancel_btn.configure(state="disabled")
            self.status_var.set("正在取消…")

    def _set_running(self, running: bool) -> None:
        self.start_btn.configure(
            state="disabled" if running else "normal"
        )
        self.cancel_btn.configure(
            state="normal" if running else "disabled"
        )

    # ---- 关闭窗口 --------------------------------------------------------

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno(
                "退出",
                "正在合成，退出将取消当前任务。\n确定退出吗？",
            ):
                return
            self._closing = True
            self.worker.request_cancel()
            # 修复 #19：关闭时给出明确提示，避免用户以为卡死。
            self.status_var.set("正在取消并退出，请稍候（最多数秒）…")
            self._wait_worker_and_close()
            return
        self._closing = True
        self.root.destroy()

    def _wait_worker_and_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.root.after(100, self._wait_worker_and_close)
        else:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()