#!/usr/bin/env python3
"""
s3_automation.py — Standalone automated GoPro processing pipeline for
s3://imu-gopro/.

Workflow:
  1. List videos under s3://imu-gopro/input/** (any nested folder depth).
  2. Skip any video already recorded in the processed-tracker JSON.
  3. Pick up to 2 unprocessed videos and process them in parallel:
       - Download to an isolated temporary workspace.
       - Extract GoPro IMU (ACCL/GYRO) from the gpmd track.
       - Run MediaPipe HandLandmarker on every frame.
       - Build a single .mcap that bundles the source video, IMU samples,
         hand landmarks, depth-availability marker and session metadata.
       - Upload ONLY <stem>.mcap to s3://imu-gopro/output/<same-nested-path>/<stem>.mcap.
       - Delete the temporary workspace for that video.
       - Record the S3 key + etag in processed.json (local) and mirror to S3.
  4. When no unprocessed videos remain, sleep 2 min and re-list. Repeat forever.
  5. If only 1 unprocessed video is found, process that one alone — don't wait
     for a second video to form a "pair".

This script is self-contained — no dependency on gopro.py. It auto-installs
its own Python packages on first run.

Requires:
  AWS credentials reachable via the default boto3 chain (env vars,
  ~/.aws/credentials, IAM role, ...) OR the bucket configured as
  publicly readable/writable on input/ and output/ (set PUBLIC_BUCKET=1).

Run:
  python s3_automation.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


# --------------------------------------------------------------------------- #
# Runtime dependency bootstrap (same approach as the original gopro.py)
# --------------------------------------------------------------------------- #
def _pip_install(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--disable-pip-version-check", *pkgs]
    )


def _ensure(pip_name: str, import_name: str | None = None):
    mod = import_name or pip_name.split("==")[0].split(">=")[0].replace("-", "_")
    try:
        return importlib.import_module(mod)
    except ImportError:
        print(f"[install] {pip_name} ...", flush=True)
        _pip_install(pip_name)
        return importlib.import_module(mod)


_ensure("numpy")
_ensure("opencv-python", "cv2")
_ensure("imageio-ffmpeg", "imageio_ffmpeg")
_ensure("mediapipe")
_ensure("mcap")
_ensure("boto3")

import cv2                    # noqa: E402
import mediapipe as mp        # noqa: E402
import imageio_ffmpeg         # noqa: E402
import boto3                  # noqa: E402
from botocore import UNSIGNED  # noqa: E402
from botocore.config import Config as BotoConfig            # noqa: E402
from botocore.exceptions import ClientError, NoCredentialsError  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BUCKET             = "imu-gopro"
REGION             = "us-east-1"
INPUT_PREFIX       = "input/"
OUTPUT_PREFIX      = "output/"
PROCESSED_S3_KEY   = "output/_system_tracker/_processed.json"
SCRIPT_DIR         = Path(__file__).resolve().parent
PROCESSED_LOCAL    = SCRIPT_DIR / "processed.json"
TEMP_ROOT          = SCRIPT_DIR / "_tmp_workspace"
VIDEO_EXTS         = {".mp4", ".mov", ".mkv", ".m4v", ".lrv", ".ts"}
IDLE_POLL_SECONDS  = 120
CONCURRENCY        = 2

# PUBLIC_BUCKET=1 (default) -> unsigned anonymous S3 requests, no creds needed.
# PUBLIC_BUCKET=0          -> use boto3 default credential chain.
PUBLIC_BUCKET = os.environ.get("PUBLIC_BUCKET", "1") == "1"

_boto_cfg = BotoConfig(
    region_name=REGION,
    retries={"max_attempts": 8, "mode": "standard"},
    connect_timeout=20,
    read_timeout=300,
    signature_version=UNSIGNED if PUBLIC_BUCKET else None,
)
s3 = boto3.client("s3", config=_boto_cfg)
_tracker_lock = Lock()


def _log(msg: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


# =========================================================================== #
#                                                                             #
#  PART 1 — GoPro video processing (inlined from gopro.py, MIT-style license) #
#                                                                             #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# GPMF parser (GoPro metadata format)
# Spec: https://github.com/gopro/gpmf-parser
# --------------------------------------------------------------------------- #
_NUMERIC = {
    "b": (">b", 1),  "B": (">B", 1),
    "s": (">h", 2),  "S": (">H", 2),
    "l": (">i", 4),  "L": (">I", 4),
    "j": (">q", 8),  "J": (">Q", 8),
    "f": (">f", 4),  "d": (">d", 8),
    "q": (">i", 4),  "Q": (">I", 4),   # fixed point
    "F": (">4s", 4),
}


def _decode_payload(t: str, ssize: int, payload: bytes):
    if t == "\x00":
        return None
    if t in ("c", "U"):
        try:
            return [payload.decode("latin-1", errors="replace").rstrip("\x00")]
        except Exception:
            return [payload]
    if t not in _NUMERIC:
        return None
    fmt, esz = _NUMERIC[t]
    if esz == 0 or ssize == 0:
        return []
    per = max(ssize // esz, 1)
    out = []
    for i in range(0, len(payload), ssize):
        chunk = payload[i:i + ssize]
        row = []
        for j in range(per):
            piece = chunk[j * esz:(j + 1) * esz]
            if len(piece) != esz:
                continue
            v = struct.unpack(fmt, piece)[0]
            if t == "q":
                v = v / 65536.0
            elif t == "Q":
                v = v / (2 ** 32)
            row.append(v)
        if not row:
            continue
        out.append(row[0] if per == 1 else row)
    return out


def _parse_klv(buf: bytes):
    items = []
    p, n = 0, len(buf)
    while p + 8 <= n:
        fourcc = buf[p:p + 4].decode("latin-1", errors="replace")
        t_byte = buf[p + 4]
        t = chr(t_byte) if t_byte != 0 else "\x00"
        ssize = buf[p + 5]
        repeat = struct.unpack(">H", buf[p + 6:p + 8])[0]
        psize = ssize * repeat
        aligned = (psize + 3) & ~3
        payload = buf[p + 8:p + 8 + psize]
        p += 8 + aligned
        children = _parse_klv(payload) if t == "\x00" and psize > 0 else None
        values = _decode_payload(t, ssize, payload) if t != "\x00" else None
        items.append({"k": fourcc, "t": t, "ssize": ssize, "n": repeat,
                      "values": values, "children": children})
    return items


# --------------------------------------------------------------------------- #
# MP4 box walker — locate the gpmd track and return its sample bytes
# --------------------------------------------------------------------------- #
def _read_box(f, end):
    pos = f.tell()
    if end - pos < 8:
        return None
    size = struct.unpack(">I", f.read(4))[0]
    box_type = f.read(4).decode("latin-1", errors="replace")
    header = 8
    if size == 1:
        size = struct.unpack(">Q", f.read(8))[0]
        header = 16
    elif size == 0:
        size = end - pos
    return {"pos": pos, "size": size, "type": box_type,
            "data_pos": pos + header, "data_end": pos + size}


def _walk_boxes(f, end):
    out = []
    while f.tell() + 8 <= end:
        box = _read_box(f, end)
        if box is None:
            break
        out.append(box)
        f.seek(box["data_end"])
    return out


def _find_gpmd_track(video_path: Path):
    with open(video_path, "rb") as f:
        f.seek(0, 2)
        file_end = f.tell()
        f.seek(0)
        top = _walk_boxes(f, file_end)
        moov = next((b for b in top if b["type"] == "moov"), None)
        if not moov:
            return None
        f.seek(moov["data_pos"])
        moov_boxes = _walk_boxes(f, moov["data_end"])
        for trak in (b for b in moov_boxes if b["type"] == "trak"):
            f.seek(trak["data_pos"])
            trak_boxes = _walk_boxes(f, trak["data_end"])
            mdia = next((b for b in trak_boxes if b["type"] == "mdia"), None)
            if not mdia:
                continue
            f.seek(mdia["data_pos"])
            mdia_boxes = _walk_boxes(f, mdia["data_end"])
            hdlr = next((b for b in mdia_boxes if b["type"] == "hdlr"), None)
            handler_type = ""
            if hdlr:
                f.seek(hdlr["data_pos"] + 8)
                handler_type = f.read(4).decode("latin-1", errors="replace")
            if handler_type != "meta":
                continue
            mdhd = next((b for b in mdia_boxes if b["type"] == "mdhd"), None)
            timescale = 1
            if mdhd:
                f.seek(mdhd["data_pos"])
                version = f.read(1)[0]
                f.read(3)
                if version == 1:
                    f.read(16)
                    timescale = struct.unpack(">I", f.read(4))[0]
                else:
                    f.read(8)
                    timescale = struct.unpack(">I", f.read(4))[0]
            minf = next((b for b in mdia_boxes if b["type"] == "minf"), None)
            if not minf:
                continue
            f.seek(minf["data_pos"])
            minf_boxes = _walk_boxes(f, minf["data_end"])
            stbl = next((b for b in minf_boxes if b["type"] == "stbl"), None)
            if not stbl:
                continue
            f.seek(stbl["data_pos"])
            stbl_boxes = _walk_boxes(f, stbl["data_end"])
            stsd = next((b for b in stbl_boxes if b["type"] == "stsd"), None)
            is_gpmd = False
            if stsd:
                f.seek(stsd["data_pos"] + 8)
                struct.unpack(">I", f.read(4))[0]
                entry_type = f.read(4).decode("latin-1", errors="replace")
                if entry_type == "gpmd":
                    is_gpmd = True
            if not is_gpmd:
                continue

            stsz = next((b for b in stbl_boxes if b["type"] == "stsz"), None)
            sizes = []
            if stsz:
                f.seek(stsz["data_pos"] + 4)
                sample_size = struct.unpack(">I", f.read(4))[0]
                count = struct.unpack(">I", f.read(4))[0]
                if sample_size == 0:
                    sizes = [struct.unpack(">I", f.read(4))[0] for _ in range(count)]
                else:
                    sizes = [sample_size] * count

            stco = next((b for b in stbl_boxes if b["type"] == "stco"), None)
            co64 = next((b for b in stbl_boxes if b["type"] == "co64"), None)
            chunk_offsets = []
            if stco:
                f.seek(stco["data_pos"] + 4)
                count = struct.unpack(">I", f.read(4))[0]
                chunk_offsets = [struct.unpack(">I", f.read(4))[0] for _ in range(count)]
            elif co64:
                f.seek(co64["data_pos"] + 4)
                count = struct.unpack(">I", f.read(4))[0]
                chunk_offsets = [struct.unpack(">Q", f.read(8))[0] for _ in range(count)]

            stsc = next((b for b in stbl_boxes if b["type"] == "stsc"), None)
            stsc_entries = []
            if stsc:
                f.seek(stsc["data_pos"] + 4)
                count = struct.unpack(">I", f.read(4))[0]
                for _ in range(count):
                    first_chunk = struct.unpack(">I", f.read(4))[0]
                    spc = struct.unpack(">I", f.read(4))[0]
                    sdi = struct.unpack(">I", f.read(4))[0]
                    stsc_entries.append((first_chunk, spc, sdi))

            stts = next((b for b in stbl_boxes if b["type"] == "stts"), None)
            durations = []
            if stts:
                f.seek(stts["data_pos"] + 4)
                count = struct.unpack(">I", f.read(4))[0]
                for _ in range(count):
                    sc = struct.unpack(">I", f.read(4))[0]
                    sd = struct.unpack(">I", f.read(4))[0]
                    durations.extend([sd] * sc)

            sample_offsets = []
            if not stsc_entries or not chunk_offsets:
                return None
            extended = []
            for i, (fc, spc, sdi) in enumerate(stsc_entries):
                next_first = (stsc_entries[i + 1][0]
                              if i + 1 < len(stsc_entries)
                              else len(chunk_offsets) + 1)
                for c in range(fc, next_first):
                    extended.append((c, spc))
            sample_idx = 0
            for chunk_one_based, spc in extended:
                chunk_zero = chunk_one_based - 1
                if chunk_zero >= len(chunk_offsets):
                    break
                off = chunk_offsets[chunk_zero]
                for _ in range(spc):
                    if sample_idx >= len(sizes):
                        break
                    sample_offsets.append(off)
                    off += sizes[sample_idx]
                    sample_idx += 1
            return {
                "sizes": sizes,
                "offsets": sample_offsets,
                "durations": durations,
                "timescale": timescale,
            }
    return None


def _read_gpmf_payloads(video_path: Path):
    info = _find_gpmd_track(video_path)
    if info is None:
        return
    sizes = info["sizes"]
    offs = info["offsets"]
    durs = info["durations"]
    ts = info["timescale"] or 1
    with open(video_path, "rb") as f:
        elapsed = 0
        for i in range(min(len(sizes), len(offs))):
            f.seek(offs[i])
            data = f.read(sizes[i])
            dur = durs[i] if i < len(durs) else 0
            t_start = elapsed / ts
            elapsed += dur
            yield i, t_start, data


# --------------------------------------------------------------------------- #
# IMU extraction (ACCL + GYRO from GPMF)
# --------------------------------------------------------------------------- #
def _find_streams(items):
    for it in items:
        if it["k"] == "STRM" and it["children"]:
            yield it["children"]
        elif it["children"]:
            yield from _find_streams(it["children"])


def _extract_imu(video_path: Path):
    out = []
    have_any = False
    for _, t_payload_start, raw in _read_gpmf_payloads(video_path):
        if not raw:
            continue
        have_any = True
        items = _parse_klv(raw)
        for stream in _find_streams(items):
            scal = [1.0]
            sensor_kind = None
            samples = None
            for it in stream:
                if it["k"] == "SCAL" and it["values"]:
                    vals = it["values"]
                    scal = [float(v) for v in (vals if isinstance(vals[0], (int, float)) else vals[0])]
                elif it["k"] == "ACCL" and it["values"]:
                    sensor_kind = "accel"
                    samples = it["values"]
                elif it["k"] == "GYRO" and it["values"]:
                    sensor_kind = "gyro"
                    samples = it["values"]
            if not samples or sensor_kind is None:
                continue
            scale_x = scal[0] if scal else 1.0
            scale_y = scal[1] if len(scal) > 1 else scale_x
            scale_z = scal[2] if len(scal) > 2 else scale_x
            n = len(samples)
            window = 1.0
            dt = window / max(n, 1)
            for k, row in enumerate(samples):
                if not isinstance(row, list) or len(row) < 3:
                    continue
                t_sec = t_payload_start + k * dt
                t_ns = int(t_sec * 1e9)
                out.append({
                    "t": t_ns,
                    "sensor": sensor_kind,
                    "x": round(row[0] / scale_x, 4) if scale_x else float(row[0]),
                    "y": round(row[1] / scale_y, 4) if scale_y else float(row[1]),
                    "z": round(row[2] / scale_z, 4) if scale_z else float(row[2]),
                    "accuracy": 3,
                })
    out.sort(key=lambda r: r["t"])
    return out, have_any


# --------------------------------------------------------------------------- #
# Video info via ffmpeg stderr probe
# --------------------------------------------------------------------------- #
def _probe_video(video_path: Path):
    res = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True,
    )
    info = {"width": None, "height": None, "fps": None,
            "duration_s": None, "creation_time": None}
    txt = res.stderr
    m = re.search(r"creation_time\s*:\s*(\S+)", txt)
    if m:
        info["creation_time"] = m.group(1)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", txt)
    if m:
        h, mn, s = m.group(1), m.group(2), m.group(3)
        info["duration_s"] = int(h) * 3600 + int(mn) * 60 + float(s)
    m = re.search(r"Stream #\d+:\d+.*Video:.*?(\d{2,5})x(\d{2,5})", txt)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt)
    if m:
        info["fps"] = float(m.group(1))
    return info


def _parse_creation_time_ms(text: str | None) -> int | None:
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Hand landmarks (MediaPipe tasks API)
# --------------------------------------------------------------------------- #
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_hand_model() -> Path:
    cache = Path.home() / ".cache" / "mediapipe"
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / "hand_landmarker.task"
    if not model.exists() or model.stat().st_size < 1024:
        import urllib.request
        _log(f"[hands] downloading model -> {model}")
        urllib.request.urlretrieve(_HAND_MODEL_URL, model)
    return model


def _extract_hand_landmarks(video_path: Path, start_ms: int, fps: float):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_hand_model()
    base = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    lines = []
    seen_hands = set()
    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        # Optimize: Downscale massive 4K/5K frames to speed up memory copying.
        # MediaPipe uses 256x256 internally, so 1080p is more than enough for accuracy.
        # Normalized output coordinates (0.0 to 1.0) remain perfectly accurate.
        h, w = frame.shape[:2]
        if w > 1080:
            scale = 1080 / w
            frame = cv2.resize(frame, (1080, int(h * scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000.0 / fps)
        result = landmarker.detect_for_video(mp_image, ts_ms)
        if result.hand_landmarks:
            hands_out = []
            for i, hand in enumerate(result.hand_landmarks):
                handedness = "Unknown"
                score = 0.0
                if result.handedness and i < len(result.handedness):
                    cls = result.handedness[i][0]
                    handedness = cls.category_name or cls.display_name or "Unknown"
                    score = float(cls.score)
                hands_out.append({
                    "handedness": handedness,
                    "score": score,
                    "keypoints": [
                        {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
                        for p in hand
                    ],
                })
                seen_hands.add(handedness)
            t_ms = int(start_ms + (frame_idx * 1000.0 / fps))
            lines.append({"t": t_ms, "hands": hands_out})
        frame_idx += 1
        if frame_idx % 200 == 0:
            _log(f"  [hands] frame {frame_idx} ({time.time() - t0:.1f}s)")
    cap.release()
    landmarker.close()
    _log(f"[hands] {len(lines)} frames with hands / {frame_idx} total, "
         f"{len(seen_hands)} distinct hand types")
    return lines, len(seen_hands), frame_idx


# --------------------------------------------------------------------------- #
# MCAP builder — attach the video + write IMU/hands/depth/session messages.
# Returns the path to the produced .mcap.
# --------------------------------------------------------------------------- #
def _build_mcap(video: Path, session_meta: dict, imu_samples: list,
                hand_lines: list, depth_record: dict, out_path: Path,
                video_name: str | None = None) -> Path:
    from mcap.writer import Writer
    _log(f"  [mcap] building {out_path.name} ...")
    t0 = time.time()

    start_ms = int(session_meta.get("start_timestamp", 0))
    start_ns = start_ms * 1_000_000
    empty_schema = b'{"type":"object"}'

    with open(out_path, "wb") as out_file:
        writer = Writer(out_file)
        writer.start(profile="owrecorder", library="owrecorder-mcap-v1")

        # attach the source video (matches gopro.py behaviour)
        with open(video, "rb") as vf:
            writer.add_attachment(
                create_time=start_ns, log_time=start_ns,
                name=video_name or video.name, media_type="video/mp4", data=vf.read(),
            )

        if imu_samples:
            sch = writer.register_schema("sensor_msgs/Imu", "jsonschema", empty_schema)
            ch = writer.register_channel(schema_id=sch, topic="/imu", message_encoding="json")
            for rec in imu_samples:
                ts_ns = int(rec["t"])
                data = json.dumps(rec, separators=(",", ":")).encode("utf-8")
                writer.add_message(channel_id=ch, log_time=ts_ns,
                                   publish_time=ts_ns, data=data)

        if hand_lines:
            sch = writer.register_schema("HandLandmarks", "jsonschema", empty_schema)
            ch = writer.register_channel(schema_id=sch, topic="/hand_landmarks", message_encoding="json")
            for rec in hand_lines:
                ts_ns = int(rec["t"]) * 1_000_000
                data = json.dumps(rec, separators=(",", ":")).encode("utf-8")
                writer.add_message(channel_id=ch, log_time=ts_ns,
                                   publish_time=ts_ns, data=data)

        sch = writer.register_schema("DepthStatus", "jsonschema", empty_schema)
        ch = writer.register_channel(schema_id=sch, topic="/depth_status", message_encoding="json")
        writer.add_message(
            channel_id=ch, log_time=start_ns, publish_time=start_ns,
            data=json.dumps(depth_record).encode("utf-8"),
        )

        sch = writer.register_schema("SessionMetadata", "jsonschema", empty_schema)
        ch = writer.register_channel(schema_id=sch, topic="/session", message_encoding="json")
        writer.add_message(
            channel_id=ch, log_time=start_ns, publish_time=start_ns,
            data=json.dumps(session_meta).encode("utf-8"),
        )

        writer.finish()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    _log(f"  [mcap] done in {time.time() - t0:.1f}s, {size_mb:.1f} MB")
    return out_path


# --------------------------------------------------------------------------- #
# Top-level: process one local video file -> produce files to upload.
# Everything stays in `sandbox`; nothing else is written to disk.
# --------------------------------------------------------------------------- #
def process_video_to_mcap(video: Path, sandbox: Path) -> Path:
    probe = _probe_video(video)
    fps        = probe["fps"]        or 30.0
    width      = probe["width"]      or 1920
    height     = probe["height"]     or 1080
    duration_s = probe["duration_s"] or 0.0

    creation_ms = _parse_creation_time_ms(probe["creation_time"])
    if creation_ms is None:
        creation_ms = int(video.stat().st_mtime * 1000) - int(duration_s * 1000)
    start_ms = creation_ms
    end_ms = creation_ms + int(duration_s * 1000)

    dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    session_id = f"video_{dt.strftime('%Y-%m-%d_%H%M')}_{start_ms}"
    _log(f"[session] {session_id}")
    _log(f"[video] {width}x{height} @ {fps:.3f} fps, duration {duration_s:.2f}s")

    # ----- IMU
    imu_samples, gpmf_any = _extract_imu(video)
    _log(f"[imu] {len(imu_samples)} samples "
         f"({'gpmd track found' if gpmf_any else 'NO gpmd track'})")

    # ----- Hand landmarks
    hand_lines, distinct_handed, _frame_count = _extract_hand_landmarks(
        video, start_ms, fps,
    )

    # ----- Depth not available on GoPro
    depth_record = {"available": False, "reason": "no_depth_sensor"}

    # ----- Session metadata
    session_meta = {
        "session_id": session_id,
        "voice_triggered": False,
        "resolution": f"{width}x{height}",
        "fps": round(float(fps), 6),
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "actual_hands_detected": int(distinct_handed),
        "imu_sample_count": len(imu_samples),
        "imu_epoch_offset_ns": start_ms * 1_000_000,
        "depth_available": False,
        "depth_frame_count": 0,
        "depth_resolution": "",
        "depth_source": "none_hardware",
        "depth_reason": "GoPro has no ToF sensor",
        "depth_recoverable_from_video": True,
        "depth_recovery_method": "monocular_estimation_with_imu_scale",
        "device_name": "GoPro",
        "source_video": video.name,
    }

    # ----- Mute Video
    muted_video = sandbox / f"{video.stem}_muted{video.suffix}"
    _log(f"[ffmpeg] muting audio for {video.name}...")
    subprocess.run([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-c:v", "copy",
        "-an",
        str(muted_video)
    ], check=True)

    mcap_path = sandbox / f"{video.stem}.mcap"
    _build_mcap(muted_video, session_meta, imu_samples, hand_lines, depth_record, mcap_path, video_name=video.name)

    return mcap_path


# =========================================================================== #
#                                                                             #
#  PART 2 — S3 pipeline (crawl, dispatch, track)                              #
#                                                                             #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Processed-tracker JSON
# --------------------------------------------------------------------------- #
def _load_tracker() -> dict:
    """Tracker shape: {"processed": {key: {"etag": str, "ts": int}}}"""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=PROCESSED_S3_KEY)
        body = resp["Body"].read().decode("utf-8")
        data = json.loads(body)
        try:
            PROCESSED_LOCAL.write_text(body, encoding="utf-8")
        except Exception:
            pass
        return data
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("NoSuchKey", "404", "AccessDenied"):
            _log(f"[warn] couldn't read S3 tracker ({code}): {e}")
    except Exception as e:
        _log(f"[warn] couldn't read S3 tracker: {e}")

    if PROCESSED_LOCAL.exists():
        try:
            return json.loads(PROCESSED_LOCAL.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"[warn] couldn't read local tracker: {e}")
    return {"processed": {}}


def _save_tracker(tracker: dict):
    tracker["updated_at"] = int(time.time())
    blob = json.dumps(tracker, indent=2, sort_keys=True)
    try:
        PROCESSED_LOCAL.write_text(blob, encoding="utf-8")
    except Exception as e:
        _log(f"[warn] couldn't write local tracker: {e}")
    try:
        s3.put_object(
            Bucket=BUCKET, Key=PROCESSED_S3_KEY,
            Body=blob.encode("utf-8"), ContentType="application/json",
        )
    except Exception as e:
        _log(f"[warn] couldn't sync tracker to S3: {e}")


def _is_processed(tracker: dict, key: str, etag: str) -> bool:
    processed = tracker.get("processed", {})
    
    # 1. If this exact path was processed before, skip it
    if key in processed:
        return True
        
    # 2. If this exact file content (by S3 ETag) was processed under any folder, skip it
    for p_data in processed.values():
        if p_data.get("etag") == etag:
            return True
            
    return False


def _mark_processed(tracker: dict, key: str, etag: str):
    with _tracker_lock:
        tracker.setdefault("processed", {})[key] = {
            "etag": etag,
            "ts": int(time.time()),
        }
        _save_tracker(tracker)


# --------------------------------------------------------------------------- #
# S3 listing + key mapping
# --------------------------------------------------------------------------- #
def _list_videos():
    videos = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=INPUT_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if Path(key).suffix.lower() not in VIDEO_EXTS:
                continue
            videos.append({
                "key": key,
                "size": int(obj["Size"]),
                "etag": obj["ETag"].strip('"'),
            })
    return videos


def _sandbox_for(key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return TEMP_ROOT / f"{Path(key).stem}_{h}"

class ProgressLogger:
    def __init__(self, action: str, total_bytes: float, name: str):
        self.action = action
        self.total = total_bytes
        self.name = name
        self.seen = 0

    def __call__(self, bytes_amount):
        self.seen += bytes_amount
        pct = int((self.seen / self.total) * 100) if self.total else 0
        # Print smoothly on every update so it never looks frozen
        sys.stdout.write(f"\r  [{self.action}] {self.name}: {self.seen/(1024*1024):.1f}MB / {self.total/(1024*1024):.1f}MB ({pct}%)    ")
        sys.stdout.flush()

def _prune_empty_s3_dirs(bucket: str, key: str, base_prefix: str):
    """Removes empty directory markers in S3 up to the base_prefix."""
    parts = key.split('/')[:-1]
    for i in range(len(parts), 0, -1):
        dir_key = "/".join(parts[:i]) + "/"
        if dir_key == base_prefix or not dir_key.startswith(base_prefix):
            break
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=dir_key, MaxKeys=2)
            contents = resp.get("Contents", [])
            
            is_empty = True
            for obj in contents:
                if obj["Key"] != dir_key:
                    is_empty = False
                    break
                    
            if is_empty:
                s3.delete_object(Bucket=bucket, Key=dir_key)
                _log(f"[cleanup] removed empty S3 folder: {dir_key}")
            else:
                break # Not empty, so parent won't be either
        except Exception as e:
            _log(f"[warn] failed to check/remove S3 folder {dir_key}: {e}")
            break

def _output_key_for(input_key: str, stem: str, file_name: str) -> str:
    """input/A/B/C.MP4 -> output/A/B/C/<file_name>"""
    rel = input_key[len(INPUT_PREFIX):]
    rel_dir = Path(rel).parent.as_posix()
    if rel_dir in ("", "."):
        return f"{OUTPUT_PREFIX}{stem}/{file_name}"
    return f"{OUTPUT_PREFIX}{rel_dir}/{stem}/{file_name}"


# --------------------------------------------------------------------------- #
# Per-video pipeline (one thread)
# --------------------------------------------------------------------------- #
def _process_one(video_obj: dict) -> tuple[str, str, bool, str | None]:
    key = video_obj["key"]
    etag = video_obj["etag"]
    sandbox = _sandbox_for(key)
    local_video = sandbox / Path(key).name

    try:
        sandbox.mkdir(parents=True, exist_ok=True)

        _log(f"[download] s3://{BUCKET}/{key} ({video_obj['size'] / (1024*1024):.1f} MB)")
        t0 = time.time()
        dl_prog = ProgressLogger("download", video_obj['size'], Path(key).name)
        s3.download_file(BUCKET, key, str(local_video), Callback=dl_prog)
        print() # new line after progress bar
        _log(f"[download] done in {time.time() - t0:.1f}s")

        _log(f"[process] {key}")
        mcap_path = process_video_to_mcap(local_video, sandbox)

        dest_key = _output_key_for(key, local_video.stem, mcap_path.name)
        _log(f"[upload] {mcap_path.name} -> s3://{BUCKET}/{dest_key}")
        
        from boto3.s3.transfer import TransferConfig
        # Disable multipart uploads (up to the AWS 5GB single PUT limit)
        # since AWS blocks multipart for anonymous users
        config = TransferConfig(multipart_threshold=5 * 1024 * 1024 * 1024)
        
        up_prog = ProgressLogger("upload", mcap_path.stat().st_size, mcap_path.name)
        s3.upload_file(
            str(mcap_path), BUCKET, dest_key,
            ExtraArgs={"ContentType": "application/x-mcap"},
            Config=config,
            Callback=up_prog
        )
        print() # new line after progress bar

        _log(f"[cleanup] deleting source video s3://{BUCKET}/{key} to manage space")
        try:
            s3.delete_object(Bucket=BUCKET, Key=key)
            _prune_empty_s3_dirs(BUCKET, key, INPUT_PREFIX)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "AccessDenied":
                _log(f"[warn] Cannot delete s3://{BUCKET}/{key} due to AccessDenied. Update your bucket policy to allow s3:DeleteObject.")
            else:
                raise

        return key, etag, True, None

    except Exception as e:
        traceback.print_exc()
        return key, etag, False, str(e)

    finally:
        try:
            if sandbox.exists():
                shutil.rmtree(sandbox, ignore_errors=True)
                _log(f"[cleanup] removed {sandbox}")
        except Exception as e:
            _log(f"[warn] cleanup failed for {sandbox}: {e}")


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def _process_batch(pending: list[dict], tracker: dict):
    batch = pending[:CONCURRENCY]
    _log(f"[batch] starting {len(batch)} video(s) in parallel (cap={CONCURRENCY})")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(_process_one, v): v for v in batch}
        for fut in as_completed(futures):
            key, etag, ok, err = fut.result()
            if ok:
                _mark_processed(tracker, key, etag)
                _log(f"[ok]   {key}")
            else:
                _log(f"[fail] {key}: {err}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
_shutdown = False


def _handle_sigint(signum, frame):
    global _shutdown
    if _shutdown:
        _log("[exit] forced exit")
        sys.exit(130)
    _shutdown = True
    _log("[exit] shutdown requested — finishing in-flight work, "
         "press Ctrl+C again to force quit")


def _sleep_interruptible(seconds: int):
    for _ in range(seconds):
        if _shutdown:
            return
        time.sleep(1)


def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        signal.signal(signal.SIGTERM, _handle_sigint)
    except (AttributeError, ValueError):
        pass

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    _log(f"[startup] bucket=s3://{BUCKET}, input='{INPUT_PREFIX}', output='{OUTPUT_PREFIX}'")
    _log(f"[startup] auth={'anonymous (PUBLIC_BUCKET=1)' if PUBLIC_BUCKET else 'IAM credentials'}")
    _log(f"[startup] concurrency={CONCURRENCY}, idle_poll={IDLE_POLL_SECONDS}s")
    _log(f"[startup] tmp workspace = {TEMP_ROOT}")

    try:
        s3.list_objects_v2(Bucket=BUCKET, Prefix=INPUT_PREFIX, MaxKeys=1)
    except NoCredentialsError:
        _log("[fatal] No AWS credentials found. Either configure creds "
             "(`aws configure`) or run with PUBLIC_BUCKET=1 and a public bucket policy.")
        sys.exit(2)
    except ClientError as e:
        _log(f"[fatal] cannot list s3://{BUCKET}/{INPUT_PREFIX}: {e}")
        sys.exit(2)

    tracker = _load_tracker()
    _log(f"[startup] {len(tracker.get('processed', {}))} videos already processed")

    idle_cycles = 0
    while not _shutdown:
        try:
            all_videos = _list_videos()
        except Exception as e:
            _log(f"[error] list_videos failed: {e}. Sleeping {IDLE_POLL_SECONDS}s.")
            _sleep_interruptible(IDLE_POLL_SECONDS)
            continue

        pending = [v for v in all_videos if not _is_processed(tracker, v["key"], v["etag"])]

        if not pending:
            idle_cycles += 1
            _log(f"[idle] no new videos. tracked={len(tracker.get('processed', {}))}, "
                 f"in_bucket={len(all_videos)}. sleeping {IDLE_POLL_SECONDS}s "
                 f"(idle cycle #{idle_cycles})")
            _sleep_interruptible(IDLE_POLL_SECONDS)
            continue

        idle_cycles = 0
        _log(f"[scan] {len(pending)} unprocessed video(s)")
        _process_batch(pending, tracker)

    _log("[exit] clean shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
