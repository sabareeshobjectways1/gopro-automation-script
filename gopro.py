#!/usr/bin/env python3
"""
extract_gopro_metadata.py

For every GoPro video in a directory (default: current dir), produce a folder
named after the video stem containing the same files as the reference layout:
    <video-stem>/<session>.json
    <video-stem>/<session>_imu.jsonl
    <video-stem>/<session>_hand_landmarks.jsonl
    <video-stem>/<session>_depth_meta.jsonl   (always "not available" — GoPro has no ToF)

The script auto-installs the Python packages it needs on first run.

Usage:
    python extract_gopro_metadata.py                         # scan current dir
    python extract_gopro_metadata.py --dir path/to/videos    # scan a folder
    python extract_gopro_metadata.py GX019585.MP4 OTHER.MP4  # process specific files
    python extract_gopro_metadata.py --force                 # overwrite existing output
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
# Runtime dependency bootstrap
# --------------------------------------------------------------------------- #
def _pip_install(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--disable-pip-version-check", *pkgs]
    )


def ensure(pip_name: str, import_name: str | None = None):
    mod = import_name or pip_name.split("==")[0].split(">=")[0].replace("-", "_")
    try:
        return importlib.import_module(mod)
    except ImportError:
        print(f"[install] {pip_name} ...", flush=True)
        _pip_install(pip_name)
        return importlib.import_module(mod)


# Light, always-required deps
ensure("numpy")
ensure("opencv-python", "cv2")
ensure("imageio-ffmpeg", "imageio_ffmpeg")
ensure("mediapipe")
ensure("mcap")

import numpy as np                  # noqa: E402
import cv2                          # noqa: E402
import mediapipe as mp              # noqa: E402
import imageio_ffmpeg               # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


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
    "q": (">i", 4),  "Q": (">I", 4),  # fixed point
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
    """Return (sample_offsets, sample_sizes, timescale) for the gpmd track, or None."""
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
            # handler
            hdlr = next((b for b in mdia_boxes if b["type"] == "hdlr"), None)
            handler_type = ""
            if hdlr:
                f.seek(hdlr["data_pos"] + 8)
                handler_type = f.read(4).decode("latin-1", errors="replace")
            if handler_type != "meta":
                continue
            # timescale from mdhd
            mdhd = next((b for b in mdia_boxes if b["type"] == "mdhd"), None)
            timescale = 1
            if mdhd:
                f.seek(mdhd["data_pos"])
                version = f.read(1)[0]
                f.read(3)  # flags
                if version == 1:
                    f.read(16)  # creation+modification
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

            # verify gpmd via stsd
            stsd = next((b for b in stbl_boxes if b["type"] == "stsd"), None)
            is_gpmd = False
            if stsd:
                f.seek(stsd["data_pos"] + 8)
                entry_size = struct.unpack(">I", f.read(4))[0]
                entry_type = f.read(4).decode("latin-1", errors="replace")
                if entry_type == "gpmd":
                    is_gpmd = True
            if not is_gpmd:
                continue

            # sample sizes
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

            # chunk offsets
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

            # samples-per-chunk
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

            # decode times (stts) — sample durations in mdia timescale
            stts = next((b for b in stbl_boxes if b["type"] == "stts"), None)
            durations = []
            if stts:
                f.seek(stts["data_pos"] + 4)
                count = struct.unpack(">I", f.read(4))[0]
                for _ in range(count):
                    sc = struct.unpack(">I", f.read(4))[0]
                    sd = struct.unpack(">I", f.read(4))[0]
                    durations.extend([sd] * sc)

            # build sample offsets list by expanding stsc/stco
            sample_offsets = []
            if not stsc_entries or not chunk_offsets:
                return None
            # extend stsc by appending sentinel
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


def read_gpmf_payloads(video_path: Path):
    """Yield (sample_index, t_start_seconds, payload_bytes) for each gpmd sample."""
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
def _flatten(items):
    for it in items:
        yield it
        if it["children"]:
            yield from _flatten(it["children"])


def _find_streams(items):
    """Yield each STRM container's child items."""
    for it in items:
        if it["k"] == "STRM" and it["children"]:
            yield it["children"]
        elif it["children"]:
            yield from _find_streams(it["children"])


def extract_imu(video_path: Path, epoch_offset_ns: int):
    """
    Returns a list of dicts compatible with the reference IMU jsonl:
        {"t": <ns>, "sensor": "accel"|"gyro", "x":..., "y":..., "z":..., "accuracy": 3}

    t is "nanoseconds since recording started" so that t + epoch_offset_ns == wall clock.
    """
    out = []
    have_any = False
    for sample_idx, t_payload_start, raw in read_gpmf_payloads(video_path):
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
            # spread samples evenly across this payload's window — payloads are ~1 Hz
            # use a nominal 1 s window when we cannot infer the next payload's start.
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
def probe_video(video_path: Path):
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
        print(f"[hands] downloading model -> {model}", flush=True)
        urllib.request.urlretrieve(_HAND_MODEL_URL, model)
    return model


def extract_hand_landmarks(video_path: Path, start_ms: int, fps: float):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_hand_model()
    base = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    # Lowered confidence thresholds to detect both hands more reliably
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
    frames_with_hands = {}  # Track frames by hand count: {1: count, 2: count}
    hand_scores_by_type = {}  # Track detection scores: {"Left": [...], "Right": [...]}
    print("[hands] running MediaPipe HandLandmarker on every frame ...", flush=True)
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000.0 / fps)
        result = landmarker.detect_for_video(mp_image, ts_ms)
        if result.hand_landmarks:
            hands_out = []
            hand_count = len(result.hand_landmarks)
            frames_with_hands[hand_count] = frames_with_hands.get(hand_count, 0) + 1
            
            for i, hand in enumerate(result.hand_landmarks):
                handedness = "Unknown"
                score = 0.0
                if result.handedness and i < len(result.handedness):
                    cls = result.handedness[i][0]
                    handedness = cls.category_name or cls.display_name or "Unknown"
                    score = float(cls.score)
                    # Track scores by handedness
                    if handedness not in hand_scores_by_type:
                        hand_scores_by_type[handedness] = []
                    hand_scores_by_type[handedness].append(score)
                
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
            print(f"  frame {frame_idx} ({time.time() - t0:.1f}s)", flush=True)
    cap.release()
    landmarker.close()
    
    # Print hand detection statistics
    print(f"[hands] Frame detection breakdown:", flush=True)
    for hand_count in sorted(frames_with_hands.keys()):
        count = frames_with_hands[hand_count]
        print(f"        {hand_count} hand(s) detected in {count} frames", flush=True)
    
    if hand_scores_by_type:
        print(f"[hands] Hand detection scores (avg):", flush=True)
        for handedness in sorted(hand_scores_by_type.keys()):
            scores = hand_scores_by_type[handedness]
            avg_score = sum(scores) / len(scores)
            print(f"        {handedness}: {avg_score:.3f} (min: {min(scores):.3f}, max: {max(scores):.3f})", flush=True)
    
    print(f"[hands] {len(lines)} frames with hands / {frame_idx} total, {len(seen_hands)} distinct hand types detected", flush=True)
    return lines, len(seen_hands), frame_idx


# --------------------------------------------------------------------------- #
# MCAP Utils
# --------------------------------------------------------------------------- #
def convert_to_mcap(video_src: Path, meta_dir: Path, session_meta: dict, output_dir: Path):
    from mcap.writer import Writer
    print(f"  • Building MCAP ...", end="", flush=True)
    t0 = time.time()

    start_ms = session_meta.get("start_timestamp", 0)
    start_ns = start_ms * 1_000_000

    mcap_path = output_dir / f"{video_src.stem}.mcap"
    with open(mcap_path, "wb") as out_file:
        writer = Writer(out_file)
        writer.start(profile="owrecorder", library="owrecorder-mcap-v1")
        empty_schema = b'{"type":"object"}'

        with open(video_src, "rb") as vf:
            writer.add_attachment(
                create_time=start_ns, log_time=start_ns,
                name=video_src.name, media_type="video/mp4", data=vf.read()
            )

        imu_file = next(meta_dir.glob("*_imu.jsonl"), None)
        if imu_file:
            sch = writer.register_schema("sensor_msgs/Imu", "jsonschema", empty_schema)
            ch = writer.register_channel(schema_id=sch, topic="/imu", message_encoding="json")
            with open(imu_file) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    rec = json.loads(line)
                    ts_ns = rec["t"]
                    writer.add_message(channel_id=ch, log_time=ts_ns, publish_time=ts_ns, data=line.encode("utf-8"))

        hand_file = next(meta_dir.glob("*_hand_landmarks.jsonl"), None)
        if hand_file:
            sch = writer.register_schema("HandLandmarks", "jsonschema", empty_schema)
            ch = writer.register_channel(schema_id=sch, topic="/hand_landmarks", message_encoding="json")
            with open(hand_file) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    rec = json.loads(line)
                    ts_ns = int(rec["t"] * 1_000_000)
                    writer.add_message(channel_id=ch, log_time=ts_ns, publish_time=ts_ns, data=line.encode("utf-8"))

        depth_file = next(meta_dir.glob("*_depth_meta.jsonl"), None)
        if depth_file:
            sch = writer.register_schema("DepthStatus", "jsonschema", empty_schema)
            ch = writer.register_channel(schema_id=sch, topic="/depth_status", message_encoding="json")
            with open(depth_file) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    writer.add_message(channel_id=ch, log_time=start_ns, publish_time=start_ns, data=line.encode("utf-8"))

        sch = writer.register_schema("SessionMetadata", "jsonschema", empty_schema)
        ch = writer.register_channel(schema_id=sch, topic="/session", message_encoding="json")
        writer.add_message(channel_id=ch, log_time=start_ns, publish_time=start_ns, data=json.dumps(session_meta).encode("utf-8"))
        writer.finish()

    size_mb = mcap_path.stat().st_size / (1024 * 1024)
    print(f" done ({time.time()-t0:.1f}s, {size_mb:.1f} MB)")
    return mcap_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_creation_time_ms(text: str | None) -> int | None:
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(text)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".lrv", ".ts"}


def discover_videos(scan_dir: Path):
    found = []
    for p in sorted(scan_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            found.append(p)
    return found


def process_video(video: Path, voice_triggered: bool, force: bool):
    out = video.parent / "metadata" / video.stem
    if out.exists() and not force:
        if any(out.glob("*.json")):
            print(f"[skip] {video.name} -> metadata/{out.name}/ already has output (use --force to redo)")
            return False
    out.mkdir(parents=True, exist_ok=True)

    probe = probe_video(video)
    fps = probe["fps"] or 30.0
    width = probe["width"] or 1920
    height = probe["height"] or 1080
    duration_s = probe["duration_s"] or 0.0
    creation_ms = parse_creation_time_ms(probe["creation_time"])
    if creation_ms is None:
        creation_ms = int(video.stat().st_mtime * 1000) - int(duration_s * 1000)

    start_ms = creation_ms
    end_ms = creation_ms + int(duration_s * 1000)
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    session = f"video_{dt.strftime('%Y-%m-%d_%H%M')}_{start_ms}"

    imu_path = out / f"{session}_imu.jsonl"
    hands_path = out / f"{session}_hand_landmarks.jsonl"
    depth_meta_path = out / f"{session}_depth_meta.jsonl"
    session_json = out / f"{session}.json"

    print(f"\n=== {video.name} -> {out.name}/ ===")
    print(f"[session] {session}")
    print(f"[video] {width}x{height} @ {fps:.3f} fps, duration {duration_s:.2f}s")

    # ----- IMU
    epoch_offset_ns = start_ms * 1_000_000
    imu_samples, gpmf_any = extract_imu(video, epoch_offset_ns)
    print(f"[imu] {len(imu_samples)} samples ({'gpmd track found' if gpmf_any else 'NO gpmd track'})")
    with open(imu_path, "w", encoding="utf-8") as f:
        for s in imu_samples:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")

    # ----- Hand landmarks
    hand_lines, distinct_handed, frame_count = extract_hand_landmarks(video, start_ms, fps)
    with open(hands_path, "w", encoding="utf-8") as f:
        for ln in hand_lines:
            f.write(json.dumps(ln, separators=(",", ":")) + "\n")

    # ----- Depth — GoPro has no depth sensor; mirror the reference's not-available marker
    with open(depth_meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"available": False, "reason": "no_depth_sensor"}) + "\n")

    # ----- session.json
    session_meta = {
        "session_id": session,
        "voice_triggered": bool(voice_triggered),
        "resolution": f"{width}x{height}",
        "fps": round(float(fps), 6),
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "actual_hands_detected": int(distinct_handed),
        "imu_file": imu_path.name,
        "imu_sample_count": len(imu_samples),
        "imu_epoch_offset_ns": epoch_offset_ns,
        "depth_available": False,
        "depth_frame_count": 0,
        "depth_resolution": "",
        "depth_source": "none_hardware",
        "depth_reason": "GoPro has no ToF sensor",
        "depth_recoverable_from_video": True,
        "depth_recovery_method": "monocular_estimation_with_imu_scale",
        "hand_landmarks_file": hands_path.name,
        "device_name": "GoPro",
        "source_video": video.name,
    }
    with open(session_json, "w", encoding="utf-8") as f:
        json.dump(session_meta, f, indent=4)

    print(f"[done] {session_json.name} | imu {len(imu_samples)} | hands {len(hand_lines)} frames")
    return True


def clean_path(p_str: str) -> Path:
    """Safely strip quotes that CMD/PowerShell might leave behind (e.g. trailing \\\")"""
    return Path(p_str.strip('"').strip("'")).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*",
                    help="video file(s) to process (default: every video in --dir)")
    ap.add_argument("--dir", default=".",
                    help="directory to scan when no videos are passed (default: cwd)")
    ap.add_argument("--voice-triggered", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if an output folder already has a session JSON")
    args = ap.parse_args()

    if args.videos:
        targets = []
        for v in args.videos:
            p = clean_path(v)
            if not p.exists():
                print(f"[warn] not found: {p}")
                continue
            targets.append(p)
    else:
        scan_dir = clean_path(args.dir)
        if not scan_dir.is_dir():
            print(f"[error] not a directory: {scan_dir}")
            input("Press Enter to exit...")
            sys.exit(1)
        targets = discover_videos(scan_dir)

    if not targets:
        print("[error] no videos found")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"Processing {len(targets)} video(s) -> Extracting Metadata + Building MCAPs")
    print(f"{'─' * 60}")
    
    output_dir = targets[0].parent / "output"
    output_dir.mkdir(exist_ok=True)
    meta_root = targets[0].parent / "metadata"

    processed, failed = 0, 0
    t_batch = time.time()
    
    for video in targets:
        try:
            # 1. Extract Metadata
            process_video(video, args.voice_triggered, args.force)
            
            # 2. Bundle into MCAP
            meta_dir = meta_root / video.stem
            session_json = next(meta_dir.glob("*.json"), None) if meta_dir.exists() else None
            
            if not session_json:
                print(f"  [error] {video.name}: No session JSON found, skipping MCAP conversion.")
                failed += 1
                continue
            
            with open(session_json) as f:
                session_meta = json.load(f)
                
            print(f"=== {video.name} -> output/{video.stem}.mcap ===")
            convert_to_mcap(video, meta_dir, session_meta, output_dir)
            processed += 1
            
        except Exception as e:
            failed += 1
            print(f"  [error] {video.name}: {e}")

    print(f"\n[batch done] successfully processed={processed} failed={failed} in {time.time() - t_batch:.1f}s")
    print(f"Metadata folder: {meta_root}")
    print(f"MCAP Output folder: {output_dir}")


if __name__ == "__main__":
    main()
