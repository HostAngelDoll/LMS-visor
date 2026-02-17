#!py -3.10
# main.py
# detector de señas LSM - reorganizado + sistema de grabación 5s -> json
# solo mano derecha del pov (izquierda / pov de la camára)
# + reconocimiento en tiempo real de samples grabados (gestures.json)

import cv2
import depthai as dai
import math
import csv
import time
import json
import os
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions
)
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python import BaseOptions
from types import SimpleNamespace


# --- config ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
CSV_PATH = "hand_data.csv"
JSON_PATH = "gestures.json"

SMOOTH_ALPHA = 0.7
EXTENDED_ANGLE_THRESH = 150
CURLED_ANGLE_THRESH = 110
MIRROR = False

RECORD_SECONDS = 5.0
SAVE_MSG_TTL = 2.0  # show saved message 2s

# recognition params
RECOG_THRESHOLD = 0.35  # lower -> stricter match
NUMERIC_WEIGHT = 0.6
BOOL_WEIGHT = 0.3
GESTURE_WEIGHT = 0.1

# smoothing storage
_prev_smoothed = {}

# numeric / boolean keys used for comparison (must match stored aggregates)
NUMERIC_KEYS = [
    "ang_idx", "ang_mid", "ang_ring", "ang_pinky",
    "d_thumb_index", "d_index_middle", "hand_size"
]

BOOL_KEYS = [
    # "is_right", 
    "thumb_left_of_index", 
    "idx_ext", 
    "mid_ext", 
    "ring_ext", 
    "pinky_ext",
    # new orientation flags
    "pointing_up",
    "pointing_down"
]

EPS = 1e-6

# ------------------- land utils ---------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_landmarks(frame, hand_landmarks):
    for landmark in hand_landmarks:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

def draw_connections(frame, hand_landmarks):
    for start, end in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[start].x * frame.shape[1])
        y1 = int(hand_landmarks[start].y * frame.shape[0])
        x2 = int(hand_landmarks[end].x * frame.shape[1])
        y2 = int(hand_landmarks[end].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


# ------------------- utility math / landmarks -------------------

def smooth_landmarks(hand_index, lands):
    coords = [SimpleNamespace(x=ld.x, y=ld.y) for ld in lands]
    prev = _prev_smoothed.get(hand_index)
    if prev is None:
        _prev_smoothed[hand_index] = coords
        return coords
    alpha = SMOOTH_ALPHA
    out = []
    for p, c in zip(prev, coords):
        nx = alpha * p.x + (1 - alpha) * c.x
        ny = alpha * p.y + (1 - alpha) * c.y
        out.append(SimpleNamespace(x=nx, y=ny))
    _prev_smoothed[hand_index] = out
    return out

def angle_pts(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0
    v = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(v))

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def dist_sm(a,b):
    return math.hypot(a.x - b.x, a.y - b.y)

def palm_size(lands):
    return distance(SimpleNamespace(x=lands[0].x, y=lands[0].y), SimpleNamespace(x=lands[9].x, y=lands[9].y))

def norm_dist(a, b, hand_s):
    return dist_sm(a, b) / (hand_s if hand_s != 0 else 1.0)

def infer_is_right_hand_sm(lands):
    try:
        cond = lands[8].x < lands[20].x
    except Exception:
        return True
    return cond if not MIRROR else not cond

def thumb_left_of_index_sm(lands, is_right):
    if is_right:
        return lands[4].x < lands[8].x
    else:
        return lands[4].x > lands[8].x

def is_finger_extended_sm(lands, mcp_idx, pip_idx, tip_idx, thresh=EXTENDED_ANGLE_THRESH):
    ang = angle_pts(lands[mcp_idx], lands[pip_idx], lands[tip_idx])
    return ang > thresh

def is_curled(lands, mcp, pip, tip, thresh=CURLED_ANGLE_THRESH):
    return angle_pts(lands[mcp], lands[pip], lands[tip]) < thresh

def is_extended(lands, mcp, pip, tip, thresh=EXTENDED_ANGLE_THRESH):
    return angle_pts(lands[mcp], lands[pip], lands[tip]) > thresh

def fingers_state_sm(lands):
    return {
        "thumb_left_of_index": thumb_left_of_index_sm(lands, infer_is_right_hand_sm(lands)),
        "index": is_finger_extended_sm(lands, 5, 6, 8),
        "middle": is_finger_extended_sm(lands, 9, 10, 12),
        "ring": is_finger_extended_sm(lands, 13, 14, 16),
        "pinky": is_finger_extended_sm(lands, 17, 18, 20),
    }

def is_fist(st):
    return not st["index"] and not st["middle"] and not st["ring"] and not st["pinky"]

def is_C_sm(lands):
    hand_s = palm_size(lands)
    return norm_dist(lands[4], lands[8], hand_s) > 0.65

# orientation helper: decide if hand points up/down based on middle fingertip vs wrist
def pointing_up_down_sm(lands, tip_idx=12, thresh_ratio=0.03):
    """
    Returns (pointing_up, pointing_down).
    Uses vertical (y) difference between tip and wrist normalized by hand_size.
    """
    hand_s = palm_size(lands)
    wrist = lands[0]
    tip = lands[tip_idx]
    dy = tip.y - wrist.y  # y increases downward: negative => tip is above wrist (pointing up)
    # threshold relative to hand size provides some robustness
    thresh = thresh_ratio
    # If hand_s is normalized (0..1 coordinates), thresh_ratio is absolute fraction of image; we keep simple check
    pointing_up = dy < -thresh
    pointing_down = dy > thresh
    return pointing_up, pointing_down

# Recognizers (same logic as original; kept for compatibility)
def recognize_MNT_sm(lands):
    INDEX = lands[8]; MIDDLE = lands[12]; RING = lands[16]; THUMB = lands[4]
    hand_s = palm_size(lands)
    d_thumb_index = dist_sm(THUMB, INDEX)
    if THUMB.y > INDEX.y and THUMB.y > MIDDLE.y and THUMB.y > RING.y:
        st = fingers_state_sm(lands)
        if not st["index"] and not st["middle"] and not st["ring"]:
            return "M"
    if THUMB.y > INDEX.y and THUMB.y > MIDDLE.y and THUMB.y < RING.y:
        st = fingers_state_sm(lands)
        if not st["index"] and not st["middle"]:
            return "N"
    ang = angle_pts(THUMB, INDEX, MIDDLE)
    if 60 < ang < 120 and d_thumb_index < hand_s * 0.6:
        return "T"
    return None

def recognize_gesture_sm(lands):
    hand_s = palm_size(lands)
    st = fingers_state_sm(lands)
    is_right = infer_is_right_hand_sm(lands)
    TH = lands[4]; IDX = lands[8]; MID = lands[12]; RING = lands[16]; PNK = lands[20]
    d_th_idx = norm_dist(TH, IDX, hand_s)
    d_idx_mid = norm_dist(IDX, MID, hand_s)
    g = recognize_MNT_sm(lands)
    if g:
        return g
    if d_th_idx < 0.28:
        if st["middle"] and st["ring"] and st["pinky"]:
            return "F"
        return "O"
    if (
        st["index"] and st["middle"] and st["ring"] and st["pinky"]
        and not st["thumb_left_of_index"]
        and norm_dist(TH, lands[0], hand_s) < 0.7
    ):
        return "B"
    if st["index"] and not st["middle"] and not st["ring"] and not st["pinky"]:
        return "D"
    if (
        st["index"]
        and not st["middle"] and not st["ring"] and not st["pinky"]
        and (
            st["thumb_left_of_index"]
            or is_extended(lands, 1, 2, 4, thresh=120)
        )
    ):
        return "L"
    if st["index"] and st["middle"] and not st["ring"] and not st["pinky"]:
        return "U" if d_idx_mid < 0.25 else "V"
    if st["index"] and st["middle"] and st["ring"] and not st["pinky"]:
        return "W"
    if st["pinky"] and st["thumb_left_of_index"] and not st["index"] and not st["middle"]:
        return "Y"
    if st["index"] and st["middle"]:
        dx = abs(IDX.x - MID.x)
        dy = abs(IDX.y - MID.y)
        if dx / hand_s < 0.18 and dy / hand_s < 0.25:
            return "R"
    if is_fist(st):
        if (
            is_curled(lands, 5, 6, 8)
            and is_curled(lands, 9, 10, 12)
        ):
            return "E"
        if abs(lands[4].x - lands[5].x) / hand_s < 0.15 and lands[4].y < lands[5].y:
            return "S"
        return "A"
    if (
        not st["index"]
        and norm_dist(lands[8], lands[6], hand_s) < 0.3
        and not st["middle"]
    ):
        return "X"
    if is_C_sm(lands):
        return "C"
    return "..."

# ------------------- extraction of properties for JSON -------------------

def extract_props_from_lands(lands):
    # lands is smoothed SimpleNamespace list (x,y)
    hand_s = palm_size(lands)
    TH = lands[4]; IDX = lands[8]; MID = lands[12]; RING = lands[16]; PNK = lands[20]
    ang_idx = angle_pts(lands[5], lands[6], lands[8])
    ang_mid = angle_pts(lands[9], lands[10], lands[12])
    ang_ring = angle_pts(lands[13], lands[14], lands[16])
    ang_pinky = angle_pts(lands[17], lands[18], lands[20])
    d_thumb_index = norm_dist(TH, IDX, hand_s)
    d_index_middle = norm_dist(IDX, MID, hand_s)
    st = fingers_state_sm(lands)
    is_right = infer_is_right_hand_sm(lands)
    gesture = recognize_gesture_sm(lands)

    # orientation (up/down) based on middle fingertip vs wrist
    pointing_up, pointing_down = pointing_up_down_sm(lands, tip_idx=12, thresh_ratio=0.03)

    props = {
        "ang_idx": ang_idx,
        "ang_mid": ang_mid,
        "ang_ring": ang_ring,
        "ang_pinky": ang_pinky,
        "d_thumb_index": d_thumb_index,
        "d_index_middle": d_index_middle,
        "hand_size": hand_s,
        "is_right": bool(is_right),
        "thumb_left_of_index": bool(st["thumb_left_of_index"]),
        "idx_ext": bool(st["index"]),
        "mid_ext": bool(st["middle"]),
        "ring_ext": bool(st["ring"]),
        "pinky_ext": bool(st["pinky"]),
        "gesture": gesture,
        "pointing_up": bool(pointing_up),
        "pointing_down": bool(pointing_down)
    }
    return props

def compute_aggregates(frames_props):
    """
    frames_props: list of props dicts (as returned by extract_props_from_lands)
    returns: dict with min/max/avg for numeric keys, and counts/proportions for booleans, and majority gesture
    """
    out = {"frames": len(frames_props)}
    if not frames_props:
        # default empty
        for k in NUMERIC_KEYS:
            out.setdefault("min_"+k, None)
            out.setdefault("max_"+k, None)
            out.setdefault("avg_"+k, None)
        for k in BOOL_KEYS:
            out.setdefault("prop_"+k, 0.0)
        out["majority_gesture"] = None
        return out

    # numeric
    for k in NUMERIC_KEYS:
        vals = [p[k] for p in frames_props if p.get(k) is not None]
        if vals:
            out["min_"+k] = min(vals)
            out["max_"+k] = max(vals)
            out["avg_"+k] = sum(vals)/len(vals)
        else:
            out["min_"+k] = out["max_"+k] = out["avg_"+k] = None

    # booleans proportions
    for k in BOOL_KEYS:
        vals = [1 if p.get(k) else 0 for p in frames_props if p.get(k) is not None]
        out["prop_"+k] = sum(vals)/len(vals) if vals else 0.0

    # majority gesture
    gestures = {}
    for p in frames_props:
        g = p.get("gesture", None)
        if g is None:
            continue
        gestures[g] = gestures.get(g, 0) + 1
    majority = max(gestures.items(), key=lambda x: x[1])[0] if gestures else None
    out["majority_gesture"] = majority
    return out

# ------------------- CSV saving (kept from original) -------------------

def save_landmarks_csv(writer, lands, frame_id, hand_idx, expected_letter="", result_label=""):
    ts = time.time()
    lands_sm = smooth_landmarks(hand_idx, lands)
    st = fingers_state_sm(lands_sm)
    is_right = infer_is_right_hand_sm(lands_sm)
    gesture = recognize_gesture_sm(lands_sm)
    hand_s = palm_size(lands_sm)
    TH = lands_sm[4]; IDX = lands_sm[8]; MID = lands_sm[12]
    RING = lands_sm[16]; PNK = lands_sm[20]

    # compute pointing up/down for CSV row
    pointing_up, pointing_down = pointing_up_down_sm(lands_sm, tip_idx=12, thresh_ratio=0.03)

    row = [ts, frame_id, hand_idx]
    # raw landmarks (original)
    for lm in lands:
        row += [lm.x, lm.y, getattr(lm, "z", 0.0)]
    # debug props
    row += [
        gesture,
        is_right,
        st["thumb_left_of_index"],
        st["index"],
        st["middle"],
        st["ring"],
        st["pinky"],
    ]
    # orientation flags (new)
    row += [
        bool(pointing_up),
        bool(pointing_down)
    ]
    # angles
    row += [
        angle_pts(lands_sm[5], lands_sm[6], lands_sm[8]),
        angle_pts(lands_sm[9], lands_sm[10], lands_sm[12]),
        angle_pts(lands_sm[13], lands_sm[14], lands_sm[16]),
        angle_pts(lands_sm[17], lands_sm[18], lands_sm[20]),
    ]
    # distances
    row += [
        norm_dist(TH, IDX, hand_s),
        norm_dist(IDX, MID, hand_s),
        hand_s
    ]
    # test metadata
    row += [expected_letter, result_label]
    writer.writerow(row)
    try:
        writer._csv.writerows([])  # noop to ensure writer exists (safe)
    except Exception:
        pass

# ------------------- gestures.json helpers -------------------

def load_gestures_json():
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_gestures_json(data):
    tmp = JSON_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, JSON_PATH)

def append_gesture_sample(letter, sample):
    data = load_gestures_json()
    if letter not in data:
        data[letter] = {"samples": []}
    data[letter]["samples"].append(sample)
    save_gestures_json(data)

# ------------------- matching logic -------------------

def score_numeric_vs_sample(cur_val, min_val, max_val, avg_val):
    """
    returns a score in [0,1]; 0 if inside [min,max], else normalized distance to closest bound.
    if min==max (tiny range) normalize by avg or hand_size scale.
    """
    if min_val is None or max_val is None:
        # no range data -> treat as neutral (0.5)
        return 0.5
    if min_val <= cur_val <= max_val:
        return 0.0
    rng = max_val - min_val
    if abs(rng) < EPS:
        # fallback normalization
        denom = abs(avg_val) + EPS
        return min(1.0, abs(cur_val - avg_val) / denom)
    # distance to closest bound
    dd = min(abs(cur_val - min_val), abs(cur_val - max_val))
    return min(1.0, dd / (rng + EPS))

def score_bool_vs_sample(cur_bool, prop_true):
    """
    prop_true: proportion of frames where boolean was true in sample (0..1)
    if prop_true >= 0.5 then expected true, else expected false.
    returns 0 if matches majority expectation, 1 otherwise.
    """
    expect = prop_true >= 0.5
    return 0.0 if (cur_bool == expect) else 1.0

def match_props_to_gestures(props, gestures_db):
    """
    props: current props dict (extract_props_from_lands)
    gestures_db: loaded JSON structure
    returns: (best_letter, best_score, best_sample_meta) or (None, None, None) if no db
    """
    if not gestures_db:
        return None, None, None

    best_score = 1e9
    best_letter = None
    best_sample = None

    for letter, info in gestures_db.items():
        # Soportar formato viejo (lista directa)
        if isinstance(info, list):
            samples = info
        # Formato nuevo {"samples": [...]}
        elif isinstance(info, dict):
            samples = info.get("samples", [])
        else:
            continue
        
        for samp in samples:
            agg = samp.get("aggregates", {})
            # numeric score
            numeric_scores = []
            for k in NUMERIC_KEYS:
                min_k = agg.get("min_" + k)
                max_k = agg.get("max_" + k)
                avg_k = agg.get("avg_" + k)
                cur_v = props.get(k)
                if cur_v is None:
                    numeric_scores.append(0.5)  # neutral penalty
                else:
                    numeric_scores.append(score_numeric_vs_sample(cur_v, min_k, max_k, avg_k))
            numeric_mean = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 1.0

            # bool score
            bool_scores = []
            for k in BOOL_KEYS:
                prop_k = agg.get("prop_" + k, 0.0)
                cur_b = bool(props.get(k, False))
                bool_scores.append(score_bool_vs_sample(cur_b, prop_k))
            bool_mean = sum(bool_scores) / len(bool_scores) if bool_scores else 1.0

            # gesture string match (majority)
            maj_g = agg.get("majority_gesture")
            gest_mismatch = 0.0
            if maj_g is None:
                gest_mismatch = 0.5
            else:
                gest_mismatch = 0.0 if (props.get("gesture") == maj_g) else 1.0

            score = NUMERIC_WEIGHT * numeric_mean + BOOL_WEIGHT * bool_mean + GESTURE_WEIGHT * gest_mismatch

            if score < best_score:
                best_score = score
                best_letter = letter
                best_sample = {
                    "sample_meta": samp,
                    "numeric_mean": numeric_mean,
                    "bool_mean": bool_mean,
                    "maj_gesture": maj_g
                }

    if best_score == 1e9:
        return None, None, None
    return best_letter, best_score, best_sample

# ------------------- MediaPipe callback & global state -------------------

last_result = None
frame = None

def callback(result, output_image, timestamp_ms):
    global frame, last_result
    last_result = result
    frame = output_image.numpy_view().copy()
    # draw landmarks on frame for visualization (non-blocking)
    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            draw_connections(frame, hand_landmarks)
            draw_landmarks(frame, hand_landmarks)

# ------------------- main loop + recording state -------------------

def main():
    global frame, last_result
    # load gestures DB (may be empty)
    gestures_db = load_gestures_json()

    # init mediapipe hand landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=callback,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    hand_landmarker = HandLandmarker.create_from_options(options)

    # depthai pipeline
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    # CSV init
    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        header = ["timestamp", "frame_id", "hand_idx"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header += [
            "gesture", "is_right", "thumb_left_of_index",
            "idx_ext", "mid_ext", "ring_ext", "pinky_ext",
            # orientation flags
            "pointing_up", "pointing_down",
            "ang_idx", "ang_mid", "ang_ring", "ang_pinky",
            "d_thumb_index", "d_index_middle", "hand_size",
            "expected_letter", "result"
        ]
        csv_writer.writerow(header)

    # recording state
    recording = False
    record_start = 0.0
    record_buffers = {}  # hand_idx -> list of props dicts (frames)
    record_expected_letter = ""
    last_save_msg = ""
    last_save_time = 0.0

    test_mode = False
    expected_letter = ""

    # for smoothing recognition display
    last_recognitions = {}  # hand_idx -> (letter, score, timestamp)

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        frame_id = 0

        while True:
            in_rgb = q.get()
            frame_bgr = in_rgb.getCvFrame()
            frame_bgr = cv2.flip(frame_bgr, 1)  # espejo horizontal
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            hand_landmarker.detect_async(mp_image, timestamp_ms=int(frame_id * 33))

            # display image (prefer processed frame if available)
            if frame is not None:
                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            else:
                display = frame_bgr.copy()

            # overlays
            y_base = 20
            # mode_line = f"Test mode: {'ON' if test_mode else 'OFF'}  (t to toggle)"
            # cv2.putText(display, mode_line, (10, y_base+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if test_mode else (0,150,150), 2)
            cv2.putText(display, f"Expected letter: {expected_letter}", (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            # msg_1 = "SPACE = save success | TAB = save fail | q = quit | Enter = record 5s"
            # cv2.putText(display, "", (10, y_base+48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # recording handling
            if recording:
                elapsed = time.time() - record_start
                remaining = max(0.0, RECORD_SECONDS - elapsed)
                cv2.putText(display, f"RECORDING [{record_expected_letter}] - {remaining:.1f}s left", (10, y_base+72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                # capture frames props into buffers using last_result
                if last_result and last_result.hand_landmarks:
                    for idx, hand_landmarks in enumerate(last_result.hand_landmarks):
                        sm_lands = smooth_landmarks(idx, hand_landmarks)
                        props = extract_props_from_lands(sm_lands)
                        record_buffers.setdefault(idx, []).append(props)
                        
                # finish
                if elapsed >= RECORD_SECONDS:
                    # compute aggregates per hand and save
                    saved_samples = []
                    for hidx, frames_props in record_buffers.items():
                        agg = compute_aggregates(frames_props)
                        sample = {
                            "timestamp": time.time(),
                            "hand_idx": hidx,
                            "duration": elapsed,
                            "frames": agg.pop("frames", len(frames_props)),
                            "aggregates": agg,
                            "raw_count": len(frames_props),
                        }
                        append_gesture_sample(record_expected_letter, sample)
                        saved_samples.append(sample)
                    # reload gestures_db to include newly saved samples
                    gestures_db = load_gestures_json()
                    last_save_msg = f"Recorded {len(saved_samples)} sample(s) for [{record_expected_letter}]"
                    last_save_time = time.time()
                    recording = False
                    record_buffers = {}
                    print(last_save_msg)

            # real-time recognition: for each detected hand compute props and match DB
            if last_result and last_result.hand_landmarks and gestures_db:
                for idx, hand_landmarks in enumerate(last_result.hand_landmarks):
                    sm_lands = smooth_landmarks(idx, hand_landmarks)
                    cur_props = extract_props_from_lands(sm_lands)
                    best_letter, best_score, best_sample = match_props_to_gestures(cur_props, gestures_db)
                    # smoothing / debounce: keep last result if recent and score similar
                    now = time.time()
                    if best_letter is not None and best_score is not None and best_score <= RECOG_THRESHOLD:
                        # accept
                        last_recognitions[idx] = (best_letter, best_score, now)
                    else:
                        # if previous recognition is recent (0.3s) keep it, else clear
                        prev = last_recognitions.get(idx)
                        if prev and (now - prev[2]) < 0.3:
                            # keep previous
                            pass
                        else:
                            last_recognitions.pop(idx, None)

                    # overlay recognition near wrist (landmark 0)
                    x = int(sm_lands[0].x * display.shape[1])
                    y = int(sm_lands[0].y * display.shape[0])
                    rec = last_recognitions.get(idx)
                    if rec:
                        letter, score, tsr = rec
                        conf = max(0.0, 1.0 - score)  # convert score->confidence
                        text = f"Rec: {letter} ({conf:.2f})"
                        cv2.putText(display, text, (x, y - 10 - idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    else:
                        cv2.putText(display, "---", (x, y - 10 - idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

            # show last save message temporarily
            if last_save_msg and (time.time() - last_save_time) < SAVE_MSG_TTL:
                cv2.putText(display, last_save_msg, (10, y_base+96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("OAK-D Hand Tracking", display)
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF

            # toggle test mode
            # if key == ord('t'):
            #     test_mode = not test_mode
            #     expected_letter = ""
            print(f"{key}") if key != 255 else None
            
            # letter selection (a-z / A-Z)
            if (ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z')):
                expected_letter = chr(key).upper()
                last_save_msg = f"Expected set: {expected_letter}"
                last_save_time = time.time()
                
            # Enter key (start recording) - support common Enter codes 13 and 10
            elif key in (13, 10):
                # start recording only if expected_letter set
                if not expected_letter:
                    last_save_msg = "Set expected letter first (press a-z)."
                    last_save_time = time.time()
                else:
                    recording = True
                    record_start = time.time()
                    record_buffers = {}
                    record_expected_letter = expected_letter
                    last_save_msg = f"Recording started for [{record_expected_letter}]"
                    last_save_time = time.time()
                    
            elif key == 32:  # SPACE
                # in test mode => treat as success save; otherwise same as before
                if last_result and last_result.hand_landmarks:
                    for idx, hand_landmarks in enumerate(last_result.hand_landmarks):
                        if test_mode:
                            res_label = "success"
                            save_landmarks_csv(csv_writer, hand_landmarks, frame_id, idx, expected_letter, res_label)
                            last_save_msg = f"Saved SUCCESS [{expected_letter}] hand {idx}"
                        else:
                            save_landmarks_csv(csv_writer, hand_landmarks, frame_id, idx, "", "captured")
                            last_save_msg = f"Saved capture hand {idx}"
                        last_save_time = time.time()
                    print(last_save_msg)
            elif key == 9:  # TAB -> mark failure
                if last_result and last_result.hand_landmarks:
                    for idx, hand_landmarks in enumerate(last_result.hand_landmarks):
                        if test_mode:
                            res_label = "fail"
                            save_landmarks_csv(csv_writer, hand_landmarks, frame_id, idx, expected_letter, res_label)
                            last_save_msg = f"Saved FAIL [{expected_letter}] hand {idx}"
                        else:
                            save_landmarks_csv(csv_writer, hand_landmarks, frame_id, idx, "", "fail")
                            last_save_msg = f"Saved FAIL hand {idx}"
                        last_save_time = time.time()
                    print(last_save_msg)
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    main()
