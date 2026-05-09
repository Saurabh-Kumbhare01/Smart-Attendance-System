import csv
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


def run_with_streamlit_when_started_as_python_file():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        return

    if get_script_run_ctx() is None:
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.port",
            "8501",
        ]
        sys.exit(streamlit_cli.main())


run_with_streamlit_when_started_as_python_file()


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAINER_DIR = BASE_DIR / "trainer"
ATTENDANCE_DIR = BASE_DIR / "attendance"
STUDENTS_FILE = BASE_DIR / "students.csv"
ATTENDANCE_HISTORY_FILE = ATTENDANCE_DIR / "attendance_history.csv"
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

face_detector = cv2.CascadeClassifier(HAAR_CASCADE)


st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="FR",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_attendance_history():
    if ATTENDANCE_HISTORY_FILE.exists():
        try:
            with open(ATTENDANCE_HISTORY_FILE, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                return {f"{row['id']}_{row['date']}": dict(row) for row in reader}
        except Exception:
            return {}
    return {}

def append_attendance_record(record):
    file_exists = ATTENDANCE_HISTORY_FILE.exists()
    fieldnames = ["id", "name", "date", "time", "confidence", "status"]
    with open(ATTENDANCE_HISTORY_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

if "attendance_running" not in st.session_state:
    st.session_state.attendance_running = False
if "attendance_records" not in st.session_state:
    st.session_state.attendance_records = load_attendance_history()
if "model" not in st.session_state:
    st.session_state.model = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "last_event" not in st.session_state:
    st.session_state.last_event = "System ready"


def get_students():
    students = {}
    if STUDENTS_FILE.exists():
        with open(STUDENTS_FILE, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    students[row[0]] = row[1]
    return students


def add_student_record(student_id, student_name):
    existing = get_students()
    if existing.get(student_id) != student_name:
        existing[student_id] = student_name
        with open(STUDENTS_FILE, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for sid, name in sorted(existing.items()):
                writer.writerow([sid, name])


def dataset_student_count():
    return len({path.name.split(".")[1] for path in DATASET_DIR.glob("User.*.*.jpg")})


def latest_attendance_file():
    files = sorted(ATTENDANCE_DIR.glob("attendance_*.xml"), reverse=True)
    return files[0].name if files else "No export yet"


def train_model():
    image_paths = list(DATASET_DIR.glob("User.*.*.jpg"))
    if not image_paths:
        st.error("No face images found. Register faces first.")
        return False

    student_ids = sorted(set(img_path.name.split(".")[1] for img_path in image_paths))
    student_id_to_int = {sid: i for i, sid in enumerate(student_ids)}

    face_samples = []
    ids = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        sid = img_path.name.split(".")[1]
        face_samples.append(img)
        ids.append(student_id_to_int[sid])

    if not face_samples:
        st.error("No valid face images found.")
        return False

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        st.error("OpenCV face recognition is unavailable. Install opencv-contrib-python.")
        return False

    recognizer.train(face_samples, np.array(ids))
    recognizer.write(str(TRAINER_DIR / "trainer.yml"))
    with open(TRAINER_DIR / "id_mapping.json", "w", encoding="utf-8") as f:
        json.dump(student_id_to_int, f)
    st.session_state.model = recognizer
    st.session_state.last_event = "Model trained successfully"
    return True


def load_model():
    trainer_path = TRAINER_DIR / "trainer.yml"
    mapping_path = TRAINER_DIR / "id_mapping.json"
    if not trainer_path.exists() or not mapping_path.exists():
        return None, None

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        st.error("OpenCV face recognition is unavailable. Install opencv-contrib-python.")
        return None, None

    recognizer.read(str(trainer_path))
    with open(mapping_path, encoding="utf-8") as f:
        student_id_to_int = json.load(f)
    return recognizer, {v: k for k, v in student_id_to_int.items()}


def save_attendance_xml():
    if not st.session_state.attendance_records:
        st.warning("No attendance records are available to save.")
        return None

    now = datetime.now()
    xml_path = ATTENDANCE_DIR / f"attendance_{now.strftime('%Y%m%d_%H%M%S')}.xml"
    root = ET.Element("Attendance")
    for record in st.session_state.attendance_records.values():
        entry = ET.SubElement(root, "Record")
        ET.SubElement(entry, "ID").text = record["id"]
        ET.SubElement(entry, "Name").text = record["name"]
        ET.SubElement(entry, "Date").text = record["date"]
        ET.SubElement(entry, "Time").text = record["time"]
        ET.SubElement(entry, "Confidence").text = record["confidence"]
        ET.SubElement(entry, "Status").text = record.get("status", "Present")

    ET.ElementTree(root).write(str(xml_path), encoding="utf-8", xml_declaration=True)
    st.session_state.last_event = f"Saved {xml_path.name}"
    return xml_path


def apply_theme():
    mode = "dark" if st.session_state.dark_mode else "light"
    bg = "#0b1220" if mode == "dark" else "#f7fafc"
    text = "#f8fafc" if mode == "dark" else "#0f172a"
    muted = "#b6c2d6" if mode == "dark" else "#64748b"
    card = "#111827" if mode == "dark" else "#ffffff"
    border = "#263449" if mode == "dark" else "#e2e8f0"
    input_bg = "#0f172a" if mode == "dark" else "#ffffff"

    st.markdown(
        f"""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            :root {{
                --app-bg: {bg};
                --app-text: {text};
                --app-muted: {muted};
                --app-card: {card};
                --app-border: {border};
                --app-input: {input_bg};
                --accent-a: #22d3ee;
                --accent-b: #8b5cf6;
                --accent-c: #34d399;
                --danger: #fb7185;
            }}

            @keyframes floatUp {{
                from {{ opacity: 0; transform: translateY(18px) scale(.98); }}
                to {{ opacity: 1; transform: translateY(0) scale(1); }}
            }}
            @keyframes pulseRing {{
                0% {{ box-shadow: 0 0 0 0 rgba(34, 211, 238, .42); }}
                70% {{ box-shadow: 0 0 0 18px rgba(34, 211, 238, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); }}
            }}
            @keyframes gradientMove {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            @keyframes scan {{
                0% {{ top: 8%; opacity: .2; }}
                50% {{ opacity: .95; }}
                100% {{ top: 88%; opacity: .2; }}
            }}
            @keyframes driftGrid {{
                from {{ background-position: 0 0, 0 0; }}
                to {{ background-position: 72px 72px, 100% 0; }}
            }}
            @keyframes shimmer {{
                0% {{ transform: translateX(-120%); }}
                100% {{ transform: translateX(120%); }}
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                background:
                    linear-gradient(rgba(148, 163, 184, .055) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(148, 163, 184, .055) 1px, transparent 1px),
                    linear-gradient(135deg, rgba(34, 211, 238, .14), transparent 34%, rgba(139, 92, 246, .13), transparent 72%, rgba(52, 211, 153, .10)),
                    var(--app-bg) !important;
                background-size: 36px 36px, 36px 36px, 180% 180%, auto;
                animation: driftGrid 22s linear infinite;
                color: var(--app-text);
            }}
            [data-testid="stHeader"] {{ background: transparent; }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, rgba(15, 23, 42, .94), rgba(30, 41, 59, .86)) !important;
                border-right: 1px solid rgba(148, 163, 184, .18);
            }}
            [data-testid="stSidebar"] * {{ color: #eef6ff !important; }}
            .block-container {{
                max-width: 1280px;
                padding-top: 1.7rem;
                padding-bottom: 3rem;
            }}
            h1, h2, h3, h4, h5, h6, p, label, span, div {{
                color: var(--app-text);
                letter-spacing: 0;
            }}
            p, .muted {{ color: var(--app-muted); }}
            .app-topbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 1rem;
                padding: .82rem .95rem;
                border: 1px solid var(--app-border);
                border-radius: 20px;
                background: rgba(15, 23, 42, .48);
                backdrop-filter: blur(22px);
                box-shadow: 0 16px 52px rgba(2, 6, 23, .18);
                animation: floatUp .45s ease both;
            }}
            .command-search {{
                flex: 1;
                display: flex;
                align-items: center;
                gap: .75rem;
                min-height: 42px;
                padding: 0 .9rem;
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, .18);
                background: rgba(255, 255, 255, .06);
                color: var(--app-muted);
            }}
            .topbar-chip {{
                display: inline-flex;
                align-items: center;
                gap: .45rem;
                min-height: 40px;
                padding: 0 .75rem;
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, .18);
                background: rgba(255, 255, 255, .065);
                font-weight: 750;
                white-space: nowrap;
            }}
            .hero-shell {{
                min-height: 430px;
                padding: clamp(1.5rem, 4vw, 4rem);
                border: 1px solid var(--app-border);
                border-radius: 30px;
                background:
                    linear-gradient(135deg, rgba(15, 23, 42, .82), rgba(30, 41, 59, .56)),
                    linear-gradient(120deg, rgba(34, 211, 238, .18), rgba(139, 92, 246, .18));
                box-shadow: 0 24px 80px rgba(2, 6, 23, .28);
                overflow: hidden;
                position: relative;
                animation: floatUp .75s ease both;
            }}
            .hero-shell:after {{
                content: "";
                position: absolute;
                inset: auto -12% -42% 44%;
                height: 340px;
                background: linear-gradient(90deg, rgba(34, 211, 238, .30), rgba(139, 92, 246, .34), rgba(52, 211, 153, .20));
                filter: blur(46px);
                transform: rotate(-10deg);
            }}
            .hero-grid {{
                display: grid;
                grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr);
                gap: clamp(1.2rem, 3vw, 2.2rem);
                align-items: center;
            }}
            .preview-board {{
                border: 1px solid rgba(255, 255, 255, .14);
                border-radius: 24px;
                background: rgba(2, 6, 23, .36);
                padding: 1rem;
                position: relative;
                z-index: 1;
                transform: perspective(1100px) rotateY(-7deg) rotateX(3deg);
                box-shadow: 0 30px 80px rgba(2, 6, 23, .38);
                transition: transform .24s ease, box-shadow .24s ease;
            }}
            .preview-board:hover {{
                transform: perspective(1100px) rotateY(-3deg) rotateX(1deg) translateY(-4px);
                box-shadow: 0 36px 95px rgba(34, 211, 238, .18);
            }}
            .preview-toolbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: .7rem;
                margin-bottom: .9rem;
            }}
            .window-dots {{
                display: flex;
                gap: .38rem;
            }}
            .window-dots span {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #67e8f9;
                opacity: .9;
            }}
            .window-dots span:nth-child(2) {{ background: #c4b5fd; }}
            .window-dots span:nth-child(3) {{ background: #86efac; }}
            .recognition-card {{
                border: 1px solid rgba(148, 163, 184, .18);
                border-radius: 18px;
                padding: 1rem;
                background: linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
            }}
            .face-frame {{
                min-height: 190px;
                border-radius: 18px;
                display: grid;
                place-items: center;
                overflow: hidden;
                position: relative;
                background:
                    linear-gradient(135deg, rgba(34,211,238,.12), rgba(139,92,246,.14)),
                    repeating-linear-gradient(45deg, rgba(255,255,255,.045) 0 1px, transparent 1px 18px);
            }}
            .face-frame:before {{
                content: "";
                width: 128px;
                height: 128px;
                border: 2px solid rgba(103,232,249,.72);
                border-radius: 32px;
                animation: pulseRing 2.1s infinite;
            }}
            .live-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: .75rem;
                margin-top: .75rem;
            }}
            .live-tile {{
                padding: .75rem;
                border-radius: 16px;
                background: rgba(255,255,255,.06);
                border: 1px solid rgba(148, 163, 184, .15);
            }}
            .confidence-bar {{
                height: 8px;
                border-radius: 99px;
                overflow: hidden;
                background: rgba(148, 163, 184, .18);
                position: relative;
            }}
            .confidence-bar span {{
                display: block;
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, #22d3ee, #8b5cf6, #34d399);
                position: relative;
                overflow: hidden;
            }}
            .confidence-bar span:after {{
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,.55), transparent);
                animation: shimmer 2.2s ease-in-out infinite;
            }}
            .eyebrow {{
                display: inline-flex;
                align-items: center;
                gap: .55rem;
                padding: .45rem .75rem;
                border: 1px solid rgba(255, 255, 255, .18);
                border-radius: 999px;
                background: rgba(255, 255, 255, .08);
                color: #dff9ff;
                font-size: .86rem;
                font-weight: 700;
            }}
            .hero-title {{
                max-width: 760px;
                margin: 1.1rem 0 .8rem;
                font-size: clamp(2.45rem, 6vw, 5.7rem);
                line-height: .96;
                font-weight: 900;
                color: #ffffff;
            }}
            .hero-copy {{
                max-width: 650px;
                color: #c9d7f2;
                font-size: 1.08rem;
                line-height: 1.75;
            }}
            .gradient-text {{
                background: linear-gradient(90deg, #67e8f9, #c4b5fd, #86efac);
                -webkit-background-clip: text;
                color: transparent;
            }}
            .glass-card, .metric-card, .camera-card {{
                border: 1px solid var(--app-border);
                border-radius: 22px;
                background: var(--app-card);
                backdrop-filter: blur(18px);
                box-shadow: 0 18px 55px rgba(2, 6, 23, .20);
                animation: floatUp .65s ease both;
                transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
            }}
            .glass-card:hover, .metric-card:hover {{
                transform: translateY(-4px) scale(1.01);
                border-color: rgba(34, 211, 238, .45);
                box-shadow: 0 28px 75px rgba(14, 165, 233, .18);
            }}
            .action-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
            }}
            .action-card {{
                min-height: 156px;
                padding: 1.1rem;
                border-radius: 22px;
                border: 1px solid var(--app-border);
                background: linear-gradient(135deg, rgba(255,255,255,.09), rgba(255,255,255,.035));
                box-shadow: 0 18px 46px rgba(2, 6, 23, .16);
                transition: transform .2s ease, border-color .2s ease;
            }}
            .action-card:hover {{
                transform: translateY(-5px);
                border-color: rgba(139, 92, 246, .48);
            }}
            .action-card i {{ color: #67e8f9; }}
            .metric-card {{
                padding: 1.2rem;
                min-height: 128px;
                position: relative;
                overflow: hidden;
            }}
            .metric-card:after {{
                content: "";
                position: absolute;
                inset: auto 0 0 0;
                height: 3px;
                background: linear-gradient(90deg, #22d3ee, #8b5cf6, #34d399);
            }}
            .metric-icon {{
                width: 44px;
                height: 44px;
                display: grid;
                place-items: center;
                border-radius: 14px;
                background: linear-gradient(135deg, rgba(34, 211, 238, .18), rgba(139, 92, 246, .24));
                color: #67e8f9;
            }}
            .metric-number {{
                margin-top: .9rem;
                font-size: 2.1rem;
                font-weight: 850;
            }}
            .nav-brand {{
                padding: .9rem 0 1.1rem;
                font-size: 1.22rem;
                font-weight: 850;
            }}
            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: .45rem;
                padding: .42rem .72rem;
                border-radius: 999px;
                background: rgba(52, 211, 153, .14);
                color: #86efac;
                font-weight: 750;
                border: 1px solid rgba(52, 211, 153, .24);
            }}
            .bootstrap-btn, div.stButton > button {{
                width: 100%;
                min-height: 44px;
                border: 0 !important;
                border-radius: 14px !important;
                background: linear-gradient(135deg, #06b6d4, #7c3aed, #10b981) !important;
                background-size: 180% 180% !important;
                animation: gradientMove 8s ease infinite;
                color: #ffffff !important;
                font-weight: 800 !important;
                box-shadow: 0 14px 34px rgba(14, 165, 233, .24);
                transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
            }}
            div.stButton > button:hover {{
                transform: translateY(-2px) scale(1.01);
                filter: brightness(1.08);
                box-shadow: 0 20px 46px rgba(124, 58, 237, .28);
            }}
            div.stButton > button:active {{ transform: translateY(1px) scale(.99); }}
            [data-baseweb="input"] input, [data-baseweb="select"] > div, textarea {{
                background: var(--app-input) !important;
                color: var(--app-text) !important;
                border: 1px solid var(--app-border) !important;
                border-radius: 14px !important;
            }}
            .camera-card {{
                min-height: 320px;
                padding: 1.1rem;
                position: relative;
                overflow: hidden;
            }}
            .camera-preview {{
                min-height: 270px;
                border-radius: 18px;
                border: 1px solid rgba(148, 163, 184, .22);
                background:
                    linear-gradient(135deg, rgba(14, 165, 233, .12), rgba(124, 58, 237, .12)),
                    repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 1px, transparent 1px 28px);
                display: grid;
                place-items: center;
                position: relative;
                overflow: hidden;
            }}
            .camera-preview:before {{
                content: "";
                position: absolute;
                left: 6%;
                right: 6%;
                height: 2px;
                background: linear-gradient(90deg, transparent, #67e8f9, transparent);
                animation: scan 2.5s ease-in-out infinite;
            }}
            .camera-preview:after {{
                content: "";
                width: 142px;
                height: 142px;
                border: 2px solid rgba(103, 232, 249, .65);
                border-radius: 28px;
                animation: pulseRing 2.2s infinite;
            }}
            .timeline-step {{
                display: flex;
                align-items: flex-start;
                gap: .8rem;
                padding: .8rem 0;
                border-bottom: 1px solid var(--app-border);
            }}
            .timeline-step:last-child {{ border-bottom: 0; }}
            .step-dot {{
                width: 34px;
                height: 34px;
                display: grid;
                place-items: center;
                flex: 0 0 34px;
                border-radius: 50%;
                background: rgba(34, 211, 238, .15);
                color: #67e8f9;
            }}
            .table-shell {{
                border: 1px solid var(--app-border);
                border-radius: 22px;
                overflow: hidden;
                background: var(--app-card);
                backdrop-filter: blur(18px);
            }}
            [data-testid="stDataFrame"] {{
                border-radius: 18px;
                overflow: hidden;
            }}
            .success-burst {{
                padding: .9rem 1rem;
                border-radius: 16px;
                background: rgba(52, 211, 153, .13);
                border: 1px solid rgba(52, 211, 153, .30);
                animation: floatUp .45s ease both, pulseRing 1.8s ease;
            }}
            .insight-strip {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: .8rem;
            }}
            .insight-item {{
                padding: .85rem;
                border: 1px solid var(--app-border);
                border-radius: 18px;
                background: rgba(255, 255, 255, .055);
            }}
            .mini-feed {{
                display: flex;
                flex-direction: column;
                gap: .7rem;
            }}
            .feed-item {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: .75rem;
                padding: .8rem;
                border-radius: 16px;
                background: rgba(255,255,255,.055);
                border: 1px solid rgba(148, 163, 184, .15);
            }}
            .feed-avatar {{
                width: 38px;
                height: 38px;
                display: grid;
                place-items: center;
                border-radius: 13px;
                background: linear-gradient(135deg, rgba(34,211,238,.22), rgba(139,92,246,.20));
                color: #e0fbff;
                font-weight: 850;
            }}

            /* Clean theme overrides */
            html, body, [data-testid="stAppViewContainer"] {{
                background: var(--app-bg) !important;
                animation: none !important;
                color: var(--app-text);
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 52%, #eef8ff 100%) !important;
                border-right: 1px solid #e5edf6;
                box-shadow: 18px 0 44px rgba(15, 23, 42, .07);
            }}
            [data-testid="stSidebar"] * {{
                color: #0f172a !important;
            }}
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
                color: #64748b !important;
            }}
            .nav-brand {{
                display: flex;
                align-items: center;
                gap: .8rem;
                margin: .15rem 0 .95rem;
                padding: .95rem;
                border: 1px solid #dbeafe;
                border-radius: 22px;
                background:
                    linear-gradient(135deg, rgba(14, 165, 233, .12), rgba(16, 185, 129, .10)),
                    #ffffff;
                box-shadow: 0 16px 34px rgba(2, 132, 199, .10);
            }}
            .nav-brand i {{
                width: 46px;
                height: 46px;
                display: inline-grid;
                place-items: center;
                border-radius: 17px;
                background: linear-gradient(135deg, #0284c7, #059669);
                color: #ffffff !important;
                box-shadow: 0 12px 24px rgba(2, 132, 199, .24);
            }}
            .nav-title {{
                display: flex;
                flex-direction: column;
                line-height: 1.15;
            }}
            .nav-title strong {{
                font-size: 1.02rem;
                font-weight: 900;
                color: #0f172a !important;
            }}
            .nav-title span {{
                margin-top: .24rem;
                font-size: .75rem;
                font-weight: 750;
                color: #64748b !important;
            }}
            .nav-caption {{
                margin: .95rem .45rem .45rem;
                font-size: .72rem;
                font-weight: 850;
                letter-spacing: .08em;
                text-transform: uppercase;
                color: #94a3b8 !important;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] {{
                display: flex;
                flex-direction: column;
                gap: .55rem;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label {{
                min-height: 52px;
                padding: .62rem .75rem .62rem 3.1rem;
                border: 1px solid transparent;
                border-radius: 18px;
                background: rgba(255, 255, 255, .56);
                position: relative;
                transition: background .18s ease, border-color .18s ease, transform .18s ease, box-shadow .18s ease;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label input {{
                opacity: 0;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stWidgetLabel"],
            [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
                display: none !important;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label:before {{
                content: "\\f015";
                font-family: "Font Awesome 6 Free";
                font-weight: 900;
                position: absolute;
                left: .72rem;
                top: 50%;
                transform: translateY(-50%);
                width: 34px;
                height: 34px;
                display: grid;
                place-items: center;
                border-radius: 12px;
                background: #f1f5f9;
                color: #64748b;
                transition: background .18s ease, color .18s ease, box-shadow .18s ease;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label:nth-child(2):before {{ content: "\\f030"; }}
            [data-testid="stSidebar"] [role="radiogroup"] label:nth-child(3):before {{ content: "\\f03d"; }}
            [data-testid="stSidebar"] [role="radiogroup"] label:nth-child(4):before {{ content: "\\f080"; }}
            [data-testid="stSidebar"] [role="radiogroup"] label:nth-child(5):before {{ content: "\\f544"; }}
            [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
                background: #ffffff;
                border-color: #dbeafe;
                transform: translateX(4px);
                box-shadow: 0 12px 26px rgba(15, 23, 42, .07);
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
                background: linear-gradient(135deg, rgba(14, 165, 233, .16), rgba(16, 185, 129, .14)), #ffffff;
                border-color: #bae6fd;
                box-shadow: inset 4px 0 0 #0284c7, 0 14px 28px rgba(2, 132, 199, .12);
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked):before {{
                background: linear-gradient(135deg, #0284c7, #059669);
                color: #ffffff;
                box-shadow: 0 10px 20px rgba(2, 132, 199, .20);
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
                color: #075985 !important;
                font-weight: 850;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label p {{
                font-weight: 760;
            }}
            [data-testid="stSidebar"] hr {{
                border-color: #e2e8f0;
                margin: 1rem 0;
            }}
            .nav-mode {{
                margin-top: .85rem;
                padding: .75rem;
                border: 1px solid #dbeafe;
                border-radius: 18px;
                background: rgba(255, 255, 255, .72);
                box-shadow: 0 10px 24px rgba(15, 23, 42, .05);
            }}
            .nav-mode p {{
                margin: 0 0 .55rem;
                font-size: .78rem;
                font-weight: 850;
                color: #64748b !important;
            }}
            .button-note {{
                margin-bottom: .8rem;
                padding: .75rem .85rem;
                border-radius: 14px;
                border: 1px solid #e2e8f0;
                background: #f8fafc;
                color: #64748b !important;
                font-size: .9rem;
            }}
            .app-topbar, .hero-shell, .glass-card, .metric-card, .camera-card, .table-shell {{
                background: var(--app-card) !important;
                border: 1px solid var(--app-border) !important;
                border-radius: 16px !important;
                box-shadow: 0 10px 28px rgba(15, 23, 42, .07) !important;
                backdrop-filter: none !important;
            }}
            .hero-shell {{
                min-height: 280px !important;
                background: linear-gradient(135deg, rgba(14, 165, 233, .10), rgba(16, 185, 129, .08)), var(--app-card) !important;
            }}
            .hero-shell:after, .preview-board, .action-grid {{
                display: none !important;
            }}
            .hero-grid {{
                grid-template-columns: 1fr !important;
            }}
            .hero-title {{
                max-width: 820px !important;
                font-size: clamp(2.1rem, 5vw, 4.25rem) !important;
                line-height: 1.04 !important;
                color: var(--app-text) !important;
            }}
            .hero-copy, p, .muted {{
                color: var(--app-muted) !important;
            }}
            .eyebrow, .topbar-chip {{
                background: rgba(14, 165, 233, .09) !important;
                border-color: rgba(14, 165, 233, .18) !important;
                color: var(--app-text) !important;
            }}
            .gradient-text {{
                background: linear-gradient(90deg, #0284c7, #059669) !important;
                -webkit-background-clip: text !important;
                color: transparent !important;
            }}
            .metric-card:after, .confidence-bar span {{
                background: linear-gradient(90deg, #0284c7, #059669) !important;
            }}
            .metric-icon, .step-dot {{
                background: rgba(14, 165, 233, .10) !important;
                color: #0284c7 !important;
            }}
            .status-pill {{
                background: rgba(16, 185, 129, .12) !important;
                border-color: rgba(16, 185, 129, .22) !important;
                color: #047857 !important;
            }}
            div.stButton > button {{
                min-height: 48px !important;
                border-radius: 15px !important;
                color: #ffffff !important;
                background:
                    linear-gradient(135deg, #0369a1 0%, #0284c7 48%, #059669 100%) !important;
                border: 1px solid rgba(255, 255, 255, .55) !important;
                box-shadow: 0 12px 26px rgba(2, 132, 199, .20) !important;
                animation: none !important;
                font-weight: 850 !important;
                letter-spacing: 0 !important;
                transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
            }}
            div.stButton > button:hover, .glass-card:hover, .metric-card:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 14px 32px rgba(15, 23, 42, .10) !important;
            }}
            div.stButton > button:hover {{
                filter: brightness(1.04) saturate(1.06) !important;
                box-shadow: 0 16px 34px rgba(5, 150, 105, .18) !important;
            }}
            div.stButton > button:active {{
                transform: translateY(0) scale(.99) !important;
                box-shadow: 0 8px 18px rgba(15, 23, 42, .10) !important;
            }}
            [data-testid="stFormSubmitButton"] button {{
                min-height: 52px !important;
                border-radius: 16px !important;
                background:
                    linear-gradient(135deg, #ffffff, #f0f9ff) !important;
                color: #075985 !important;
                border: 1px solid #bae6fd !important;
                box-shadow: 0 12px 26px rgba(2, 132, 199, .12) !important;
                font-weight: 900 !important;
            }}
            [data-testid="stFormSubmitButton"] button:hover {{
                background:
                    linear-gradient(135deg, #ecfeff, #f0fdf4) !important;
                color: #064e3b !important;
                border-color: #86efac !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 16px 34px rgba(5, 150, 105, .14) !important;
            }}
            [data-testid="stDownloadButton"] button {{
                min-height: 48px !important;
                border-radius: 15px !important;
                color: #075985 !important;
                background: #ffffff !important;
                border: 1px solid #bae6fd !important;
                box-shadow: 0 10px 22px rgba(15, 23, 42, .08) !important;
                font-weight: 850 !important;
            }}
            [data-testid="stDownloadButton"] button:hover {{
                color: #064e3b !important;
                background: #f0fdf4 !important;
                border-color: #86efac !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 14px 30px rgba(5, 150, 105, .12) !important;
            }}
            .attendance-actions div.stButton > button,
            .attendance-actions div.stButton > button * {{
                color: #ffffff !important;
            }}
            .camera-preview {{
                background: linear-gradient(135deg, rgba(14, 165, 233, .08), rgba(16, 185, 129, .07)) !important;
                border-radius: 14px !important;
            }}
            .camera-preview:before {{
                background: linear-gradient(90deg, transparent, #0284c7, transparent) !important;
            }}
            .camera-preview:after {{
                border-color: rgba(2, 132, 199, .60) !important;
            }}
            .compact-camera {{
                max-width: 620px;
                min-height: 220px !important;
                margin-left: auto;
            }}
            .compact-camera .camera-preview {{
                min-height: 190px !important;
            }}
            .compact-camera .camera-preview:after {{
                width: 104px !important;
                height: 104px !important;
                border-radius: 22px !important;
            }}
            .insight-item, .feed-item {{
                background: rgba(148, 163, 184, .06) !important;
            }}
            .feed-avatar {{
                background: rgba(14, 165, 233, .12) !important;
                color: #0284c7 !important;
            }}
            @media (max-width: 768px) {{
                .hero-shell {{ min-height: 520px; padding: 1.3rem; border-radius: 22px; }}
                .hero-title {{ font-size: 2.45rem; }}
                .metric-card {{ min-height: 110px; }}
                .hero-grid, .action-grid, .insight-strip {{ grid-template-columns: 1fr; }}
                .app-topbar {{ align-items: stretch; flex-direction: column; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_navigation():
    with st.sidebar:
        st.markdown(
            """
            <div class="nav-brand">
                <i class="fa-solid fa-face-smile-beam me-2"></i>
                <div class="nav-title">
                    <strong>Smart Attendance</strong>
                    <span>Face recognition</span>
                </div>
            </div>
            <div class="nav-caption">Workspace</div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["Home", "Register", "Attendance", "Dashboard", "Train Model"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<div class="nav-mode"><p>Appearance</p>', unsafe_allow_html=True)
        st.toggle("Dark mode", key="dark_mode")
        st.markdown("</div>", unsafe_allow_html=True)
        return page


def metric_card(icon, label, value, help_text):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="d-flex align-items-center justify-content-between">
                <div>
                    <p class="muted mb-0">{label}</p>
                    <div class="metric-number">{value}</div>
                </div>
                <div class="metric-icon"><i class="{icon}"></i></div>
            </div>
            <p class="muted mb-0">{help_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(page):
    students = get_students()
    model_state = "Model ready" if (TRAINER_DIR / "trainer.yml").exists() else "Train needed"
    now = datetime.now().strftime("%d %b, %I:%M %p")
    st.markdown(
        f"""
        <div class="app-topbar">
            <div class="command-search">
                <i class="fa-solid fa-magnifying-glass-chart"></i>
                <span>{escape(page)} - {len(students)} registered students - {model_state}</span>
            </div>
            <div class="d-flex flex-wrap gap-2">
                <span class="topbar-chip"><i class="fa-solid fa-clock"></i>{now}</span>
                <span class="topbar-chip"><i class="fa-solid fa-signal"></i>{escape(st.session_state.last_event)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def quick_actions():
    model_state = "Ready" if (TRAINER_DIR / "trainer.yml").exists() else "Pending"
    st.markdown(
        f"""
        <div class="action-grid mt-3">
            <div class="action-card">
                <i class="fa-solid fa-camera fa-xl mb-3"></i>
                <h4>Register Flow</h4>
                <p class="muted mb-0">Guided capture with scanner animation, progress feedback, and saved face samples.</p>
            </div>
            <div class="action-card">
                <i class="fa-solid fa-brain fa-xl mb-3"></i>
                <h4>Recognition Core</h4>
                <p class="muted mb-0">Current model status: <strong>{model_state}</strong>. Train whenever new faces are added.</p>
            </div>
            <div class="action-card">
                <i class="fa-solid fa-chart-simple fa-xl mb-3"></i>
                <h4>Live Dashboard</h4>
                <p class="muted mb-0">Attendance cards, current-session table, CSV snapshot, and XML export.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recent_activity(records):
    if not records:
        st.markdown(
            """
            <div class="mini-feed">
                <div class="feed-item">
                    <div class="d-flex align-items-center gap-3">
                        <div class="feed-avatar"><i class="fa-solid fa-sparkles"></i></div>
                        <div><strong>Ready for first scan</strong><p class="muted mb-0">Attendance activity will appear here.</p></div>
                    </div>
                    <span class="status-pill">Idle</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    items = []
    for record in records[-5:][::-1]:
        name = escape(record.get("name") or "Student")
        initials = "".join(part[:1] for part in name.split()[:2]).upper() or "FR"
        confidence = escape(record.get("confidence", "0"))
        time_value = escape(record.get("time", "--:--"))
        items.append(
            f"""
            <div class="feed-item">
                <div class="d-flex align-items-center gap-3">
                    <div class="feed-avatar">{initials}</div>
                    <div><strong>{name}</strong><p class="muted mb-0">Confidence {confidence} - {time_value}</p></div>
                </div>
                <span class="status-pill">Present</span>
            </div>
            """
        )
    st.markdown(f'<div class="mini-feed">{"".join(items)}</div>', unsafe_allow_html=True)


def hero():
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-grid position-relative z-1">
                <div>
                    <span class="eyebrow">
                        <i class="fa-solid fa-user-check"></i>
                        Smart attendance system
                    </span>
                    <h1 class="hero-title">
                        Face Recognition <span class="gradient-text">Attendance</span>
                    </h1>
                    <p class="hero-copy">
                        Register students, train the face recognizer, mark attendance with the camera, and review records from a clean dashboard.
                    </p>
                    <div class="d-flex flex-wrap gap-3 mt-4">
                        <span class="status-pill"><i class="fa-solid fa-video"></i> Live camera capture</span>
                        <span class="status-pill"><i class="fa-solid fa-bolt"></i> Automatic matching</span>
                        <span class="status-pill"><i class="fa-solid fa-file-code"></i> XML exports</span>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def home_page():
    hero()
    st.write("")

    students = get_students()
    records = list(st.session_state.attendance_records.values())
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in records if r["date"] == today_str]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("fa-solid fa-users", "Students", len(students), "Registered in the local roster")
    with col2:
        metric_card("fa-solid fa-database", "Face Sets", dataset_student_count(), "Students with captured samples")
    with col3:
        metric_card("fa-solid fa-user-check", "Today Present", len(today_records), "Marked today")
    with col4:
        metric_card("fa-solid fa-brain", "Model", "Ready" if (TRAINER_DIR / "trainer.yml").exists() else "Train", "Recognizer status")

    st.markdown(
        """
        <div class="glass-card p-4 mt-3">
            <h3 class="mb-3">Workflow</h3>
            <div class="timeline-step">
                <div class="step-dot"><i class="fa-solid fa-id-card"></i></div>
                <div><strong>Register</strong><p class="muted mb-0">Enter student details and capture face samples.</p></div>
            </div>
            <div class="timeline-step">
                <div class="step-dot"><i class="fa-solid fa-brain"></i></div>
                <div><strong>Train</strong><p class="muted mb-0">Train the recognizer after adding students.</p></div>
            </div>
            <div class="timeline-step">
                <div class="step-dot"><i class="fa-solid fa-camera-retro"></i></div>
                <div><strong>Attendance</strong><p class="muted mb-0">Start camera scanning and mark present students.</p></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def register_page():
    st.markdown('<h1>Register Face</h1><p class="muted">Capture a clean face dataset with guided progress and immediate feedback.</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-strip mb-3">
            <div class="insight-item"><strong>01</strong><p class="muted mb-0">Enter student identity</p></div>
            <div class="insight-item"><strong>02</strong><p class="muted mb-0">Open camera scanner</p></div>
            <div class="insight-item"><strong>03</strong><p class="muted mb-0">Capture 5 samples</p></div>
            <div class="insight-item"><strong>04</strong><p class="muted mb-0">Train recognizer</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    form_col, camera_col = st.columns([0.92, 1.08], gap="large")

    with form_col:
        st.markdown('<div class="glass-card p-4 white-action attendance-actions">', unsafe_allow_html=True)
        st.markdown('<div class="button-note"><i class="fa-solid fa-camera me-2"></i>Capture 5 face samples for a new student.</div>', unsafe_allow_html=True)
        with st.form("register_form"):
            student_id = st.text_input("Student ID", placeholder="e.g. CS-1042")
            student_name = st.text_input("Student Name", placeholder="e.g. Aditi Sharma")
            submitted = st.form_submit_button("Register and Capture Faces")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="glass-card p-4 mt-3">
                <h4>Capture Tips</h4>
                <p class="muted mb-2"><i class="fa-solid fa-lightbulb me-2"></i>Face the camera with even lighting.</p>
                <p class="muted mb-2"><i class="fa-solid fa-eye me-2"></i>Keep your face inside the scanner frame.</p>
                <p class="muted mb-0"><i class="fa-solid fa-circle-notch me-2"></i>The system saves 5 samples per student.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with camera_col:
        st.markdown(
            """
            <div class="camera-card">
                <div class="camera-preview">
                    <div class="position-absolute text-center">
                        <i class="fa-solid fa-camera fa-2x mb-3" style="color:#67e8f9"></i>
                        <p class="muted mb-0">Camera activates when capture starts</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        if not student_id or not student_name:
            st.error("Please enter both Student ID and Student Name.")
            return

        add_student_record(student_id, student_name)
        st.session_state.last_event = f"Capturing {student_name}"
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_placeholder = st.empty()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Cannot open camera.")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        sample_count = 0
        required = 5
        attempts = 0
        max_attempts = 900
        failed_reads = 0

        while sample_count < required and attempts < max_attempts:
            attempts += 1
            ret, frame = cap.read()
            if not ret:
                failed_reads += 1
                status_text.warning("Camera frame not available. Retrying...")
                if failed_reads >= 20:
                    st.error("Failed to capture camera frames from the webcam.")
                    break
                time.sleep(0.08)
                continue

            failed_reads = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(80, 80),
            )

            if len(faces) == 0:
                status_text.info(f"Looking for a face... Captured {sample_count}/{required} samples.")

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
                filename = DATASET_DIR / f"User.{student_id}.{sample_count + 1}.jpg"
                saved = cv2.imwrite(str(filename), face_img)
                if saved:
                    sample_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (52, 211, 153), 2)
                    cv2.putText(frame, f"Captured {sample_count}/{required}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    progress_bar.progress(sample_count / required)
                    status_text.markdown(f'<div class="success-burst">Face detected. Captured {sample_count}/{required} samples.</div>', unsafe_allow_html=True)
                else:
                    status_text.error("Could not save the captured image. Check dataset folder permissions.")
                break
            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.08)

        cap.release()
        if sample_count == required:
            st.session_state.last_event = f"Registered {student_name}"
            st.success(f"Face capture complete for {student_name}. Saved {sample_count}/{required} images.")
        else:
            st.session_state.last_event = f"Capture incomplete for {student_name}"
            st.warning(f"Capture stopped after saving {sample_count}/{required} images. Keep your face centered and try again.")


def train_page():
    st.markdown('<h1>Train Model</h1><p class="muted">Refresh the recognizer after adding or updating student face samples.</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("fa-solid fa-images", "Images", len(list(DATASET_DIR.glob("User.*.*.jpg"))), "Saved face samples")
    with col2:
        metric_card("fa-solid fa-database", "Identities", dataset_student_count(), "Unique students in dataset")
    with col3:
        metric_card("fa-solid fa-brain", "Model", "Ready" if (TRAINER_DIR / "trainer.yml").exists() else "Missing", "Recognizer status")

    st.write("")
    left, right = st.columns([0.8, 1.2], gap="large")
    with left:
        if st.button("Train Recognizer"):
            with st.spinner("Training the recognition model..."):
                success = train_model()
            if success:
                st.success("Model trained successfully.")
    with right:
        st.markdown(
            """
            <div class="glass-card p-4">
                <h4>Training Notes</h4>
                <p class="muted mb-0">
                    The recognizer maps each registered student ID to an internal numeric label, trains on the grayscale face crops, and stores both the model and mapping under the trainer folder.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def attendance_page():
    st.markdown('<h1>Mark Attendance</h1><p class="muted">Start a live recognition session. Student details fill automatically when confidence is high.</p>', unsafe_allow_html=True)

    control_col, scan_col = st.columns([0.75, 1.25], gap="large")
    with control_col:
        st.markdown('<div class="glass-card p-4 white-action">', unsafe_allow_html=True)
        st.markdown('<div class="button-note"><i class="fa-solid fa-video me-2"></i>Control the live attendance scanner.</div>', unsafe_allow_html=True)
        start = st.button("Start Attendance")
        stop = st.button("Stop Attendance")
        st.markdown("</div>", unsafe_allow_html=True)

        if start:
            st.session_state.attendance_running = True
            st.session_state.last_event = "Attendance scan running"
        if stop:
            st.session_state.attendance_running = False
            st.session_state.last_event = "Attendance scan stopped"

        records = list(st.session_state.attendance_records.values())
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_records = [r for r in records if r["date"] == today_str]
        st.write("")
        metric_card("fa-solid fa-user-check", "Marked Today", len(today_records), "Unique recognized students today")

    with scan_col:
        st.markdown(
            """
            <div class="camera-card compact-camera">
                <div class="camera-preview">
                    <div class="position-absolute text-center">
                        <i class="fa-solid fa-circle-notch fa-spin fa-2x mb-3" style="color:#67e8f9"></i>
                        <p class="muted mb-0">Detection feedback appears below when scanning starts</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.attendance_running:
        recognizer, int_to_student_id = load_model()
        if not recognizer:
            st.error("Model not trained. Please train the recognizer first.")
            st.session_state.attendance_running = False
            return

        students = get_students()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera.")
            st.session_state.attendance_running = False
            return

        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        while st.session_state.attendance_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5)
            if len(faces) == 0:
                status_placeholder.info("Searching for a face...")

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
                id_pred, confidence = recognizer.predict(face_img)
                student_id = int_to_student_id.get(id_pred, "Unrecognized")
                name = students.get(student_id, "Unrecognized")

                if confidence < 70:
                    timestamp = datetime.now()
                    date_str = timestamp.strftime("%Y-%m-%d")
                    record_key = f"{student_id}_{date_str}"
                    if record_key not in st.session_state.attendance_records:
                        new_record = {
                            "id": student_id,
                            "name": name,
                            "date": date_str,
                            "time": timestamp.strftime("%H:%M:%S"),
                            "confidence": f"{confidence:.1f}",
                            "status": "Present",
                        }
                        st.session_state.attendance_records[record_key] = new_record
                        append_attendance_record(new_record)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (52, 211, 153), 2)
                    status_placeholder.markdown(
                        f'<div class="success-burst">Recognized {name} ({student_id}) with confidence {confidence:.1f}.</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.last_event = f"Marked {name}"
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (251, 113, 133), 2)
                    status_placeholder.error("Face detected, but confidence is too low.")

                cv2.putText(frame, f"{name} {confidence:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                break

            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.1)

        cap.release()


def dashboard_page():
    st.markdown('<h1>Attendance Dashboard</h1><p class="muted">Monitor attendance records and history.</p>', unsafe_allow_html=True)

    records = list(st.session_state.attendance_records.values())
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in records if r["date"] == today_str]
    students = get_students()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("fa-solid fa-user-graduate", "Roster", len(students), "Students registered")
    with col2:
        metric_card("fa-solid fa-clipboard-check", "Present Today", len(today_records), "Marked today")
    with col3:
        rate = f"{round((len(today_records) / len(students)) * 100)}%" if students else "0%"
        metric_card("fa-solid fa-chart-line", "Coverage", rate, "Today's coverage")

    st.write("")
    
    tab1, tab2 = st.tabs(["Today's Attendance", "Full History"])
    
    with tab1:
        if today_records:
            df_today = pd.DataFrame(today_records)
            df_today = df_today.sort_values(by=["time"], ascending=False)
            st.markdown('<div class="table-shell p-3 mt-3">', unsafe_allow_html=True)
            st.dataframe(df_today, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            left, right = st.columns([0.35, 0.65])
            with left:
                if st.button("Save Today's XML"):
                    xml_path = save_attendance_xml()
                    if xml_path:
                        st.success(f"Saved to {xml_path.name}.")
            with right:
                csv_data = df_today.to_csv(index=False).encode("utf-8")
                st.download_button("Download Today's CSV", csv_data, f"attendance_{today_str}.csv", "text/csv")
        else:
            st.markdown(
                """
                <div class="glass-card p-5 text-center">
                    <i class="fa-solid fa-clipboard-list fa-3x mb-3" style="color:#67e8f9"></i>
                    <h3>No attendance today</h3>
                    <p class="muted mb-0">Start an attendance scan to populate today's list.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab2:
        if records:
            df_all = pd.DataFrame(records)
            df_all = df_all.sort_values(by=["date", "time"], ascending=[False, False])
            st.markdown('<div class="table-shell p-3 mt-3">', unsafe_allow_html=True)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            csv_all_data = df_all.to_csv(index=False).encode("utf-8")
            st.download_button("Download Full History CSV", csv_all_data, "attendance_history_export.csv", "text/csv")
        else:
            st.markdown(
                """
                <div class="glass-card p-5 text-center">
                    <i class="fa-solid fa-clock-rotate-left fa-3x mb-3" style="color:#67e8f9"></i>
                    <h3>No history available</h3>
                    <p class="muted mb-0">Historical records will appear here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


apply_theme()
page = sidebar_navigation()
render_topbar(page)

if page == "Home":
    home_page()
elif page == "Register":
    register_page()
elif page == "Attendance":
    attendance_page()
elif page == "Dashboard":
    dashboard_page()
else:
    train_page()
