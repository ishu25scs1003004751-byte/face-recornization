import os
import re
import json
import base64
import hashlib
from io import BytesIO
from PIL import Image
import pandas as pd
import streamlit as st
from datetime import datetime
from groq import Groq

# ------------------------------------------------------------------
# 1. Page config (sidebar hidden: navigation happens on the home screen)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Attendance",
    layout="wide",
    page_icon="🪪",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# 2. Storage
# ------------------------------------------------------------------
STUDENTS_DIR = "students_data"
RECORDS_DIR = "attendance_records"
DB_FILE = "students.json"

os.makedirs(STUDENTS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump({}, f)


def load_students():
    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_students(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)


SUBJECTS_FILE = "subjects.json"
DEFAULT_SUBJECTS = ["Data Structures", "Operating Systems", "Machine Learning", "Python"]
BASE_COLS = ["Roll No", "Student Name", "Department"]
DATE_FILE_RE = r"^Attendance_\d{4}-\d{2}-\d{2}\.xlsx$"
TIME_SUFFIX = " (Time)"


def subject_cols(df):
    return [c for c in df.columns if c not in BASE_COLS and not str(c).endswith(TIME_SUFFIX)]


def save_subjects(data):
    with open(SUBJECTS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_subjects():
    if not os.path.exists(SUBJECTS_FILE):
        save_subjects(DEFAULT_SUBJECTS)
    with open(SUBJECTS_FILE, "r") as f:
        return json.load(f)


# ------------------------------------------------------------------
# 3. Groq client
# ------------------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Groq API key is missing. Add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-120b"


def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


VISION_MODEL = st.secrets.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def to_data_url(image_bytes, max_side=640):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + encode_image_to_base64(buf.getvalue())


def identify_student(live_bytes, students):
    """Ask the Groq vision model which registered student is in the live photo.
    Returns (roll_no or None, message). Registered photos are sent in batches of 4."""
    cands = [s for s in students.values() if os.path.exists(s.get("image_path", ""))]
    if not cands:
        return None, "No registered photos to compare with. Select the student yourself."
    live_url = to_data_url(live_bytes)
    found = []
    for i in range(0, len(cands), 4):
        batch = cands[i:i + 4]
        prompt = (
            "You compare faces for a school attendance system. The LAST image is a live photo. "
            "The earlier images are registered students, in this order: "
            + "; ".join(f"image {n + 1} = roll {c['roll_no']}" for n, c in enumerate(batch))
            + ". Decide whether the live photo shows the same person as one of the registered images. "
            'Reply with JSON only: {"match": "<roll number or NONE>", "confidence": "high|medium|low"}'
        )
        content = [{"type": "text", "text": prompt}]
        for c in batch:
            with open(c["image_path"], "rb") as f:
                content.append({"type": "image_url", "image_url": {"url": to_data_url(f.read())}})
        content.append({"type": "image_url", "image_url": {"url": live_url}})
        resp = client.chat.completions.create(
            model=VISION_MODEL, messages=[{"role": "user", "content": content}], temperature=0, max_tokens=100
        )
        m = re.search(r"\{.*?\}", resp.choices[0].message.content or "", re.S)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
        except ValueError:
            continue
        roll = str(data.get("match", "NONE")).strip()
        conf = str(data.get("confidence", "")).lower()
        if roll in students and conf in ("high", "medium"):
            found.append((roll, conf))

    highs = [r for r, c in found if c == "high"]
    pick = highs[0] if len(highs) == 1 else (found[0][0] if len(found) == 1 else None)
    if pick:
        conf = dict(found)[pick]
        return pick, f"AI suggestion: {students[pick]['name']} ({conf} confidence). Please confirm."
    return None, "The AI could not find a confident match. Select the student yourself."


# ------------------------------------------------------------------
# 4. Styling
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --paper: #F4F6F3;
    --card: #FFFFFF;
    --ink: #14292B;
    --muted: #5E6E6D;
    --line: #DCE3DF;
    --teal: #0E7C7B;
    --teal-dark: #0A5F5E;
    --teal-soft: #DFF1EF;
    --amber: #C77700;
    --amber-soft: #FDF0DC;
    --indigo: #3F51B5;
    --indigo-soft: #E6E9F8;
    --red: #B3261E;
    --red-soft: #FBE9E7;
}

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp { background: var(--paper); color: var(--ink); }
.block-container { padding-top: 1.6rem; max-width: 1150px; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"], div[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapsedControl"] { display: none; }

/* Hero (home) */
.hero {
    background: linear-gradient(120deg, #0A5F5E 0%, #0E7C7B 55%, #2A9D8F 100%);
    border-radius: 22px; padding: 38px 40px; color: #fff; margin-bottom: 26px;
    position: relative; overflow: hidden;
}
.hero::after {
    content: "🪪"; position: absolute; right: 34px; top: 12px; font-size: 7.5rem; opacity: 0.18;
}
.hero h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 6px 0; color: #fff; }
.hero p { margin: 0 0 22px 0; font-size: 1.05rem; opacity: 0.9; max-width: 52ch; }
.hero-stats { display: flex; gap: 14px; flex-wrap: wrap; }
.hero-stat { background: rgba(255,255,255,0.16); border-radius: 14px; padding: 10px 18px; min-width: 130px; }
.hero-stat b { display: block; font-size: 1.6rem; font-weight: 800; }
.hero-stat span { font-size: 0.85rem; opacity: 0.85; }

/* Home option cards */
.opt {
    background: var(--card); border: 1px solid var(--line); border-radius: 20px;
    padding: 26px 24px 18px 24px; min-height: 215px;
}
.opt-icon {
    width: 58px; height: 58px; border-radius: 16px; display: flex; align-items: center;
    justify-content: center; font-size: 1.9rem; margin-bottom: 14px;
}
.opt h3 { margin: 0 0 6px 0; font-size: 1.3rem; font-weight: 800; color: var(--ink); }
.opt p { margin: 0; color: var(--muted); font-size: 0.95rem; line-height: 1.5; }
.opt-reg .opt-icon { background: var(--teal-soft); }
.opt-live .opt-icon { background: var(--amber-soft); }
.opt-view .opt-icon { background: var(--indigo-soft); }
.opt-reg { border-top: 5px solid var(--teal); }
.opt-live { border-top: 5px solid var(--amber); }
.opt-view { border-top: 5px solid var(--indigo); }

/* Page header */
.page-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; color: var(--ink); }
.page-sub { color: var(--muted); font-size: 1rem; margin: 0.25rem 0 1.4rem 0; max-width: 60ch; }

/* Cards */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
    padding: 20px 22px; margin-bottom: 14px; }
.card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; }
.card-note { color: var(--muted); font-size: 0.9rem; margin-bottom: 12px; }

/* Result panel */
.result-ok { background: var(--teal-soft); border: 1px solid var(--teal); border-radius: 16px; padding: 20px 22px; }
.result-ok h3 { margin: 0 0 8px 0; color: var(--teal-dark); font-size: 1.25rem; }
.result-ok .row { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0;
    border-bottom: 1px solid rgba(14,124,123,0.2); font-size: 0.95rem; }
.result-ok .row:last-child { border-bottom: none; }
.result-ok .row span:first-child { color: var(--muted); }
.result-wait { background: var(--card); border: 1px dashed var(--line); border-radius: 16px;
    padding: 28px 22px; color: var(--muted); text-align: center; }

/* Student row */
.student-row { display: flex; align-items: center; gap: 12px; padding: 4px 0; }
.avatar { width: 42px; height: 42px; border-radius: 50%; background: var(--teal-soft); color: var(--teal-dark);
    display: flex; align-items: center; justify-content: center; font-weight: 800; flex-shrink: 0; }
.student-name { font-weight: 700; }
.student-meta { color: var(--muted); font-size: 0.85rem; }

/* Metrics */
div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; }
div[data-testid="stMetricLabel"] p { color: var(--muted); font-weight: 500; }
div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: var(--teal); color: #fff; font-weight: 600; border: none;
    border-radius: 12px; padding: 0.7rem 1.2rem; width: 100%; transition: background 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--teal-dark); color: #fff; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid #9ADAD6; outline-offset: 2px; }
.stButton > button[kind="secondary"] { background: #fff; color: var(--ink); border: 1px solid var(--line); }
.stButton > button[kind="secondary"]:hover { background: var(--paper); color: var(--ink); }
.stButton > button[kind="primary"].danger, .del-zone .stButton > button { }

.stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div { border-radius: 10px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }

@media (max-width: 700px) {
    .hero { padding: 26px 22px; } .hero h1 { font-size: 1.7rem; } .hero::after { display: none; }
    .page-title { font-size: 1.5rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 5. Navigation state
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None


def go(page):
    st.session_state.page = page
    st.session_state.pending_delete = None


def page_header(title, subtitle):
    top = st.columns([1, 5])
    with top[0]:
        st.button("← Home", key="back_home", on_click=go, args=("home",), type="secondary")
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def card_start(title=None, note=None):
    html = '<div class="card">'
    if title:
        html += f'<div class="card-title">{title}</div>'
    if note:
        html += f'<div class="card-note">{note}</div>'
    st.markdown(html, unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


students_db = load_students()
page = st.session_state.page

# ==================================================================
# HOME
# ==================================================================
if page == "home":
    today = datetime.now().strftime("%Y-%m-%d")
    today_file = os.path.join(RECORDS_DIR, f"Attendance_{today}.xlsx")
    sheets_today = 0
    if os.path.exists(today_file):
        sheets_today = len(subject_cols(pd.read_excel(today_file, nrows=0)))
    total_sheets = len([f for f in os.listdir(RECORDS_DIR) if re.match(DATE_FILE_RE, f)])

    st.markdown(
        f"""
<div class="hero">
  <h1>Attendance, made simple</h1>
  <p>Register students, mark attendance with the camera, and check reports. All in one place.</p>
  <div class="hero-stats">
    <div class="hero-stat"><b>{len(students_db)}</b><span>Registered students</span></div>
    <div class="hero-stat"><b>{sheets_today}</b><span>Subjects marked today</span></div>
    <div class="hero-stat"><b>{total_sheets}</b><span>Dates recorded</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            '<div class="opt opt-reg"><div class="opt-icon">📝</div><h3>Registration</h3>'
            "<p>Add new students with a photo, or remove students who have left.</p></div>",
            unsafe_allow_html=True,
        )
        st.button("Open registration", key="go_reg", on_click=go, args=("register",))
    with c2:
        st.markdown(
            '<div class="opt opt-live"><div class="opt-icon">📸</div><h3>Live attendance</h3>'
            "<p>Choose the subject first, then scan a face to mark the student present.</p></div>",
            unsafe_allow_html=True,
        )
        st.button("Start attendance", key="go_live", on_click=go, args=("live",))
    with c3:
        st.markdown(
            '<div class="opt opt-view"><div class="opt-icon">📊</div><h3>View attendance</h3>'
            "<p>See daily sheets, download Excel files, and check each student's percentage.</p></div>",
            unsafe_allow_html=True,
        )
        st.button("View reports", key="go_view", on_click=go, args=("view",))

    st.caption(f"Powered by Groq · {MODEL_NAME}")

# ==================================================================
# REGISTRATION (add + delete)
# ==================================================================
elif page == "register":
    page_header("Registration", "Add a new student, or remove one from the list below.")

    tab_add, tab_manage = st.tabs(["➕ Add student", f"🗂️ Registered students ({len(students_db)})"])

    with tab_add:
        col_form, col_cam = st.columns([1.1, 1], gap="large")

        with col_form:
            card_start("Student details")
            roll_no = st.text_input("Roll number", placeholder="e.g. 21CSE045")
            student_name = st.text_input("Full name", placeholder="e.g. Ishu Mishra")
            department = st.text_input("Department", "CSE (AI & ML)")

            st.caption("Subjects are chosen later, when you take attendance.")
            card_end()

        with col_cam:
            card_start("Photo", "Face the camera in good light.")
            camera_image = st.camera_input("Take registration photo", label_visibility="collapsed")
            card_end()

        if st.button("Save student", key="save_student"):
            if not roll_no or not student_name or camera_image is None:
                st.error("Roll number, name and photo are required.")
            else:
                with st.spinner("Saving student..."):
                    try:
                        img_filename = f"{roll_no}_{student_name.replace(' ', '_')}.jpg"
                        img_path = os.path.join(STUDENTS_DIR, img_filename)
                        with open(img_path, "wb") as f:
                            f.write(camera_image.getbuffer())

                        students_db[roll_no] = {
                            "name": student_name,
                            "roll_no": roll_no,
                            "department": department,
                            "image_path": img_path,
                        }
                        save_students(students_db)
                        st.success(f"{student_name} ({roll_no}) is registered.")
                    except Exception as e:
                        st.error(f"Could not save the student: {e}")

    with tab_manage:
        if not students_db:
            st.info("No students registered yet. Use the Add student tab to register the first one.")
        else:
            for r_no, s in list(students_db.items()):
                initials = "".join(w[0] for w in s["name"].split()[:2]).upper()
                is_pending = st.session_state.pending_delete == r_no

                card_start()
                c_img, c_info, c_btn = st.columns([0.7, 4, 1.4], vertical_alignment="center")
                with c_img:
                    if os.path.exists(s.get("image_path", "")):
                        st.image(s["image_path"], width=58)
                    else:
                        st.markdown(f'<div class="avatar">{initials}</div>', unsafe_allow_html=True)
                with c_info:
                    st.markdown(
                        f'<div class="student-name">{s["name"]}</div>'
                        f'<div class="student-meta">{s["roll_no"]} · {s["department"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c_btn:
                    if not is_pending:
                        if st.button("🗑️ Delete", key=f"del_{r_no}", type="secondary"):
                            st.session_state.pending_delete = r_no
                            st.rerun()

                if is_pending:
                    st.warning(f"Delete {s['name']} ({r_no})? Their photo and profile will be removed. "
                               "Old attendance sheets stay as they are.")
                    y, n = st.columns(2)
                    with y:
                        if st.button("Yes, delete", key=f"yes_{r_no}"):
                            try:
                                if os.path.exists(s.get("image_path", "")):
                                    os.remove(s["image_path"])
                            except OSError:
                                pass
                            students_db.pop(r_no, None)
                            save_students(students_db)
                            st.session_state.pending_delete = None
                            st.rerun()
                    with n:
                        if st.button("Cancel", key=f"no_{r_no}", type="secondary"):
                            st.session_state.pending_delete = None
                            st.rerun()
                card_end()

# ==================================================================
# LIVE ATTENDANCE
# ==================================================================
elif page == "live":
    page_header("Live attendance", "Choose the subject, take the photo, confirm the student, then mark present.")

    subjects = load_subjects()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    PICK = "— Choose a subject —"
    NONE_OPT = "— None —"
    PICK_STUDENT = "— Select student —"

    for k, v in (("cam_n", 0), ("last_result", None), ("ai_sig", None), ("ai_note", "")):
        if k not in st.session_state:
            st.session_state[k] = v

    if "pending_subject" in st.session_state:
        st.session_state["live_subject"] = st.session_state.pop("pending_subject")
    if st.session_state.pop("reset_student", False):
        st.session_state["live_student"] = PICK_STUDENT
        st.session_state.ai_sig = None
        st.session_state.ai_note = ""
    if st.session_state.get("live_subject") not in [PICK] + subjects:
        st.session_state["live_subject"] = PICK
    if st.session_state.get("rm_sub") not in [NONE_OPT] + subjects:
        st.session_state["rm_sub"] = NONE_OPT

    card_start(
        "Subject",
        f"Today is {now.strftime('%d %b %Y')}. Attendance goes into today's sheet, in this subject's own column.",
    )
    picked = st.selectbox("Which subject are you taking attendance for?", [PICK] + subjects, key="live_subject")
    current_subject = None if picked == PICK else picked
    if current_subject:
        st.caption(f"Selected subject: **{current_subject}**")

    with st.expander("Add or remove a subject"):
        a1, a2 = st.columns([3, 1], vertical_alignment="bottom")
        with a1:
            new_sub = st.text_input("New subject name", placeholder="e.g. Cloud Computing", key="new_sub")
        with a2:
            add_clicked = st.button("Add", key="add_sub")
        if add_clicked:
            name = new_sub.strip()
            if not name:
                st.error("Type a subject name first.")
            elif name in BASE_COLS or name.endswith(TIME_SUFFIX) or name.lower() in [x.lower() for x in subjects]:
                st.warning("That subject already exists, or the name is not allowed.")
            else:
                subjects.append(name)
                save_subjects(subjects)
                st.session_state["pending_subject"] = name
                st.rerun()
        if subjects:
            r1, r2 = st.columns([3, 1], vertical_alignment="bottom")
            with r1:
                to_remove = st.selectbox("Remove a subject", [NONE_OPT] + subjects, key="rm_sub")
            with r2:
                rm_clicked = st.button("Remove", key="rm_btn", type="secondary")
            if rm_clicked and to_remove != NONE_OPT:
                subjects.remove(to_remove)
                save_subjects(subjects)
                st.rerun()
    card_end()

    col_cam, col_info = st.columns([1.2, 1], gap="large")

    with col_cam:
        card_start("Camera", "Take the student's photo.")
        live_photo = st.camera_input(
            "Scan face", label_visibility="collapsed", key=f"live_cam_{st.session_state.cam_n}"
        )
        card_end()

        # Run the AI check once for every new photo
        if live_photo is not None and students_db:
            photo_bytes = live_photo.getvalue()
            sig = hashlib.md5(photo_bytes).hexdigest()
            if st.session_state.ai_sig != sig:
                st.session_state.ai_sig = sig
                with st.spinner("Identifying the student..."):
                    try:
                        roll, note = identify_student(photo_bytes, students_db)
                    except Exception as e:
                        roll, note = None, f"The AI check did not work ({e}). Select the student yourself."
                st.session_state["live_student"] = roll if roll else PICK_STUDENT
                st.session_state.ai_note = note

    if st.session_state.get("live_student") not in [PICK_STUDENT] + list(students_db):
        st.session_state["live_student"] = PICK_STUDENT

    with col_info:
        card_start("Student", "Check who this is. Change it if the suggestion is wrong.")
        if live_photo is not None and st.session_state.ai_note:
            st.caption(st.session_state.ai_note)
        chosen = st.selectbox(
            "Student",
            [PICK_STUDENT] + list(students_db),
            key="live_student",
            format_func=lambda r: r if r == PICK_STUDENT else f"{students_db[r]['name']} ({r})",
            label_visibility="collapsed",
        )
        chosen_roll = None if chosen == PICK_STUDENT else chosen
        if chosen_roll and os.path.exists(students_db[chosen_roll].get("image_path", "")):
            st.image(students_db[chosen_roll]["image_path"], width=110, caption="Registered photo")
        card_end()
        mark_clicked = st.button("Mark present", key="mark_btn")

        st.markdown("#### Result")
        if st.session_state.last_result:
            st.markdown(st.session_state.last_result, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-wait">Take a photo, check the student, then press Mark present.</div>',
                unsafe_allow_html=True,
            )

    marked_ok = False
    if mark_clicked:
        if not current_subject:
            st.error("Choose a subject first.")
        elif live_photo is None:
            st.error("Take the student's photo first.")
        elif not chosen_roll:
            st.error("Select which student this is.")
        else:
            try:
                st_info = students_db[chosen_roll]
                time_str = now.strftime("%H:%M:%S")
                excel_file = os.path.join(RECORDS_DIR, f"Attendance_{date_str}.xlsx")

                # One sheet per date: a row per student, a column per subject
                if os.path.exists(excel_file):
                    df = pd.read_excel(excel_file, dtype={"Roll No": str})
                else:
                    df = pd.DataFrame(columns=BASE_COLS)

                known = set(df["Roll No"].astype(str))
                new_rows = [
                    {"Roll No": str(d["roll_no"]), "Student Name": d["name"], "Department": d["department"]}
                    for d in students_db.values()
                    if str(d["roll_no"]) not in known
                ]
                if new_rows:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

                tcol = current_subject + TIME_SUFFIX
                if current_subject not in df.columns:
                    df[current_subject] = "Absent"
                for c in subject_cols(df):
                    df[c] = df[c].fillna("Absent")
                    t = c + TIME_SUFFIX
                    if t not in df.columns:
                        df[t] = "-"
                    df[t] = df[t].fillna("-")
                # Keep each subject's status column next to its time column
                df = df[BASE_COLS + [x for c in subject_cols(df) for x in (c, c + TIME_SUFFIX)]]

                row = df["Roll No"].astype(str) == str(st_info["roll_no"])
                already = bool((df.loc[row, current_subject] == "Present").any())
                df.loc[row, current_subject] = "Present"
                if not already:
                    df.loc[row, tcol] = time_str
                df.to_excel(excel_file, index=False)

                total_students = len(df)
                present_count = int((df[current_subject] == "Present").sum())
                pct = round(present_count / total_students * 100, 1) if total_students else 0
                heading = "Already marked present" if already else "Marked present"

                st.session_state.last_result = f"""
<div class="result-ok">
  <h3>✓ {heading}</h3>
  <div class="row"><span>Name</span><b>{st_info['name']}</b></div>
  <div class="row"><span>Roll number</span><b>{st_info['roll_no']}</b></div>
  <div class="row"><span>Subject</span><b>{current_subject}</b></div>
  <div class="row"><span>Date and time</span><b>{date_str} · {time_str}</b></div>
  <div class="row"><span>{current_subject} so far</span><b>{present_count}/{total_students} present ({pct}%)</b></div>
</div>
"""
                st.session_state.cam_n += 1
                st.session_state["reset_student"] = True
                marked_ok = True
            except Exception as e:
                st.error(f"Could not mark attendance: {e}")
    if marked_ok:
        st.rerun()

    today_file = os.path.join(RECORDS_DIR, f"Attendance_{date_str}.xlsx")
    if os.path.exists(today_file):
        with st.expander("Today's sheet", expanded=True):
            st.dataframe(
                pd.read_excel(today_file, dtype={"Roll No": str}), use_container_width=True, hide_index=True
            )

# ==================================================================
# VIEW ATTENDANCE
# ==================================================================
elif page == "view":
    page_header("View attendance", "See any date, any subject, or any student, with the time each class was marked.")

    excel_files = sorted([f for f in os.listdir(RECORDS_DIR) if re.match(DATE_FILE_RE, f)])

    def file_date(fn):
        return fn[len("Attendance_"):-len(".xlsx")]

    def read_sheet(fn):
        d = pd.read_excel(os.path.join(RECORDS_DIR, fn), dtype={"Roll No": str})
        return d.drop_duplicates(subset=["Roll No"])

    def time_of(row_or_df, subject):
        tc = subject + TIME_SUFFIX
        return str(row_or_df[tc]) if tc in row_or_df.index else "-"

    def colour_status(val):
        v = str(val)
        if v.startswith("Present"):
            return "background-color: #DFF1EF; color: #0A5F5E; font-weight: 600"
        if v.startswith("Absent"):
            return "background-color: #FDF0DC; color: #9A5B00; font-weight: 600"
        return ""

    def show_cell(status, t):
        return f"{status} · {t}" if status == "Present" and t not in ("-", "nan", "") else status

    if not excel_files:
        st.info("No attendance sheets yet. Take attendance first and the sheets will appear here.")
    else:
        tab_day, tab_subject, tab_student, tab_pct = st.tabs(
            ["By date", "By subject (date-wise)", "By student (date-wise)", "Percentages"]
        )

        # ---- By date: one sheet, each subject has a status and a time column ----
        with tab_day:
            sel_date = st.selectbox("Date", [file_date(f) for f in reversed(excel_files)])
            selected_file = f"Attendance_{sel_date}.xlsx"
            df = read_sheet(selected_file)
            sub_cols = subject_cols(df)

            m1, m2 = st.columns(2)
            m1.metric("Students", len(df))
            m2.metric("Subjects taken", len(sub_cols))

            if sub_cols:
                summ = pd.DataFrame(
                    [
                        {
                            "Subject": c,
                            "Present": int((df[c] == "Present").sum()),
                            "Absent": int((df[c] != "Present").sum()),
                            "Attendance %": round((df[c] == "Present").mean() * 100, 1),
                        }
                        for c in sub_cols
                    ]
                )
                st.markdown("#### Subject summary")
                st.dataframe(
                    summ,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Attendance %": st.column_config.ProgressColumn(
                            "Attendance %", min_value=0, max_value=100, format="%.1f%%"
                        )
                    },
                )

            st.markdown(f"#### Student sheet for {sel_date}")
            st.dataframe(
                df.style.map(colour_status, subset=sub_cols) if sub_cols else df,
                use_container_width=True,
                hide_index=True,
            )
            with open(os.path.join(RECORDS_DIR, selected_file), "rb") as f:
                st.download_button(
                    "Download this date's Excel",
                    data=f.read(),
                    file_name=selected_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        # ---- By subject: students down the side, dates across ----
        with tab_subject:
            all_subjects = sorted(
                {
                    c
                    for fn in excel_files
                    for c in subject_cols(pd.read_excel(os.path.join(RECORDS_DIR, fn), nrows=0))
                }
            )
            if not all_subjects:
                st.info("No subject has been marked yet.")
            else:
                subj = st.selectbox("Subject", all_subjects, key="view_subject")
                f_stat, f_disp = [], []
                for fn in excel_files:
                    d = read_sheet(fn)
                    if subj not in d.columns:
                        continue
                    stat = d[subj].astype(str)
                    tc = subj + TIME_SUFFIX
                    tms = d[tc].astype(str) if tc in d.columns else pd.Series("-", index=d.index)
                    key = pd.MultiIndex.from_frame(d[["Roll No", "Student Name"]])
                    f_stat.append(pd.Series(stat.values, index=key, name=file_date(fn)))
                    f_disp.append(
                        pd.Series([show_cell(a, b) for a, b in zip(stat, tms)], index=key, name=file_date(fn))
                    )

                g_stat = pd.concat(f_stat, axis=1).fillna("-")
                g_disp = pd.concat(f_disp, axis=1).fillna("-")
                date_cols = list(g_disp.columns)
                g_disp["Present"] = (g_stat == "Present").sum(axis=1)
                g_disp["Classes"] = g_stat.isin(["Present", "Absent"]).sum(axis=1)
                g_disp["Attendance %"] = (
                    (g_disp["Present"] / g_disp["Classes"].where(g_disp["Classes"] > 0) * 100).round(1).fillna(0.0)
                )
                grid = g_disp.reset_index()

                st.caption(f"{subj}: {len(date_cols)} date(s). Present cells show the time attendance was marked.")
                st.dataframe(
                    grid.style.map(colour_status, subset=date_cols),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Attendance %": st.column_config.ProgressColumn(
                            "Attendance %", min_value=0, max_value=100, format="%.1f%%"
                        )
                    },
                )
                buf = BytesIO()
                grid.to_excel(buf, index=False)
                st.download_button(
                    f"Download {subj} report",
                    data=buf.getvalue(),
                    file_name=f"{subj.replace(' ', '_')}_datewise.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        # ---- By student: dates down the side, subjects across, with times ----
        with tab_student:
            people = {}
            for fn in excel_files:
                for _, r in read_sheet(fn)[["Roll No", "Student Name"]].iterrows():
                    people[str(r["Roll No"])] = r["Student Name"]
            roll_pick = st.selectbox(
                "Student", list(people), format_func=lambda r: f"{people[r]} ({r})", key="view_student"
            )

            rows = []
            for fn in excel_files:
                d = read_sheet(fn)
                m = d[d["Roll No"].astype(str) == roll_pick]
                if m.empty:
                    continue
                r = m.iloc[0]
                for c in subject_cols(d):
                    rows.append(
                        {"Date": file_date(fn), "Subject": c, "Status": str(r[c]), "Time": time_of(r, c)}
                    )

            if not rows:
                st.info("No subject has been marked for this student yet.")
            else:
                log = pd.DataFrame(rows)
                held = len(log)
                pres = int((log["Status"] == "Present").sum())
                k1, k2, k3 = st.columns(3)
                k1.metric("Classes held", held)
                k2.metric("Present", pres)
                k3.metric("Attendance", f"{round(pres / held * 100, 1)}%")

                log["Shown"] = [show_cell(s_, t_) for s_, t_ in zip(log["Status"], log["Time"])]
                grid = (
                    log.pivot_table(index="Date", columns="Subject", values="Shown", aggfunc="first")
                    .fillna("-")
                    .sort_index(ascending=False)
                    .reset_index()
                )
                grid.columns.name = None
                st.markdown("#### Date-wise attendance")
                st.dataframe(
                    grid.style.map(colour_status, subset=[c for c in grid.columns if c != "Date"]),
                    use_container_width=True,
                    hide_index=True,
                )

                per_sub = (
                    log.assign(P=(log["Status"] == "Present").astype(int))
                    .groupby("Subject")["P"]
                    .agg(Classes="count", Present="sum")
                    .reset_index()
                )
                per_sub["Attendance %"] = (per_sub["Present"] / per_sub["Classes"] * 100).round(1)
                st.markdown("#### Subject-wise")
                st.dataframe(
                    per_sub,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Attendance %": st.column_config.ProgressColumn(
                            "Attendance %", min_value=0, max_value=100, format="%.1f%%"
                        )
                    },
                )

        # ---- Percentages for every student ----
        with tab_pct:
            records = []
            for fn in excel_files:
                d = read_sheet(fn)
                for c in subject_cols(d):
                    part = d[["Roll No", "Student Name"]].copy()
                    part["Subject"] = c
                    part["Present"] = (d[c] == "Present").astype(int)
                    records.append(part)

            if not records:
                st.info("No subject has been marked yet.")
            else:
                long_df = pd.concat(records, ignore_index=True)
                agg = (
                    long_df.groupby(["Roll No", "Student Name", "Subject"])["Present"]
                    .mean().mul(100).round(1).reset_index()
                )
                out = agg.pivot_table(
                    index=["Roll No", "Student Name"], columns="Subject", values="Present"
                ).reset_index()
                out.columns.name = None
                overall = (
                    long_df.groupby(["Roll No", "Student Name"])["Present"].mean().mul(100).round(1).reset_index()
                ).rename(columns={"Present": "Overall %"})
                out = out.merge(overall, on=["Roll No", "Student Name"])
                out["Note"] = out["Overall %"].apply(lambda p: "Below 75%" if p < 75 else "OK")

                st.caption("Each subject column shows the percentage of that subject's classes the student attended.")
                pct_cols = [c for c in out.columns if c not in ("Roll No", "Student Name", "Note")]
                st.dataframe(
                    out,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.1f%%")
                        for c in pct_cols
                    },
                )
