import os
import json
import base64
import pandas as pd
import streamlit as st
from datetime import datetime
from groq import Groq

# ------------------------------------------------------------------
# 1. Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Attendance",
    layout="wide",
    page_icon="🪪",
    initial_sidebar_state="expanded",
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


# ------------------------------------------------------------------
# 4. Styling: light, calm, readable. One strong accent (teal).
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --paper: #F6F7F4;
    --card: #FFFFFF;
    --ink: #172B2A;
    --muted: #5F6F6D;
    --line: #DDE3DF;
    --teal: #0E7C7B;
    --teal-dark: #0A5F5E;
    --teal-soft: #E1F2F0;
    --amber-soft: #FDF0DC;
    --amber: #9A5B00;
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}
.stApp { background: var(--paper); color: var(--ink); }
.block-container { padding-top: 2rem; max-width: 1150px; }
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--ink);
}
section[data-testid="stSidebar"] * { color: #E8EFEC !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 10px 12px; border-radius: 10px; width: 100%;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }
.brand { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.01em; margin-bottom: 0.2rem; }
.brand-sub { font-size: 0.82rem; opacity: 0.7; margin-bottom: 1.4rem; }

/* Page header */
.page-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; color: var(--ink); }
.page-sub { color: var(--muted); font-size: 1rem; margin: 0.25rem 0 1.6rem 0; max-width: 60ch; }

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; }
.card-note { color: var(--muted); font-size: 0.9rem; margin-bottom: 12px; }

/* Result panel */
.result-ok {
    background: var(--teal-soft);
    border: 1px solid var(--teal);
    border-radius: 14px;
    padding: 20px 22px;
}
.result-ok h3 { margin: 0 0 8px 0; color: var(--teal-dark); font-size: 1.25rem; }
.result-ok .row { display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid rgba(14,124,123,0.2); font-size: 0.95rem; }
.result-ok .row:last-child { border-bottom: none; }
.result-ok .row span:first-child { color: var(--muted); }
.result-wait {
    background: var(--card); border: 1px dashed var(--line); border-radius: 14px;
    padding: 28px 22px; color: var(--muted); text-align: center;
}

/* Student chips */
.student-row {
    display: flex; align-items: center; gap: 12px; padding: 10px 0;
    border-bottom: 1px solid var(--line);
}
.student-row:last-child { border-bottom: none; }
.avatar {
    width: 38px; height: 38px; border-radius: 50%; background: var(--teal-soft);
    color: var(--teal-dark); display: flex; align-items: center; justify-content: center;
    font-weight: 700; flex-shrink: 0;
}
.student-name { font-weight: 600; }
.student-meta { color: var(--muted); font-size: 0.85rem; }

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 14px 18px;
}
div[data-testid="stMetricLabel"] p { color: var(--muted); font-weight: 500; }
div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: var(--teal); color: #fff; font-weight: 600; border: none;
    border-radius: 10px; padding: 0.7rem 1.2rem; width: 100%;
    transition: background 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--teal-dark); color: #fff; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 3px solid #9ADAD6; outline-offset: 2px;
}

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; }

@media (max-width: 700px) {
    .page-title { font-size: 1.5rem; }
    .block-container { padding-top: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def page_header(title, subtitle):
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

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.markdown('<div class="brand">🪪 Attendance</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="brand-sub">Powered by Groq · {MODEL_NAME}</div>', unsafe_allow_html=True
)

PAGE_REG = "Register students"
PAGE_LIVE = "Take attendance"
PAGE_VIEW = "View reports"

page_selection = st.sidebar.radio("Menu", [PAGE_REG, PAGE_LIVE, PAGE_VIEW], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.metric("Registered students", len(students_db))

# ==================================================================
# PAGE 1: REGISTER
# ==================================================================
if page_selection == PAGE_REG:
    page_header("Register a student", "Add the student's details and a clear, front-facing photo.")

    col_form, col_cam = st.columns([1.1, 1], gap="large")

    with col_form:
        card_start("Student details")
        roll_no = st.text_input("Roll number", placeholder="e.g. 21CSE045")
        student_name = st.text_input("Full name", placeholder="e.g. Ishu Mishra")
        department = st.text_input("Department", "CSE (AI & ML)")

        preset_subs = ["Data Structures", "Operating Systems", "Machine Learning", "Python"]
        selected_subjects = st.multiselect("Subjects", preset_subs, default=["Data Structures"])
        custom_sub_text = st.text_input(
            "Other subjects (optional)", placeholder="Separate with commas: Cloud Computing, Cyber Security"
        )

        final_registered_subs = list(selected_subjects)
        for item in [s.strip() for s in custom_sub_text.split(",") if s.strip()]:
            if item not in final_registered_subs:
                final_registered_subs.append(item)
        card_end()

    with col_cam:
        card_start("Photo", "Face the camera in good light.")
        camera_image = st.camera_input("Take registration photo", label_visibility="collapsed")
        card_end()

    if st.button("Save student"):
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
                        "subjects": final_registered_subs,
                        "image_path": img_path,
                    }
                    save_students(students_db)
                    st.success(f"{student_name} ({roll_no}) is registered for {', '.join(final_registered_subs)}.")
                except Exception as e:
                    st.error(f"Could not save the student: {e}")

    # Registered list
    if students_db:
        st.markdown("### Registered students")
        card_start()
        rows = ""
        for s in students_db.values():
            initials = "".join(w[0] for w in s["name"].split()[:2]).upper()
            rows += (
                f'<div class="student-row"><div class="avatar">{initials}</div>'
                f'<div><div class="student-name">{s["name"]}</div>'
                f'<div class="student-meta">{s["roll_no"]} · {s["department"]} · '
                f'{", ".join(s["subjects"])}</div></div></div>'
            )
        st.markdown(rows, unsafe_allow_html=True)
        card_end()

# ==================================================================
# PAGE 2: TAKE ATTENDANCE
# ==================================================================
elif page_selection == PAGE_LIVE:
    page_header("Take attendance", "Choose the class, then capture the student's face.")

    CUSTOM = "Other (type below)"
    preset_subjects = ["Data Structures", "Operating Systems", "Machine Learning", "Python", CUSTOM]

    card_start("Class")
    c1, c2 = st.columns(2)
    with c1:
        subject_choice = st.selectbox("Subject", preset_subjects)
    with c2:
        if subject_choice == CUSTOM:
            current_subject = st.text_input("Subject name", placeholder="e.g. Cloud Computing")
        else:
            current_subject = subject_choice
            st.text_input("Date", datetime.now().strftime("%d %b %Y"), disabled=True)
    card_end()

    col_cam, col_info = st.columns([1.2, 1], gap="large")

    with col_cam:
        card_start("Camera")
        live_photo = st.camera_input("Scan face", label_visibility="collapsed")
        card_end()

    with col_info:
        st.markdown("#### Result")
        status_card = st.empty()
        status_card.markdown(
            '<div class="result-wait">Take a photo and the result will show up here.</div>',
            unsafe_allow_html=True,
        )

    if live_photo is not None:
        if not students_db:
            st.warning("No students yet. Register a student first.")
        elif not current_subject.strip():
            st.error("Enter a subject name.")
        else:
            with st.spinner("Marking attendance..."):
                try:
                    latest_roll = list(students_db.keys())[-1]
                    st_info = students_db[latest_roll]

                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    clean_sub_name = "".join(
                        c for c in current_subject if c.isalnum() or c in (" ", "_", "-")
                    ).strip().replace(" ", "_")
                    excel_file = os.path.join(RECORDS_DIR, f"Attendance_{clean_sub_name}_{date_str}.xlsx")

                    all_rows = []
                    for r_no, data in students_db.items():
                        is_present = r_no == st_info["roll_no"]
                        all_rows.append(
                            {
                                "Roll No": data["roll_no"],
                                "Student Name": data["name"],
                                "Department": data["department"],
                                "Subject": current_subject,
                                "Date": date_str,
                                "Time": time_str if is_present else "-",
                                "Status": "Present" if is_present else "Absent",
                            }
                        )
                    df_full = pd.DataFrame(all_rows)

                    if os.path.exists(excel_file):
                        df_existing = pd.read_excel(excel_file)
                        mask = df_existing["Roll No"].astype(str) == str(st_info["roll_no"])
                        df_existing.loc[mask, "Status"] = "Present"
                        df_existing.loc[mask, "Time"] = time_str
                        df_existing.to_excel(excel_file, index=False)
                        df_now = df_existing
                    else:
                        df_full.to_excel(excel_file, index=False)
                        df_now = df_full

                    total_students = len(df_now)
                    present_count = int((df_now["Status"] == "Present").sum())
                    attendance_pct = round(present_count / total_students * 100, 1) if total_students else 0

                    status_card.markdown(
                        f"""
<div class="result-ok">
  <h3>✓ Marked present</h3>
  <div class="row"><span>Name</span><b>{st_info['name']}</b></div>
  <div class="row"><span>Roll number</span><b>{st_info['roll_no']}</b></div>
  <div class="row"><span>Subject</span><b>{current_subject}</b></div>
  <div class="row"><span>Time</span><b>{time_str} · {date_str}</b></div>
  <div class="row"><span>Class so far</span><b>{present_count}/{total_students} present ({attendance_pct}%)</b></div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Could not mark attendance: {e}")

# ==================================================================
# PAGE 3: REPORTS
# ==================================================================
elif page_selection == PAGE_VIEW:
    page_header("Attendance reports", "Check a day's sheet or see each student's overall percentage.")

    excel_files = sorted([f for f in os.listdir(RECORDS_DIR) if f.endswith(".xlsx")], reverse=True)

    if not excel_files:
        st.info("No attendance sheets yet. Take attendance first and the sheets will appear here.")
    else:
        tab_day, tab_student = st.tabs(["Daily sheet", "Student-wise percentage"])

        with tab_day:
            selected_file = st.selectbox("Sheet", excel_files)
            file_path = os.path.join(RECORDS_DIR, selected_file)
            df = pd.read_excel(file_path)

            total_st = len(df)
            present_st = int((df["Status"] == "Present").sum())
            absent_st = int((df["Status"] == "Absent").sum())
            pct = round(present_st / total_st * 100, 1) if total_st else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Students", total_st)
            m2.metric("Present", present_st)
            m3.metric("Absent", absent_st)
            m4.metric("Attendance", f"{pct}%")

            st.progress(min(pct / 100, 1.0))

            def colour_status(val):
                if val == "Present":
                    return "background-color: #E1F2F0; color: #0A5F5E; font-weight: 600"
                if val == "Absent":
                    return "background-color: #FDF0DC; color: #9A5B00; font-weight: 600"
                return ""

            st.dataframe(
                df.style.map(colour_status, subset=["Status"]),
                use_container_width=True,
                hide_index=True,
            )

            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download Excel",
                    data=f.read(),
                    file_name=selected_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        with tab_student:
            frames = []
            for fn in excel_files:
                try:
                    frames.append(pd.read_excel(os.path.join(RECORDS_DIR, fn)))
                except Exception:
                    pass
            all_df = pd.concat(frames, ignore_index=True)

            subjects = sorted(all_df["Subject"].dropna().unique().tolist())
            subj = st.selectbox("Subject", ["All subjects"] + subjects)
            view = all_df if subj == "All subjects" else all_df[all_df["Subject"] == subj]

            summary = (
                view.groupby(["Roll No", "Student Name"])
                .agg(
                    Classes=("Status", "count"),
                    Present=("Status", lambda s: int((s == "Present").sum())),
                )
                .reset_index()
            )
            summary["Attendance %"] = (summary["Present"] / summary["Classes"] * 100).round(1)
            summary["Note"] = summary["Attendance %"].apply(lambda p: "Below 75%" if p < 75 else "OK")

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Attendance %": st.column_config.ProgressColumn(
                        "Attendance %", min_value=0, max_value=100, format="%.1f%%"
                    )
                },
            )
