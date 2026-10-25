import os
import json
import base64
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
    sheets_today = len([f for f in os.listdir(RECORDS_DIR) if f.endswith(f"{today}.xlsx")])
    total_sheets = len([f for f in os.listdir(RECORDS_DIR) if f.endswith(".xlsx")])

    st.markdown(
        f"""
<div class="hero">
  <h1>Attendance, made simple</h1>
  <p>Register students, mark attendance with the camera, and check reports. All in one place.</p>
  <div class="hero-stats">
    <div class="hero-stat"><b>{len(students_db)}</b><span>Registered students</span></div>
    <div class="hero-stat"><b>{sheets_today}</b><span>Classes marked today</span></div>
    <div class="hero-stat"><b>{total_sheets}</b><span>Total sheets</span></div>
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
            "<p>Pick the class, scan a face with the camera, and mark the student present.</p></div>",
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
                            "subjects": final_registered_subs,
                            "image_path": img_path,
                        }
                        save_students(students_db)
                        st.success(f"{student_name} ({roll_no}) is registered for {', '.join(final_registered_subs)}.")
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
                        f'<div class="student-meta">{s["roll_no"]} · {s["department"]} · '
                        f'{", ".join(s["subjects"])}</div>',
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
    page_header("Live attendance", "Choose the class, then capture the student's face.")

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
# VIEW ATTENDANCE
# ==================================================================
elif page == "view":
    page_header("View attendance", "Check a day's sheet or see each student's overall percentage.")

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
                    return "background-color: #DFF1EF; color: #0A5F5E; font-weight: 600"
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
