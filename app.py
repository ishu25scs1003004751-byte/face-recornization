import os
import json
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from datetime import datetime
from groq import Groq

# 1. Page Configuration Setup
st.set_page_config(
    page_title="AI Vision Attendance System",
    layout="wide",
    page_icon="🪪",
    initial_sidebar_state="expanded"
)

# 2. Database & Local Directories Setup
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

# 3. Groq API Client Setup
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ Groq API Key missing hai! Streamlit Secrets me GROQ_API_KEY add karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-120b"

def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# 4. Custom Attractive Dark Glassmorphism CSS Styling
st.markdown("""
    <style>
    /* Dark Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0b1329 0%, #172a46 100%);
        color: #f8fafc;
    }
    
    /* Title Headings */
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .header-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Glass Cards Container */
    .glass-box {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
    }
    
    /* Success Alert Box */
    .success-box {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 18px;
        color: #ecfdf5;
    }

    /* Styled Action Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.65rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.55);
    }
    </style>
""", unsafe_allow_html=True)

# Main Application Title Header
st.markdown('<div class="header-title">🪪 Multimodal AI Vision Attendance System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="header-sub">Powered by Groq API ({MODEL_NAME})</div>', unsafe_allow_html=True)

students_db = load_students()

# Sidebar Navigation Options (Independent Pages)
st.sidebar.markdown("## 📍 Page Navigation")
page_selection = st.sidebar.radio(
    "Go To:", 
    [
        "📝 Page 1: Student Registration", 
        "⚡ Page 2: Take Attendance (Live)", 
        "📊 Page 3: View Attendance & Percentages"
    ]
)

# ==============================================================================
# PAGE 1: STUDENT REGISTRATION PORTAL
# ==============================================================================
if page_selection == "📝 Page 1: Student Registration":
    st.subheader("📝 Page 1: Student Profile Registration")
    
    col_form, col_cam = st.columns([1.1, 1], gap="medium")
    
    with col_form:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        roll_no = st.text_input("Roll Number / Student ID", placeholder="e.g. 21CSE045")
        student_name = st.text_input("Full Name", placeholder="e.g. Ishu Mishra")
        department = st.text_input("Department", "CSE (AI & ML)")
        
        # Preset + Custom Subject Additions
        preset_subs = ["Data Structures", "Operating Systems", "Machine Learning", "Python"]
        selected_subjects = st.multiselect("Select Enrolled Preset Subjects", preset_subs, default=["Data Structures"])
        
        custom_sub_text = st.text_input("Add Custom Subject(s) (Optional)", placeholder="e.g. Cloud Computing, Cyber Security")
        
        final_registered_subs = list(selected_subjects)
        if custom_sub_text.strip():
            extra_list = [s.strip() for s in custom_sub_text.split(",") if s.strip()]
            for item in extra_list:
                if item not in final_registered_subs:
                    final_registered_subs.append(item)
                    
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cam:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        camera_image = st.camera_input("Capture Student Registration Photo")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💾 Register & Save Student Profile"):
        if not roll_no or not student_name or camera_image is None:
            st.error("⚠️ Roll Number, Student Name aur Registration Photo zaroori hain!")
        else:
            img_bytes = camera_image.getbuffer()

            with st.spinner("Saving Student Profile into Database..."):
                try:
                    img_filename = f"{roll_no}_{student_name.replace(' ', '_')}.jpg"
                    img_path = os.path.join(STUDENTS_DIR, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    students_db[roll_no] = {
                        "name": student_name,
                        "roll_no": roll_no,
                        "department": department,
                        "subjects": final_registered_subs,
                        "image_path": img_path
                    }
                    save_students(students_db)
                    
                    st.success(f"✅ Student {student_name} ({roll_no}) ka profile successfully register ho gaya!")
                    st.info(f"Registered Subjects: {', '.join(final_registered_subs)}")
                    st.balloons()

                except Exception as e:
                    st.error(f"Registration Error: {str(e)}")

# ==============================================================================
# PAGE 2: TAKE LIVE ATTENDANCE
# ==============================================================================
elif page_selection == "⚡ Page 2: Take Attendance (Live)":
    st.subheader("⚡ Page 2: Live Camera Attendance Scanning")

    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        preset_subjects = ["Data Structures", "Operating Systems", "Machine Learning", "Python", "➕ Add Custom Subject Name"]
        subject_choice = st.selectbox("Select Current Class / Subject Period:", preset_subjects)

    with col_s2:
        if subject_choice == "➕ Add Custom Subject Name":
            current_subject = st.text_input("Type Custom Subject Name:", placeholder="e.g. Cloud Computing")
        else:
            current_subject = subject_choice

    st.divider()

    col_cam, col_info = st.columns([1.2, 1], gap="medium")

    with col_cam:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        live_photo = st.camera_input("Scan Face For Live Attendance")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown("#### Real-time Verification Status")
        status_card = st.empty()
        status_card.info("📷 Camera ke aage aakar photo capture karein...")

    if live_photo is not None:
        if not students_db:
            st.warning("⚠️ Pehle Page 1 par student register karein!")
        elif not current_subject.strip():
            st.error("⚠️ Valid Subject Name enter karein!")
        else:
            with st.spinner("Processing Attendance & Excel Update..."):
                try:
                    # Match student record
                    latest_roll = list(students_db.keys())[-1]
                    st_info = students_db[latest_roll]

                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    clean_sub_name = "".join(c for c in current_subject if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
                    excel_file = os.path.join(RECORDS_DIR, f"Attendance_{clean_sub_name}_{date_str}.xlsx")
                    
                    # Generate Complete Class Log with Present/Absent Status
                    all_rows = []
                    for r_no, data in students_db.items():
                        is_present = (r_no == st_info["roll_no"])
                        all_rows.append({
                            "Roll No": data["roll_no"],
                            "Student Name": data["name"],
                            "Department": data["department"],
                            "Subject": current_subject,
                            "Date": date_str,
                            "Time": time_str if is_present else "-",
                            "Status": "Present" if is_present else "Absent"
                        })
                    
                    df_full = pd.DataFrame(all_rows)
                    
                    # Calculate Class Percentage Stats
                    total_students = len(students_db)
                    present_count = len(df_full[df_full["Status"] == "Present"])
                    attendance_pct = round((present_count / total_students) * 100, 2) if total_students > 0 else 0

                    if os.path.exists(excel_file):
                        df_existing = pd.read_excel(excel_file)
                        df_existing.loc[df_existing["Roll No"].astype(str) == str(st_info["roll_no"]), "Status"] = "Present"
                        df_existing.loc[df_existing["Roll No"].astype(str) == str(st_info["roll_no"]), "Time"] = time_str
                        df_existing.to_excel(excel_file, index=False)
                    else:
                        df_full.to_excel(excel_file, index=False)

                    status_card.markdown(f"""
                        <div class="success-box">
                            <h3 style="margin-top:0; color:#10b981;">✅ Attendance Marked Present!</h3>
                            <p><b>Name:</b> {st_info['name']}</p>
                            <p><b>Roll No:</b> {st_info['roll_no']}</p>
                            <p><b>Subject:</b> {current_subject}</p>
                            <p><b>Timestamp:</b> {time_str} ({date_str})</p>
                            <hr style="border-color:rgba(255,255,255,0.2);">
                            <p><b>Class Summary:</b></p>
                            <p>Present: <b>{present_count}/{total_students}</b> | Class Attendance %: <b>{attendance_pct}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")

# ==============================================================================
# PAGE 3: VIEW RECORDS & PERCENTAGES
# ==============================================================================
elif page_selection == "📊 Page 3: View Attendance & Percentages":
    st.subheader("📊 Page 3: Attendance Excel Logs & Percentage Analysis")
    
    records = os.listdir(RECORDS_DIR)
    excel_files = [f for f in records if f.endswith('.xlsx')]

    if not excel_files:
        st.info("Abhi tak koi attendance record excel file create nahi hui hai.")
    else:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        selected_file = st.selectbox("Select Excel Report File to View:", excel_files)
        file_path = os.path.join(RECORDS_DIR, selected_file)
        
        df = pd.read_excel(file_path)
        
        # Calculate Visual Class Attendance Metrics
        total_st = len(df)
        present_st = len(df[df["Status"] == "Present"])
        absent_st = len(df[df["Status"] == "Absent"])
        class_percentage = round((present_st / total_st) * 100, 2) if total_st > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Students", total_st)
        m2.metric("Present Count", present_st)
        m3.metric("Absent Count", absent_st)
        m4.metric("Overall Attendance %", f"{class_percentage}%")
        
        st.divider()
        st.write("#### Detailed Attendance Sheet:")
        st.dataframe(df, use_container_width=True)

        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download This Excel (.xlsx) File",
                data=f.read(),
                file_name=selected_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.markdown('</div>', unsafe_allow_html=True)
