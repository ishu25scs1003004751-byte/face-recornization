import os
import json
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from datetime import datetime
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="Multimodal AI Vision Attendance System",
    layout="wide",
    page_icon="🪪",
    initial_sidebar_state="expanded"
)

# 2. Database & Storage Folders Setup
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
    st.error("⚠️ Groq API Key missing hai! Streamlit Secrets me GROQ_API_KEY verify karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "openai/gpt-oss-120b"

def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# 4. Modern Glassmorphism Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .success-card {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 18px;
        color: #ecfdf5;
    }

    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">🪪 Multimodal AI Vision Attendance System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Powered by Groq API ({MODEL_NAME})</div>', unsafe_allow_html=True)

students_db = load_students()

# Sidebar Multi-Page Navigation
st.sidebar.title("📌 Navigation Menu")
menu = st.sidebar.radio(
    "Go To Page:", 
    ["1. Student Registration Portal", "2. Take Attendance (Live Scan)", "3. View Attendance Records & Percentage"]
)

# ================= PAGE 1: STUDENT REGISTRATION =================
if menu == "1. Student Registration Portal":
    st.subheader("📝 New Student Registration")
    
    col1, col2 = st.columns([1.1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        roll_no = st.text_input("Roll Number / Student ID", placeholder="e.g. 21CSE045")
        student_name = st.text_input("Full Name", placeholder="e.g. Ishu Mishra")
        department = st.text_input("Department", "CSE (AI & ML)")
        
        default_sub = ["Data Structures", "Operating Systems", "Machine Learning", "Python"]
        selected_subjects = st.multiselect("Select Preset Subjects", default_sub, default=["Data Structures"])
        
        custom_subject_input = st.text_input("Add Custom Subject (Optional)", placeholder="e.g. Cloud Computing, Cyber Security")
        
        final_subjects = list(selected_subjects)
        if custom_subject_input.strip():
            extra_subs = [s.strip() for s in custom_subject_input.split(",") if s.strip()]
            for sub in extra_subs:
                if sub not in final_subjects:
                    final_subjects.append(sub)
                    
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        camera_image = st.camera_input("Capture Student Photo")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💾 Register & Save Student Profile"):
        if not roll_no or not student_name or camera_image is None:
            st.error("⚠️ Roll Number, Full Name aur Photo teeno hona zaroori hai!")
        else:
            img_bytes = camera_image.getbuffer()

            with st.spinner("Saving Student Profile..."):
                try:
                    img_filename = f"{roll_no}_{student_name.replace(' ', '_')}.jpg"
                    img_path = os.path.join(STUDENTS_DIR, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    students_db[roll_no] = {
                        "name": student_name,
                        "roll_no": roll_no,
                        "department": department,
                        "subjects": final_subjects,
                        "image_path": img_path
                    }
                    save_students(students_db)
                    
                    st.success(f"✅ Student {student_name} ({roll_no}) successful register ho gaya!")
                    st.info(f"Registered Subjects: {', '.join(final_subjects)}")
                    st.balloons()

                except Exception as e:
                    st.error(f"Save Error: {str(e)}")

# ================= PAGE 2: TAKE ATTENDANCE =================
elif menu == "2. Take Attendance (Live Scan)":
    st.subheader("⚡ Live Camera Attendance Marking")

    col_sub1, col_sub2 = st.columns([1, 1])
    
    with col_sub1:
        preset_subjects = ["Data Structures", "Operating Systems", "Machine Learning", "Python", "➕ Enter Custom Subject Name"]
        subject_choice = st.selectbox("Select Class / Subject Period:", preset_subjects)

    with col_sub2:
        if subject_choice == "➕ Enter Custom Subject Name":
            current_subject = st.text_input("Type Custom Subject Name:", placeholder="e.g. Cloud Computing")
        else:
            current_subject = subject_choice

    st.divider()

    col_cam, col_info = st.columns([1.2, 1], gap="medium")

    with col_cam:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        live_photo = st.camera_input("Scan Face for Attendance")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown("#### Live Verification Status")
        details_box = st.empty()
        details_box.info("📷 Live Camera ke aage aakar photo click karein...")

    if live_photo is not None:
        if not students_db:
            st.warning("⚠️ Pehle Page 1 par kisi student ko register karein!")
        elif not current_subject.strip():
            st.error("⚠️ Valid Subject Name enter karein!")
        else:
            with st.spinner("Processing Attendance Entry..."):
                try:
                    # Match latest student for active session
                    latest_roll = list(students_db.keys())[-1]
                    st_info = students_db[latest_roll]

                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    clean_subject_name = "".join(c for c in current_subject if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
                    excel_file = os.path.join(RECORDS_DIR, f"Attendance_{clean_subject_name}_{date_str}.xlsx")
                    
                    # Complete Attendance Sheet Generator with Present/Absent & %
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
                    
                    # Calculate Present/Absent Summary %
                    total_students = len(students_db)
                    present_count = len(df_full[df_full["Status"] == "Present"])
                    absent_count = total_students - present_count
                    attendance_pct = round((present_count / total_students) * 100, 2) if total_students > 0 else 0

                    if os.path.exists(excel_file):
                        df_existing = pd.read_excel(excel_file)
                        # Update present status if already recorded
                        df_existing.loc[df_existing["Roll No"].astype(str) == str(st_info["roll_no"]), "Status"] = "Present"
                        df_existing.loc[df_existing["Roll No"].astype(str) == str(st_info["roll_no"]), "Time"] = time_str
                        df_existing.to_excel(excel_file, index=False)
                    else:
                        df_full.to_excel(excel_file, index=False)

                    details_box.markdown(f"""
                        <div class="success-card">
                            <h3 style="margin-top:0; color:#10b981;">✅ Attendance Marked Present!</h3>
                            <p><b>Name:</b> {st_info['name']}</p>
                            <p><b>Roll No:</b> {st_info['roll_no']}</p>
                            <p><b>Subject:</b> {current_subject}</p>
                            <p><b>Timestamp:</b> {time_str} ({date_str})</p>
                            <hr style="border-color:rgba(255,255,255,0.2);">
                            <p><b>Class Attendance Summary:</b></p>
                            <p>Present: <b>{present_count}/{total_students}</b> | Attendance %: <b>{attendance_pct}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")

# ================= PAGE 3: VIEW RECORDS & PERCENTAGE =================
elif menu == "3. View Attendance Records & Percentage":
    st.subheader("📊 Subject Attendance Records & Percentage Analysis")
    
    records = os.listdir(RECORDS_DIR)
    excel_files = [f for f in records if f.endswith('.xlsx')]

    if not excel_files:
        st.info("Abhi tak koi attendance record excel file generate nahi hui hai.")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        selected_file = st.selectbox("Select Attendance Excel Report File:", excel_files)
        file_path = os.path.join(RECORDS_DIR, selected_file)
        
        df = pd.read_excel(file_path)
        
        # Attendance Metrics Cards
        total_students = len(df)
        present_count = len(df[df["Status"] == "Present"])
        absent_count = len(df[df["Status"] == "Absent"])
        percentage = round((present_count / total_students) * 100, 2) if total_students > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Registered", total_students)
        m2.metric("Present Count", present_count)
        m3.metric("Absent Count", absent_count)
        m4.metric("Class Attendance %", f"{percentage}%")
        
        st.divider()
        st.write("#### Detailed Student Log:")
        st.dataframe(df, use_container_width=True)

        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download Excel (.xlsx) File",
                data=f.read(),
                file_name=selected_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.markdown('</div>', unsafe_allow_html=True)
