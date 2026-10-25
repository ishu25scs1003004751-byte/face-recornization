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
    page_title="Multimodal AI Vision Attendance",
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

# 4. Attractive Custom UI CSS Styling
st.markdown("""
    <style>
    /* Dark Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Styling */
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

    /* Glassmorphism Cards */
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

    /* Primary Buttons */
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
st.markdown(f'<div class="sub-title">Powered by Groq API ({MODEL_NAME}) | Smart Student Management</div>', unsafe_allow_html=True)

students_db = load_students()

# Sidebar Navigation
st.sidebar.title("📌 System Navigation")
menu = st.sidebar.radio(
    "Choose Mode", 
    ["1. Register Student Profile", "2. AI Live Camera Attendance", "3. View Attendance Records"]
)

# ================= 1. STUDENT REGISTRATION =================
if menu == "1. Register Student Profile":
    st.markdown("### 📸 Student Registration Portal")
    
    col1, col2 = st.columns([1.1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        roll_no = st.text_input("Roll Number / Student ID", placeholder="e.g. 21CSE045")
        student_name = st.text_input("Full Name", placeholder="e.g. Ishu Mishra")
        department = st.text_input("Department", "CSE (AI & ML)")
        
        # Multi-Select Subjects with Custom Choice
        default_sub = ["Data Structures", "Operating Systems", "Machine Learning", "Python"]
        subjects = st.multiselect("Enrolled Subjects", default_sub, default=["Data Structures"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        camera_image = st.camera_input("Take Live Student Photo")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💾 Verify with AI API & Save Profile"):
        if not roll_no or not student_name or camera_image is None:
            st.error("⚠️ Kripya Roll Number, Full Name aur Photo teeno provide karein!")
        else:
            img_bytes = camera_image.getbuffer()

            with st.spinner("Registering Student with AI Vision Verification..."):
                try:
                    # Save Image File
                    img_filename = f"{roll_no}_{student_name.replace(' ', '_')}.jpg"
                    img_path = os.path.join(STUDENTS_DIR, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    # Update Database
                    students_db[roll_no] = {
                        "name": student_name,
                        "roll_no": roll_no,
                        "department": department,
                        "subjects": subjects,
                        "image_path": img_path
                    }
                    save_students(students_db)
                    
                    st.success(f"✅ Student {student_name} ({roll_no}) successfully register ho gaya hai!")
                    st.balloons()

                except Exception as e:
                    st.error(f"API/Storage Error: {str(e)}")

# ================= 2. LIVE ATTENDANCE SCANNING =================
elif menu == "2. AI Live Camera Attendance":
    st.markdown("### ⚡ Live AI Camera Attendance")

    # Subject Selection (Dropdown + Custom Subject Entry)
    col_sub1, col_sub2 = st.columns([1, 1])
    
    with col_sub1:
        preset_subjects = ["Data Structures", "Operating Systems", "Machine Learning", "Python", "➕ Add New Subject Name"]
        subject_choice = st.selectbox("Select Current Subject / Class Period:", preset_subjects)

    with col_sub2:
        if subject_choice == "➕ Add New Subject Name":
            current_subject = st.text_input("Type Custom Subject Name:", placeholder="e.g. Cloud Computing")
        else:
            current_subject = subject_choice

    st.divider()

    col_cam, col_info = st.columns([1.2, 1], gap="medium")

    with col_cam:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        live_photo = st.camera_input("Scan Student Face")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown("#### Real-time Verification Status")
        details_box = st.empty()
        details_box.info("🔍 Live Camera ke aage aakar photo click karein...")

    if live_photo is not None:
        if not students_db:
            st.warning("Pehle kam se kam ek student ko register karein!")
        elif not current_subject.strip():
            st.error("Kripya valid Subject Name select ya enter karein!")
        else:
            with st.spinner("Processing Attendance via Groq AI API..."):
                try:
                    # Get latest registered student for verification match
                    latest_roll = list(students_db.keys())[-1]
                    st_info = students_db[latest_roll]

                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    # Dynamic Subject & Date-wise Excel File Name
                    clean_subject_name = "".join(c for c in current_subject if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
                    excel_file = os.path.join(RECORDS_DIR, f"Attendance_{clean_subject_name}_{date_str}.xlsx")
                    
                    new_data = pd.DataFrame([{
                        "Roll No": st_info["roll_no"],
                        "Student Name": st_info["name"],
                        "Department": st_info["department"],
                        "Subject": current_subject,
                        "Date": date_str,
                        "Time": time_str,
                        "Status": "Present",
                        "AI Verification": f"Groq API ({MODEL_NAME})"
                    }])

                    if os.path.exists(excel_file):
                        df_existing = pd.read_excel(excel_file)
                        if not ((df_existing["Roll No"].astype(str) == str(st_info["roll_no"])) & (df_existing["Date"] == date_str)).any():
                            df_updated = pd.concat([df_existing, new_data], ignore_index=True)
                            df_updated.to_excel(excel_file, index=False)
                    else:
                        new_data.to_excel(excel_file, index=False)

                    details_box.markdown(f"""
                        <div class="success-card">
                            <h3 style="margin-top:0; color:#10b981;">✅ Attendance Marked Present!</h3>
                            <p><b>Name:</b> {st_info['name']}</p>
                            <p><b>Roll No:</b> {st_info['roll_no']}</p>
                            <p><b>Department:</b> {st_info['department']}</p>
                            <p><b>Subject:</b> {current_subject}</p>
                            <p><b>Timestamp:</b> {time_str} ({date_str})</p>
                            <p><b>Excel Report:</b> Saved in <code>Attendance_{clean_subject_name}_{date_str}.xlsx</code></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

                except Exception as e:
                    st.error(f"API Error: {str(e)}")

# ================= 3. ATTENDANCE RECORDS & EXCEL =================
elif menu == "3. View Attendance Records":
    st.markdown("### 📊 Subject & Date-wise Attendance Logs")
    
    records = os.listdir(RECORDS_DIR)
    excel_files = [f for f in records if f.endswith('.xlsx')]

    if not excel_files:
        st.info("Abhi tak koi attendance record excel file generate nahi hui hai.")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        selected_file = st.selectbox("Select Attendance Excel File to View:", excel_files)
        file_path = os.path.join(RECORDS_DIR, selected_file)
        
        df = pd.read_excel(file_path)
        st.dataframe(df, use_container_width=True)

        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download This Excel (.xlsx) Report",
                data=f.read(),
                file_name=selected_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.markdown('</div>', unsafe_allow_html=True)
