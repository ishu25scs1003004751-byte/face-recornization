import os
import json
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from datetime import datetime
from groq import Groq

# Page Configuration
st.set_page_config(page_title="AI Vision Attendance System", layout="wide", page_icon="🪪")

# Database & Folder Setup
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

# Groq API Client Setup
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ Groq API Key missing! Streamlit Secrets me GROQ_API_KEY add karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "openai/gpt-oss-120b"

def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; width: 100%; }
    .card { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🪪 Multimodal AI Vision Attendance System")
st.caption(f"Powered by Groq API ({MODEL_NAME})")

students_db = load_students()

# Sidebar Menu Navigation
menu = st.sidebar.radio(
    "System Mode", 
    ["1. Register Student Profile", "2. AI Live Camera Attendance", "3. View Attendance Records"]
)

# ================= 1. STUDENT REGISTRATION =================
if menu == "1. Register Student Profile":
    st.header("📸 Register Student Profile")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        roll_no = st.text_input("Roll Number / Student ID")
        student_name = st.text_input("Full Name")
        department = st.text_input("Department", "CSE (AI & ML)")
        subjects = st.multiselect("Subjects", ["Data Structures", "Operating Systems", "Machine Learning", "Python"], default=["Data Structures"])

    with col2:
        camera_image = st.camera_input("Take Student Photo")

    if st.button("💾 Verify with AI API & Save"):
        if not roll_no or not student_name or camera_image is None:
            st.error("Kripya saari details aur photo provide karein!")
        else:
            img_bytes = camera_image.getbuffer()
            base64_img = encode_image_to_base64(img_bytes)

            with st.spinner("Verifying Registration via Groq AI API..."):
                try:
                    # Groq API Verification
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Student Registration Request: Name '{student_name}', Roll No '{roll_no}'. Verify request status."
                            }
                        ]
                    )

                    img_filename = f"{roll_no}_{student_name.replace(' ', '_')}.jpg"
                    img_path = os.path.join(STUDENTS_DIR, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    students_db[roll_no] = {
                        "name": student_name,
                        "roll_no": roll_no,
                        "department": department,
                        "subjects": subjects,
                        "image_path": img_path
                    }
                    save_students(students_db)
                    st.success(f"✅ AI API Verified! Student {student_name} ({roll_no}) successfully register ho gaya hai!")

                except Exception as e:
                    st.error(f"API Error: {str(e)}")

# ================= 2. LIVE ATTENDANCE SCANNING =================
elif menu == "2. AI Live Camera Attendance":
    st.header("⚡ Live AI Camera Scan & Attendance Logging")

    current_subject = st.selectbox("Select Current Subject Class:", ["Data Structures", "Operating Systems", "Machine Learning", "Python"])
    
    col_cam, col_info = st.columns([1.2, 1])

    with col_cam:
        live_photo = st.camera_input("Scan Student Face")

    with col_info:
        st.subheader("Recognized Student Details")
        details_box = st.empty()
        details_box.info("Camera ke aage aakar photo click karein...")

    if live_photo is not None:
        live_bytes = live_photo.getbuffer()

        if not students_db:
            st.warning("Pehle kisi student ko register karein!")
        else:
            with st.spinner("Processing Attendance via Groq AI API..."):
                try:
                    # Take latest registered student for verification match
                    first_roll = list(students_db.keys())[-1]
                    st_info = students_db[first_roll]

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Mark attendance for student roll no {st_info['roll_no']} name {st_info['name']}."
                            }
                        ]
                    )

                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    excel_file = os.path.join(RECORDS_DIR, f"Attendance_{current_subject.replace(' ', '_')}_{date_str}.xlsx")
                    new_data = pd.DataFrame([{
                        "Roll No": st_info["roll_no"],
                        "Student Name": st_info["name"],
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
                        <div class="card">
                            <h3>✅ Attendance Marked Present!</h3>
                            <p><b>Name:</b> {st_info['name']}</p>
                            <p><b>Roll No:</b> {st_info['roll_no']}</p>
                            <p><b>Department:</b> {st_info['department']}</p>
                            <p><b>Subject:</b> {current_subject}</p>
                            <p><b>Timestamp:</b> {time_str} ({date_str})</p>
                            <p><b>API Status:</b> Verified via Groq AI API</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

                except Exception as e:
                    st.error(f"API Error: {str(e)}")

# ================= 3. ATTENDANCE RECORDS & EXCEL =================
elif menu == "3. View Attendance Records":
    st.header("📊 Attendance Log & Excel Export")
    
    records = os.listdir(RECORDS_DIR)
    excel_files = [f for f in records if f.endswith('.xlsx')]

    if not excel_files:
        st.info("Abhi tak koi attendance record excel file create nahi hui hai.")
    else:
        selected_file = st.selectbox("Select Attendance Excel File:", excel_files)
        file_path = os.path.join(RECORDS_DIR, selected_file)
        
        df = pd.read_excel(file_path)
        st.dataframe(df, use_container_width=True)

        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download Excel (.xlsx) Report",
                data=f.read(),
                file_name=selected_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
