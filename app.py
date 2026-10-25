import os
import json
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from datetime import datetime
from groq import Groq

# Page Config
st.set_page_config(page_title="AI Vision Attendance System", layout="wide", page_icon="🪪")

# Database & Records Setup
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

# Groq Vision API Client
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ Groq API Key missing hai! Streamlit Secrets me GROQ_API_KEY add karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Helper function to encode image to Base64
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
st.caption("Powered by Groq Llama-3.2 Vision API")

students_db = load_students()

# Sidebar Navigation
menu = st.sidebar.radio(
    "System Mode", 
    ["1. Register Student Profile", "2. AI Live Camera Attendance", "3. View Attendance Records"]
)

# ================= 1. STUDENT REGISTRATION =================
if menu == "1. Register Student Profile":
    st.header("📸 Register Student (Groq AI Vision)")
    
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

            with st.spinner("Analyzing Face Features using Groq Vision API..."):
                try:
                    # Vision API Call to verify face presence
                    response = client.chat.completions.create(
                       model="llama-3.2-11b-vision-instruct"
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Is there a clear human face visible in this image? Answer ONLY 'YES' or 'NO'."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                            ]
                        }]
                    )
                    api_result = response.choices[0].message.content.strip()

                    if "YES" in api_result.upper():
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
                    else:
                        st.error("❌ AI API Ko photo me clear face nahi mila. Dobara photo lein.")
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
        live_b64 = encode_image_to_base64(live_bytes)

        if not students_db:
            st.warning("Pehle kisi student ko register karein!")
        else:
            with st.spinner("Sending Frame to Groq Vision AI API..."):
                try:
                    # Compare captured face with registered students via Vision LLM API
                    prompt = f"""
                    You are an AI Attendance Verification System.
                    I am sending you a live camera image.
                    Registered Students in Database: {json.dumps([{ 'roll_no': k, 'name': v['name'] } for k, v in students_db.items()])}
                    
                    Analyze the face in the image and tell me which registered student matches best.
                    Output JSON only in this format:
                    {{
                        "matched": true/false,
                        "roll_no": "student_roll_no",
                        "name": "student_name",
                        "confidence": "percentage"
                    }}
                    """

                    response = client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{live_b64}"}}
                            ]
                        }]
                    )
                    
                    result_text = response.choices[0].message.content.strip()
                    
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        res_data = json.loads(json_match.group(0))
                        
                        if res_data.get("matched") and res_data.get("roll_no") in students_db:
                            st_info = students_db[res_data["roll_no"]]
                            now = datetime.now()
                            date_str = now.strftime("%Y-%m-%d")
                            time_str = now.strftime("%H:%M:%S")

                            # Save to Excel File
                            excel_file = os.path.join(RECORDS_DIR, f"Attendance_{current_subject.replace(' ', '_')}_{date_str}.xlsx")
                            new_data = pd.DataFrame([{
                                "Roll No": st_info["roll_no"],
                                "Student Name": st_info["name"],
                                "Subject": current_subject,
                                "Date": date_str,
                                "Time": time_str,
                                "Status": "Present",
                                "AI Verification": "Groq Llama Vision API"
                            }])

                            if os.path.exists(excel_file):
                                df_existing = pd.read_excel(excel_file)
                                # Check duplicate
                                if not ((df_existing["Roll No"].astype(str) == str(st_info["roll_no"])) & (df_existing["Date"] == date_str)).any():
                                    df_updated = pd.concat([df_existing, new_data], ignore_index=True)
                                    df_updated.to_excel(excel_file, index=False)
                            else:
                                new_data.to_excel(excel_file, index=False)

                            # Display UI Card
                            details_box.markdown(f"""
                                <div class="card">
                                    <h3>✅ Attendance Marked Present!</h3>
                                    <p><b>Name:</b> {st_info['name']}</p>
                                    <p><b>Roll No:</b> {st_info['roll_no']}</p>
                                    <p><b>Department:</b> {st_info['department']}</p>
                                    <p><b>Subject:</b> {current_subject}</p>
                                    <p><b>Timestamp:</b> {time_str} ({date_str})</p>
                                    <p><b>API Status:</b> Verified via Groq Llama 3.2 Vision API</p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            details_box.error("❌ Unknown Face! Student database me match nahi hua.")
                    else:
                        details_box.warning("AI API response read karne me issue aaya. Dobara try karein.")

                except Exception as e:
                    st.error(f"Vision API Error: {str(e)}")

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

        # Download Excel Button
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download Excel (.xlsx) Report",
                data=f.read(),
                file_name=selected_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
