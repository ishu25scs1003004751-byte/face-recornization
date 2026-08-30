import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="AI Smart Attendance System", page_icon="📸", layout="wide")

st.title("📸 AI Face Recognition Attendance System")
st.write("Webcam ke samne aayein. System aapka face recognize karke real-time attendance mark kar dega.")

# Data directory setup for images and attendance sheet
FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(FACES_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# Helper function to load known face encodings
@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(FACES_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0].replace("_", " ").title()
                known_names.append(name)
                
    return known_encodings, known_names

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Check if already marked for today
    already_marked = ((df["Name"] == name) & (df["Date"] == today_date)).any()

    if not already_marked:
        new_entry = pd.DataFrame([{"Name": name, "Date": today_date, "Time": current_time}])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True, current_time
    return False, None

# Sidebar Setup
st.sidebar.title("⚙️ System Control")
mode = st.sidebar.radio("Navigation", ["Mark Attendance (Scanner)", "Register New Face", "View Attendance Logs"])

known_encodings, known_names = load_known_faces()

# MODE 1: MARK ATTENDANCE
if mode == "Mark Attendance (Scanner)":
    st.subheader("📷 Face Scanner")
    img_file_buffer = st.camera_input("Capture photo to mark attendance")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if not face_encodings:
            st.warning("⚠️ Koi face detect nahi hua! Frame mein sahi se saamne aayein.")
        else:
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                if name != "Unknown":
                    marked, time_str = mark_attendance(name)
                    if marked:
                        st.success(f"✅ Attendance Marked for **{name}** at {time_str}!")
                        st.balloons()
                    else:
                        st.info(f"ℹ️ **{name}** ki attendance aaj ke liye pehle se marked hai.")
                else:
                    st.error("❌ Unregistered Face! Pehle 'Register New Face' tab par jaakar registration karein.")

# MODE 2: REGISTER NEW FACE
elif mode == "Register New Face":
    st.subheader("👤 Register Student/Employee")
    new_name = st.text_input("Enter Full Name:")
    new_photo = st.camera_input("Take Photo for Registration")

    if st.button("Save Face Data"):
        if new_name and new_photo:
            clean_name = new_name.strip().replace(" ", "_").lower()
            file_path = os.path.join(FACES_DIR, f"{clean_name}.jpg")
            
            with open(file_path, "wb") as f:
                f.write(new_photo.getbuffer())
                
            st.success(f"🎉 **{new_name}** ka face record successfully save ho gaya hai!")
            st.cache_resource.clear()
        else:
            st.warning("Kripya Naam aur Photo dono provide karein.")

# MODE 3: VIEW ATTENDANCE LOGS
elif mode == "View Attendance Logs":
    st.subheader("📋 Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df_logs = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df_logs, use_container_width=True)
        
        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Attendance CSV Sheet",
            data=csv_data,
            file_name=f"attendance_report_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv"
        )
