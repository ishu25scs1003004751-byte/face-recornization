import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import face_recognition
from datetime import datetime

# Page Configuration (Dark High-Tech UI)
st.set_page_config(page_title="Smart AI Attendance System", layout="wide", page_icon="🪪")

# Folder & Database Setup
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

# Custom High-Tech Styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; width: 100%; }
    .card { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

st.title("🪪 AI Face Recognition Attendance System")

# Sidebar Navigation
menu = st.sidebar.radio(
    "Select System Mode", 
    ["1. Register New Student (Live Camera)", "2. Live Attendance Scanner", "3. Attendance Records & Excel"]
)

students_db = load_students()

# ================= STEP 1: LIVE STUDENT REGISTRATION =================
if menu == "1. Register New Student (Live Camera)":
    st.header("📸 Student Face & Profile Registration")
    st.write("Student ki details enter karein aur webcam se clear face photo capture karein.")

    col1, col2 = st.columns([1, 1])

    with col1:
        roll_no = st.text_input("Roll Number / ID")
        student_name = st.text_input("Full Name")
        department = st.text_input("Department / Branch", "CSE (AI & ML)")
        subjects = st.multiselect(
            "Enrolled Subjects", 
            ["Data Structures", "Operating Systems", "Machine Learning", "Python", "Mathematics"], 
            default=["Data Structures"]
        )

    with col2:
        st.subheader("Live Camera Capture")
        camera_image = st.camera_input("Take a photo of the student")

    if st.button("💾 Register & Save Student Permanently"):
        if not roll_no or not student_name or camera_image is None:
            st.error("Kripya Roll No, Name aur Live Camera Photo teeno provide karein!")
        else:
            # Save Image File
            img_filename = f"{roll_no}_{student_name.replace(' ', '_')}.jpg"
            img_path = os.path.join(STUDENTS_DIR, img_filename)

            with open(img_path, "wb") as f:
                f.write(camera_image.getbuffer())

            # AI Model Face Detection Verification
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                st.error("❌ Photo me koi face detect nahi hua! Kripya dobara clear light me photo lein.")
                os.remove(img_path)
            else:
                # Save details into JSON Database
                students_db[roll_no] = {
                    "name": student_name,
                    "roll_no": roll_no,
                    "department": department,
                    "subjects": subjects,
                    "image_path": img_path
                }
                save_students(students_db)
                st.success(f"✅ Student {student_name} ({roll_no}) successfully register ho gaya hai aur database me save ho gaya!")

    st.subheader("📋 Registered Students Database")
    if students_db:
        st.json(students_db)
    else:
        st.info("Abhi tak koi student register nahi hua hai.")
