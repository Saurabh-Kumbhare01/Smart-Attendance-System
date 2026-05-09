import csv
import json
import os
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import END, LEFT, RIGHT, SUNKEN, TOP, BOTTOM, BOTH, X, Y, Text, Tk, Button, Entry, Frame, Label, messagebox, Scrollbar

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAINER_DIR = BASE_DIR / "trainer"
ATTENDANCE_DIR = BASE_DIR / "attendance"
STUDENTS_FILE = BASE_DIR / "students.csv"
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

face_detector = cv2.CascadeClassifier(HAAR_CASCADE)


class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition Attendance System")
        self.master.geometry("900x620")
        self.master.resizable(False, False)

        self.attendance_records = {}
        self.model = None
        self.current_capture_thread = None
        self.recognition_thread = None
        self.stop_recognition = threading.Event()

        self.create_widgets()

    def create_widgets(self):
        top_frame = Frame(self.master, pady=10)
        top_frame.pack(fill=X)

        FieldFrame = Frame(top_frame)
        FieldFrame.pack(side=LEFT, padx=12)

        Label(FieldFrame, text="Student ID:", font=("Arial", 12)).grid(row=0, column=0, sticky="w")
        self.id_entry = Entry(FieldFrame, font=("Arial", 12), width=22)
        self.id_entry.grid(row=0, column=1, pady=4)

        Label(FieldFrame, text="Student Name:", font=("Arial", 12)).grid(row=1, column=0, sticky="w")
        self.name_entry = Entry(FieldFrame, font=("Arial", 12), width=22)
        self.name_entry.grid(row=1, column=1, pady=4)

        button_frame = Frame(top_frame)
        button_frame.pack(side=RIGHT, padx=12)

        Button(button_frame, text="Register Face", font=("Arial", 11), width=18, command=self.start_capture).pack(pady=4)
        Button(button_frame, text="Train Recognizer", font=("Arial", 11), width=18, command=self.train_model).pack(pady=4)
        Button(button_frame, text="Start Attendance", font=("Arial", 11), width=18, command=self.start_attendance).pack(pady=4)
        Button(button_frame, text="Save Attendance XML", font=("Arial", 11), width=18, command=self.save_attendance_xml).pack(pady=4)
        Button(button_frame, text="Exit", font=("Arial", 11), width=18, command=self.close).pack(pady=4)

        middle_frame = Frame(self.master, bd=2, relief=SUNKEN)
        middle_frame.pack(fill=BOTH, padx=12, pady=10, expand=True)

        self.video_label = Label(middle_frame)
        self.video_label.pack(fill=BOTH, expand=True)

        bottom_frame = Frame(self.master)
        bottom_frame.pack(fill=BOTH, padx=12, pady=(0, 10), expand=False)

        self.log_text = Text(bottom_frame, height=11, font=("Consolas", 10), wrap="word")
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(bottom_frame, command=self.log_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log("Ready. Enter Student ID and Name, then register or begin attendance.")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(END, f"[{timestamp}] {message}\n")
        self.log_text.see(END)

    def start_capture(self):
        student_id = self.id_entry.get().strip()
        student_name = self.name_entry.get().strip()
        if not student_id or not student_name:
            messagebox.showwarning("Input required", "Please enter both Student ID and Name.")
            return
        if self.current_capture_thread and self.current_capture_thread.is_alive():
            messagebox.showinfo("Capture running", "Face capture is already running.")
            return
        self.current_capture_thread = threading.Thread(target=self.capture_faces, args=(student_id, student_name), daemon=True)
        self.current_capture_thread.start()

    def capture_faces(self, student_id: str, student_name: str):
        self.log(f"Opening webcam for face capture for {student_name} ({student_id}).")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log("Unable to open webcam.")
            messagebox.showerror("Webcam error", "Could not open the webcam.")
            return

        sample_count = 0
        required_samples = 30
        last_save_time = time.time()

        while sample_count < required_samples:
            ret, frame = cap.read()
            if not ret:
                self.log("Failed to read from webcam.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                sample_count += 1
                face_img = gray[y : y + h, x : x + w]
                face_img = cv2.resize(face_img, (200, 200))
                filename = DATASET_DIR / f"User.{student_id}.{sample_count}.jpg"
                cv2.imwrite(str(filename), face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured {sample_count}/{required_samples}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                break

            self.show_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if sample_count >= required_samples:
                break
            if time.time() - last_save_time > 0.1:
                last_save_time = time.time()

        cap.release()
        cv2.destroyAllWindows()
        self.add_student_record(student_id, student_name)
        self.log(f"Face capture finished: {sample_count} images saved for {student_name}.")
        messagebox.showinfo("Capture complete", f"Captured {sample_count} face images for {student_name}.")

    def add_student_record(self, student_id: str, student_name: str):
        existing = {}
        if STUDENTS_FILE.exists():
            with open(STUDENTS_FILE, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        existing[row[0]] = row[1]
        if existing.get(student_id) != student_name:
            existing[student_id] = student_name
            with open(STUDENTS_FILE, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                for sid, name in sorted(existing.items()):
                    writer.writerow([sid, name])
            self.log(f"Saved student record: {student_id} -> {student_name}.")

    def train_model(self):
        self.log("Training face recognizer from dataset...")
        image_paths = list(DATASET_DIR.glob("User.*.*.jpg"))
        if not image_paths:
            messagebox.showwarning("Dataset empty", "No face images found in the dataset. Register faces first.")
            self.log("Training aborted: no dataset images found.")
            return

        student_ids = sorted(set(img_path.name.split(".")[1] for img_path in image_paths))
        student_id_to_int = {sid: i for i, sid in enumerate(student_ids)}

        face_samples = []
        ids = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sid = img_path.name.split(".")[1]
            label = student_id_to_int[sid]
            face_samples.append(img)
            ids.append(label)

        if not face_samples:
            messagebox.showwarning("Training error", "No valid grayscale face images were found.")
            self.log("Training aborted: invalid face samples.")
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            messagebox.showerror(
                "OpenCV missing contrib",
                "The installed OpenCV package does not include face recognition. Install opencv-contrib-python."
            )
            self.log("Training aborted: OpenCV face module unavailable.")
            return

        recognizer.train(face_samples, np.array(ids))
        trainer_path = TRAINER_DIR / "trainer.yml"
        recognizer.write(str(trainer_path))
        mapping_path = TRAINER_DIR / "id_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(student_id_to_int, f)
        self.model = recognizer
        self.log(f"Training complete. Model saved to {trainer_path}.")
        messagebox.showinfo("Training complete", "Face recognizer has been trained successfully.")

    def get_students(self):
        students = {}
        if STUDENTS_FILE.exists():
            with open(STUDENTS_FILE, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        students[row[0]] = row[1]
        return students

    def start_attendance(self):
        trainer_path = TRAINER_DIR / "trainer.yml"
        if not trainer_path.exists():
            messagebox.showwarning("Model missing", "Please train the recognizer before starting attendance.")
            return
        if self.recognition_thread and self.recognition_thread.is_alive():
            messagebox.showinfo("Attendance running", "Attendance recognition is already running.")
            return
        self.stop_recognition.clear()
        self.recognition_thread = threading.Thread(target=self.mark_attendance, daemon=True)
        self.recognition_thread.start()

    def mark_attendance(self):
        self.log("Starting attendance recognition. Press the 'Exit' button to stop.")
        students = self.get_students()
        self.attendance_records = {}

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(TRAINER_DIR / "trainer.yml"))

        mapping_path = TRAINER_DIR / "id_mapping.json"
        if not mapping_path.exists():
            self.log("ID mapping file not found.")
            messagebox.showerror("Mapping missing", "ID mapping file not found. Please retrain the model.")
            return
        with open(mapping_path) as f:
            student_id_to_int = json.load(f)
            int_to_student_id = {v: k for k, v in student_id_to_int.items()}

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log("Unable to open webcam for attendance.")
            messagebox.showerror("Webcam error", "Could not open the webcam.")
            return

        while not self.stop_recognition.is_set():
            ret, frame = cap.read()
            if not ret:
                self.log("Failed to read webcam frame during attendance.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = gray[y : y + h, x : x + w]
                face_img = cv2.resize(face_img, (200, 200))
                id_pred, confidence = recognizer.predict(face_img)
                student_id = int_to_student_id.get(id_pred, "Unknown")
                name = students.get(student_id, "Unknown")
                if confidence < 70:
                    status = "Present"
                    if student_id not in self.attendance_records:
                        timestamp = datetime.now()
                        self.attendance_records[student_id] = {
                            "id": student_id,
                            "name": name,
                            "date": timestamp.strftime("%Y-%m-%d"),
                            "time": timestamp.strftime("%H:%M:%S"),
                            "confidence": f"{confidence:.1f}",
                            "status": status,
                        }
                        self.log(f"Marked attendance: {name} ({student_id}) confidence {confidence:.1f}.")
                else:
                    name = "Unknown"
                    status = "Unrecognized"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if status == "Present" else (0, 0, 255), 2)
                cv2.putText(frame, f"{name} {confidence:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            self.show_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.log("Attendance session ended.")

    def save_attendance_xml(self):
        if not self.attendance_records:
            messagebox.showwarning("No attendance", "No attendance records are available to save.")
            return

        now = datetime.now()
        xml_path = ATTENDANCE_DIR / f"attendance_{now.strftime('%Y%m%d_%H%M%S')}.xml"
        root = ET.Element("Attendance")
        for record in self.attendance_records.values():
            entry = ET.SubElement(root, "Record")
            ET.SubElement(entry, "ID").text = record["id"]
            ET.SubElement(entry, "Name").text = record["name"]
            ET.SubElement(entry, "Date").text = record["date"]
            ET.SubElement(entry, "Time").text = record["time"]
            ET.SubElement(entry, "Confidence").text = record["confidence"]
            ET.SubElement(entry, "Status").text = record["status"]

        tree = ET.ElementTree(root)
        tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)
        self.log(f"Saved attendance XML to {xml_path}.")
        messagebox.showinfo("Attendance saved", f"Attendance sheet saved to:\n{xml_path}")

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((880, 380))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def close(self):
        self.stop_recognition.set()
        self.master.quit()


if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
