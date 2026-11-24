import cv2
import numpy as np
import os
import json
import csv
from datetime import datetime

# -------------------- PROJECT FOLDERS --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")

MODEL_FILE = os.path.join(MODEL_DIR, "trainer.yml")
LABELS_FILE = os.path.join(MODEL_DIR, "labels.json")
CSV_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")

# Lower LBPH distance means a closer match.
# This is a starting value; lighting and cameras affect it.

RECOGNITION_THRESHOLD = 65

for folder in [DATASET_DIR, MODEL_DIR, ATTENDANCE_DIR]:
    os.makedirs(folder, exist_ok=True)

# OpenCV's built-in face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def open_camera():
    """Try the default webcam. Change 0 to 1 if your USB camera is not found."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Camera not found. Check the webcam connection.")
        return None
    return camera

def register_student():
    print("\n--- STUDENT REGISTRATION ---")
    student_id = input("Enter a unique student ID (example: 101): ").strip()
    student_name = input("Enter student name: ").strip()

    if not student_id or not student_name:
        print("ID and name cannot be empty.")
        return

    student_folder = os.path.join(DATASET_DIR, student_id)
    if os.path.exists(student_folder):
        print("This ID is already registered. Use a different ID.")
        return

    os.makedirs(student_folder)
    camera = open_camera()
    if camera is None:
        os.rmdir(student_folder)
        return

    sample_count = 0
    print("\nLook at the camera. Slowly turn your face a little left and right.")
    print("The program will try to save 30 clear face samples.")
    print("Press Q to stop early.\n")

    while sample_count < 30:
        success, frame = camera.read()
        if not success:
            print("Could not read from the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            sample_count += 1
            image_path = os.path.join(
                student_folder, f"{sample_count}.jpg"
            )
            cv2.imwrite(image_path, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Samples: {sample_count}/30",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )
            # Save only one detected face per camera frame.
            break

        cv2.imshow("Register Student - Press Q to stop", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    if sample_count < 10:
        print("Not enough samples. Please register again in better lighting.")
        for filename in os.listdir(student_folder):
            os.remove(os.path.join(student_folder, filename))
        os.rmdir(student_folder)
        return

 # Store the student's name separately.

    name_file = os.path.join(student_folder, "name.txt")
    with open(name_file, "w", encoding="utf-8") as file:
        file.write(student_name)

    print(f"Registration complete: {student_name} ({student_id})")
    print("Next: choose option 2 to train the model.")

def train_model():
    print("\n--- TRAINING THE FACE MODEL ---")
    face_samples = []
    face_labels = []
    label_to_student = {}
    current_label = 0

    if not os.path.exists(DATASET_DIR):
        print("Dataset folder is missing.")
        return

    student_folders = [
        folder for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ]

    for student_id in sorted(student_folders):
        student_folder = os.path.join(DATASET_DIR, student_id)
        name_file = os.path.join(student_folder, "name.txt")

        if not os.path.exists(name_file):
            continue

        with open(name_file, "r", encoding="utf-8") as file:
            student_name = file.read().strip()

        label_to_student[str(current_label)] = {
            "student_id": student_id,
            "name": student_name
        }

        for filename in os.listdir(student_folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(student_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (200, 200))
            face_samples.append(image)
            face_labels.append(current_label)

        current_label += 1

    if len(face_samples) == 0:
        print("No face samples found. Register at least one student first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(
        face_samples,
        np.array(face_labels, dtype=np.int32)
    )

    recognizer.write(MODEL_FILE)
    with open(LABELS_FILE, "w", encoding="utf-8") as file:
        json.dump(label_to_student, file, indent=4)

    print("Training completed successfully!")
    print(f"Saved {len(face_samples)} face samples.")
    print("Next: choose option 3 to start attendance.")

def mark_attendance(student_id, student_name):
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    file_exists = os.path.exists(CSV_FILE)
    already_marked = False

    if file_exists:
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
        for row in csv.reader:
            if row.get("Student ID") == student_id and row.get("Date") == today:
                already_marked = True
                break
    if already_marked:
        return "Already marked"

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Student ID", "Name", "Date", "Time", "Status"])
        writer.writerow([student_id, student_name, today, current_time, "Present"])

    return "Attendance marked"

def start_attendance():
    print("\n--- STARTING ATTENDANCE ---")

    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print("Model not found. Register students and train the model first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    with open(LABELS_FILE, "r", encoding="utf-8") as file:
        label_to_student = json.load(file)

    camera = open_camera()
    if camera is None:
        return

    print("Attendance is running. Press Q to quit.")
    last_label = None
    stable_count = 0
    confirmed_label = None

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read from the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            last_label = None
            stable_count = 0
            confirmed_label = None

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, distance = recognizer.predict(face)

            if str(label) in label_to_student and distance < RECOGNITION_THRESHOLD:
                student = label_to_student[str(label)]
                display_name = student["name"]
                display_text = f"{display_name} - checking"

            if label == last_label:
                stable_count += 1
            else:
                last_label = label
                stable_count = 1

            if stable_count >= 5 and confirmed_label != label:
                result = mark_attendance(
                    student["student_id"], student["name"]
                )

                confirmed_label = label
                display_text = f"{display_name}: {result}"
            elif confirmed_label == label:
                display_text = f"{display_name}: attendance checked"

            box_color = (0, 200, 0)
        else:
            display_text = "Unknown face"
            box_color = (0, 0, 255)
            last_label = None
            stable_count = 0
            confirmed_label = None

        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(
            frame, display_text, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2
        )

        cv2.imshow("AI Smart Attendance - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"Attendance file location: {CSV_FILE}")


def main():
     while True:
       print("\n========== AI SMART ATTENDANCE ==========")
       print("1. Register a student")
       print("2. Train the face-recognition model")
       print("3. Start attendance")
       print("4. Exit")

       choice = input("Enter your choice (1-4): ").strip()

       if choice == "1":
            register_student()
       elif choice == "2":
            train_model()
       elif choice == "3":
            start_attendance()
       elif choice == "4":
            print("Program closed. Goodbye!")
            break
       else:
           print("Invalid choice. Please enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()