# Face Recognition Attendance System

This Python app uses OpenCV for live webcam face capture, registration, training, and attendance marking. Attendance output can be saved as an XML sheet. It includes a redesigned Streamlit web interface with responsive cards, animated camera panels, dark mode, dashboard metrics, and Bootstrap/Tailwind-inspired styling.

## Features
- Live webcam face registration
- Face dataset creation and training using OpenCV LBPH recognizer
- Real-time attendance marking with face recognition
- Downloadable attendance sheet in XML format
- Modern Streamlit web UI for capture, training, attendance, and dashboard views
- Dark mode, glassmorphism cards, hover states, animated scan/loading feedback, and responsive navigation
- Simple Tkinter GUI is still available for desktop use

## Setup
1. Install Python 3.10+.
2. Open a terminal in this project folder.
3. Create a virtual environment (recommended):

```bash
python -m venv venv
.\\venv\\Scripts\\activate
```

4. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage
1. Run the modern web app:

```bash
streamlit run app_streamlit.py
```

2. Or run the legacy desktop app:

```bash
python app.py
```

3. Enter a student ID and name on the Register page.
4. Capture face images, then open Train Model and train the recognizer.
5. Open Attendance to start scanning with the webcam.
6. Open Dashboard to review records and export attendance XML.

## Notes
- The webcam must be available and accessible to the Python app.
- Use `opencv-contrib-python` so the face recognition module is available.
- Attendance XML files are saved inside the `attendance/` folder.
- Runtime-generated artifacts like `dataset/`, `trainer/`, `attendance/`, and `students.csv` are ignored by git and recreated when the app runs.

## Project Structure
- `app_streamlit.py`: Modern web application
- `app.py`: Legacy Tkinter desktop application
- `dataset/`: Stored face images
- `trainer/`: Recognizer model files
- `attendance/`: Saved XML attendance sheets
- `students.csv`: Registered student names and IDs
- `requirements.txt`: Python package dependencies
