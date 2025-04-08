# Fitness Tracker Pro

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg) <!-- Optional: Adjust version if needed -->
<!-- ![License](https://img.shields.io/badge/License-MIT-yellow.svg) --> <!-- Optional: Uncomment if you add an MIT license file -->

A real-time fitness tracker using Python, OpenCV, and MediaPipe to monitor exercise form, count repetitions, and provide feedback for various exercises using a webcam or video file input.

<!-- Optional: Add a screenshot or GIF of the application in action! -->
<!-- [Screenshot/GIF placeholder - Replace this line with an actual image tag once you have one] -->
<!-- Example: ![App Screenshot](path/to/your/screenshot.png) -->

## Description

This application leverages computer vision to analyze body pose during workouts. It tracks key body landmarks using Google's MediaPipe Pose solution to:

*   Detect the current exercise being performed (from a predefined list).
*   Count repetitions based on defined angle thresholds for specific joints.
*   Provide real-time feedback on form correctness (e.g., back straightness, joint angles).
*   Offer visual guides (GIFs) for selected exercises before starting (for webcam input).
*   Work with either a live webcam feed or a pre-recorded video file.

## Features

*   **Real-time Pose Estimation:** Utilizes MediaPipe Pose for accurate body landmark tracking.
*   **Exercise Tracking:** Currently supports:
    *   Bicep Curl (counts left/right arm separately)
    *   Squat
    *   Push Up
    *   Pull Up
    *   Deadlift
*   **Repetition Counting:** Automatic rep counting based on exercise-specific movement patterns and angle thresholds.
*   **Form Feedback:** Provides on-screen warnings for common form issues (e.g., back angle during squats/deadlifts, push-up body alignment).
*   **Multiple Video Sources:** Choose between using your webcam or analyzing a local video file (`.mp4`, `.avi`, etc.).
*   **Exercise Guides:** Displays animated GIFs to demonstrate proper form before starting an exercise (when using webcam).
*   **User Interface:** Simple UI built with OpenCV for selecting source, exercise, and viewing stats/feedback.

## Tech Stack

*   **Python** (3.8+)
*   **OpenCV** (`opencv-python`) - For video capture, image processing, and UI display.
*   **MediaPipe** (`mediapipe`) - For pose estimation.
*   **NumPy** (`numpy`) - For numerical operations (angles, coordinates).
*   **Imageio** (`imageio`) - For loading and handling GIF files.
*   **Tkinter** (Built-in Python library) - Used for the file selection dialog.

## Directory Structure

```
.
├── Fitness_Tracker_ui.py  # Main application script with UI & GIF guides (Recommended Entry Point)
├── fitness_tracker.py     # (Note: Seems like a simpler version without GIF guides?)
├── GIFs/                  # Contains exercise guide GIFs (REQUIRED)
│   ├── bicep.gif
│   ├── squats.gif
│   ├── pushup.gif
│   ├── pullup.gif
│   └── deadlift.gif
├── videos/                # (Optional) Place sample input videos here for testing
├── venv/                  # Python virtual environment (Should be in .gitignore)
├── requirements.txt       # List of Python dependencies (Recommended)
├── .gitignore             # Specifies intentionally untracked files
├── README.md              # This file
└── LICENSE                # (Optional) Project License file
```

**Important:**
*   The `GIFs` folder and its contents are **required** for the exercise guide feature in `Fitness_Tracker_ui.py`.
*   The `venv` folder should **not** be committed to Git (ensure it's listed in your `.gitignore`).

## Prerequisites

*   **Python:** Version 3.8 or higher recommended.
*   **Git:** For cloning the repository.
*   **Pip:** Python package installer (usually comes with Python).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/a1harfoush/Fitness_Tracker_Pro.git
    cd Fitness_Tracker_Pro
    ```

2.  **Create and activate a virtual environment (Recommended):**
    *   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**
    *   **(Recommended Way) Using `requirements.txt`:**
        *(First, ensure you've created `requirements.txt` using `pip freeze > requirements.txt` while your venv is active)*
        ```bash
        pip install -r requirements.txt
        ```
    *   **(Alternative Way) Install manually:**
        ```bash
        pip install opencv-python mediapipe numpy imageio tk
        ```
        *(Note: `tk` might already be included with your Python installation, but explicitly listing can sometimes help)*

4.  **Ensure GIF files are present:** Verify that the `GIFs` folder exists in the project root and contains the necessary `.gif` files (`bicep.gif`, `squats.gif`, etc.).

## Usage

1.  **Activate the virtual environment** (if not already active):
    *   macOS/Linux: `source venv/bin/activate`
    *   Windows: `.\venv\Scripts\activate`

2.  **Run the main application script:**
    ```bash
    python Fitness_Tracker_ui.py
    ```

3.  **Follow the on-screen UI:**
    *   **Home Screen:** Choose "Use Webcam" or "Load Video File".
    *   **Exercise Select Screen:** Click to select the desired exercise, then click "Start".
    *   **Guide Screen (Webcam Only):** A GIF demonstrating the exercise will play for a few seconds. Click "Start Exercise" or wait for it to potentially auto-transition (if implemented).
    *   **Tracking Screen:** Perform the exercise. View rep counts, stage (UP/DOWN/HOLD), and form feedback. You can switch exercises using the buttons at the top or return to the Home screen using the 'H' button.
    *   Press 'Q' on your keyboard at any time to quit the application.

## Configuration

Angle thresholds and other parameters for exercise detection and form feedback are defined as constants near the top of the `Fitness_Tracker_ui.py` script. You can modify these values to tune the sensitivity and accuracy for your specific needs or body type.

## License

<!-- State your license here. Example: -->
This project is licensed under the MIT License - see the `LICENSE` file for details (if applicable).

<!-- Or if no license file yet: -->
<!-- License details TBD. -->

## Acknowledgements

*   **MediaPipe** by Google for the powerful pose estimation framework.
