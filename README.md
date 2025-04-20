Okay, here's an updated README.md file reflecting the significant enhancements in your Python script (profiles, sets, rest periods, stats, etc.).

```markdown
# Fitness Tracker Pro (Enhanced Version)

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
<!-- ![License](https://img.shields.io/badge/License-MIT-yellow.svg) --> <!-- Optional: Uncomment if you add an MIT license file -->

A comprehensive, real-time fitness tracker using Python, OpenCV, and MediaPipe. Monitor exercise form, count repetitions across **sets**, track **rest periods**, manage user **profiles**, save **statistics**, and get visual feedback using a webcam or video file input.

<!-- Optional: Add a screenshot or GIF of the application in action! -->
<!-- [Screenshot/GIF placeholder - Replace this line with an actual image tag once you have one] -->
<!-- Example: ![App Screenshot](path/to/your/screenshot.png) -->

## Description

This application leverages computer vision to analyze body pose during workouts. It tracks key body landmarks using Google's MediaPipe Pose solution to:

*   **Manage User Profiles:** Create and select user profiles to track progress individually.
*   **Configure Workouts:** Define the number of sets, repetitions per set, and rest time for structured workouts (webcam mode).
*   **Track Exercises:** Detect the current exercise being performed (from a predefined list).
*   **Count Repetitions:** Automatically count reps within each set based on defined angle thresholds.
*   **Monitor Form:** Provide real-time feedback on form correctness (e.g., back straightness, joint angles).
*   **Guide Performance:** Offer visual guides (GIFs) for selected exercises before starting (webcam mode).
*   **Manage Rest:** Display a countdown timer during rest periods between sets (webcam mode).
*   **Save Statistics:** Persistently store workout data (total reps, estimated calories) per user per exercise in JSON files.
*   **Visualize Stats:** Display a pie chart summarizing calorie distribution across exercises for the selected user.
*   **Support Multiple Sources:** Work with either a live webcam feed or a pre-recorded video file (video file mode has limited features - no sets/stats saving).

## Features

*   **Real-time Pose Estimation:** Utilizes MediaPipe Pose for accurate body landmark tracking.
*   **User Profiles:** Create, select, and store user profiles (`profiles.json`) with basic info (age, height, weight used for calorie estimates).
*   **Workout Configuration:** Set targets for Sets, Reps per Set, and Rest Time via an interactive UI before starting a webcam session.
*   **Structured Workouts:** Guides the user through configured sets and rest periods.
*   **Exercise Tracking:** Currently supports:
    *   Bicep Curl (counts left/right arm separately)
    *   Squat
    *   Push Up
    *   Pull Up
    *   Deadlift
*   **Repetition Counting:** Automatic rep counting within the current set.
*   **Form Feedback:** Provides on-screen warnings for common form issues.
*   **Multiple Video Sources:** Choose between using your webcam (full features) or analyzing a local video file (basic tracking, no sets/stats/guides).
*   **Exercise Guides (Webcam Mode):** Displays animated GIFs to demonstrate proper form before starting an exercise.
*   **Rest Timer (Webcam Mode):** Visual countdown timer between sets.
*   **Persistent Statistics (Webcam Mode):** Saves total reps and estimated calories per user/exercise to `stats.json`.
*   **Statistics Visualization:** View a pie chart of calorie distribution via the "View Stats" option.
*   **Interactive UI:** Built with OpenCV, handling different application states (Home, Select Exercise, Set Config, Guide, Tracking, Rest, Stats).

For detailed information on how pose estimation is used for rep counting and form correction, see [POSE_ESTIMATION_DETAILS.md](POSE_ESTIMATION_DETAILS.md).


## Tech Stack

*   **Python** (3.8+)
*   **OpenCV** (`opencv-python`) - For video capture, image processing, and UI display.
*   **MediaPipe** (`mediapipe`) - For pose estimation.
*   **NumPy** (`numpy`) - For numerical operations (angles, coordinates).
*   **Matplotlib** (`matplotlib`) - For generating the statistics pie chart.
*   **Imageio** (`imageio`) - For loading and handling GIF files.
*   **Tkinter** (Built-in Python library) - Used for profile popups and file selection dialogs.
*   **JSON** (Built-in Python library) - For saving/loading profiles and stats.

## Directory Structure

```
.
├── your_script_name.py    # Main application script (e.g., fitness_tracker_pro.py)
├── GIFs/                  # Contains exercise guide GIFs (REQUIRED)
│   ├── bicep.gif
│   ├── squats.gif
│   ├── pushup.gif
│   ├── pullup.gif
│   └── deadlift.gif
├── profiles.json          # Stores user profile data (Created automatically)
├── stats.json             # Stores user statistics (Created automatically)
├── videos/                # (Optional) Place sample input videos here for testing
├── venv/                  # Python virtual environment (Should be in .gitignore)
├── requirements.txt       # List of Python dependencies (Recommended)
├── .gitignore             # Specifies intentionally untracked files
├── README.md              # This file
└── LICENSE                # (Optional) Project License file
```

**Important:**
*   Replace `your_script_name.py` with the actual name of your main Python file.
*   The `GIFs` folder and its contents are **required** for the exercise guide feature.
*   `profiles.json` and `stats.json` will be created in the same directory as the script when you first run it or create profiles/save stats.
*   The `venv` folder should **not** be committed to Git (ensure it's listed in your `.gitignore`).

## Prerequisites

*   **Python:** Version 3.8 or higher recommended.
*   **Git:** For cloning the repository.
*   **Pip:** Python package installer (usually comes with Python).

## Installation

1.  **Clone the repository:**
    ```bash
    # Replace with your actual repository URL if different
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
        *(First, create/update `requirements.txt` while your venv is active: `pip freeze > requirements.txt`)*
        ```bash
        pip install -r requirements.txt
        ```
    *   **(Alternative Way) Install manually:**
        ```bash
        pip install opencv-python mediapipe numpy matplotlib imageio
        ```
        *(Note: Tkinter and JSON are typically included with Python)*

4.  **Ensure GIF files are present:** Verify that the `GIFs` folder exists in the project root and contains the necessary `.gif` files (`bicep.gif`, `squats.gif`, etc.).

## Usage

1.  **Activate the virtual environment** (if not already active):
    *   macOS/Linux: `source venv/bin/activate`
    *   Windows: `.\venv\Scripts\activate`

2.  **Run the main application script:**
    ```bash
    # Replace with your actual script name
    python your_script_name.py
    ```

3.  **Follow the on-screen UI:**
    *   **Home Screen:**
        *   Select or Create a User Profile. Profile selection is required before starting a workout.
        *   View statistics for the selected user ("View Stats").
        *   Choose workout source: "Start Webcam Workout" (full features) or "Load Video (No Stats)" (limited features).
    *   **(Webcam Flow):**
        *   **Exercise Select:** Choose the desired exercise.
        *   **Set Configuration:** Adjust the number of Sets, Reps per Set, and Rest Time using the +/- buttons. Click "Confirm & Start".
        *   **Guide:** A GIF demonstrates the exercise. Click "Start Exercise" or wait.
        *   **Tracking:** Perform the exercise for the current set. View reps, set number, stage (UP/DOWN), and form feedback. Switch exercises (resets set config) or go Home (ends session, saves stats).
        *   **Rest:** After completing reps for a set (if not the last set), a rest timer starts. You can skip the rest or wait for it to finish.
        *   The cycle repeats for all sets. Session stats are saved upon returning Home or finishing all sets.
    *   **(Video File Flow):**
        *   **Exercise Select:** Choose the exercise matching the video content.
        *   **Tracking:** View basic rep counting and form feedback for the duration of the video. No sets, rest periods, or statistics are tracked/saved.
    *   **Stats Screen:** Displays the pie chart (if stats exist for the user). Go "Back to Home".
    *   Press 'Q' on your keyboard at any time to quit the application. Data is typically saved when ending a webcam session cleanly via the Home button or completing all sets.

## Configuration

Parameters for exercise detection, form feedback, MET values, and UI appearance are defined as constants near the top of the `your_script_name.py` script.

*   **Rep Counting Thresholds:** Adjust `*_ENTER_ANGLE`, `*_EXIT_ANGLE` constants.
*   **Form Correction Thresholds:** Modify `BACK_ANGLE_THRESHOLD_*`, `PUSHUP_BODY_STRAIGHT_*`, etc.
*   **Set/Rep/Rest Defaults:** Change `target_sets`, `target_reps_per_set`, `target_rest_time` initial values.
*   **Data Files:** `PROFILES_FILE`, `STATS_FILE` specify the names for saved data.
*   **MET Values:** `MET_VALUES` dictionary maps exercises to Metabolic Equivalent Task values for calorie estimation.
*   **UI:** Colors, fonts, layout constants can be tweaked.

## License

<!-- State your license here. Example: -->
This project is licensed under the MIT License - see the `LICENSE` file for details (if applicable).

<!-- Or if no license file yet: -->
<!-- License details TBD. -->

## Acknowledgements

*   **MediaPipe** by Google for the powerful pose estimation framework.
*   **OpenCV** for the versatile computer vision library.
```

**Key Changes Made:**

*   Updated the main description and features list to include Profiles, Sets, Reps, Rest, Stats, JSON saving, Matplotlib visualization, and the interactive UI flow.
*   Clarified the difference in features between Webcam and Video File modes.
*   Added `matplotlib` to the Tech Stack and manual installation instructions.
*   Updated the Directory Structure to reflect the single main script and the automatically generated JSON files.
*   Significantly expanded the "Usage" section to describe the new UI flow, including profile management, set configuration, and the rest timer.
*   Expanded the "Configuration" section to mention the new constants related to sets, rest, stats files, and MET values.
*   Replaced placeholder script names with `your_script_name.py` – **remember to replace this with the actual filename** (e.g., `fitness_tracker_pro.py`).