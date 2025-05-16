# Fitness Tracker Pro

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)

A comprehensive, real-time fitness tracker using Python, OpenCV, and MediaPipe. Monitor exercise form, count repetitions across **sets**, track **rest periods**, manage user **profiles**, save **statistics**, get visual feedback, and interact with an **AI Fitness Coach**. Uses webcam or video file input.

![image](https://github.com/user-attachments/assets/0b439fd5-5e9e-445d-afce-249bc8d5388c)
![Screenshots](https://github.com/user-attachments/assets/6480f03a-9a6b-4616-af89-cfd2a13b8c0e)

## Description

This application leverages computer vision to analyze body pose during workouts. It tracks key body landmarks using Google's MediaPipe Pose solution to:

* **Manage User Profiles:** Create and select user profiles to track progress individually.
* **Configure Workouts:** Define the number of sets, repetitions per set, and rest time for structured workouts (webcam mode).
* **Track Exercises:** Detect the current exercise being performed (from a predefined list).
* **Count Repetitions:** Automatically count reps within each set based on defined angle thresholds with hysteresis and EMA smoothing for stability.
* **Monitor Form:** Provide real-time feedback on form correctness (e.g., back straightness, joint angles, squat depth, push-up body alignment, bicep arm stability).
* **Guide Performance:** Offer visual guides (GIFs) for selected exercises before starting (webcam mode).
* **Manage Rest:** Display a countdown timer during rest periods between sets (webcam mode).
* **AI Fitness Coach:** Interact with an AI assistant (powered by Google Gemini) for personalized advice, motivation, and answers to fitness-related questions based on your profile and stats.
* **Save Statistics:** Persistently store workout data (total reps, estimated calories, total active time, last workout configuration, logged sessions) per user per exercise in JSON files.
* **Visualize Stats:** Display a pie chart summarizing calorie distribution and a bar chart for total active time per exercise for the selected user.
* **Support Multiple Sources:** Work with either a live webcam feed (full features) or a pre-recorded video file (video file mode has limited features - no sets/stats saving/AI coach).

## Features

* **Real-time Pose Estimation:** Utilizes MediaPipe Pose for accurate body landmark tracking.
* **User Profiles:** Create, select, and store user profiles (`profiles.json`) with basic info (age, height, weight, gender used for calorie estimates and AI context).
* **AI Fitness Coach:**
    * Powered by Google Gemini.
    * Answers questions about tracked progress, stats, and workout history.
    * Explains exercises, form, and muscles worked.
    * Discusses basic workout principles and nutritional concepts.
    * Provides motivation and personalized tips based on user data.
    * Requires a `GOOGLE_API_KEY` in a `.env` file.
* **Workout Configuration:** Set targets for Sets, Reps per Set, and Rest Time via an interactive UI before starting a webcam session.
* **Structured Workouts:** Guides the user through configured sets and rest periods.
* **Exercise Tracking:** Currently supports:
    * Bicep Curl (counts left/right arm separately, checks upper arm stability)
    * Squat (checks knee valgus, chest forward lean)
    * Push Up (checks body straightness)
    * Pull Up (checks chin over wrist and elbow angle)
    * Deadlift (checks back angle during lift and lockout)
* **Advanced Repetition Counting:**
    * Automatic rep counting within the current set.
    * Incorporates Exponential Moving Average (EMA) for smoother angle detection.
    * Uses hysteresis in angle thresholds to prevent miscounts due to minor oscillations.
* **Form Feedback:** Provides on-screen warnings for common form issues based on configurable angle thresholds.
* **Multiple Video Sources:** Choose between using your webcam (full features) or analyzing a local video file (basic tracking, no sets/stats/guides/AI).
* **Exercise Guides (Webcam Mode):** Displays animated GIFs to demonstrate proper form before starting an exercise.
* **Rest Timer (Webcam Mode):** Visual countdown timer between sets.
* **Persistent Statistics (Webcam Mode):**
    * Saves total reps, estimated calories, total active time per exercise, and last used workout configuration (sets, reps, rest) to `stats.json`.
    * Logs recent workout sessions including date, duration, and exercises performed.
* **Statistics Visualization:** View charts for calorie distribution and total active time per exercise via the "View Stats" option.
* **Interactive UI:** Built with OpenCV, handling different application states (Home, Select Exercise, Set Config, Guide, Tracking, Rest, Stats, AI Coach).

For detailed information on how pose estimation is used for rep counting and form correction, see [POSE_ESTIMATION_DETAILS.md](POSE_ESTIMATION_DETAILS.md).

## Tech Stack

* **Python** (3.8+)
* **OpenCV** (`opencv-python`) - For video capture, image processing, and UI display.
* **MediaPipe** (`mediapipe`) - For pose estimation.
* **NumPy** (`numpy`) - For numerical operations (angles, coordinates).
* **Matplotlib** (`matplotlib`) - For generating the statistics charts.
* **Imageio** (`imageio`) - For loading and handling GIF files.
* **Tkinter** (Built-in Python library) - Used for profile popups and file selection dialogs.
* **google-generativeai** (`google-generativeai`) - For interacting with the Gemini AI model.
* **python-dotenv** (`python-dotenv`) - For managing environment variables (like API keys).
* **JSON** (Built-in Python library) - For saving/loading profiles and stats.

## Directory Structure

```

.
├── fitness\_tracker\_ui.py  \# Main application script
├── GIFs/                  \# Contains exercise guide GIFs (REQUIRED)
│   ├── bicep.gif
│   ├── squats.gif
│   ├── pushup.gif
│   ├── pullup.gif
│   └── deadlift.gif
├── profiles.json          \# Stores user profile data (Created automatically)
├── stats.json             \# Stores user statistics (Created automatically)
├── .env                   \# Stores environment variables like GOOGLE\_API\_KEY (User must create)
├── videos/                \# (Optional) Place sample input videos here for testing
├── venv/                  \# Python virtual environment (Should be in .gitignore)
├── requirements.txt       \# List of Python dependencies
├── .gitignore             \# Specifies intentionally untracked files
├── POSE\_ESTIMATION\_DETAILS.md   \# Details on the calculations and angles
└── README.md              \# This file

````

## Prerequisites

* **Python:** Version 3.8 or higher recommended.
* **Git:** For cloning the repository.
* **Pip:** Python package installer (usually comes with Python).
* **Google API Key:** For AI Coach functionality. You need to obtain an API key for the Gemini model from Google AI Studio and store it in a `.env` file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/a1harfoush/Fitness_Tracker_Pro.git](https://github.com/a1harfoush/Fitness_Tracker_Pro.git)
    cd Fitness_Tracker_Pro
    ```

2.  **Create and activate a virtual environment (Recommended):**
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**
    * Use `requirements.txt` (if provided and up-to-date) or install manually:
        ```bash
        pip install opencv-python mediapipe numpy matplotlib imageio google-generativeai python-dotenv
        ```
        *(Note: Tkinter and JSON are typically included with Python)*

4.  **Create `.env` file for AI Coach:**
    * In the project root directory, create a file named `.env`.
    * Add your Google API Key to this file:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```
        Replace `"YOUR_API_KEY_HERE"` with your actual Gemini API key.

5.  **Ensure GIF files are present:** Verify that the `GIFs` folder exists in the project root and contains the necessary `.gif` files (`bicep.gif`, `squats.gif`, etc.).

## Usage

1.  **Activate the virtual environment** (if not already active).

2.  **Run the main application script:**
    ```bash
    python fitness_tracker_ui.py
    ```

3.  **Follow the on-screen UI:**
    * **Home Screen:**
        * Select or Create a User Profile. Profile selection is required before starting a webcam workout or using the AI Coach.
        * View statistics for the selected user ("View Stats").
        * Access the "AI Coach" for fitness advice (requires profile and API key).
        * Choose workout source: "Start Webcam Workout" (full features) or "Load Video (No Stats)" (limited features).
    * **(Webcam Flow):**
        * **Exercise Select:** Choose the desired exercise.
        * **Set Configuration:** Adjust the number of Sets, Reps per Set, and Rest Time using the +/- buttons. Click "Confirm & Start". Or choose "Start Free Play" to skip set configuration and guides.
        * **Guide (if not Free Play):** A GIF demonstrates the exercise. Click "Start Exercise" or wait.
        * **Tracking:** Perform the exercise for the current set (or indefinitely in Free Play). View reps, set number (if applicable), stage (UP/DOWN), and form feedback. Switch exercises (resets set config/ends free play for that exercise) or go Home (ends session, saves stats).
        * **Rest (if not Free Play):** After completing reps for a set (if not the last set), a rest timer starts. You can skip the rest or wait for it to finish.
        * The cycle repeats for all sets. Session stats are saved upon returning Home or finishing all sets.
    * **(Video File Flow):**
        * **Exercise Select:** Choose the exercise matching the video content.
        * **Tracking:** View basic rep counting and form feedback for the duration of the video. No sets, rest periods, AI coach, or statistics are tracked/saved.
    * **Stats Screen:** Displays charts for calorie and time distribution, and a summary of exercise stats and recent sessions (if data exists for the user). Go "Back to Home".
    * **AI Coach Screen:**
        * Type your fitness-related questions in the input field.
        * The AI will use your profile and saved statistics to provide personalized responses.
        * Scroll through the chat history.
        * Return "Back" to the Home screen.
    * Press 'Q' on your keyboard at any time to quit the application. Data is typically saved when ending a webcam session cleanly via the Home button or completing all sets.

## Configuration

Parameters for exercise detection, form feedback, MET values, AI Coach, and UI appearance are defined as constants near the top of the `fitness_tracker_ui.py` script.

* **Rep Counting Thresholds:** Adjust `*_ENTER_ANGLE`, `*_EXIT_ANGLE` constants (e.g., `BICEP_UP_ENTER_ANGLE`).
* **Form Correction Thresholds:** Modify `BACK_ANGLE_THRESHOLD_*`, `PUSHUP_BODY_STRAIGHT_*`, `SQUAT_KNEE_VALGUS_THRESHOLD`, `BICEP_UPPER_ARM_VERT_DEVIATION`, etc.
* **EMA Smoothing:** `EMA_ALPHA` controls the smoothing factor for angle calculations.
* **Set/Rep/Rest Defaults:** Change `target_sets`, `target_reps_per_set`, `target_rest_time` initial values.
* **Data Files:** `PROFILES_FILE`, `STATS_FILE` specify the names for saved data.
* **MET Values:** `MET_VALUES` dictionary maps exercises to Metabolic Equivalent Task values for calorie estimation.
* **AI Coach:**
    * `GEMINI_MODEL_NAME` specifies the Gemini model to use.
    * `GOOGLE_API_KEY_ENV_VAR` defines the environment variable name for the API key.
    * `sys_prompt` contains the system prompt for the AI Coach, defining its role and capabilities.
* **UI:** Colors, fonts, layout constants (`COLORS`, `FONT`, various `_SCALE` and `_MARGIN` constants) can be tweaked.

## Acknowledgements

* **MediaPipe** by Google for the powerful pose estimation framework.
* **OpenCV** for the versatile computer vision library.
* **Google Gemini** for the generative AI model powering the AI Coach.
