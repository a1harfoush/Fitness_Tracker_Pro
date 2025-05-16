# --- Imports ---
import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk # Added ttk
import platform
import traceback
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting to buffer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io # For plot rendering to buffer
import imageio # For GIF loading
import google.generativeai as genai # For Chatbot
from dotenv import load_dotenv # For .env file
from collections import defaultdict # For stats aggregation

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration Constants ---

# Rep Counting Thresholds (Incorporating Hysteresis) - FROM fitness_tracker.py
# Format: {EXERCISE}_{STATE}_{ACTION/LIMIT}_{ANGLE/CONDITION}
# ENTER = Threshold to enter the state
# EXIT = Threshold to go back from the state (making it harder to oscillate)

# Bicep Curl
BICEP_UP_ENTER_ANGLE = 55       # Angle to enter 'UP' state (contracted)
BICEP_UP_EXIT_ANGLE = 70        # Angle to exit 'UP' state
BICEP_DOWN_ENTER_ANGLE = 155    # Angle to enter 'DOWN' state (extended)
BICEP_DOWN_EXIT_ANGLE = 140     # Angle to exit 'DOWN' state

# Squat
SQUAT_UP_ENTER_ANGLE = 165      # Angle to enter 'UP' state (standing)
SQUAT_UP_EXIT_ANGLE = 155       # Angle to exit 'UP' state
SQUAT_DOWN_ENTER_ANGLE = 100    # Angle to enter 'DOWN' state (squatted)
SQUAT_DOWN_EXIT_ANGLE = 110     # Angle to exit 'DOWN' state

# Push Up
PUSHUP_UP_ENTER_ANGLE = 155     # Angle to enter 'UP' state (extended arms)
PUSHUP_UP_EXIT_ANGLE = 145      # Angle to exit 'UP' state
PUSHUP_DOWN_ENTER_ANGLE = 95    # Angle to enter 'DOWN' state (chest low)
PUSHUP_DOWN_EXIT_ANGLE = 105    # Angle to exit 'DOWN' state

# Pull Up
PULLUP_UP_ENTER_ELBOW_ANGLE = 80 # Elbow angle to enter 'UP' state
PULLUP_UP_EXIT_ELBOW_ANGLE = 95  # Elbow angle to exit 'UP' state
PULLUP_DOWN_ENTER_ANGLE = 160    # Elbow angle to enter 'DOWN' state
PULLUP_DOWN_EXIT_ANGLE = 150    # Elbow angle to exit 'DOWN' state
PULLUP_CHIN_ABOVE_WRIST = True   # Keep this simple check for UP state confirmation

# Deadlift
DEADLIFT_UP_ENTER_ANGLE = 168   # Hip/Knee angle to enter 'UP' state (lockout)
DEADLIFT_UP_EXIT_ANGLE = 158    # Hip/Knee angle to exit 'UP' state
DEADLIFT_DOWN_ENTER_HIP_ANGLE = 120 # Hip angle to enter 'DOWN' state (bottom)
DEADLIFT_DOWN_ENTER_KNEE_ANGLE = 135 # Knee angle to enter 'DOWN' state
DEADLIFT_DOWN_EXIT_HIP_ANGLE = 130   # Hip angle to exit 'DOWN' state
DEADLIFT_DOWN_EXIT_KNEE_ANGLE = 145 # Knee angle to exit 'DOWN' state

# --- Form Correction Thresholds --- FROM fitness_tracker.py
# Back Posture (Deviation from Vertical UP - degrees)
BACK_ANGLE_THRESHOLD_BICEP = 20       # Stricter for Bicep Curls
BACK_ANGLE_THRESHOLD_SQUAT = 45       # Max lean during squat
BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT = 15 # Very strict at lockout
BACK_ANGLE_THRESHOLD_DEADLIFT_LIFT = 60   # Max lean during the lift itself (approx)

# Pushup Specific
PUSHUP_BODY_STRAIGHT_MIN = 150 # Min angle Shoulder-Hip-Knee (actually Shoulder-Hip-Ankle in merged)
PUSHUP_BODY_STRAIGHT_MAX = 190 # Max angle Shoulder-Hip-Knee (actually Shoulder-Hip-Ankle in merged)

# Squat Specific
SQUAT_KNEE_VALGUS_THRESHOLD = 0.05 # Max normalized horizontal distance diff (knee vs ankle) - adjust!
SQUAT_CHEST_FORWARD_THRESHOLD = 0.1 # Max normalized horizontal distance (shoulder ahead of knee) - adjust!

# Bicep Specific
BICEP_UPPER_ARM_VERT_DEVIATION = 25 # Max degrees deviation from vertical down

# --- EMA Smoothing --- FROM fitness_tracker.py
EMA_ALPHA = 0.3 # Smoothing factor (higher means less smoothing, faster response)

# Statistics Constants (FROM Fitness_Tracker_ui.py, retained)
PROFILES_FILE = 'profiles.json'; STATS_FILE = 'stats.json'
MET_VALUES = {"BICEP CURL": 3.5, "SQUAT": 5.0, "PUSH UP": 4.0, "PULL UP": 8.0, "DEADLIFT": 6.0, "DEFAULT": 4.0}
STATS_SET_KEYS = ['last_config_sets', 'last_config_reps', 'last_config_rest']
TOTAL_TIME_MINUTES_KEY = 'total_active_time_minutes'

# --- UI Constants --- (FROM Fitness_Tracker_ui.py, retained and primary)
GIF_DIR = "GIFs"
EXERCISES = ["BICEP CURL", "SQUAT", "PUSH UP", "PULL UP", "DEADLIFT"]
EXERCISE_GIF_MAP = {
    "BICEP CURL": "bicep.gif", "SQUAT": "squats.gif", "PUSH UP": "pushup.gif",
    "PULL UP": "pullup.gif", "DEADLIFT": "deadlift.gif"
}
GENDERS = ["Male", "Female", "Other/Prefer not to say"]

COLORS = { # Merged, Fitness_Tracker_ui.py prioritized for UI
    "background": (245, 245, 245), "primary_text": (20, 20, 20),
    "secondary_text": (100, 100, 100), "accent_blue": (0, 122, 255),
    "accent_green": (52, 199, 89), "accent_red": (255, 59, 48),
    "accent_orange": (255, 149, 0), "accent_purple": (175, 82, 222),
    
    "chat_user_bubble_bg": (0, 122, 255), "chat_user_bubble_text": (255, 255, 255),
    "chat_ai_bubble_bg": (229, 229, 234), "chat_ai_bubble_text": (20, 20, 20),
    "chat_ai_text_color": (88, 86, 214), # Retained, but bubble colors preferred

    "button_bg_normal": (229, 229, 234), "button_bg_active": (0, 122, 255),
    "button_bg_profile": (88, 86, 214), "button_bg_stats": (255, 149, 0),
    "button_bg_freeplay": (255, 149, 0), "button_bg_chat": (175, 82, 222),
    "button_text_normal": (20, 20, 20), "button_text_active": (255, 255, 255),
    "overlay_bg": (30, 30, 30, 200), # Darker overlay from Fitness_Tracker_ui.py for tracking screen
    "landmark_vis": (52, 199, 89),    # Good landmark color
    "landmark_issue": (255, 59, 48),  # Bad landmark color
    "connection": (142, 142, 147),    # Default connection color
    "profile_text": (88, 86, 214), "timer_text": (0, 122, 255),
    "plot_text_color": (20,20,20), "plot_bg_color": (240,240,240)
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 1.6; SELECT_TITLE_SCALE = 1.1; BUTTON_TEXT_SCALE = 0.65
STATUS_TEXT_SCALE = 0.6; REP_TEXT_SCALE = 1.4; FEEDBACK_TEXT_SCALE = 0.7
LARGE_TIMER_SCALE = 3.0; STATS_TEXT_SCALE = 0.5; STATS_TITLE_SCALE = 0.7
LINE_THICKNESS = 2; BUTTON_HEIGHT = 55; BUTTON_MARGIN = 20
CORNER_RADIUS = 15; OVERLAY_ALPHA = 0.85; PLUS_MINUS_BTN_SIZE = 40
CHAT_TEXT_THICKNESS = 1

# --- Chatbot UI Constants ---
CHAT_TEXT_SCALE = 0.55
CHAT_LINE_SPACING_FACTOR = 1.6
INTER_MESSAGE_PADDING_FACTOR = 0.7
CHAT_INPUT_HEIGHT = 45; CHAT_INPUT_MARGIN = 10
CHAT_INPUT_BG_COLOR = (220, 220, 220); CHAT_INPUT_TEXT_COLOR = COLORS["primary_text"]
CHAT_PLACEHOLDER_COLOR = COLORS["secondary_text"]; CHAT_CURSOR_COLOR = COLORS["primary_text"]
CHAT_SCROLL_BUTTON_SIZE = 35
CHAT_SCROLL_AREA_PADDING = 10
CHAT_MESSAGE_START_X_OFFSET = 15
CHAT_BUBBLE_PADDING_X = 12
CHAT_BUBBLE_PADDING_Y = 8
CHAT_BUBBLE_MAX_WIDTH_FACTOR = 0.75
CHAT_BUBBLE_CORNER_RADIUS = 10

# --- Chatbot Core Constants ---
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
MAX_CHAT_HISTORY_PAIRS = 7
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
sys_prompt = """
You are 'Fitness Tracker Pro AI Coach', a specialized and friendly AI assistant integrated into the 'Fitness Tracker Pro' application. Your primary role is to help users with their fitness journey by providing personalized advice, motivation, and clear explanations.

You have access to the following user information:
User Profile: Username, Age, Gender, Height (cm), Weight (kg).
Workout Statistics (Overall & Per Exercise):
Total accumulated repetitions.
Total estimated calories burned (calculated using MET values, user weight, and active exercise duration).
Total active time spent per exercise.
Last used workout configuration (sets, reps, rest time) for each exercise.
Recent Workout Sessions: A summary of recent sessions including date, overall duration, specific exercises performed, and the duration spent on each exercise within those sessions.
Current Activity (if applicable): The user's currently selected exercise or workout segment within the app.

Use all provided data to give informed and personalized responses.

Your expertise and main goals include:
Answering questions about the user's tracked progress, statistics, and workout history.
Helping users interpret their stats and understand their progress.
Explaining common exercises (such as bicep curls, squats, push-ups, pull-ups, deadlifts), including general form guidance (e.g., 'for squats, aim to keep your chest up and back straight') and the primary muscles worked.
Discussing basic workout principles like progressive overload, the importance of rest and recovery, consistency, and how to select exercises for different fitness goals.
Providing motivation, encouragement, and positive reinforcement.
Offering general, actionable tips on how to improve based on their data (e.g., 'I see you've increased your squat reps by 5 since last week, great job! To continue progressing, you could consider...').
Discussing basic fitness-related nutritional concepts, such as the role of protein for muscle repair and the importance of hydration during workouts.

Important Limitations & Safety:
Crucially, do NOT give specific medical advice, diagnose injuries, or create specific meal plans or dietary prescriptions.
If a question is outside your fitness expertise (e.g., medical diagnosis, financial advice, complex non-fitness topics), politely state that you are specialized in fitness and exercise guidance and cannot answer that specific query. You can then offer to help with a fitness-related question instead.
Always prioritize safety and responsible fitness practices in your advice. If a user describes poor form or risky behavior, gently guide them towards safer alternatives.

When responding:
Be encouraging, positive, empathetic, and clear.
Keep responses relatively concise and actionable where possible.
Refer to yourself as 'AI Coach' or 'your fitness assistant'.
Important Formatting Constraint: Do NOT use simple list markers like '-' or '*' in your responses. Instead, integrate lists naturally into sentences or use numbered lists if absolutely necessary and appropriate for clarity.
"""

# --- Mediapipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Global Variables (Mostly from Fitness_Tracker_ui.py, core logic vars integrated) ---
app_mode = "HOME"
source_type = None; cap = None; video_source_selected = False; is_webcam_source = False
try:
    tk_root_main = tk.Tk(); tk_root_main.withdraw()
    default_win_width, default_win_height = 1280, 720
    target_win_width, target_win_height = default_win_width, default_win_height
except Exception:
    default_win_width, default_win_height = 1280, 720
    target_win_width, target_win_height = default_win_width, default_win_height
    tk_root_main = None
actual_win_width, actual_win_height = target_win_width, target_win_height
canvas = None; last_frame_for_rest = None
current_exercise = EXERCISES[0]
counter, stage = 0, None; counter_left, counter_right = 0, 0; stage_left, stage_right = None, None # Core logic vars
feedback_list = ["Select a profile or create a new one."]
last_rep_time, last_rep_time_left, last_rep_time_right = 0, 0, 0 # Core logic vars
rep_cooldown = 0.5; form_correct_overall = True; form_issues_details = set(); ema_angles = {} # Core logic vars
target_sets = 3; target_reps_per_set = 10; target_rest_time = 30
current_set_number = 1; rest_start_time = None; set_config_confirmed = False
current_user = None; user_profiles = {}; user_stats = {}

session_start_time = None
session_reps = {} # Reps per exercise THIS session
session_exercise_start_time = None
session_exercise_durations = defaultdict(float)

guide_gif_frames = []; guide_gif_reader = None; guide_gif_index = 0
guide_last_frame_time = 0; guide_frame_delay = 0.1; guide_start_time = 0; guide_duration = 5

stats_pie_image = None
stats_time_plot_image = None

gemini_model = None; gemini_chat_session = None; chat_messages = []
is_llm_thinking = False; last_chat_error = None; chat_input_text = ""
chat_input_active = False; chat_scroll_offset_y = 0
chat_total_content_height = 0; chat_visible_area_height = 0


# --- Helper Functions ---

# EMA Calculation - MERGED FROM fitness_tracker.py (explicit init)
def update_ema(current_value, key, storage_dict):
    if not isinstance(current_value, (int, float)):
        return current_value # Cannot smooth non-numeric types

    if key not in storage_dict or storage_dict[key] is None:
        storage_dict[key] = float(current_value) # Initialize as float
    else:
        prev_ema = storage_dict[key]
        storage_dict[key] = EMA_ALPHA * float(current_value) + (1 - EMA_ALPHA) * prev_ema
    return storage_dict[key]

# Angle Calculation - MERGED FROM fitness_tracker.py (more robust)
def calculate_angle(a, b, c, use_3d=False):
    """Calculates angle between three points (2D or 3D), using raw coords."""
    # Input coords format: [x, y, z, visibility]
    if not all(coord[3] > 0.1 for coord in [a,b,c]): # Check visibility
        return 0 # Return 0 if any point is barely visible

    a_np, b_np, c_np = np.array(a[:3]), np.array(b[:3]), np.array(c[:3]) # Use first 3 (x,y,z)

    dims = 3 if use_3d else 2
    a_np, b_np, c_np = a_np[:dims], b_np[:dims], c_np[:dims]

    vec_ba = a_np - b_np
    vec_bc = c_np - b_np

    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6: return 0 # Avoid division by zero (was 0 in original, using small epsilon)

    dot_product = np.dot(vec_ba, vec_bc)
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return int(angle)

# Coordinate Retrieval - MERGED FROM fitness_tracker.py (raw coords + visibility)
def get_coords(landmarks, landmark_name):
    """Safely retrieves landmark coordinates and visibility. Uses RAW coordinates."""
    # landmarks is expected to be results.pose_landmarks.landmark (the list of landmarks)
    try:
        lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]
        return [lm.x, lm.y, lm.z, lm.visibility]
    except Exception:
        return [0, 0, 0, 0] # Return neutral value with 0 visibility

# Segment Vertical Angle - MERGED FROM fitness_tracker.py (specific use for bicep)
def get_segment_vertical_angle(p1_coords, p2_coords):
     """Calculates the angle of the segment p1->p2 relative to vertical DOWN (degrees)."""
     if p1_coords[3] < 0.5 or p2_coords[3] < 0.5: return None # Need visible points
     # Use only x, y for 2D angle calculation
     vec = np.array(p2_coords[:2]) - np.array(p1_coords[:2]) # Vector p1 -> p2
     norm = np.linalg.norm(vec)
     if norm < 1e-6: return None # Was 0 in original

     # Vertical vector pointing DOWN on screen is (0, 1)
     vec_vert_down = np.array([0, 1])
     dot_prod = np.dot(vec, vec_vert_down)

     # Calculate angle in degrees
     angle_rad = np.arccos(np.clip(dot_prod / norm, -1.0, 1.0))
     angle_deg = np.degrees(angle_rad)
     # Angle near 0 means pointing down, near 180 means pointing up
     return angle_deg

# Feedback and Form Issue Adders (FROM Fitness_Tracker_ui.py, retained)
def add_feedback(msg,is_warn=False):
    global form_correct_overall,feedback_list; prefix="WARN: " if is_warn else "INFO: "; full=prefix+msg
    if full not in feedback_list: feedback_list.append(full)
    if is_warn: form_correct_overall=False
def add_form_issue(part): form_issues_details.add(part)


# --- BMR Calculation (FROM Fitness_Tracker_ui.py, retained) ---
def calculate_bmr(profile):
    weight = profile.get("weight", 0); height = profile.get("height", 0); age = profile.get("age", 0); gender = profile.get("gender", GENDERS[-1])
    if not all([weight > 0, height > 0, age > 0]): return 0
    if gender == "Male": bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender == "Female": bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else: bmr = (10 * weight) + (6.25 * height) - (5 * age) - 78 # Average approximation
    return max(0, bmr)

# --- Data I/O Functions (FROM Fitness_Tracker_ui.py, retained) ---
def load_data():
    global user_profiles, user_stats
    try:
        if os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE,'r')as f:user_profiles=json.load(f)
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE,'r')as f:user_stats=json.load(f)
        for username, profile in user_profiles.items():
            if 'gender' not in profile: profile['gender'] = GENDERS[-1]
    except Exception as e: print(f"Error loading data: {e}")
def save_data():
    try:
        with open(PROFILES_FILE,'w')as f:json.dump(user_profiles,f,indent=4)
        with open(STATS_FILE,'w')as f:json.dump(user_stats,f,indent=4)
    except Exception as e: print(f"Error saving data: {e}")

# --- Profile Management Popups (FROM Fitness_Tracker_ui.py, retained) ---
def create_profile_popup():
    global user_profiles, user_stats, current_user, feedback_list, stats_pie_image, stats_time_plot_image
    if not tk_root_main: print("Error: Tkinter root not available for popup."); return
    popup = tk.Toplevel(tk_root_main); popup.title("Create New Profile"); popup.geometry("350x300");
    popup.attributes('-topmost', True); popup.resizable(False, False)
    frame = tk.Frame(popup, padx=10, pady=10); frame.pack(expand=True, fill="both")
    tk.Label(frame, text="Username:", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew"); username_entry = tk.Entry(frame); username_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(frame, text="Age:", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew"); age_entry = tk.Entry(frame); age_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(frame, text="Height (cm):", anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="ew"); height_entry = tk.Entry(frame); height_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(frame, text="Weight (kg):", anchor="w").grid(row=3, column=0, padx=5, pady=5, sticky="ew"); weight_entry = tk.Entry(frame); weight_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(frame, text="Gender:", anchor="w").grid(row=4, column=0, padx=5, pady=5, sticky="ew")
    gender_var = tk.StringVar(popup); gender_var.set(GENDERS[0])
    gender_dropdown = ttk.Combobox(frame, textvariable=gender_var, values=GENDERS, state="readonly"); gender_dropdown.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
    frame.columnconfigure(1, weight=1)
    def submit_profile():
        global current_user, feedback_list, stats_pie_image, stats_time_plot_image
        username = username_entry.get().strip(); age_str = age_entry.get().strip(); height_str = height_entry.get().strip(); weight_str = weight_entry.get().strip(); gender = gender_var.get()
        if not username: messagebox.showerror("Input Error", "Username cannot be empty.", parent=popup); return
        if username in user_profiles: messagebox.showerror("Input Error", "Username already exists.", parent=popup); return
        if not gender or gender not in GENDERS: messagebox.showerror("Input Error", "Please select a valid gender.", parent=popup); return
        try:
            age = int(age_str) if age_str else 0; height = float(height_str) if height_str else 0.0; weight = float(weight_str) if weight_str else 0.0
            if age < 0 or height < 0 or weight < 0: raise ValueError("Values cannot be negative.")
        except ValueError: messagebox.showerror("Input Error", "Age, Height, Weight must be valid positive numbers.", parent=popup); return
        user_profiles[username] = {"age": age, "height": height, "weight": weight, "gender": gender}
        if username not in user_stats: user_stats[username] = {}
        if current_user != username: reset_chat_related_state(); stats_pie_image = None; stats_time_plot_image = None
        current_user = username; print(f"Profile created for {username} ({gender}). Set as current user."); save_data()
        feedback_list = [f"Welcome, {current_user}! Select workout source."]; popup.destroy()
    submit_button = tk.Button(frame, text="Create & Select", command=submit_profile, width=15); submit_button.grid(row=5, column=0, columnspan=2, pady=15)
    username_entry.focus_set()

def select_profile_popup():
    global current_user, feedback_list, stats_pie_image, stats_time_plot_image
    if not tk_root_main: return
    if not user_profiles: messagebox.showinfo("No Profiles", "No profiles found. Create one.", parent=tk_root_main); create_profile_popup(); return
    popup = tk.Toplevel(tk_root_main); popup.title("Select Profile"); popup.geometry("300x150"); popup.attributes('-topmost', True); popup.resizable(False, False)
    frame = tk.Frame(popup, padx=10, pady=10); frame.pack(expand=True, fill="both"); tk.Label(frame, text="Select User:").pack(pady=5)
    selected_user_tk = tk.StringVar(popup); profile_options = list(user_profiles.keys())
    if current_user and current_user in profile_options: selected_user_tk.set(current_user)
    elif profile_options: selected_user_tk.set(profile_options[0])
    else: popup.destroy(); return
    profile_menu = tk.OptionMenu(frame, selected_user_tk, *profile_options); profile_menu.pack(pady=10, fill="x")
    def submit_selection():
        global current_user, feedback_list, stats_pie_image, stats_time_plot_image
        chosen_user = selected_user_tk.get()
        if chosen_user in user_profiles:
            if current_user != chosen_user: reset_chat_related_state(); stats_pie_image = None; stats_time_plot_image = None
            current_user = chosen_user; feedback_list = [f"Welcome back, {current_user}! Select workout source."]; popup.destroy()
    select_button = tk.Button(frame, text="Select", command=submit_selection, width=10); select_button.pack(pady=10)

# --- Statistics Generation (FROM Fitness_Tracker_ui.py, retained) ---
def generate_stats_pie_image(target_w, target_h):
    global current_user, user_stats
    bg_color_rgb = tuple(np.array(COLORS['plot_bg_color']) / 255.0); text_color_rgb = tuple(np.array(COLORS['plot_text_color']) / 255.0)
    if not current_user or current_user not in user_stats:
        img = np.full((target_h, target_w, 3), COLORS['plot_bg_color'], dtype=np.uint8); msg = "No User Selected or No Stats"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 1); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.7, COLORS['plot_text_color'], 1, cv2.LINE_AA); return img
    stats = user_stats[current_user]
    if not stats:
        img = np.full((target_h, target_w, 3), COLORS['plot_bg_color'], dtype=np.uint8); msg = f"No Stats Recorded for {current_user}"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 1); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.7, COLORS['plot_text_color'], 1, cv2.LINE_AA); return img
    labels, calories = [], []; total_calories = 0.0
    for exercise, data in stats.items():
        if isinstance(data, dict):
            cal = data.get("total_calories", 0.0); reps = data.get("total_reps", 0)
            if cal > 0 or reps > 0: labels.append(exercise); calories.append(cal if cal > 0 else 0.01); total_calories += cal
    if not calories:
        img = np.full((target_h, target_w, 3), COLORS['plot_bg_color'], dtype=np.uint8); msg = f"No Calories Recorded for {current_user}"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 1); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.7, COLORS['plot_text_color'], 1, cv2.LINE_AA); return img
    try: plt.style.use('seaborn-v0_8-pastel')
    except OSError: plt.style.use('default')
    fig, ax = plt.subplots(figsize=(target_w / 100, target_h / 100), dpi=100); fig.patch.set_facecolor(bg_color_rgb); ax.set_facecolor(bg_color_rgb)
    defined_color_names = ['accent_blue', 'accent_green', 'accent_orange', 'accent_purple', 'accent_red']; pie_colors_from_names = [tuple(np.array(COLORS[c_name][::-1]) / 255.0) for c_name in defined_color_names]; custom_gray_bgr_tuple = (80, 80, 80); custom_gray_rgb_float = tuple(np.array(custom_gray_bgr_tuple)[::-1] / 255.0); available_plot_colors = pie_colors_from_names + [custom_gray_rgb_float]; num_calories_slices = len(calories); pie_colors_final = [available_plot_colors[i % len(available_plot_colors)] for i in range(num_calories_slices)]
    wedges, texts, autotexts = ax.pie(calories, autopct='%1.1f%%', startangle=90, pctdistance=0.80, colors=pie_colors_final, wedgeprops=dict(width=0.4, edgecolor='w'))
    for autotext in autotexts: autotext.set_color('white'); autotext.set_fontsize(8); autotext.set_weight('bold')
    ax.set_title(f'Calorie Distribution\nTotal: {total_calories:.1f} kcal', fontsize=10, color=text_color_rgb, pad=10); ax.axis('equal'); plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf); buf.seek(0); img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8); buf.close(); plt.close(fig); img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.shape[0] != target_h or img_bgr.shape[1] != target_w:
        if img_bgr is not None and img_bgr.shape[0] > 0 and img_bgr.shape[1] > 0: img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else: img = np.full((target_h, target_w, 3), COLORS['plot_bg_color'], dtype=np.uint8); msg = "Chart Error"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 1); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2), FONT, 0.7, COLORS['plot_text_color'], 1, cv2.LINE_AA); return img
    return img_bgr

def generate_time_plot_image(target_w, target_h):
    global current_user, user_stats
    bg_color_rgb = tuple(np.array(COLORS['plot_bg_color']) / 255.0); text_color_rgb = tuple(np.array(COLORS['plot_text_color']) / 255.0)
    if not current_user or current_user not in user_stats:
        img = np.full((target_h, target_w, 3), COLORS['plot_bg_color'], dtype=np.uint8); msg = "No User for Time Plot"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 1); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.7, COLORS['plot_text_color'], 1, cv2.LINE_AA); return img
    exercise_times = {}
    for ex_name in EXERCISES: data = user_stats[current_user].get(ex_name, {});  (exercise_times.update({ex_name: data.get(TOTAL_TIME_MINUTES_KEY,0.0)}) if isinstance(data,dict) else None)
    valid_exercises = {ex: time for ex, time in exercise_times.items() if time > 0}
    if not valid_exercises:
        img = np.full((target_h, target_w, 3), COLORS['plot_bg_color'], dtype=np.uint8); msg = f"No Time Recorded for {current_user}"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 1); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.7, COLORS['plot_text_color'], 1, cv2.LINE_AA); return img
    labels = list(valid_exercises.keys()); times = list(valid_exercises.values())
    try: plt.style.use('seaborn-v0_8-pastel')
    except OSError: plt.style.use('default')
    fig, ax = plt.subplots(figsize=(target_w / 100, target_h / 100), dpi=100); fig.patch.set_facecolor(bg_color_rgb); ax.set_facecolor(bg_color_rgb)
    bar_colors_def = ['accent_green', 'accent_orange', 'accent_purple', 'accent_red', 'accent_blue']; bar_colors = [tuple(np.array(COLORS[c_name][::-1]) / 255.0) for c_name in bar_colors_def]; final_bar_colors = [bar_colors[i % len(bar_colors)] for i in range(len(labels))]
    bars = ax.bar(labels, times, color=final_bar_colors); ax.set_ylabel('Total Time (minutes)', fontsize=9, color=text_color_rgb); ax.set_title('Total Active Time per Exercise', fontsize=10, color=text_color_rgb, pad=10); ax.tick_params(axis='x', rotation=15, labelsize=8, colors=text_color_rgb); ax.tick_params(axis='y', labelsize=8, colors=text_color_rgb); ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars: yval = bar.get_height(); (ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=7, color=text_color_rgb) if yval > 0 else None)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO(); FigureCanvas(fig).print_png(buf); buf.seek(0); img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8); buf.close(); plt.close(fig); img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.shape[0] != target_h or img_bgr.shape[1] != target_w:
        if img_bgr is not None and img_bgr.shape[0] > 0 and img_bgr.shape[1] > 0: img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else: img = np.full((target_h, target_w, 3), COLORS['plot_bg_color'], dtype=np.uint8); msg = "Chart Error"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 1); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2), FONT, 0.7, COLORS['plot_text_color'], 1, cv2.LINE_AA); return img
    return img_bgr

# --- Drawing Helper Functions (FROM Fitness_Tracker_ui.py, retained) ---
def draw_rounded_rectangle(img,pt1,pt2,color,thick,r):
    x1,y1=pt1;x2,y2=pt2;r=max(0,min(r,abs(x2-x1)//2,abs(y2-y1)//2))
    x1c,x2c=min(x1,x2),max(x1,x2);y1c,y2c=min(y1,y2),max(y1,y2)
    x1c,y1c=max(0,x1c),max(0,y1c);x2c,y2c=min(img.shape[1]-1,x2c),min(img.shape[0]-1,y2c)
    if x1c>=x2c or y1c>=y2c: return
    if thick<0:
        cv2.rectangle(img,(x1c+r,y1c),(x2c-r,y2c),color,-1);cv2.rectangle(img,(x1c,y1c+r),(x2c,y2c-r),color,-1)
        cv2.circle(img,(x1c+r,y1c+r),r,color,-1);cv2.circle(img,(x2c-r,y1c+r),r,color,-1);cv2.circle(img,(x2c-r,y2c-r),r,color,-1);cv2.circle(img,(x1c+r,y2c-r),r,color,-1)
    elif thick>0:
        cv2.ellipse(img,(x1c+r,y1c+r),(r,r),180,0,90,color,thick);cv2.ellipse(img,(x2c-r,y1c+r),(r,r),270,0,90,color,thick);cv2.ellipse(img,(x2c-r,y2c-r),(r,r),0,0,90,color,thick);cv2.ellipse(img,(x1c+r,y2c-r),(r,r),90,0,90,color,thick)
        if x1c+r<x2c-r:cv2.line(img,(x1c+r,y1c),(x2c-r,y1c),color,thick)
        if y1c+r<y2c-r:cv2.line(img,(x2c,y1c+r),(x2c,y2c-r),color,thick)
        if x1c+r<x2c-r:cv2.line(img,(x2c-r,y2c),(x1c+r,y2c),color,thick)
        if y1c+r<y2c-r:cv2.line(img,(x1c,y2c-r),(x1c,y1c+r),color,thick)
def draw_semi_transparent_rect(img,pt1,pt2,color_alpha):
    x1,y1=pt1;x2,y2=pt2;x1,x2=min(x1,x2),max(x1,x2);y1,y2=min(y1,y2),max(y1,y2)
    x1,y1=max(0,x1),max(0,y1);x2,y2=min(img.shape[1],x2),min(img.shape[0],y2)
    if x1>=x2 or y1>=y2: return
    sub=img[y1:y2,x1:x2];
    if sub.size==0: return
    color=np.array(color_alpha[:3],dtype=np.uint8);alpha=color_alpha[3]/255.0 if len(color_alpha)>3 else OVERLAY_ALPHA
    rect=np.full(sub.shape,color,dtype=np.uint8);res=cv2.addWeighted(sub,1.0-alpha,rect,alpha,0.0)
    img[y1:y2,x1:x2]=res

# --- GIF Loading Function (FROM Fitness_Tracker_ui.py, retained) ---
def load_guide_gif(ex_name):
    global guide_gif_frames,guide_gif_reader,guide_gif_index,guide_frame_delay,feedback_list; guide_gif_frames=[];guide_gif_index=0
    if not os.path.exists(GIF_DIR):
        try: os.makedirs(GIF_DIR); print(f"Created GIF directory at: {GIF_DIR}"); feedback_list = [f"Created GIF directory. Please add GIFs."]
        except OSError as e: print(f"Error creating GIF directory '{GIF_DIR}': {e}"); feedback_list = [f"Error creating GIF folder."]; return False
    if ex_name not in EXERCISE_GIF_MAP: print(f"Warning: No GIF mapping for {ex_name}"); return False
    gif_filename = EXERCISE_GIF_MAP[ex_name]; gif_path = os.path.join(GIF_DIR, gif_filename)
    if not os.path.exists(gif_path): print(f"Error: GIF file not found at {gif_path}"); feedback_list = [f"Guide GIF not found: {gif_filename}"]; return False
    try:
        guide_gif_reader = imageio.get_reader(gif_path)
        try: meta = guide_gif_reader.get_meta_data(); duration = meta.get('duration', 100); guide_frame_delay = duration / 1000.0; guide_frame_delay = np.clip(guide_frame_delay, 0.02, 1.0)
        except: guide_frame_delay = 0.1
        for frame in guide_gif_reader:
            if frame.ndim == 2: frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3: frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 4: frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else: continue
            guide_gif_frames.append(frame_bgr)
        if hasattr(guide_gif_reader, 'close'): guide_gif_reader.close()
        if not guide_gif_frames: print(f"Error: No frames loaded from GIF {gif_path}"); return False
        print(f"Loaded {len(guide_gif_frames)} frames for {ex_name} (delay: {guide_frame_delay:.3f}s)")
        return True
    except FileNotFoundError: print(f"Error: GIF file not found at {gif_path}"); feedback_list = [f"Guide GIF not found: {gif_filename}"]; return False
    except Exception as e: print(f"Error loading GIF {gif_path}: {e}"); traceback.print_exc(); feedback_list = [f"Error loading guide: {gif_filename}"]; guide_gif_reader = None; return False

# --- Chatbot Core Functions (FROM Fitness_Tracker_ui.py, retained) ---
def configure_gemini():
    global gemini_model, last_chat_error, gemini_chat_session
    last_chat_error = None; gemini_model = None; gemini_chat_session = None
    api_key = os.getenv(GOOGLE_API_KEY_ENV_VAR)
    if not api_key: last_chat_error = f"{GOOGLE_API_KEY_ENV_VAR} not set in .env file."; (messagebox.showerror("API Key Error", f"{last_chat_error} Chat disabled.", parent=tk_root_main) if tk_root_main and tk_root_main.winfo_exists() else None); print(f"ERROR: {last_chat_error}"); return False
    try: genai.configure(api_key=api_key); gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME); print(f"Gemini client configured: {GEMINI_MODEL_NAME}."); return True
    except Exception as e: last_chat_error = f"Gemini Init Error: {type(e).__name__}"; (messagebox.showerror("API Init Error", f"Failed to configure Gemini:\n{e}", parent=tk_root_main) if tk_root_main and tk_root_main.winfo_exists() else None); print(f"ERROR: Gemini config failed: {e}"); traceback.print_exc(); return False
def gather_context_for_llm(username):
    global user_profiles, user_stats, STATS_SET_KEYS
    if not username or username not in user_profiles: return "No user profile selected or found."
    profile_data=user_profiles.get(username,{}); stats_data=user_stats.get(username,{})
    gender_str = profile_data.get('gender', 'N/A'); gender_display = "Not Specified" if gender_str == GENDERS[-1] else gender_str
    context=f"User Profile for {username}:\n- Age: {profile_data.get('age','N/A')}\n- Gender: {gender_display}\n- Height (cm): {profile_data.get('height','N/A')}\n- Weight (kg): {profile_data.get('weight','N/A')}\n\n"
    if not stats_data: context+="User Statistics: No stats recorded yet.\n"
    else:
        context+="User Statistics Summary:\n"
        for ex,data in stats_data.items():
            if not isinstance(data, dict): continue
            reps,cals=data.get('total_reps',0),data.get('total_calories',0.0); time_min = data.get(TOTAL_TIME_MINUTES_KEY, 0.0); lcs,lcr,lcrt=data.get(STATS_SET_KEYS[0]),data.get(STATS_SET_KEYS[1]),data.get(STATS_SET_KEYS[2])
            cfg=f" (Last Config: {lcs}x{lcr}, {lcrt}s rest)" if lcs is not None else ""; time_str = f", {time_min:.1f} total mins" if time_min > 0 else ""; context+=f"- {ex}: {reps} total reps, {cals:.1f} total kcal{time_str}{cfg}\n"
        logged_sessions = stats_data.get('logged_sessions', [])
        if logged_sessions:
            context += f"\nRecent Workout Sessions ({len(logged_sessions)} total):\n"
            for i, session in enumerate(logged_sessions[-3:]): exercises_str = ", ".join(session.get('exercises_performed',[])); context += f"  - Session {len(logged_sessions)-i} ({session.get('date','N/A')}): {session.get('duration_minutes',0):.1f} mins. Exercises: {exercises_str}\n"
    return context.strip()
def get_llm_response(user_prompt_with_context_and_question):
    global last_chat_error, gemini_model, gemini_chat_session, chat_messages
    last_chat_error = None
    if not gemini_model and not configure_gemini(): return None
    try:
        if gemini_chat_session is None:
            print("Starting new Gemini chat session..."); api_history = []
            for msg in chat_messages:
                if msg['role'] == 'system': continue
                if msg['role'] == 'user': api_history.append({'role': 'user', 'parts': [{'text': msg['content']}]})
                elif msg['role'] == 'assistant': api_history.append({'role': 'model', 'parts': [{'text': msg['content']}]})
            history_for_init = api_history[:-1] if api_history and api_history[-1]['role'] == 'user' else api_history; start_idx = max(0, len(history_for_init) - (MAX_CHAT_HISTORY_PAIRS * 2)); limited_history_for_init = history_for_init[start_idx:]; gemini_chat_session = gemini_model.start_chat(history=limited_history_for_init)
        print(f"Sending message to Gemini: '{user_prompt_with_context_and_question[:150].replace(os.linesep,' ')}...'")
        response = gemini_chat_session.send_message(user_prompt_with_context_and_question)
        try: answer = response.text.strip(); print("Gemini Response Text Received."); return answer
        except ValueError:
            print(f"Warning: Gemini response.text extraction failed. Checking for block reasons.")
            if response.prompt_feedback and response.prompt_feedback.block_reason: last_chat_error = f"Blocked: {response.prompt_feedback.block_reason}"
            elif response.candidates and response.candidates[0].finish_reason != "STOP": last_chat_error = f"Finished: {response.candidates[0].finish_reason}"
            else: last_chat_error = "API Error: Invalid or empty response content."
            return None
        except Exception as e_text: print(f"ERROR: Gemini response text extraction: {e_text}"); last_chat_error = "API Error: Response text issue"; return None
    except Exception as e:
        err_str = str(e).lower()
        if any(s in err_str for s in ["api key", "permission", "authentication"]): last_chat_error = "Auth Error: API Key/Permissions?"; gemini_chat_session = None
        elif any(s in err_str for s in ["quota", "rate limit"]): last_chat_error = "Quota/Rate Limit."
        elif any(s in err_str for s in ["connection", "network", "dns"]): last_chat_error = "API Connection Error."
        elif "resource has been exhausted" in err_str: last_chat_error = "Resource Exhausted."
        else: last_chat_error = f"Gemini API Error: {type(e).__name__}"
        print(f"ERROR: Gemini chat: {e}"); traceback.print_exc()
        if "is not iterable" in err_str and "start_chat" in str(e.__traceback__): print("Hint: This might be due to an old version of google-generativeai. Try updating it.")
        return None

# --- Reset State Helper (FROM Fitness_Tracker_ui.py, core logic vars added) ---
def reset_exercise_state():
    global counter, stage, counter_left, counter_right, stage_left, stage_right
    global last_rep_time, last_rep_time_left, last_rep_time_right
    global form_correct_overall, form_issues_details, ema_angles
    global session_exercise_start_time, session_exercise_durations

    counter, counter_left, counter_right = 0, 0, 0
    stage, stage_left, stage_right = None, None, None
    ct = time.time(); last_rep_time,last_rep_time_left,last_rep_time_right = ct,ct,ct
    form_correct_overall = True; form_issues_details.clear(); ema_angles.clear() # Core logic resets
    session_exercise_start_time = None; session_exercise_durations = defaultdict(float)

def reset_chat_related_state():
    global chat_messages, is_llm_thinking, last_chat_error, chat_input_text, chat_input_active, chat_scroll_offset_y, gemini_chat_session
    chat_messages = []; is_llm_thinking = False; last_chat_error = None; chat_input_text = ""; chat_input_active = False; chat_scroll_offset_y = 0; gemini_chat_session = None; print("Chat state and session reset.")

def finalize_last_exercise_duration(exercise_name):
    global session_exercise_start_time, session_exercise_durations
    if exercise_name and session_exercise_start_time:
        duration_seconds = time.time() - session_exercise_start_time
        session_exercise_durations[exercise_name] += duration_seconds
        print(f"Finalized duration for {exercise_name}: {duration_seconds:.1f}s added. Total for ex: {session_exercise_durations[exercise_name]:.1f}s")
    session_exercise_start_time = None

# --- End Session Helper (FROM Fitness_Tracker_ui.py, retained) ---
def end_session():
    global session_start_time, session_reps, app_mode, cap, video_source_selected, is_webcam_source, feedback_list, current_user, source_type, target_sets, target_reps_per_set, target_rest_time, set_config_confirmed, stats_pie_image, stats_time_plot_image, session_exercise_start_time, session_exercise_durations
    print("Ending session...")
    finalize_last_exercise_duration(current_exercise)
    if current_user and session_start_time is not None and is_webcam_source:
        session_overall_duration_sec = time.time() - session_start_time; session_overall_duration_min = round(session_overall_duration_sec / 60.0, 1)
        if current_user not in user_stats: user_stats[current_user] = {}
        profile = user_profiles.get(current_user, {}); weight_kg = profile.get("weight", 0)
        total_session_reps_all_exercises = 0; active_exercises_in_session = list(session_exercise_durations.keys())
        for ex, duration_sec_for_ex in session_exercise_durations.items():
            reps_this_ex_this_session = session_reps.get(ex, 0); total_session_reps_all_exercises += reps_this_ex_this_session; ex_key_upper = ex.upper()
            if ex not in user_stats[current_user]: user_stats[current_user][ex] = {"total_reps": 0, "total_calories": 0.0, TOTAL_TIME_MINUTES_KEY: 0.0}
            user_stats[current_user][ex]["total_reps"] = user_stats[current_user][ex].get("total_reps", 0) + reps_this_ex_this_session
            duration_min_for_ex = duration_sec_for_ex / 60.0; user_stats[current_user][ex][TOTAL_TIME_MINUTES_KEY] = user_stats[current_user][ex].get(TOTAL_TIME_MINUTES_KEY, 0.0) + duration_min_for_ex
            if weight_kg > 0: met = MET_VALUES.get(ex_key_upper, MET_VALUES["DEFAULT"]); duration_hours_for_ex = duration_min_for_ex / 60.0; calories_for_ex_active_time = met * weight_kg * duration_hours_for_ex; user_stats[current_user][ex]["total_calories"] = user_stats[current_user][ex].get("total_calories", 0.0) + calories_for_ex_active_time
            if set_config_confirmed and reps_this_ex_this_session > 0: user_stats[current_user][ex][STATS_SET_KEYS[0]] = target_sets; user_stats[current_user][ex][STATS_SET_KEYS[1]] = target_reps_per_set; user_stats[current_user][ex][STATS_SET_KEYS[2]] = target_rest_time
            elif not set_config_confirmed and reps_this_ex_this_session > 0:
                for key_s in STATS_SET_KEYS:
                    if key_s in user_stats[current_user][ex]:
                        del user_stats[current_user][ex][key_s]
        if total_session_reps_all_exercises > 0 or active_exercises_in_session:
            session_log_entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "duration_minutes": session_overall_duration_min, "exercises_performed": active_exercises_in_session, "total_reps_all_exercises": total_session_reps_all_exercises, "exercise_durations_sec": dict(session_exercise_durations)}
            if 'logged_sessions' not in user_stats[current_user]: user_stats[current_user]['logged_sessions'] = []
            user_stats[current_user]['logged_sessions'].append(session_log_entry)
        save_data(); stats_pie_image = None; stats_time_plot_image = None
    session_start_time=None; session_reps={}; session_exercise_start_time = None; session_exercise_durations = defaultdict(float)
    app_mode="HOME"; (cap.release() if cap else None); cap=None; video_source_selected,is_webcam_source,source_type=False,False,None
    feedback_list=["Session Ended."]; feedback_list.append(f"Welcome back, {current_user}." if current_user else "Select or create a profile.")
    reset_exercise_state(); current_set_number = 1; set_config_confirmed = False

# --- Mouse Callback (FROM Fitness_Tracker_ui.py, retained structure) ---
def mouse_callback(event, x, y, flags, param):
    global app_mode, current_exercise, feedback_list, video_source_selected, cap, source_type, guide_start_time, current_user, session_start_time, session_reps, is_webcam_source, stats_pie_image, stats_time_plot_image, target_sets, target_reps_per_set, target_rest_time, set_config_confirmed, current_set_number, rest_start_time, chat_input_active, chat_input_text, is_llm_thinking, chat_messages, last_chat_error, chat_scroll_offset_y, chat_visible_area_height, chat_total_content_height, session_exercise_start_time, session_exercise_durations
    canvas_w = param.get('canvas_w', actual_win_width); canvas_h = param.get('canvas_h', actual_win_height)
    if event != cv2.EVENT_LBUTTONDOWN: return
    if app_mode == "HOME":
        num_top_btns = 4; btn_w_home = int(canvas_w * 0.18); btn_h_home = BUTTON_HEIGHT; gap_home = BUTTON_MARGIN // 2; total_top_btn_width = num_top_btns * btn_w_home + (num_top_btns - 1) * gap_home; start_x_profile = (canvas_w - total_top_btn_width) // 2; title_text_calc = "Fitness Tracker Pro"; (_, th_title_calc), _ = cv2.getTextSize(title_text_calc, FONT, TITLE_SCALE, LINE_THICKNESS + 1); ty_title_base = int(canvas_h * 0.1) + th_title_calc; user_text_example = f"User: {current_user}" if current_user else "User: None Selected"; (_, th_user_calc), _ = cv2.getTextSize(user_text_example, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); ty_user_calc = ty_title_base + th_title_calc + int(BUTTON_MARGIN * 0.75); profile_y_home = ty_user_calc + th_user_calc + BUTTON_MARGIN // 2; select_btn_x = start_x_profile; create_btn_x = select_btn_x + btn_w_home + gap_home; stats_btn_x = create_btn_x + btn_w_home + gap_home; chat_btn_x_home = stats_btn_x + btn_w_home + gap_home; src_btn_w, src_btn_h = int(canvas_w*0.35), int(BUTTON_HEIGHT*1.2); src_btn_x = (canvas_w-src_btn_w)//2; webcam_btn_y = profile_y_home + btn_h_home + BUTTON_MARGIN*2; video_btn_y = webcam_btn_y + src_btn_h + BUTTON_MARGIN
        if profile_y_home <= y <= profile_y_home + btn_h_home:
            if select_btn_x <= x <= select_btn_x + btn_w_home: select_profile_popup(); return
            if create_btn_x <= x <= create_btn_x + btn_w_home: create_profile_popup(); return
            if stats_btn_x <= x <= stats_btn_x + btn_w_home: stats_pie_image=None; stats_time_plot_image=None; app_mode="STATS"; return
            if chat_btn_x_home <= x <= chat_btn_x_home + btn_w_home:
                 if not current_user: messagebox.showwarning("Login Required", "Select/create profile for AI Chat.", parent=tk_root_main); return
                 app_mode = "CHAT"; chat_input_active = False; chat_scroll_offset_y = 0;
                 if not chat_messages or chat_messages[0].get('role') != 'system': chat_messages.insert(0, {"role": "system", "content": sys_prompt})
                 elif chat_messages[0].get('role') == 'system': chat_messages[0]['content'] = sys_prompt
                 if len(chat_messages) <= 1 or not any(msg['role'] == 'user' for msg in chat_messages): welcome_text = f"Coach: Hello {current_user}! How can I help you achieve your fitness goals today?"; (chat_messages.append({"role": "assistant", "content": welcome_text.split(': ',1)[1]}) if not chat_messages or (len(chat_messages) > 0 and chat_messages[-1].get('content') != welcome_text.split(': ',1)[1]) else None)
                 feedback_list = ["Ask the AI assistant. Type your question below."]; return
        elif webcam_btn_y <= y <= webcam_btn_y + src_btn_h and src_btn_x <= x <= src_btn_x + src_btn_w:
            if not current_user: messagebox.showwarning("Profile Needed","Select/create profile.",parent=tk_root_main);return
            cap=cv2.VideoCapture(0); cap=(cv2.VideoCapture(1) if not cap or not cap.isOpened() else cap)
            if cap and cap.isOpened(): source_type='webcam';is_webcam_source=True; session_start_time=time.time(); session_reps={}; session_exercise_durations = defaultdict(float); session_exercise_start_time = None; app_mode="EXERCISE_SELECT";feedback_list=["Select an exercise"]; reset_exercise_state(); current_set_number = 1; set_config_confirmed = False
            else: feedback_list=["Error: Webcam not found."];cap=None;is_webcam_source=False; return
        elif video_btn_y <= y <= video_btn_y + src_btn_h and src_btn_x <= x <= src_btn_x + src_btn_w:
            if not current_user: messagebox.showwarning("Profile Needed","Select/create profile.",parent=tk_root_main);return
            video_path=filedialog.askopenfilename(parent=tk_root_main,title="Select Video File",filetypes=[("Video Files","*.mp4 *.avi *.mov")])
            if video_path: cap=cv2.VideoCapture(video_path)
            if cap and cap.isOpened(): source_type='video';is_webcam_source=False;video_source_selected=True; session_start_time = None; session_reps={}; session_exercise_durations = defaultdict(float); session_exercise_start_time = None; app_mode="EXERCISE_SELECT";feedback_list=["Select an exercise (Video Mode)"]; reset_exercise_state(); current_set_number = 1; set_config_confirmed = False
            else: feedback_list=[f"Error opening video."];cap=None; return
    elif app_mode == "EXERCISE_SELECT":
         h_ex, w_ex = canvas_h, canvas_w; (_, th_title_ex), _ = cv2.getTextSize("Select Exercise", FONT, SELECT_TITLE_SCALE, LINE_THICKNESS + 1); ty_title_ex = BUTTON_MARGIN*3; item_h_ex = BUTTON_HEIGHT + BUTTON_MARGIN//2; list_h_ex = len(EXERCISES)*item_h_ex; start_y_ex = ty_title_ex + th_title_ex + BUTTON_MARGIN*2; btn_w_ex = int(w_ex*0.4); btn_x_ex = w_ex//2 - btn_w_ex//2; start_btn_w_ex, start_btn_h_ex = 200, BUTTON_HEIGHT; start_btn_x_ex = w_ex//2 - start_btn_w_ex//2; start_btn_y_ex = start_y_ex + list_h_ex + BUTTON_MARGIN; fp_btn_y_ex = start_btn_y_ex + start_btn_h_ex + BUTTON_MARGIN//2; back_btn_w_ex,back_btn_h_ex=150,BUTTON_HEIGHT;back_btn_x_ex=BUTTON_MARGIN*2;back_btn_y_ex=h_ex-back_btn_h_ex-BUTTON_MARGIN*2; clicked_ex_btn=False; previous_exercise = current_exercise
         for i,ex_name in enumerate(EXERCISES):
             if btn_x_ex<=x<=btn_x_ex+btn_w_ex and start_y_ex+i*item_h_ex<=y<=start_y_ex+i*item_h_ex+BUTTON_HEIGHT:
                 if current_exercise!=ex_name: finalize_last_exercise_duration(current_exercise); current_exercise=ex_name; reset_exercise_state(); session_exercise_start_time = None
                 clicked_ex_btn=True;break
         if not clicked_ex_btn and start_btn_x_ex<=x<=start_btn_x_ex+start_btn_w_ex and start_btn_y_ex<=y<=start_btn_y_ex+start_btn_h_ex:
             finalize_last_exercise_duration(previous_exercise); reset_exercise_state(); current_set_number = 1; app_mode=("SET_SELECTION" if source_type=='webcam' else "TRACKING"); (session_exercise_start_time := time.time() if app_mode=="TRACKING" and is_webcam_source else None); feedback_list = ["Configure sets." if source_type=='webcam' else f"Start {current_exercise} (Video)"]; return
         elif not clicked_ex_btn and source_type=='webcam' and start_btn_x_ex<=x<=start_btn_x_ex+start_btn_w_ex and fp_btn_y_ex<=y<=fp_btn_y_ex+start_btn_h_ex:
             finalize_last_exercise_duration(previous_exercise); reset_exercise_state();set_config_confirmed=False; current_set_number = 1; app_mode=("GUIDE" if load_guide_gif(current_exercise) else "TRACKING"); (session_exercise_start_time := time.time() if app_mode=="TRACKING" and is_webcam_source else None); guide_start_time = time.time(); feedback_list=[f"Guide: {current_exercise} (Free Play)" if app_mode=="GUIDE" else f"Start {current_exercise} (Free Play)"]; return
         elif not clicked_ex_btn and back_btn_x_ex<=x<=back_btn_x_ex+back_btn_w_ex and back_btn_y_ex<=y<=back_btn_y_ex+back_btn_h_ex: finalize_last_exercise_duration(current_exercise); end_session(); return
    elif app_mode == "SET_SELECTION":
        h_set, w_set = canvas_h, canvas_w; (_, th_title_set), _ = cv2.getTextSize(f"Configure: {current_exercise}", FONT, SELECT_TITLE_SCALE*0.9, LINE_THICKNESS+1); ty_title_set = int(h_set*0.15); content_w_set = int(w_set*0.5); content_x_set = (w_set-content_w_set)//2; item_y_start_set = ty_title_set+th_title_set+BUTTON_MARGIN*2; item_h_set = BUTTON_HEIGHT+5; val_x_set = content_x_set+180+10; minus_x_set = val_x_set+60+10; plus_x_set = minus_x_set+PLUS_MINUS_BTN_SIZE+10; btn_y_off_set = (BUTTON_HEIGHT-PLUS_MINUS_BTN_SIZE)//2; sets_btn_y_set=item_y_start_set+btn_y_off_set;reps_btn_y_set=item_y_start_set+item_h_set+btn_y_off_set;rest_btn_y_set=item_y_start_set+2*item_h_set+btn_y_off_set
        if minus_x_set<=x<minus_x_set+PLUS_MINUS_BTN_SIZE:
            if sets_btn_y_set<=y<sets_btn_y_set+PLUS_MINUS_BTN_SIZE: target_sets=max(1,target_sets-1);return
            if reps_btn_y_set<=y<reps_btn_y_set+PLUS_MINUS_BTN_SIZE: target_reps_per_set=max(1,target_reps_per_set-1);return
            if rest_btn_y_set<=y<rest_btn_y_set+PLUS_MINUS_BTN_SIZE: target_rest_time=max(0,target_rest_time-5);return
        if plus_x_set<=x<plus_x_set+PLUS_MINUS_BTN_SIZE:
            if sets_btn_y_set<=y<sets_btn_y_set+PLUS_MINUS_BTN_SIZE: target_sets+=1;return
            if reps_btn_y_set<=y<reps_btn_y_set+PLUS_MINUS_BTN_SIZE: target_reps_per_set+=1;return
            if rest_btn_y_set<=y<rest_btn_y_set+PLUS_MINUS_BTN_SIZE: target_rest_time+=5;return
        confirm_btn_w_set,confirm_btn_h_set=200,BUTTON_HEIGHT;confirm_btn_x_set=w_set//2-confirm_btn_w_set//2;confirm_btn_y_set=item_y_start_set+3*item_h_set+BUTTON_MARGIN
        if confirm_btn_x_set<=x<=confirm_btn_x_set+confirm_btn_w_set and confirm_btn_y_set<=y<=confirm_btn_y_set+confirm_btn_h_set: set_config_confirmed=True;current_set_number=1; app_mode=("GUIDE" if load_guide_gif(current_exercise) else "TRACKING"); (session_exercise_start_time := time.time() if app_mode=="TRACKING" and is_webcam_source else None); guide_start_time = time.time(); feedback_list=[f"Guide: Set {current_set_number}/{target_sets}" if app_mode=="GUIDE" else f"Start Set {current_set_number}/{target_sets}"]; return
        back_btn_w_set,back_btn_h_set=150,BUTTON_HEIGHT;back_btn_x_set=BUTTON_MARGIN*2;back_btn_y_set=h_set-back_btn_h_set-BUTTON_MARGIN*2
        if back_btn_x_set<=x<=back_btn_x_set+back_btn_w_set and back_btn_y_set<=y<=back_btn_y_set+back_btn_h_set: app_mode="EXERCISE_SELECT";feedback_list=["Select an exercise."]; return
    elif app_mode == "CHAT":
        h_chat, w_chat = canvas_h, canvas_w; title_bar_actual_height = BUTTON_MARGIN + int(SELECT_TITLE_SCALE * 0.8 * 25) + BUTTON_MARGIN // 2; input_field_x = BUTTON_MARGIN * 2 + CHAT_INPUT_MARGIN; input_field_h = CHAT_INPUT_HEIGHT; back_button_total_height = BUTTON_HEIGHT + BUTTON_MARGIN * 2; input_field_y = h_chat - back_button_total_height - input_field_h - CHAT_INPUT_MARGIN; input_field_w = w_chat - (BUTTON_MARGIN * 4) - (CHAT_INPUT_MARGIN * 2) - 80; send_btn_w, send_btn_h = 70, CHAT_INPUT_HEIGHT; send_btn_x = input_field_x + input_field_w + CHAT_INPUT_MARGIN; send_btn_y = input_field_y; back_btn_w_chat, back_btn_h_chat = 150, BUTTON_HEIGHT; back_btn_x_chat = BUTTON_MARGIN * 2; back_btn_y_chat = h_chat - back_btn_h_chat - BUTTON_MARGIN; chat_area_x_start_coord = BUTTON_MARGIN * 2; chat_area_content_width_coord = w_chat - (BUTTON_MARGIN * 4) - CHAT_SCROLL_BUTTON_SIZE - BUTTON_MARGIN; scroll_btn_x_coord = chat_area_x_start_coord + chat_area_content_width_coord + CHAT_SCROLL_AREA_PADDING + BUTTON_MARGIN // 2; chat_area_y_start_coord = title_bar_actual_height + CHAT_SCROLL_AREA_PADDING //2; chat_area_h_coord_for_scroll = input_field_y - CHAT_INPUT_MARGIN - chat_area_y_start_coord - CHAT_SCROLL_AREA_PADDING // 2; scroll_up_y_coord = chat_area_y_start_coord; scroll_down_y_coord = chat_area_y_start_coord + chat_area_h_coord_for_scroll - CHAT_SCROLL_BUTTON_SIZE; clicked_on_chat_button_this_event = False
        if input_field_x <= x <= input_field_x + input_field_w and input_field_y <= y <= input_field_y + input_field_h: chat_input_active = True; clicked_on_chat_button_this_event = True; return
        if send_btn_x <= x <= send_btn_x + send_btn_w and send_btn_y <= y <= send_btn_y + send_btn_h:
            clicked_on_chat_button_this_event = True
            if chat_input_text.strip() and not is_llm_thinking: chat_messages.append({"role": "user", "content": chat_input_text.strip()}); is_llm_thinking = True; chat_input_text = ""; last_chat_error = None; chat_scroll_offset_y = 0
            return
        if back_btn_x_chat <= x <= back_btn_x_chat + back_btn_w_chat and back_btn_y_chat <= y <= back_btn_y_chat + back_btn_h_chat: clicked_on_chat_button_this_event = True; app_mode = "HOME"; chat_input_active = False; feedback_list = [f"Welcome back, {current_user}."] if current_user else ["Select profile."]; return
        if scroll_btn_x_coord <= x <= scroll_btn_x_coord + CHAT_SCROLL_BUTTON_SIZE and scroll_up_y_coord <= y <= scroll_up_y_coord + CHAT_SCROLL_BUTTON_SIZE: clicked_on_chat_button_this_event = True; scroll_amount = int(chat_visible_area_height * 0.5); chat_scroll_offset_y += scroll_amount; max_scroll_val = max(0, chat_total_content_height - chat_visible_area_height); chat_scroll_offset_y = min(chat_scroll_offset_y, max_scroll_val); return
        if scroll_btn_x_coord <= x <= scroll_btn_x_coord + CHAT_SCROLL_BUTTON_SIZE and scroll_down_y_coord <= y <= scroll_down_y_coord + CHAT_SCROLL_BUTTON_SIZE: clicked_on_chat_button_this_event = True; scroll_amount = int(chat_visible_area_height * 0.5); chat_scroll_offset_y -= scroll_amount; chat_scroll_offset_y = max(0, chat_scroll_offset_y); return
        if not clicked_on_chat_button_this_event and chat_input_active: chat_input_active = False; return
    elif app_mode == "GUIDE":
        start_btn_w_g,start_btn_h_g=250,BUTTON_HEIGHT;start_btn_x_g=canvas_w//2-start_btn_w_g//2;start_btn_y_g=canvas_h-start_btn_h_g-BUTTON_MARGIN*2
        if start_btn_x_g<=x<=start_btn_x_g+start_btn_w_g and start_btn_y_g<=y<=start_btn_y_g+start_btn_h_g: app_mode="TRACKING"; (session_exercise_start_time := time.time() if is_webcam_source else None); feedback_list=[f"Start Set {current_set_number}/{target_sets}" if set_config_confirmed else f"Start {current_exercise} (Free Play)"];return
    elif app_mode == "TRACKING":
        try: total_btn_w_t=canvas_w-2*BUTTON_MARGIN;btn_w_t=max(50,(total_btn_w_t-(len(EXERCISES)-1)*(BUTTON_MARGIN//2))//len(EXERCISES))
        except ZeroDivisionError: btn_w_t=100
        home_btn_sz_t=50;home_btn_x_t=canvas_w-home_btn_sz_t-BUTTON_MARGIN;home_btn_y_t=canvas_h-home_btn_sz_t-BUTTON_MARGIN;clicked_top_btn_t=False
        for i,ex_t_name in enumerate(EXERCISES):
            if BUTTON_MARGIN+i*(btn_w_t+BUTTON_MARGIN//2)<=x<=BUTTON_MARGIN+i*(btn_w_t+BUTTON_MARGIN//2)+btn_w_t and BUTTON_MARGIN<=y<=BUTTON_MARGIN+BUTTON_HEIGHT:
                clicked_top_btn_t=True
                if current_exercise!=ex_t_name: finalize_last_exercise_duration(current_exercise); current_exercise=ex_t_name; reset_exercise_state(); app_mode="EXERCISE_SELECT"; session_exercise_start_time = None; feedback_list=["Exercise changed. Select configuration or Free Play."]; break
        if not clicked_top_btn_t and home_btn_x_t<=x<=home_btn_x_t+home_btn_sz_t and home_btn_y_t<=y<=home_btn_y_t+home_btn_sz_t: finalize_last_exercise_duration(current_exercise); end_session();return
    elif app_mode == "REST":
        skip_btn_w_r,skip_btn_h_r=180,BUTTON_HEIGHT;skip_btn_x_r=canvas_w//2-skip_btn_w_r//2;skip_btn_y_r=canvas_h//2+int(LARGE_TIMER_SCALE*25) + BUTTON_MARGIN # Adjusted y for timer text
        if skip_btn_x_r<=x<=skip_btn_x_r+skip_btn_w_r and skip_btn_y_r<=y<=skip_btn_y_r+skip_btn_h_r: app_mode="TRACKING"; (session_exercise_start_time := time.time() if is_webcam_source else None); feedback_list=[f"Start Set {current_set_number}/{target_sets}"];return
        home_btn_sz_r=50;home_btn_x_r=canvas_w-home_btn_sz_r-BUTTON_MARGIN;home_btn_y_r=canvas_h-home_btn_sz_r-BUTTON_MARGIN
        if home_btn_x_r<=x<=home_btn_x_r+home_btn_sz_r and home_btn_y_r<=y<=home_btn_y_r+home_btn_sz_r: end_session();return
    elif app_mode == "STATS":
        back_btn_w_s,back_btn_h_s=150,BUTTON_HEIGHT;back_btn_x_s=BUTTON_MARGIN*2;back_btn_y_s=canvas_h-back_btn_h_s-BUTTON_MARGIN*2
        if back_btn_x_s<=x<=back_btn_x_s+back_btn_w_s and back_btn_y_s<=y<=back_btn_y_s+back_btn_h_s: app_mode="HOME";stats_pie_image=None;stats_time_plot_image=None;return

# --- UI Drawing Functions ---
def draw_home_ui(canvas):
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]
    title_text = "Fitness Tracker Pro"; (tw_title, th_title), _ = cv2.getTextSize(title_text, FONT, TITLE_SCALE, LINE_THICKNESS + 1); tx = (w - tw_title) // 2; ty_title_base = int(h * 0.1) + th_title; cv2.putText(canvas, title_text, (tx, ty_title_base), FONT, TITLE_SCALE, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)
    user_text = f"User: {current_user}" if current_user else "User: None Selected"; (tw_user, th_user), _ = cv2.getTextSize(user_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); tx_user = (w - tw_user) // 2; ty_user = ty_title_base + th_title + int(BUTTON_MARGIN * 0.75); cv2.putText(canvas, user_text, (tx_user, ty_user), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["profile_text"], LINE_THICKNESS, cv2.LINE_AA)
    num_top_btns = 4; btn_w = int(w * 0.18); btn_h = BUTTON_HEIGHT; gap = BUTTON_MARGIN // 2; total_top_btn_width = num_top_btns * btn_w + (num_top_btns - 1) * gap; start_x_profile = (w - total_top_btn_width) // 2; profile_y = ty_user + th_user + BUTTON_MARGIN // 2
    select_btn_x = start_x_profile; draw_rounded_rectangle(canvas, (select_btn_x, profile_y), (select_btn_x + btn_w, profile_y + btn_h), COLORS["button_bg_profile"], -1, CORNER_RADIUS); btn_text_s = "Select Profile"; (tw_s, th_s), _ = cv2.getTextSize(btn_text_s, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text_s, (select_btn_x + (btn_w - tw_s) // 2, profile_y + (btn_h + th_s) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    create_btn_x = select_btn_x + btn_w + gap; draw_rounded_rectangle(canvas, (create_btn_x, profile_y), (create_btn_x + btn_w, profile_y + btn_h), COLORS["button_bg_profile"], -1, CORNER_RADIUS); btn_text_c = "Create Profile"; (tw_c, th_c), _ = cv2.getTextSize(btn_text_c, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text_c, (create_btn_x + (btn_w - tw_c) // 2, profile_y + (btn_h + th_c) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    stats_btn_x = create_btn_x + btn_w + gap; draw_rounded_rectangle(canvas, (stats_btn_x, profile_y), (stats_btn_x + btn_w, profile_y + btn_h), COLORS["button_bg_stats"], -1, CORNER_RADIUS); btn_text_st = "View Stats"; (tw_st, th_st), _ = cv2.getTextSize(btn_text_st, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text_st, (stats_btn_x + (btn_w - tw_st) // 2, profile_y + (btn_h + th_st) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    chat_btn_x = stats_btn_x + btn_w + gap; draw_rounded_rectangle(canvas, (chat_btn_x, profile_y), (chat_btn_x + btn_w, profile_y + btn_h), COLORS["button_bg_chat"], -1, CORNER_RADIUS); btn_text_ch = "AI Coach"; (tw_ch, th_ch), _ = cv2.getTextSize(btn_text_ch, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text_ch, (chat_btn_x + (btn_w - tw_ch) // 2, profile_y + (btn_h + th_ch) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    src_btn_w, src_btn_h = int(w*0.35),int(BUTTON_HEIGHT*1.2);src_btn_x=(w-src_btn_w)//2;webcam_btn_y=profile_y+btn_h+BUTTON_MARGIN*2;video_btn_y=webcam_btn_y+src_btn_h+BUTTON_MARGIN
    draw_rounded_rectangle(canvas,(src_btn_x,webcam_btn_y),(src_btn_x+src_btn_w,webcam_btn_y+src_btn_h),COLORS["accent_green"],-1,CORNER_RADIUS);btn_text_wb="Start Webcam Workout";(tw_wb,th_wb),_=cv2.getTextSize(btn_text_wb,FONT,BUTTON_TEXT_SCALE*1.1,LINE_THICKNESS);cv2.putText(canvas,btn_text_wb,(src_btn_x+(src_btn_w-tw_wb)//2,webcam_btn_y+(src_btn_h+th_wb)//2),FONT,BUTTON_TEXT_SCALE*1.1,COLORS["button_text_active"],LINE_THICKNESS,cv2.LINE_AA)
    draw_rounded_rectangle(canvas,(src_btn_x,video_btn_y),(src_btn_x+src_btn_w,video_btn_y+src_btn_h),COLORS["accent_blue"],-1,CORNER_RADIUS);btn_text_vid="Load Video (No Stats)";(tw_vid,th_vid),_=cv2.getTextSize(btn_text_vid,FONT,BUTTON_TEXT_SCALE*1.1,LINE_THICKNESS);cv2.putText(canvas,btn_text_vid,(src_btn_x+(src_btn_w-tw_vid)//2,video_btn_y+(src_btn_h+th_vid)//2),FONT,BUTTON_TEXT_SCALE*1.1,COLORS["button_text_active"],LINE_THICKNESS,cv2.LINE_AA)
    feedback_str=" | ".join(feedback_list) if feedback_list else "Ready";(tw_fb,th_fb),_=cv2.getTextSize(feedback_str,FONT,FEEDBACK_TEXT_SCALE,LINE_THICKNESS);fx=(w-tw_fb)//2;fy=h-BUTTON_MARGIN*2-th_fb;feedback_color=COLORS["accent_red"] if any(s in feedback_str.lower() for s in ["error", "please select", "needed"]) else COLORS["secondary_text"];cv2.putText(canvas,feedback_str,(fx,fy),FONT,FEEDBACK_TEXT_SCALE,feedback_color,LINE_THICKNESS,cv2.LINE_AA)
    quit_text="Press 'Q' to Quit";(tw_q,th_q),_=cv2.getTextSize(quit_text,FONT,0.6,1);cv2.putText(canvas,quit_text,(w-tw_q-20,h-th_q-10),FONT,0.6,COLORS["secondary_text"],1,cv2.LINE_AA)

def draw_exercise_select_ui(canvas):
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]; title_text = "Select Exercise"; (tw, th), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE, LINE_THICKNESS + 1); tx = (w - tw) // 2; ty = BUTTON_MARGIN * 3; cv2.putText(canvas, title_text, (tx, ty), FONT, SELECT_TITLE_SCALE, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)
    item_height = BUTTON_HEIGHT + BUTTON_MARGIN // 2; list_h = len(EXERCISES) * item_height; start_y = ty + th + BUTTON_MARGIN * 2; button_w = int(w * 0.4); button_x = w // 2 - button_w // 2
    for i, ex in enumerate(EXERCISES):
        btn_y = start_y + i * item_height; is_active = (ex == current_exercise); bg_color = COLORS["button_bg_active"] if is_active else COLORS["button_bg_normal"]; text_color = COLORS["button_text_active"] if is_active else COLORS["button_text_normal"]; border_color = COLORS["button_text_active"] if is_active else COLORS["secondary_text"]
        draw_rounded_rectangle(canvas, (button_x, btn_y), (button_x + button_w, btn_y + BUTTON_HEIGHT), bg_color, -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (button_x, btn_y), (button_x + button_w, btn_y + BUTTON_HEIGHT), border_color, 1, CORNER_RADIUS)
        (tw_ex, th_ex), _ = cv2.getTextSize(ex, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); tx_ex = button_x + max(0, (button_w - tw_ex) // 2); ty_ex = btn_y + (BUTTON_HEIGHT + th_ex) // 2; cv2.putText(canvas, ex, (tx_ex, ty_ex), FONT, BUTTON_TEXT_SCALE * 1.1, text_color, LINE_THICKNESS, cv2.LINE_AA)
    start_btn_w, start_btn_h = 200, BUTTON_HEIGHT; start_btn_x = w // 2 - start_btn_w // 2; start_btn_y = start_y + list_h + BUTTON_MARGIN; start_btn_text = "Configure Sets" if source_type == 'webcam' else "Start Video"; draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS); (tw, th), _ = cv2.getTextSize(start_btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, start_btn_text, (start_btn_x + (start_btn_w - tw) // 2, start_btn_y + (start_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    if source_type == 'webcam':
        free_play_btn_w, free_play_btn_h = 200, BUTTON_HEIGHT; free_play_btn_x = start_btn_x; free_play_btn_y = start_btn_y + start_btn_h + BUTTON_MARGIN // 2; draw_rounded_rectangle(canvas, (free_play_btn_x, free_play_btn_y), (free_play_btn_x + free_play_btn_w, free_play_btn_y + free_play_btn_h), COLORS["button_bg_freeplay"], -1, CORNER_RADIUS); btn_text = "Start Free Play"; (tw_fp, th_fp), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (free_play_btn_x + (free_play_btn_w - tw_fp) // 2, free_play_btn_y + (free_play_btn_h + th_fp) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2; draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); btn_text = "Back to Home"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (back_btn_x + (back_btn_w - tw) // 2, back_btn_y + (back_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)
    feedback_str = " | ".join(feedback_list); (tw_fb, th_fb), _ = cv2.getTextSize(feedback_str, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS); fx = (w - tw_fb) // 2; fy = h - BUTTON_MARGIN - th_fb; feedback_color = COLORS["accent_red"] if "Error" in feedback_str else COLORS["secondary_text"]; cv2.putText(canvas, feedback_str, (fx, fy), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)

def draw_set_selection_ui(canvas):
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]; title_text = f"Configure: {current_exercise}"; (tw, th_title), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE * 0.9, LINE_THICKNESS + 1); tx = (w - tw) // 2; ty_title = int(h * 0.15); cv2.putText(canvas, title_text, (tx, ty_title), FONT, SELECT_TITLE_SCALE * 0.9, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)
    content_w = int(w * 0.5); content_x = (w - content_w) // 2; item_y_start = ty_title + th_title + BUTTON_MARGIN * 2; item_h = BUTTON_HEIGHT + 5; label_w = 180; value_w = 60; value_x = content_x + label_w + 10; minus_btn_x = value_x + value_w + 10; plus_btn_x = minus_btn_x + PLUS_MINUS_BTN_SIZE + 10; btn_y_offset = (BUTTON_HEIGHT - PLUS_MINUS_BTN_SIZE) // 2
    sets_y = item_y_start; sets_btn_y = sets_y + btn_y_offset; cv2.putText(canvas, "Number of Sets:", (content_x, sets_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA); cv2.putText(canvas, str(target_sets), (value_x, sets_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["accent_blue"], LINE_THICKNESS + 1, cv2.LINE_AA); draw_rounded_rectangle(canvas, (minus_btn_x, sets_btn_y), (minus_btn_x + PLUS_MINUS_BTN_SIZE, sets_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5); (tw_m, th_m), _ = cv2.getTextSize("-", FONT, 1.0, 2); cv2.putText(canvas, "-", (minus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_m)//2, sets_btn_y + (PLUS_MINUS_BTN_SIZE + th_m)//2), FONT, 1.0, COLORS["primary_text"], 2); draw_rounded_rectangle(canvas, (plus_btn_x, sets_btn_y), (plus_btn_x + PLUS_MINUS_BTN_SIZE, sets_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5); (tw_p, th_p), _ = cv2.getTextSize("+", FONT, 1.0, 2); cv2.putText(canvas, "+", (plus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_p)//2, sets_btn_y + (PLUS_MINUS_BTN_SIZE + th_p)//2), FONT, 1.0, COLORS["primary_text"], 2)
    reps_y = item_y_start + item_h; reps_btn_y = reps_y + btn_y_offset; cv2.putText(canvas, "Reps per Set:", (content_x, reps_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA); cv2.putText(canvas, str(target_reps_per_set), (value_x, reps_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["accent_blue"], LINE_THICKNESS + 1, cv2.LINE_AA); draw_rounded_rectangle(canvas, (minus_btn_x, reps_btn_y), (minus_btn_x + PLUS_MINUS_BTN_SIZE, reps_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5); (tw_m, th_m), _ = cv2.getTextSize("-", FONT, 1.0, 2); cv2.putText(canvas, "-", (minus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_m)//2, reps_btn_y + (PLUS_MINUS_BTN_SIZE + th_m)//2), FONT, 1.0, COLORS["primary_text"], 2); draw_rounded_rectangle(canvas, (plus_btn_x, reps_btn_y), (plus_btn_x + PLUS_MINUS_BTN_SIZE, reps_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5); (tw_p, th_p), _ = cv2.getTextSize("+", FONT, 1.0, 2); cv2.putText(canvas, "+", (plus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_p)//2, reps_btn_y + (PLUS_MINUS_BTN_SIZE + th_p)//2), FONT, 1.0, COLORS["primary_text"], 2)
    rest_y = item_y_start + 2 * item_h; rest_btn_y = rest_y + btn_y_offset; cv2.putText(canvas, "Rest Time (sec):", (content_x, rest_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA); cv2.putText(canvas, str(target_rest_time), (value_x, rest_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["accent_blue"], LINE_THICKNESS + 1, cv2.LINE_AA); draw_rounded_rectangle(canvas, (minus_btn_x, rest_btn_y), (minus_btn_x + PLUS_MINUS_BTN_SIZE, rest_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5); (tw_m, th_m), _ = cv2.getTextSize("-", FONT, 1.0, 2); cv2.putText(canvas, "-", (minus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_m)//2, rest_btn_y + (PLUS_MINUS_BTN_SIZE + th_m)//2), FONT, 1.0, COLORS["primary_text"], 2); draw_rounded_rectangle(canvas, (plus_btn_x, rest_btn_y), (plus_btn_x + PLUS_MINUS_BTN_SIZE, rest_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5); (tw_p, th_p), _ = cv2.getTextSize("+", FONT, 1.0, 2); cv2.putText(canvas, "+", (plus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_p)//2, rest_btn_y + (PLUS_MINUS_BTN_SIZE + th_p)//2), FONT, 1.0, COLORS["primary_text"], 2)
    confirm_btn_w, confirm_btn_h = 200, BUTTON_HEIGHT; confirm_btn_x = w // 2 - confirm_btn_w // 2; confirm_btn_y = item_y_start + 3 * item_h + BUTTON_MARGIN; draw_rounded_rectangle(canvas, (confirm_btn_x, confirm_btn_y), (confirm_btn_x + confirm_btn_w, confirm_btn_y + confirm_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS); btn_text = "Confirm & Start"; (tw_c, th_c), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); cv2.putText(canvas, btn_text, (confirm_btn_x + (confirm_btn_w - tw_c) // 2, confirm_btn_y + (confirm_btn_h + th_c) // 2), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2; draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); btn_text = "Back"; (tw_b, th_b), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (back_btn_x + (back_btn_w - tw_b) // 2, back_btn_y + (back_btn_h + th_b) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)
    feedback_str = " | ".join(feedback_list); (tw_fb, th_fb), _ = cv2.getTextSize(feedback_str, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS); fx = (w - tw_fb) // 2; fy = h - BUTTON_MARGIN - th_fb; feedback_color = COLORS["accent_red"] if "Error" in feedback_str else COLORS["secondary_text"]; cv2.putText(canvas, feedback_str, (fx, fy), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)

def draw_guide_ui(canvas):
    global guide_gif_index, guide_last_frame_time
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]; mode_info = f"(Set {current_set_number}/{target_sets})" if set_config_confirmed else "(Free Play)"; title = f"Guide: {current_exercise} {mode_info}"; (tw_title, th_title), _ = cv2.getTextSize(title, FONT, TITLE_SCALE * 0.8, LINE_THICKNESS); cv2.putText(canvas, title, (BUTTON_MARGIN, BUTTON_MARGIN + th_title), FONT, TITLE_SCALE * 0.8, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA)
    gif_area_y_start = BUTTON_MARGIN * 2 + th_title; gif_area_h = h - gif_area_y_start - BUTTON_MARGIN * 4 - BUTTON_HEIGHT; gif_area_w = w - BUTTON_MARGIN * 2; gif_area_x_start = BUTTON_MARGIN
    if guide_gif_frames:
        current_time = time.time()
        if current_time - guide_last_frame_time >= guide_frame_delay: guide_gif_index = (guide_gif_index + 1) % len(guide_gif_frames); guide_last_frame_time = current_time
        frame_g=guide_gif_frames[guide_gif_index]; frame_h_g,frame_w_g=frame_g.shape[:2]
        if frame_w_g>0 and frame_h_g>0:
            scale_g=min(gif_area_w/frame_w_g,gif_area_h/frame_h_g);new_w_g,new_h_g=int(frame_w_g*scale_g),int(frame_h_g*scale_g)
            if new_w_g>0 and new_h_g>0:
                try: display_frame_g=cv2.resize(frame_g,(new_w_g,new_h_g),interpolation=cv2.INTER_LINEAR); ox_g,oy_g=BUTTON_MARGIN+(gif_area_w-new_w_g)//2,gif_area_y_start+(gif_area_h-new_h_g)//2
                except Exception as e: print(f"Error resizing/displaying GIF frame: {e}"); display_frame_g=None; ox_g,oy_g=0,0 #Prevent crash
                if display_frame_g is not None and oy_g>=0 and ox_g>=0 and oy_g+new_h_g<=h and ox_g+new_w_g<=w: canvas[oy_g:oy_g+new_h_g,ox_g:ox_g+new_w_g]=display_frame_g
    else: no_gif_text = "Exercise GIF not available."; (tw_ng, th_ng),_ = cv2.getTextSize(no_gif_text, FONT, 0.8, 2); cv2.putText(canvas, no_gif_text, (gif_area_x_start + (gif_area_w - tw_ng)//2, gif_area_y_start + (gif_area_h + th_ng)//2), FONT, 0.8, COLORS["secondary_text"], 2, cv2.LINE_AA)
    start_btn_w, start_btn_h = 250, BUTTON_HEIGHT; start_btn_x = w // 2 - start_btn_w // 2; start_btn_y = h - start_btn_h - BUTTON_MARGIN * 2; draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS); btn_text = "Start Exercise"; (tw_st, th_st), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (start_btn_x + (start_btn_w - tw_st) // 2, start_btn_y + (start_btn_h + th_st) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    elapsed = time.time() - guide_start_time; remaining = max(0, guide_duration - elapsed); skip_text = f"Starting in {remaining:.0f}s (Click Start)" if remaining > 0 else "Click Start Now"; (tw_skip, th_skip), _ = cv2.getTextSize(skip_text, FONT, FEEDBACK_TEXT_SCALE * 0.9, 1); cv2.putText(canvas, skip_text, (start_btn_x + (start_btn_w - tw_skip)//2, start_btn_y - th_skip - 5), FONT, FEEDBACK_TEXT_SCALE * 0.9, COLORS["secondary_text"], 1, cv2.LINE_AA)

def draw_tracking_ui(canvas,frame,results):
    global last_frame_for_rest
    h, w = canvas.shape[:2]; frame_h, frame_w = frame.shape[:2]; canvas[:] = (10,10,10) # Dark background for video
    ox, oy, sw, sh = 0, 0, w, h
    if frame_w > 0 and frame_h > 0:
        scale = min(w / frame_w, h / frame_h); sw, sh = int(frame_w * scale), int(frame_h * scale); ox, oy = (w - sw) // 2, (h - sh) // 2; interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        try:
            if sw > 0 and sh > 0:
                resized_frame = cv2.resize(frame, (sw, sh), interpolation=interp)
                # Pass results.pose_landmarks (actual MediaPipe object)
                draw_pose_landmarks_on_frame(resized_frame, results.pose_landmarks if results else None, mp_pose.POSE_CONNECTIONS, form_issues_details)
                if oy >= 0 and ox >= 0 and oy + sh <= h and ox + sw <= w: canvas[oy:oy + sh, ox:ox + sw] = resized_frame
                else: print("Warning: Video ROI calc error.")
                last_frame_for_rest = canvas.copy()
            else: print("Warning: Invalid resize dimensions."); last_frame_for_rest = canvas.copy()
        except Exception as e: print(f"Error resizing/drawing landmarks/placing frame: {e}"); err_txt = "Video Display Error"; (tw, th),_ = cv2.getTextSize(err_txt, FONT, 1.0, 2); cv2.putText(canvas, err_txt, ((w-tw)//2, (h+th)//2), FONT, 1.0, COLORS['accent_red'], 2, cv2.LINE_AA); last_frame_for_rest = canvas.copy()
    else: err_txt = "Invalid Frame Input"; (tw, th),_ = cv2.getTextSize(err_txt, FONT, 1.0, 2); cv2.putText(canvas, err_txt, ((w-tw)//2, (h+th)//2), FONT, 1.0, COLORS['accent_red'], 2, cv2.LINE_AA); last_frame_for_rest = canvas.copy()
    
    overlay_canvas = np.zeros_like(canvas) # Transparent overlay for UI elements
    try: total_button_width = w - 2 * BUTTON_MARGIN; btn_w = max(50, (total_button_width - (len(EXERCISES) - 1) * (BUTTON_MARGIN // 2)) // len(EXERCISES))
    except ZeroDivisionError: btn_w = 100
    for i, ex in enumerate(EXERCISES):
        bx = BUTTON_MARGIN + i * (btn_w + BUTTON_MARGIN // 2); bxe = bx + btn_w; is_active = (ex == current_exercise); bg_color = COLORS["button_bg_active"] if is_active else COLORS["button_bg_normal"]; text_color = COLORS["button_text_active"] if is_active else COLORS["button_text_normal"]
        draw_rounded_rectangle(overlay_canvas, (bx, BUTTON_MARGIN), (bxe, BUTTON_MARGIN + BUTTON_HEIGHT), bg_color, -1, CORNER_RADIUS)
        (tw_ex, th_ex), _ = cv2.getTextSize(ex, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); tx = bx + max(0, (btn_w - tw_ex) // 2); ty = BUTTON_MARGIN + (BUTTON_HEIGHT + th_ex) // 2; cv2.putText(overlay_canvas, ex, (tx, ty), FONT, BUTTON_TEXT_SCALE, text_color, LINE_THICKNESS, cv2.LINE_AA)
    
    is_bicep = current_exercise == "BICEP CURL"; sb_h = 200 if is_bicep else 170; sb_w = 400 if is_bicep else 320; sb_x, sb_y = BUTTON_MARGIN, BUTTON_MARGIN * 2 + BUTTON_HEIGHT; sb_xe, sb_ye = sb_x + sb_w, sb_y + sb_h; sb_xe = min(sb_xe, w - BUTTON_MARGIN); sb_ye = min(sb_ye, h - BUTTON_MARGIN); sb_w = sb_xe - sb_x; sb_h = sb_ye - sb_y
    if sb_w > 0 and sb_h > 0:
        draw_semi_transparent_rect(overlay_canvas, (sb_x, sb_y), (sb_xe, sb_ye), COLORS["overlay_bg"]); draw_rounded_rectangle(overlay_canvas, (sb_x, sb_y), (sb_xe, sb_ye), COLORS["secondary_text"], 1, CORNER_RADIUS)
        line_h = 25; label_color = COLORS["primary_text"]; value_color = COLORS["primary_text"]; stage_color = COLORS["primary_text"]; v_pad = 20
        user_display = f"User: {current_user}" if current_user else "User: -"; cv2.putText(overlay_canvas, user_display, (sb_x + 15, sb_y + v_pad), FONT, STATUS_TEXT_SCALE * 0.9, COLORS["background"], 1, cv2.LINE_AA); v_pad += line_h
        cv2.putText(overlay_canvas, 'EXERCISE:', (sb_x + 15, sb_y + v_pad), FONT, STATUS_TEXT_SCALE, COLORS["background"], 1, cv2.LINE_AA); cv2.putText(overlay_canvas, current_exercise, (sb_x + 110, sb_y + v_pad), FONT, STATUS_TEXT_SCALE, COLORS["accent_green"], LINE_THICKNESS, cv2.LINE_AA); v_pad += line_h
        mode_text = f"SET: {current_set_number}/{target_sets}" if set_config_confirmed else "MODE: Free Play"; cv2.putText(overlay_canvas, mode_text, (sb_x + 15, sb_y + v_pad), FONT, STATUS_TEXT_SCALE, COLORS["background"], 1, cv2.LINE_AA); v_pad += line_h
        display_stage = stage if stage is not None else "INIT"; display_stage_l = stage_left if stage_left is not None else "INIT"; display_stage_r = stage_right if stage_right is not None else "INIT"; rep_target_str = f"/{target_reps_per_set}" if set_config_confirmed else ""
        if is_bicep:
            rep_y = sb_y + v_pad; stage_y = rep_y + line_h + 5; col1_x = sb_x + 15; col2_x = sb_x + sb_w // 2 - 10
            cv2.putText(overlay_canvas, f'L REPS: {counter_left}{rep_target_str}', (col1_x, rep_y), FONT, STATUS_TEXT_SCALE, COLORS["background"], 1, cv2.LINE_AA); cv2.putText(overlay_canvas, 'L STAGE:', (col1_x, stage_y), FONT, STATUS_TEXT_SCALE * 0.9, COLORS["background"], 1, cv2.LINE_AA); cv2.putText(overlay_canvas, display_stage_l, (col1_x + 80, stage_y), FONT, STATUS_TEXT_SCALE, COLORS["accent_blue"], LINE_THICKNESS, cv2.LINE_AA)
            cv2.putText(overlay_canvas, f'R REPS: {counter_right}{rep_target_str}', (col2_x, rep_y), FONT, STATUS_TEXT_SCALE, COLORS["background"], 1, cv2.LINE_AA); cv2.putText(overlay_canvas, 'R STAGE:', (col2_x, stage_y), FONT, STATUS_TEXT_SCALE * 0.9, COLORS["background"], 1, cv2.LINE_AA); cv2.putText(overlay_canvas, display_stage_r, (col2_x + 80, stage_y), FONT, STATUS_TEXT_SCALE, COLORS["accent_blue"], LINE_THICKNESS, cv2.LINE_AA)
        else:
            rep_y = sb_y + v_pad; stage_y = rep_y + line_h + 5; rep_text = f"REPS: {counter}{rep_target_str}"; cv2.putText(overlay_canvas, rep_text, (sb_x + 15, rep_y), FONT, STATUS_TEXT_SCALE * 1.1, COLORS["background"], 1, cv2.LINE_AA); cv2.putText(overlay_canvas, 'STAGE:', (sb_x + 15, stage_y), FONT, STATUS_TEXT_SCALE, COLORS["background"], 1, cv2.LINE_AA); cv2.putText(overlay_canvas, display_stage, (sb_x + 100, stage_y), FONT, STATUS_TEXT_SCALE * 1.1, COLORS["accent_blue"], LINE_THICKNESS, cv2.LINE_AA)
    
    fb_h = 65; home_btn_size = 50; fb_w = w - 2 * BUTTON_MARGIN - home_btn_size - BUTTON_MARGIN; fb_x, fb_y = BUTTON_MARGIN, h - fb_h - BUTTON_MARGIN; fb_xe, fb_ye = fb_x + fb_w, fb_y + fb_h;
    if fb_w > 0 and fb_h > 0:
        draw_semi_transparent_rect(overlay_canvas, (fb_x, fb_y), (fb_xe, fb_ye), COLORS["overlay_bg"]); draw_rounded_rectangle(overlay_canvas, (fb_x, fb_y), (fb_xe, fb_ye), COLORS["secondary_text"], 1, CORNER_RADIUS)
        warnings = [f.replace("WARN: ", "") for f in feedback_list if "WARN:" in f]; infos = [f.replace("INFO: ", "") for f in feedback_list if "INFO:" in f and "WARN:" not in f]; display_feedback = ""; feedback_color = COLORS["accent_blue"]
        if warnings: display_feedback = "WARN: " + " | ".join(sorted(list(set(warnings)))); feedback_color = COLORS["accent_red"]
        elif infos: display_feedback = " | ".join(sorted(list(set(infos)))); feedback_color = COLORS["accent_green"] # Changed to green for info
        elif stage is None and app_mode == "TRACKING": display_feedback = "Initializing..."; feedback_color = COLORS["secondary_text"]
        elif app_mode == "TRACKING": display_feedback = "Status OK"; feedback_color = COLORS["accent_green"]
        else: display_feedback = "..."; feedback_color = COLORS["secondary_text"]
        max_feedback_chars = int(fb_w / (FEEDBACK_TEXT_SCALE * 10)); # Adjusted divisor
        if len(display_feedback) > max_feedback_chars > 3 : display_feedback = display_feedback[:max_feedback_chars - 3] + "..."
        (tw_fb, th_fb), _ = cv2.getTextSize(display_feedback, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS); cv2.putText(overlay_canvas, display_feedback, (fb_x + 15, fb_y + (fb_h + th_fb) // 2), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)
    
    hb_x = w - home_btn_size - BUTTON_MARGIN; hb_y = h - home_btn_size - BUTTON_MARGIN; draw_rounded_rectangle(overlay_canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["accent_red"], -1, CORNER_RADIUS // 2); icon_size = int(home_btn_size * 0.4); icon_x1 = hb_x + (home_btn_size - icon_size) // 2; icon_y1 = hb_y + (home_btn_size - icon_size) // 2; cv2.rectangle(overlay_canvas, (icon_x1, icon_y1), (icon_x1 + icon_size, icon_y1 + icon_size), COLORS["button_text_active"], -1) # Simple square icon
    
    try: # Blend overlay using mask
        gray_overlay = cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2GRAY); _, mask = cv2.threshold(gray_overlay, 5, 255, cv2.THRESH_BINARY); mask_inv = cv2.bitwise_not(mask); bg = cv2.bitwise_and(canvas, canvas, mask=mask_inv); fg = cv2.bitwise_and(overlay_canvas, overlay_canvas, mask=mask); cv2.add(bg, fg, dst=canvas)
    except Exception as e: print(f"Error blending overlay: {e}")

def draw_rest_ui(canvas):
    h, w = canvas.shape[:2]
    if last_frame_for_rest is not None and last_frame_for_rest.shape == canvas.shape: canvas[:] = last_frame_for_rest
    else: canvas[:] = (20, 20, 20) # Fallback dark background
    overlay_color = COLORS["overlay_bg"]; draw_semi_transparent_rect(canvas, (0, 0), (w, h), overlay_color) # Use consistent overlay
    rest_text = "REST"; (tw_r, th_r), _ = cv2.getTextSize(rest_text, FONT, TITLE_SCALE * 0.8, LINE_THICKNESS + 1); tx_r = (w - tw_r) // 2; ty_r = int(h * 0.25); cv2.putText(canvas, rest_text, (tx_r, ty_r), FONT, TITLE_SCALE * 0.8, COLORS["background"], LINE_THICKNESS + 1, cv2.LINE_AA)
    time_elapsed = time.time() - rest_start_time if rest_start_time else 0; time_remaining = max(0, target_rest_time - time_elapsed); timer_text = f"{time_remaining:.0f}"; (tw_t, th_t), _ = cv2.getTextSize(timer_text, FONT, LARGE_TIMER_SCALE, LINE_THICKNESS + 2); tx_t = (w - tw_t) // 2; ty_t = h // 2 + th_t // 2; cv2.putText(canvas, timer_text, (tx_t, ty_t), FONT, LARGE_TIMER_SCALE, COLORS["timer_text"], LINE_THICKNESS + 2, cv2.LINE_AA)
    next_set_text = f"Next: Set {current_set_number}/{target_sets} - {current_exercise}"; (tw_n, th_n), _ = cv2.getTextSize(next_set_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); tx_n = (w - tw_n) // 2; ty_n = ty_r + th_r + BUTTON_MARGIN; cv2.putText(canvas, next_set_text, (tx_n, ty_n), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["secondary_text"], LINE_THICKNESS, cv2.LINE_AA)
    skip_btn_w, skip_btn_h = 180, BUTTON_HEIGHT; skip_btn_x = w // 2 - skip_btn_w // 2; skip_btn_y = ty_t + th_t // 2 + BUTTON_MARGIN; draw_rounded_rectangle(canvas, (skip_btn_x, skip_btn_y), (skip_btn_x + skip_btn_w, skip_btn_y + skip_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); btn_text = "Skip Rest"; (tw_s, th_s), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (skip_btn_x + (skip_btn_w - tw_s) // 2, skip_btn_y + (skip_btn_h + th_s) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)
    home_btn_size = 50; hb_x = w - home_btn_size - BUTTON_MARGIN; hb_y = h - home_btn_size - BUTTON_MARGIN; draw_rounded_rectangle(canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["accent_red"], -1, CORNER_RADIUS // 2); icon_size = int(home_btn_size * 0.4); icon_x1 = hb_x + (home_btn_size - icon_size) // 2; icon_y1 = hb_y + (home_btn_size - icon_size) // 2; cv2.rectangle(canvas, (icon_x1, icon_y1), (icon_x1 + icon_size, icon_y1 + icon_size), COLORS["button_text_active"], -1)

def draw_stats_ui(canvas):
    global stats_pie_image, stats_time_plot_image; h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]
    title_text = f"Statistics for {current_user}" if current_user else "Statistics"; (tw_title, th_title), _ = cv2.getTextSize(title_text, FONT, TITLE_SCALE * 0.9, LINE_THICKNESS + 1); tx = (w - tw_title) // 2; ty = BUTTON_MARGIN * 2 + th_title; cv2.putText(canvas, title_text, (tx, ty), FONT, TITLE_SCALE * 0.9, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)
    plot_area_y_start = ty + th_title + BUTTON_MARGIN; plot_height = int(h * 0.30); plot_gap = BUTTON_MARGIN * 2; pie_chart_w = int(w * 0.40); pie_chart_x_start = BUTTON_MARGIN * 3
    cv2.putText(canvas, "Calorie Breakdown", (pie_chart_x_start, plot_area_y_start - 5), FONT, STATS_TITLE_SCALE, COLORS["primary_text"], 1, cv2.LINE_AA)
    if stats_pie_image is None and current_user: stats_pie_image = generate_stats_pie_image(pie_chart_w, plot_height)
    if stats_pie_image is not None:
        if stats_pie_image.shape[0] == plot_height and stats_pie_image.shape[1] == pie_chart_w: canvas[plot_area_y_start : plot_area_y_start + plot_height, pie_chart_x_start : pie_chart_x_start + pie_chart_w] = stats_pie_image
        else: cv2.rectangle(canvas, (pie_chart_x_start, plot_area_y_start), (pie_chart_x_start+pie_chart_w, plot_area_y_start+plot_height), COLORS['plot_bg_color'], -1); cv2.putText(canvas, "Pie Chart Error", (pie_chart_x_start+10, plot_area_y_start+plot_height//2), FONT, 0.6, COLORS['plot_text_color'],1)
    time_plot_w = int(w * 0.40); time_plot_x_start = pie_chart_x_start + pie_chart_w + BUTTON_MARGIN * 3; text_list_y_start_offset = 0
    if time_plot_x_start + time_plot_w > w - BUTTON_MARGIN * 3: time_plot_x_start = pie_chart_x_start; plot_area_y_start_time = plot_area_y_start + plot_height + plot_gap + 20; text_list_y_start_offset = plot_height + plot_gap + 20
    else: plot_area_y_start_time = plot_area_y_start
    cv2.putText(canvas, "Time per Exercise (mins)", (time_plot_x_start, plot_area_y_start_time - 5), FONT, STATS_TITLE_SCALE, COLORS["primary_text"], 1, cv2.LINE_AA)
    if stats_time_plot_image is None and current_user: stats_time_plot_image = generate_time_plot_image(time_plot_w, plot_height)
    if stats_time_plot_image is not None:
        if stats_time_plot_image.shape[0] == plot_height and stats_time_plot_image.shape[1] == time_plot_w: canvas[plot_area_y_start_time : plot_area_y_start_time + plot_height, time_plot_x_start : time_plot_x_start + time_plot_w] = stats_time_plot_image
        else: cv2.rectangle(canvas, (time_plot_x_start, plot_area_y_start_time), (time_plot_x_start+time_plot_w, plot_area_y_start_time+plot_height), COLORS['plot_bg_color'], -1); cv2.putText(canvas, "Time Chart Error", (time_plot_x_start+10, plot_area_y_start_time+plot_height//2), FONT, 0.6, COLORS['plot_text_color'],1)
    text_list_y_start = plot_area_y_start + plot_height + text_list_y_start_offset + plot_gap; text_list_x_start = BUTTON_MARGIN * 3; text_line_h = int(STATS_TEXT_SCALE * 45); current_y_text = text_list_y_start
    cv2.putText(canvas, "Exercise Summary:", (text_list_x_start, current_y_text - 10), FONT, STATS_TITLE_SCALE, COLORS["primary_text"], 1, cv2.LINE_AA); current_y_text += text_line_h //2
    if current_user and current_user in user_stats and user_stats[current_user]:
        exercises_data_exists = False
        for ex_name_iter in EXERCISES:
            data = user_stats[current_user].get(ex_name_iter)
            if isinstance(data, dict):
                exercises_data_exists = True; reps = data.get("total_reps", 0); cals = data.get("total_calories", 0.0); time_mins = data.get(TOTAL_TIME_MINUTES_KEY, 0.0); lcs, lcr, lcrt = data.get(STATS_SET_KEYS[0]), data.get(STATS_SET_KEYS[1]), data.get(STATS_SET_KEYS[2])
                line1 = f"- {ex_name_iter}: {reps} reps, {cals:.1f} kcal, {time_mins:.1f} mins"; cv2.putText(canvas, line1, (text_list_x_start, current_y_text), FONT, STATS_TEXT_SCALE, COLORS['primary_text'], 1, cv2.LINE_AA); current_y_text += text_line_h
                if lcs is not None: line2 = f"   (Last Config: {lcs}x{lcr}, {lcrt}s rest)"; cv2.putText(canvas, line2, (text_list_x_start, current_y_text), FONT, STATS_TEXT_SCALE * 0.9, COLORS['secondary_text'], 1, cv2.LINE_AA); current_y_text += text_line_h
                else: current_y_text += int(text_line_h * 0.1)
                if current_y_text > h - BUTTON_HEIGHT - BUTTON_MARGIN * 4: break
        if not exercises_data_exists and not user_stats[current_user].get('logged_sessions'): cv2.putText(canvas,"No exercise data recorded yet.",(text_list_x_start,current_y_text),FONT,STATS_TEXT_SCALE,COLORS['secondary_text'],1,cv2.LINE_AA); current_y_text += text_line_h
        current_y_text += BUTTON_MARGIN; cv2.putText(canvas, "Recent Sessions:", (text_list_x_start, current_y_text - 10), FONT, STATS_TITLE_SCALE, COLORS["primary_text"], 1, cv2.LINE_AA); current_y_text += text_line_h // 2
        logged_sessions = user_stats[current_user].get('logged_sessions', [])
        if logged_sessions:
            for session in reversed(logged_sessions[-3:]):
                if current_y_text > h - BUTTON_HEIGHT - BUTTON_MARGIN * 3: cv2.putText(canvas,"...more sessions exist",(text_list_x_start,current_y_text),FONT,STATS_TEXT_SCALE*0.8,COLORS['secondary_text'],1,cv2.LINE_AA); break
                date_str = session.get('date', 'N/A').split(' ')[0]; duration_str = f"{session.get('duration_minutes', 0):.1f} min"; exercises_str = ", ".join(session.get('exercises_performed', [])); (exercises_str := exercises_str[:37] + "..." if len(exercises_str) > 40 else exercises_str)
                session_line = f"- {date_str}: {duration_str} ({exercises_str})"; cv2.putText(canvas, session_line, (text_list_x_start, current_y_text), FONT, STATS_TEXT_SCALE*0.9, COLORS['secondary_text'], 1, cv2.LINE_AA); current_y_text += int(text_line_h * 0.8)
        else: cv2.putText(canvas, "No workout sessions logged.", (text_list_x_start, current_y_text), FONT, STATS_TEXT_SCALE, COLORS['secondary_text'], 1, cv2.LINE_AA)
    else: cv2.putText(canvas, "No user data available.", (text_list_x_start, current_y_text), FONT, STATS_TEXT_SCALE, COLORS['secondary_text'], 1, cv2.LINE_AA)
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2; draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); btn_text_b = "Back to Home"; (tw_b, th_b), _ = cv2.getTextSize(btn_text_b, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text_b, (back_btn_x + (back_btn_w - tw_b) // 2, back_btn_y + (back_btn_h + th_b) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)

def get_wrapped_text_lines_and_dims(text_content, max_line_width_pixels, text_scale, line_spacing_factor, text_thickness=1):
    lines_for_block, max_actual_width_for_block, current_y_offset = [], 0, 0; raw_lines = text_content.split('\n'); (_, text_height_raw), _ = cv2.getTextSize("Ay", FONT, text_scale, text_thickness); line_spacing_pixels = int(text_height_raw * line_spacing_factor); (line_spacing_pixels := text_height_raw + 2 if line_spacing_pixels <= text_height_raw else line_spacing_pixels)
    for line_idx, line_content in enumerate(raw_lines):
        is_list_item = line_content.strip().startswith(("- ", "* ")); display_line_content = line_content; prefix = "•  " if is_list_item else ""; (display_line_content := line_content.strip()[2:].strip() if is_list_item else display_line_content)
        words = display_line_content.split(' ');
        if not words or (len(words)==1 and not words[0]): (lines_for_block.append(""), (current_y_offset := current_y_offset + line_spacing_pixels) if line_idx < len(raw_lines)-1 else None); continue
        current_line_for_wrapping = prefix
        for i, word in enumerate(words):
            test_line = current_line_for_wrapping + word + " "; (tw_test, _), _ = cv2.getTextSize(test_line, FONT, text_scale, text_thickness)
            if tw_test > max_line_width_pixels and current_line_for_wrapping != prefix: final_line = current_line_for_wrapping.strip(); lines_for_block.append(final_line); (tw_final, _), _ = cv2.getTextSize(final_line, FONT, text_scale, text_thickness); max_actual_width_for_block = max(max_actual_width_for_block, tw_final); current_y_offset += line_spacing_pixels; current_line_for_wrapping = prefix + word + " "
            else: current_line_for_wrapping = test_line
        final_line_tail = current_line_for_wrapping.strip()
        if final_line_tail: lines_for_block.append(final_line_tail); (tw_tail, _), _ = cv2.getTextSize(final_line_tail, FONT, text_scale, text_thickness); max_actual_width_for_block = max(max_actual_width_for_block, tw_tail)
        current_y_offset += line_spacing_pixels
    total_height_of_block = current_y_offset - (line_spacing_pixels - text_height_raw) if lines_for_block else current_y_offset
    return lines_for_block, max_actual_width_for_block, total_height_of_block

def draw_chat_ui(canvas):
    global last_chat_error, chat_messages, current_user, is_llm_thinking, chat_input_text, chat_input_active, chat_scroll_offset_y, chat_total_content_height, chat_visible_area_height
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]; title_text = f"AI Coach ({current_user})" if current_user else "AI Coach"; (tw_title, th_title_text), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE * 0.8, LINE_THICKNESS); tx_title = (w - tw_title) // 2; ty_title_top = BUTTON_MARGIN; title_bar_height = th_title_text + BUTTON_MARGIN; cv2.putText(canvas, title_text, (tx_title, ty_title_top + th_title_text + BUTTON_MARGIN // 2 - 5), FONT, SELECT_TITLE_SCALE * 0.8, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA)
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = h - back_btn_h - BUTTON_MARGIN; draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); btn_text_b = "Back"; (tw_b, th_b), _ = cv2.getTextSize(btn_text_b, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text_b, (back_btn_x + (back_btn_w - tw_b) // 2, back_btn_y + (back_btn_h + th_b) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)
    status_x = back_btn_x + back_btn_w + BUTTON_MARGIN * 2; status_y = back_btn_y + back_btn_h // 2; status_text_val = ("Coach is thinking..." if is_llm_thinking else (f"Error: {last_chat_error}" if last_chat_error else "")); status_color = (COLORS["secondary_text"] if is_llm_thinking else (COLORS["accent_red"] if last_chat_error else COLORS["primary_text"]))
    if status_text_val: (tw_st, th_st), _ = cv2.getTextSize(status_text_val, FONT, FEEDBACK_TEXT_SCALE*0.9,1); cv2.putText(canvas, status_text_val, (status_x, status_y + th_st//2), FONT, FEEDBACK_TEXT_SCALE*0.9, status_color, 1, cv2.LINE_AA)
    input_field_x = BUTTON_MARGIN * 2 + CHAT_INPUT_MARGIN; input_field_y = back_btn_y - CHAT_INPUT_MARGIN - CHAT_INPUT_HEIGHT; input_field_w = w - (BUTTON_MARGIN * 4) - (CHAT_INPUT_MARGIN * 2) - 80; input_field_h = CHAT_INPUT_HEIGHT; draw_rounded_rectangle(canvas, (input_field_x, input_field_y), (input_field_x + input_field_w, input_field_y + input_field_h), CHAT_INPUT_BG_COLOR, -1, CORNER_RADIUS//2); draw_rounded_rectangle(canvas, (input_field_x, input_field_y), (input_field_x + input_field_w, input_field_y + input_field_h), COLORS["secondary_text"] if not chat_input_active else COLORS["accent_blue"], 2 if chat_input_active else 1, CORNER_RADIUS//2)
    display_input_text = chat_input_text; current_text_color = CHAT_INPUT_TEXT_COLOR; (display_input_text := display_input_text + "|" if chat_input_active and int(time.time() * 1.5) % 2 == 0 else display_input_text); (display_input_text := "Type your question..." if not chat_input_text and not chat_input_active else display_input_text); (current_text_color := CHAT_PLACEHOLDER_COLOR if not chat_input_text and not chat_input_active else current_text_color); (_, th_input_text), _ = cv2.getTextSize("Ay", FONT, CHAT_TEXT_SCALE, 1); temp_display_input_text = display_input_text; (tw_input_disp, _), _ = cv2.getTextSize(temp_display_input_text, FONT, CHAT_TEXT_SCALE, 1)
    while tw_input_disp > input_field_w - 15 and len(temp_display_input_text) > 1: temp_display_input_text = temp_display_input_text[1:]; (tw_input_disp, _), _ = cv2.getTextSize(temp_display_input_text, FONT, CHAT_TEXT_SCALE, 1)
    cv2.putText(canvas, temp_display_input_text, (input_field_x + 10, input_field_y + input_field_h // 2 + th_input_text // 2), FONT, CHAT_TEXT_SCALE, current_text_color, 1, cv2.LINE_AA)
    send_btn_w, send_btn_h = 70, CHAT_INPUT_HEIGHT; send_btn_x = input_field_x + input_field_w + CHAT_INPUT_MARGIN; send_btn_y = input_field_y; draw_rounded_rectangle(canvas, (send_btn_x, send_btn_y), (send_btn_x + send_btn_w, send_btn_y + send_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS//2); btn_text_send = "Send"; (tw_send, th_send), _ = cv2.getTextSize(btn_text_send, FONT, BUTTON_TEXT_SCALE*0.8,1); cv2.putText(canvas, btn_text_send, (send_btn_x + (send_btn_w - tw_send) // 2, send_btn_y + (send_btn_h + th_send) // 2), FONT, BUTTON_TEXT_SCALE*0.8, COLORS["button_text_active"], 1, cv2.LINE_AA)
    chat_area_x_start = BUTTON_MARGIN * 2; chat_area_y_start = title_bar_height + CHAT_SCROLL_AREA_PADDING // 2; chat_area_w_total = w - (BUTTON_MARGIN * 4); chat_area_content_w = chat_area_w_total - CHAT_SCROLL_BUTTON_SIZE - BUTTON_MARGIN; chat_area_h = input_field_y - CHAT_INPUT_MARGIN - chat_area_y_start - CHAT_SCROLL_AREA_PADDING // 2; chat_visible_area_height = chat_area_h
    ALL_MESSAGES_CANVAS_HEIGHT = max(actual_win_height * 4, 3000); all_messages_canvas = np.full((ALL_MESSAGES_CANVAS_HEIGHT, chat_area_content_w, 3), COLORS["background"], dtype=np.uint8); current_y_on_all_msg_canvas = CHAT_SCROLL_AREA_PADDING; (_, _line_h_calc_raw_height), _ = cv2.getTextSize("Ay", FONT, CHAT_TEXT_SCALE, CHAT_TEXT_THICKNESS); _inter_msg_padding = int(_line_h_calc_raw_height * CHAT_LINE_SPACING_FACTOR * INTER_MESSAGE_PADDING_FACTOR) + CHAT_BUBBLE_PADDING_Y; _line_height_for_drawing = int(_line_h_calc_raw_height * CHAT_LINE_SPACING_FACTOR); (_line_height_for_drawing := _line_h_calc_raw_height + 2 if _line_height_for_drawing <= _line_h_calc_raw_height else _line_height_for_drawing); max_bubble_width = int(chat_area_content_w * CHAT_BUBBLE_MAX_WIDTH_FACTOR); text_width_for_wrapping = max_bubble_width - (2 * CHAT_BUBBLE_PADDING_X) - 5
    for msg_data in chat_messages:
        if msg_data['role'] == 'system': continue
        is_user_msg = msg_data['role'] == 'user'; bubble_bg_color = COLORS["chat_user_bubble_bg"] if is_user_msg else COLORS["chat_ai_bubble_bg"]; bubble_text_color = COLORS["chat_user_bubble_text"] if is_user_msg else COLORS["chat_ai_bubble_text"]; text_for_bubble = msg_data['content']
        lines, text_w, text_h = get_wrapped_text_lines_and_dims(text_for_bubble, text_width_for_wrapping, CHAT_TEXT_SCALE, CHAT_LINE_SPACING_FACTOR, CHAT_TEXT_THICKNESS)
        bubble_w = text_w + 2 * CHAT_BUBBLE_PADDING_X; bubble_h = text_h + 2 * CHAT_BUBBLE_PADDING_Y; bubble_h = max(bubble_h, _line_h_calc_raw_height + 2 * CHAT_BUBBLE_PADDING_Y); bubble_x = (chat_area_content_w - bubble_w - CHAT_SCROLL_AREA_PADDING) if is_user_msg else CHAT_SCROLL_AREA_PADDING
        draw_rounded_rectangle(all_messages_canvas, (bubble_x, current_y_on_all_msg_canvas), (bubble_x + bubble_w, current_y_on_all_msg_canvas + bubble_h), bubble_bg_color, -1, CHAT_BUBBLE_CORNER_RADIUS)
        text_draw_y = current_y_on_all_msg_canvas + CHAT_BUBBLE_PADDING_Y + _line_h_calc_raw_height
        for line_str in lines: cv2.putText(all_messages_canvas, line_str, (bubble_x + CHAT_BUBBLE_PADDING_X, text_draw_y), FONT, CHAT_TEXT_SCALE, bubble_text_color, CHAT_TEXT_THICKNESS, cv2.LINE_AA); text_draw_y += _line_height_for_drawing
        current_y_on_all_msg_canvas += bubble_h + _inter_msg_padding
    chat_total_content_height = current_y_on_all_msg_canvas; max_allowed_scroll_up = max(0, chat_total_content_height - chat_visible_area_height); chat_scroll_offset_y = np.clip(chat_scroll_offset_y, 0, max_allowed_scroll_up); src_y1 = int(max(0, (chat_total_content_height - chat_visible_area_height) - chat_scroll_offset_y)); src_y2 = int(min(src_y1 + chat_visible_area_height, chat_total_content_height)) # Use chat_total_content_height
    if chat_area_h > 0 and chat_area_content_w > 0 and src_y2 > src_y1:
        content_slice = all_messages_canvas[src_y1:src_y2, 0:chat_area_content_w]; h_slice, w_slice = content_slice.shape[:2]; dest_x = chat_area_x_start + CHAT_SCROLL_AREA_PADDING // 2; dest_y = chat_area_y_start
        if h_slice > 0 and w_slice > 0:
            dest_y_final = dest_y + (chat_area_h - h_slice) if h_slice < chat_area_h else dest_y; h_to_blit = min(h_slice, chat_area_h) if h_slice >= chat_area_h else h_slice
            canvas[dest_y_final : dest_y_final + h_to_blit, dest_x : dest_x + w_slice] = content_slice[:h_to_blit, :] if h_slice >= chat_area_h else content_slice
    cv2.rectangle(canvas, (chat_area_x_start + CHAT_SCROLL_AREA_PADDING // 2 -1, chat_area_y_start -1), (chat_area_x_start + CHAT_SCROLL_AREA_PADDING // 2 + chat_area_content_w +1, chat_area_y_start + chat_area_h +1), COLORS["secondary_text"], 1)
    scroll_btn_x = chat_area_x_start + chat_area_content_w + CHAT_SCROLL_AREA_PADDING + BUTTON_MARGIN // 2; scroll_up_y = chat_area_y_start; draw_rounded_rectangle(canvas, (scroll_btn_x, scroll_up_y), (scroll_btn_x + CHAT_SCROLL_BUTTON_SIZE, scroll_up_y + CHAT_SCROLL_BUTTON_SIZE), COLORS["button_bg_normal"], -1, 5); cv2.putText(canvas, "^", (scroll_btn_x + CHAT_SCROLL_BUTTON_SIZE//4 +2, scroll_up_y + int(CHAT_SCROLL_BUTTON_SIZE*0.75) -2), FONT, 0.9, COLORS["primary_text"], 2)
    scroll_down_y = chat_area_y_start + chat_area_h - CHAT_SCROLL_BUTTON_SIZE; draw_rounded_rectangle(canvas, (scroll_btn_x, scroll_down_y), (scroll_btn_x + CHAT_SCROLL_BUTTON_SIZE, scroll_down_y + CHAT_SCROLL_BUTTON_SIZE), COLORS["button_bg_normal"], -1, 5); cv2.putText(canvas, "v", (scroll_btn_x + CHAT_SCROLL_BUTTON_SIZE//4 +2, scroll_down_y + int(CHAT_SCROLL_BUTTON_SIZE*0.75) +2), FONT, 0.9, COLORS["primary_text"], 2)

# --- Pose Landmark Drawing --- MERGED FROM fitness_tracker.py (more advanced)
def draw_pose_landmarks_on_frame(target_image, landmarks_list, connections, form_issue_details):
    """Draws pose landmarks on the target_image, highlighting issues."""
    if not landmarks_list: return # Check if landmark list exists (MediaPipe.python.solution_base.SolutionOutputs)

    h_img, w_img = target_image.shape[:2]
    if h_img == 0 or w_img == 0: return

    # Use color definitions from Fitness_Tracker_ui.py's COLORS dictionary
    default_landmark_spec = mp_drawing.DrawingSpec(color=COLORS["landmark_vis"], thickness=1, circle_radius=2)
    problem_landmark_spec = mp_drawing.DrawingSpec(color=COLORS["landmark_issue"], thickness=-1, circle_radius=4) # Filled, larger
    default_connection_spec = mp_drawing.DrawingSpec(color=COLORS["connection"], thickness=1)
    problem_connection_spec = mp_drawing.DrawingSpec(color=COLORS["landmark_issue"], thickness=2) # Thicker error connection

    relevant_joint_indices = set()
    # Mapping uses landmark ENUM value (integer)
    mapping = { # Updated to use mp_pose.PoseLandmark for clarity if needed, but direct values are fine
        "BACK": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
        "LEFT_KNEE": [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
        "RIGHT_KNEE": [mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        "LEFT_ELBOW": [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
        "RIGHT_ELBOW": [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
        "LEFT_UPPER_ARM": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value],
        "RIGHT_UPPER_ARM": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        "BODY": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value], # Extended for pushup body line
        "HIPS": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    }
    num_landmarks_available = len(landmarks_list.landmark) if landmarks_list.landmark else 0
    for part_name in form_issue_details:
        if part_name in mapping:
            for lm_index_value in mapping[part_name]:
                 if 0 <= lm_index_value < num_landmarks_available:
                    relevant_joint_indices.add(lm_index_value)

    custom_connection_specs = {}
    if connections: # connections is mp_pose.POSE_CONNECTIONS
        for connection_pair in connections: # Each connection is a tuple (start_idx_enum, end_idx_enum)
            # Convert enum to value if they are not already integers
            start_idx = connection_pair[0].value if hasattr(connection_pair[0], 'value') else connection_pair[0]
            end_idx = connection_pair[1].value if hasattr(connection_pair[1], 'value') else connection_pair[1]
            
            current_spec = default_connection_spec
            if start_idx in relevant_joint_indices or end_idx in relevant_joint_indices:
                current_spec = problem_connection_spec
            custom_connection_specs[connection_pair] = current_spec
    
    try:
        mp_drawing.draw_landmarks(
            image=target_image,
            landmark_list=landmarks_list, # This is the SolutionOutputs.pose_landmarks object
            connections=connections,
            landmark_drawing_spec=default_landmark_spec, # Default for all landmarks initially
            connection_drawing_spec=custom_connection_specs # Use the custom dict for connections
        )
        # Redraw problematic landmarks on top with problem_spec
        if landmarks_list.landmark: # Ensure landmark data exists
            for idx_value in relevant_joint_indices:
                 if idx_value < num_landmarks_available: # Check index bounds again
                     lm = landmarks_list.landmark[idx_value]
                     if lm.visibility > 0.5: # Only redraw visible problematic landmarks
                         cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                         cv2.circle(target_image, (np.clip(cx,0,w_img-1), np.clip(cy,0,h_img-1)),
                                    problem_landmark_spec.circle_radius,
                                    problem_landmark_spec.color,
                                    problem_landmark_spec.thickness)
    except Exception as draw_error: print(f"Error during mp_drawing.draw_landmarks: {draw_error}")


# --- Main Application Loop ---
load_data()
if not configure_gemini(): print("WARNING: Gemini could not be configured. Chatbot will be non-functional.")

window_name = 'Fitness Tracker Pro - AI Coach v2'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
if platform.system() == "Windows": cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else: cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL); cv2.resizeWindow(window_name, default_win_width, default_win_height)

try: rect_init = cv2.getWindowImageRect(window_name); actual_win_width, actual_win_height = (rect_init[2], rect_init[3]) if rect_init and len(rect_init) == 4 and rect_init[2] > 0 and rect_init[3] > 0 else (default_win_width, default_win_height)
except Exception: actual_win_width, actual_win_height = default_win_width, default_win_height
callback_param = {'canvas_w': actual_win_width, 'canvas_h': actual_win_height}
cv2.setMouseCallback(window_name, mouse_callback, callback_param)

pose = None
try: pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55, model_complexity=1)
except Exception as e: print(f"FATAL: Pose model init: {e}"); exit()

while True:
    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect and len(rect)==4 and rect[2]>1 and rect[3]>1 and (rect[2]!=actual_win_width or rect[3]!=actual_win_height):
             actual_win_width,actual_win_height=rect[2],rect[3]; callback_param['canvas_w'],callback_param['canvas_h']=actual_win_width,actual_win_height; stats_pie_image=None; stats_time_plot_image=None; last_frame_for_rest=None; print(f"Window Resized: {actual_win_width}x{actual_win_height}")
    except cv2.error as e: 
        error_message = str(e).lower() # Make it case-insensitive
        if "null window" in error_message or "invalid window" in error_message:
            print(f"Window closed or invalid, exiting loop: {e}")
            break # Exit the loop
        else:
            print(f"Window state check error (cv2.error): {e}")
    except Exception as e_resize: print(f"Error checking window size: {e_resize}")

    if tk_root_main:
        try:
            tk_root_main.update_idletasks()
            tk_root_main.update()
        except tk.TclError:
            tk_root_main = None

    canvas_created_for_mode = False
    if app_mode == "REST" and last_frame_for_rest is not None and last_frame_for_rest.shape[0]==actual_win_height and last_frame_for_rest.shape[1]==actual_win_width: canvas = last_frame_for_rest.copy(); canvas_created_for_mode = True
    if not canvas_created_for_mode:
         if actual_win_height <= 0 or actual_win_width <= 0: time.sleep(0.1); continue
         canvas = np.zeros((actual_win_height, actual_win_width, 3), dtype=np.uint8); canvas[:] = COLORS['background']

    if app_mode == "HOME": draw_home_ui(canvas)
    elif app_mode == "EXERCISE_SELECT": draw_exercise_select_ui(canvas)
    elif app_mode == "SET_SELECTION": draw_set_selection_ui(canvas)
    elif app_mode == "GUIDE":
        if time.time() - guide_start_time > guide_duration and guide_gif_frames: app_mode="TRACKING"; (session_exercise_start_time := time.time() if is_webcam_source else None); feedback_list=[f"Start Set {current_set_number}/{target_sets}" if set_config_confirmed else f"Start {current_exercise} (Free Play)"]
        else: draw_guide_ui(canvas)
    elif app_mode == "STATS": draw_stats_ui(canvas)
    elif app_mode == "CHAT":
        if is_llm_thinking:
            context_str = gather_context_for_llm(current_user); system_prompt_content = chat_messages[0]['content'] if chat_messages and chat_messages[0]['role'] == 'system' else ""; latest_question_content = chat_messages[-1]['content'] if chat_messages and chat_messages[-1]["role"] == "user" else ""
            if latest_question_content:
                full_prompt_for_turn = f"System Instructions (for AI Fitness Coach):\n{system_prompt_content}\n\nUser Context (Profile & Stats):\n{context_str}\n\n---\nUser's Current Question: {latest_question_content}"; ai_response = get_llm_response(full_prompt_for_turn)
                chat_messages.append({"role": "assistant", "content": ai_response if ai_response else f"Sorry, I couldn't get a response. ({last_chat_error or 'Unknown API issue'})"}); chat_scroll_offset_y = 0
            else: last_chat_error = "Internal: No user question to process."; chat_messages.append({"role": "assistant", "content": "It seems there was no question to send."})
            is_llm_thinking = False
        draw_chat_ui(canvas)
    elif app_mode == "REST":
        if rest_start_time and time.time()-rest_start_time >= target_rest_time: app_mode="TRACKING"; (session_exercise_start_time := time.time() if is_webcam_source else None); feedback_list=[f"Start Set {current_set_number}/{target_sets}"]
        else: draw_rest_ui(canvas)
    elif app_mode == "TRACKING":
        if not cap or not cap.isOpened(): feedback_list=["Error: Video source lost."];end_session();continue
        ret,frame=cap.read()
        if not ret: (end_session() if source_type=='video' else (feedback_list:=["Error: Webcam frame."], end_session())); continue
        if frame.shape[0]<=0 or frame.shape[1]<=0: continue
        if session_exercise_start_time is None and is_webcam_source: session_exercise_start_time = time.time()
        
        results=None
        try: img_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);img_rgb.flags.writeable=False;results=pose.process(img_rgb)
        except Exception as e: feedback_list=["Error processing pose."];draw_tracking_ui(canvas,frame,None);continue
        
        feedback_list=[];form_correct_overall=True;form_issues_details.clear()
        try:
            if results and results.pose_landmarks:
                landmarks_mp_obj = results.pose_landmarks # This is the MediaPipe SolutionOutputs object
                landmarks_raw_list = landmarks_mp_obj.landmark # This is the list of NormalizedLandmark
                
                sh_l, sh_r=get_coords(landmarks_raw_list,'LEFT_SHOULDER'), get_coords(landmarks_raw_list,'RIGHT_SHOULDER'); hip_l, hip_r=get_coords(landmarks_raw_list,'LEFT_HIP'), get_coords(landmarks_raw_list,'RIGHT_HIP'); kn_l, kn_r=get_coords(landmarks_raw_list,'LEFT_KNEE'), get_coords(landmarks_raw_list,'RIGHT_KNEE'); an_l, an_r=get_coords(landmarks_raw_list,'LEFT_ANKLE'), get_coords(landmarks_raw_list,'RIGHT_ANKLE'); el_l, el_r=get_coords(landmarks_raw_list,'LEFT_ELBOW'), get_coords(landmarks_raw_list,'RIGHT_ELBOW'); wr_l, wr_r=get_coords(landmarks_raw_list,'LEFT_WRIST'), get_coords(landmarks_raw_list,'RIGHT_WRIST'); nose=get_coords(landmarks_raw_list, 'NOSE')
                
                # Calculate RAW angles
                angle_l_elbow_raw=calculate_angle(sh_l, el_l, wr_l); angle_r_elbow_raw=calculate_angle(sh_r, el_r, wr_r); angle_l_knee_raw=calculate_angle(hip_l, kn_l, an_l); angle_r_knee_raw=calculate_angle(hip_r, kn_r, an_r); angle_l_hip_raw=calculate_angle(sh_l, hip_l, kn_l); angle_r_hip_raw=calculate_angle(sh_r, hip_r, kn_r)
                angle_l_body_shk_raw = calculate_angle(sh_l, hip_l, kn_l); angle_r_body_shk_raw = calculate_angle(sh_r, hip_r, kn_r) # Shoulder-Hip-Knee
                angle_l_body_sha_raw = calculate_angle(sh_l, hip_l, an_l); angle_r_body_sha_raw = calculate_angle(sh_r, hip_r, an_r) # Shoulder-Hip-Ankle (for pushup)
                
                # Apply EMA Smoothing
                l_elbow_logic=update_ema(angle_l_elbow_raw,"LEFT_ELBOW",ema_angles); r_elbow_logic=update_ema(angle_r_elbow_raw,"RIGHT_ELBOW",ema_angles); l_knee_logic=update_ema(angle_l_knee_raw,"LEFT_KNEE",ema_angles); r_knee_logic=update_ema(angle_r_knee_raw,"RIGHT_KNEE",ema_angles); l_hip_logic=update_ema(angle_l_hip_raw,"LEFT_HIP",ema_angles); r_hip_logic=update_ema(angle_r_hip_raw,"RIGHT_HIP",ema_angles)
                l_body_shk_logic=update_ema(angle_l_body_shk_raw,"LEFT_BODY_SHK",ema_angles); r_body_shk_logic=update_ema(angle_r_body_shk_raw,"RIGHT_BODY_SHK",ema_angles)
                l_body_sha_logic=update_ema(angle_l_body_sha_raw,"LEFT_BODY_SHA",ema_angles); r_body_sha_logic=update_ema(angle_r_body_sha_raw,"RIGHT_BODY_SHA",ema_angles)

                # Averaged EMA Angles for Logic
                avg_knee_angle=(l_knee_logic+r_knee_logic)/2 if(kn_l[3]>0.5 and kn_r[3]>0.5)else(l_knee_logic if kn_l[3]>0.5 else r_knee_logic if kn_r[3]>0.5 else 90)
                avg_hip_angle=(l_hip_logic+r_hip_logic)/2 if(hip_l[3]>0.5 and hip_r[3]>0.5)else(l_hip_logic if hip_l[3]>0.5 else r_hip_logic if hip_r[3]>0.5 else 90)
                avg_elbow_angle=(l_elbow_logic+r_elbow_logic)/2 if(el_l[3]>0.5 and el_r[3]>0.5)else(l_elbow_logic if el_l[3]>0.5 else r_elbow_logic if el_r[3]>0.5 else 180)
                avg_body_angle_shk=(l_body_shk_logic+r_body_shk_logic)/2 if(hip_l[3]>0.5 and hip_r[3]>0.5)else(l_body_shk_logic if hip_l[3]>0.5 else r_body_shk_logic if hip_r[3]>0.5 else 180) # SHK for general body
                avg_body_angle_sha=(l_body_sha_logic+r_body_sha_logic)/2 if(hip_l[3]>0.5 and hip_r[3]>0.5 and an_l[3]>0.5 and an_r[3]>0.5) else (l_body_sha_logic if hip_l[3]>0.5 and an_l[3]>0.5 else (r_body_sha_logic if hip_r[3]>0.5 and an_r[3]>0.5 else 180)) # SHA for pushup straightness

                # Master Form Check: Back Angle (Vertical) - FROM fitness_tracker.py
                vertical_back_ok=True; back_angle_vertical=90 # Default to upright
                vis_sh=[s for s in[sh_l,sh_r]if s[3]>0.6]; vis_hip=[h for h in[hip_l,hip_r]if h[3]>0.6]
                if len(vis_sh)>0 and len(vis_hip)>0:
                    sh_avg_pt=np.mean([s[:2]for s in vis_sh],axis=0); hip_avg_pt=np.mean([h[:2]for h in vis_hip],axis=0); vec_hs=sh_avg_pt-hip_avg_pt; vec_vert=np.array([0,-1]); norm_hs=np.linalg.norm(vec_hs)
                    if norm_hs > 1e-6: back_angle_vertical=np.degrees(np.arccos(np.clip(np.dot(vec_hs,vec_vert)/norm_hs,-1.0,1.0)))
                
                current_vert_thresh = 90 # Default lenient
                if current_exercise == "BICEP CURL": current_vert_thresh = BACK_ANGLE_THRESHOLD_BICEP
                elif current_exercise == "SQUAT": current_vert_thresh = BACK_ANGLE_THRESHOLD_SQUAT
                if current_exercise not in ["DEADLIFT", "PUSH UP"] and back_angle_vertical > current_vert_thresh: # Exclude deadlift/pushup from this general check
                     vertical_back_ok=False; add_feedback(f"Back Angle ({back_angle_vertical:.0f}° > {current_vert_thresh}°)",True); add_form_issue("BACK")

                ct = time.time(); rep_counted_this_frame = False; set_completed_this_frame = False

                # === BICEP CURL === (Using fitness_tracker.py logic)
                if current_exercise == "BICEP CURL":
                    # Form Check: Upper Arm Stability (using get_segment_vertical_angle)
                    upper_arm_vert_angle_l = get_segment_vertical_angle(sh_l, el_l) # Angle with vertical DOWN
                    upper_arm_vert_angle_r = get_segment_vertical_angle(sh_r, el_r)
                    if upper_arm_vert_angle_l is not None and abs(upper_arm_vert_angle_l) > BICEP_UPPER_ARM_VERT_DEVIATION and abs(upper_arm_vert_angle_l - 180) > BICEP_UPPER_ARM_VERT_DEVIATION : # Deviates from pointing down (0) or up (180)
                         add_feedback("L Upper Arm Still", True); add_form_issue("LEFT_UPPER_ARM"); form_correct_overall=False
                    if upper_arm_vert_angle_r is not None and abs(upper_arm_vert_angle_r) > BICEP_UPPER_ARM_VERT_DEVIATION and abs(upper_arm_vert_angle_r - 180) > BICEP_UPPER_ARM_VERT_DEVIATION:
                         add_feedback("R Upper Arm Still", True); add_form_issue("RIGHT_UPPER_ARM"); form_correct_overall=False
                    if not vertical_back_ok: form_correct_overall=False # Also consider general back lean

                    # Rep Counting (Left)
                    if all(p[3] > 0.5 for p in [sh_l, el_l, wr_l]):
                        if stage_left is None: stage_left = "INIT" # Or "DOWN" if starting extended
                        if l_elbow_logic > BICEP_DOWN_ENTER_ANGLE and stage_left != "DOWN": stage_left = "DOWN"
                        elif l_elbow_logic < BICEP_UP_ENTER_ANGLE and stage_left == "DOWN":
                            stage_left = "UP";
                            if form_correct_overall and ct - last_rep_time_left > rep_cooldown:
                                counter_left += 1; last_rep_time_left = ct; rep_counted_this_frame = True
                                if is_webcam_source: session_reps[current_exercise] = session_reps.get(current_exercise, 0) + 1 # Count bilateral as one rep for session
                                add_feedback(f"L: Rep {counter_left}!", False)
                                if set_config_confirmed and counter_left >= target_reps_per_set and counter_right >= target_reps_per_set: set_completed_this_frame = True
                            elif not form_correct_overall: add_feedback("L: Fix Form", True)
                        if stage_left == "UP" and l_elbow_logic > BICEP_UP_EXIT_ANGLE: add_feedback("L: Lower More", False) # Hysteresis feedback
                        if stage_left == "DOWN" and l_elbow_logic < BICEP_DOWN_EXIT_ANGLE: add_feedback("L: Curl Higher", False) # Hysteresis feedback

                    # Rep Counting (Right)
                    if all(p[3] > 0.5 for p in [sh_r, el_r, wr_r]):
                       if stage_right is None: stage_right = "INIT"
                       if r_elbow_logic > BICEP_DOWN_ENTER_ANGLE and stage_right != "DOWN": stage_right = "DOWN"
                       elif r_elbow_logic < BICEP_UP_ENTER_ANGLE and stage_right == "DOWN":
                           stage_right = "UP"
                           if form_correct_overall and ct - last_rep_time_right > rep_cooldown:
                               counter_right += 1; last_rep_time_right = ct
                               if is_webcam_source and not rep_counted_this_frame : session_reps[current_exercise] = session_reps.get(current_exercise, 0) + 1 # Count if left didn't already log it for session
                               rep_counted_this_frame = True # Mark a rep was counted for this frame (either L or R)
                               add_feedback(f"R: Rep {counter_right}!", False)
                               if set_config_confirmed and counter_left >= target_reps_per_set and counter_right >= target_reps_per_set: set_completed_this_frame = True
                           elif not form_correct_overall: add_feedback("R: Fix Form", True)
                       if stage_right == "UP" and r_elbow_logic > BICEP_UP_EXIT_ANGLE: add_feedback("R: Lower More", False)
                       if stage_right == "DOWN" and r_elbow_logic < BICEP_DOWN_EXIT_ANGLE: add_feedback("R: Curl Higher", False)
                
                # === OTHER EXERCISES (SQUAT, PUSH UP, PULL UP, DEADLIFT) ===
                else:
                    current_form_ok_for_rep = True # Specific form for this exercise rep
                    if not vertical_back_ok and current_exercise in ["SQUAT"]: # General back check for squat
                        current_form_ok_for_rep = False

                    if current_exercise == "SQUAT":
                        if kn_l[3]>0.5 and an_l[3]>0.5 and kn_l[0] < an_l[0] - SQUAT_KNEE_VALGUS_THRESHOLD: add_feedback("L Knee In?", True); add_form_issue("LEFT_KNEE"); current_form_ok_for_rep=False
                        if kn_r[3]>0.5 and an_r[3]>0.5 and kn_r[0] > an_r[0] + SQUAT_KNEE_VALGUS_THRESHOLD: add_feedback("R Knee Out?", True); add_form_issue("RIGHT_KNEE"); current_form_ok_for_rep=False
                        # Chest forward relative to knee (X-coord)
                        avg_shoulder_x = (sh_l[0] + sh_r[0]) / 2 if sh_l[3] > 0.5 and sh_r[3] > 0.5 else (sh_l[0] if sh_l[3] > 0.5 else sh_r[0])
                        avg_knee_x = (kn_l[0] + kn_r[0]) / 2 if kn_l[3] > 0.5 and kn_r[3] > 0.5 else (kn_l[0] if kn_l[3] > 0.5 else kn_r[0])
                        if stage == "DOWN" and avg_shoulder_x < avg_knee_x - SQUAT_CHEST_FORWARD_THRESHOLD: add_feedback("Chest Up More", True); add_form_issue("BACK"); current_form_ok_for_rep=False
                        up_angle, down_angle = SQUAT_UP_ENTER_ANGLE, SQUAT_DOWN_ENTER_ANGLE
                        up_exit, down_exit = SQUAT_UP_EXIT_ANGLE, SQUAT_DOWN_EXIT_ANGLE
                        logic_angle = avg_knee_angle
                        up_feedback, down_feedback = "Stand Fully", "Go Deeper"
                        # Visible check for squat: at least one full leg (hip, knee, ankle) for avg_knee_angle
                        visible_check = (hip_l[3] > 0.5 and kn_l[3] > 0.5 and an_l[3] > 0.5) or \
                                        (hip_r[3] > 0.5 and kn_r[3] > 0.5 and an_r[3] > 0.5)

                    elif current_exercise == "PUSH UP":
                         if avg_body_angle_sha < PUSHUP_BODY_STRAIGHT_MIN or avg_body_angle_sha > PUSHUP_BODY_STRAIGHT_MAX: add_feedback(f"Body Straight ({avg_body_angle_sha:.0f}deg)", True); add_form_issue("BODY"); current_form_ok_for_rep=False
                        # Relaxed visibility check for push-up rep counting: only elbows and shoulders needed.
                        # Body straightness (SHA) check above will use available landmarks or default.
                                # Push-up specific form check: Body Straightness (SHA angle)
                         # Only check body straightness if ankles are visible
                         if an_l[3] > 0.5 and an_r[3] > 0.5:
                             if avg_body_angle_sha < PUSHUP_BODY_STRAIGHT_MIN or avg_body_angle_sha > PUSHUP_BODY_STRAIGHT_MAX:
                                 add_feedback(f"Body Straight ({avg_body_angle_sha:.0f}deg)", True); add_form_issue("BODY"); current_form_ok_for_rep=False

                         up_angle, down_angle = PUSHUP_UP_ENTER_ANGLE, PUSHUP_DOWN_ENTER_ANGLE
                         # Visible check for push-up: at least one full arm (shoulder, elbow, wrist) for avg_elbow_angle
                         visible_check = (sh_l[3] > 0.5 and el_l[3] > 0.5 and wr_l[3] > 0.5) or \
                                         (sh_r[3] > 0.5 and el_r[3] > 0.5 and wr_r[3] > 0.5)
                         up_exit, down_exit = PUSHUP_UP_EXIT_ANGLE, PUSHUP_DOWN_EXIT_ANGLE
                         logic_angle = avg_elbow_angle
                         up_feedback, down_feedback = "Extend Arms", "Lower Chest"

                    elif current_exercise == "PULL UP":
                        avg_wrist_y = (wr_l[1] + wr_r[1]) / 2 if (wr_l[3] > 0.5 and wr_r[3] > 0.5) else (wr_l[1] if wr_l[3] > 0.5 else (wr_r[1] if wr_r[3] > 0.5 else 1.0)) # Default below nose
                        nose_y = nose[1] if nose[3] > 0.5 else 0.0 # Default above wrist
                        
                        # For Pull Up, "UP" state is low elbow angle and chin above wrist, "DOWN" state is high elbow angle
                        up_condition_met = (avg_elbow_angle < PULLUP_UP_ENTER_ELBOW_ANGLE) and (nose_y < avg_wrist_y if PULLUP_CHIN_ABOVE_WRIST else True)
                        down_condition_met = avg_elbow_angle > PULLUP_DOWN_ENTER_ANGLE
                        
                        # Rep counting: transition from DOWN to UP
                        up_angle_for_logic = PULLUP_UP_ENTER_ELBOW_ANGLE # This is a bit inverted for standard logic
                        down_angle_for_logic = PULLUP_DOWN_ENTER_ANGLE
                        # Hysteresis for pull-ups
                        up_exit_angle = PULLUP_UP_EXIT_ELBOW_ANGLE
                        down_exit_angle = PULLUP_DOWN_EXIT_ANGLE
                        up_feedback, down_feedback = "Pull Higher", "Extend Arms"
                        visible_check = (el_l[3] > 0.5 or el_r[3] > 0.5) and nose[3] > 0.5 and (wr_l[3]>0.5 or wr_r[3]>0.5)

                    elif current_exercise == "DEADLIFT":
                        lockout_form_ok, lift_form_ok = True, True
                        if stage == "DOWN" or (stage == "INIT" and avg_hip_angle < 150): # Check during lift/setup
                            if back_angle_vertical > BACK_ANGLE_THRESHOLD_DEADLIFT_LIFT: lift_form_ok = False; add_feedback(f"Back Round? ({back_angle_vertical:.0f}deg)", True); add_form_issue("BACK")
                        is_potentially_up = avg_hip_angle > DEADLIFT_UP_EXIT_ANGLE and avg_knee_angle > DEADLIFT_UP_EXIT_ANGLE
                        if stage == "UP" or (stage != "UP" and is_potentially_up): # Check near/at lockout (stage != "UP" handles coming up from DOWN)
                            if back_angle_vertical > BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT: lockout_form_ok = False; add_feedback(f"Lockout Back ({back_angle_vertical:.0f}deg)", True); add_form_issue("BACK")
                        current_form_ok_for_rep = lockout_form_ok and lift_form_ok
                        
                        up_angle, down_angle_hip, down_angle_knee = DEADLIFT_UP_ENTER_ANGLE, DEADLIFT_DOWN_ENTER_HIP_ANGLE, DEADLIFT_DOWN_ENTER_KNEE_ANGLE
                        up_exit, down_exit_hip, down_exit_knee = DEADLIFT_UP_EXIT_ANGLE, DEADLIFT_DOWN_EXIT_HIP_ANGLE, DEADLIFT_DOWN_EXIT_KNEE_ANGLE
                        up_feedback, down_feedback = "Lockout Hips/Knees", "Lower Bar"
                        visible_check = all(p[3] > 0.5 for p in [kn_l, kn_r, hip_l, hip_r, sh_l, sh_r])

                    # Generic Rep Counting Logic (adapted for pull-ups and deadlift)
                    if visible_check:
                        if stage is None: stage = "INIT" # Or "UP" for exercises starting at top

                        # Define conditions for UP and DOWN state based on exercise
                        is_up_state_achieved, is_down_state_achieved = False, False
                        if current_exercise == "PULL UP":
                            is_up_state_achieved = (avg_elbow_angle < PULLUP_UP_ENTER_ELBOW_ANGLE) and (nose_y < avg_wrist_y if PULLUP_CHIN_ABOVE_WRIST else True)
                            is_down_state_achieved = avg_elbow_angle > PULLUP_DOWN_ENTER_ANGLE
                        elif current_exercise == "DEADLIFT":
                            is_up_state_achieved = avg_hip_angle > DEADLIFT_UP_ENTER_ANGLE and avg_knee_angle > DEADLIFT_UP_ENTER_ANGLE
                            is_down_state_achieved = avg_hip_angle < DEADLIFT_DOWN_ENTER_HIP_ANGLE and avg_knee_angle < DEADLIFT_DOWN_ENTER_KNEE_ANGLE
                        else: # Squat, Push Up
                            is_up_state_achieved = logic_angle > up_angle
                            is_down_state_achieved = logic_angle < down_angle
                        
                        # State Transitions
                        if current_exercise == "PULL UP": # Pull up logic is: Down (extended) -> Up (pulled) -> Rep -> Down
                            if is_down_state_achieved and stage != "DOWN": # Entering DOWN (extended)
                                stage = "DOWN"
                            elif is_up_state_achieved and stage == "DOWN": # Moving from DOWN to UP
                                stage = "UP"
                                if current_form_ok_for_rep and ct - last_rep_time > rep_cooldown:
                                    counter += 1; last_rep_time = ct; rep_counted_this_frame = True
                                    if is_webcam_source: session_reps[current_exercise] = session_reps.get(current_exercise, 0) + 1
                                    add_feedback(f"Rep {counter}!", False)
                                    if set_config_confirmed and counter >= target_reps_per_set: set_completed_this_frame = True
                                elif not current_form_ok_for_rep: add_feedback("Fix Form!", True)
                            # Hysteresis feedback for Pull Up
                            if stage == "UP" and avg_elbow_angle > PULLUP_UP_EXIT_ELBOW_ANGLE: add_feedback(up_feedback, False)
                            if stage == "DOWN" and avg_elbow_angle < PULLUP_DOWN_EXIT_ANGLE: add_feedback(down_feedback, False)
                        else: # Squat, Push Up, Deadlift (Up -> Down -> Rep -> Up)
                            if is_up_state_achieved and stage != "UP": # Typically means just completed a rep and returned to UP
                                if stage == "DOWN": # This is the rep count moment
                                    if current_form_ok_for_rep and ct - last_rep_time > rep_cooldown:
                                        counter += 1; last_rep_time = ct; rep_counted_this_frame = True
                                        if is_webcam_source: session_reps[current_exercise] = session_reps.get(current_exercise, 0) + 1
                                        add_feedback(f"Rep {counter}!", False)
                                        if set_config_confirmed and counter >= target_reps_per_set: set_completed_this_frame = True
                                    elif not current_form_ok_for_rep: add_feedback("Fix Form!", True)
                                stage = "UP"
                            elif is_down_state_achieved and stage != "DOWN": # Moving from UP to DOWN
                                stage = "DOWN"
                            
                            # Hysteresis feedback for Squat, Push Up, Deadlift
                            if stage == "UP":
                                if current_exercise == "DEADLIFT" and not (avg_hip_angle > DEADLIFT_UP_EXIT_ANGLE and avg_knee_angle > DEADLIFT_UP_EXIT_ANGLE): add_feedback(up_feedback, False)
                                elif current_exercise != "DEADLIFT" and logic_angle < up_exit : add_feedback(up_feedback, False)
                            if stage == "DOWN":
                                if current_exercise == "DEADLIFT" and not (avg_hip_angle < DEADLIFT_DOWN_EXIT_HIP_ANGLE and avg_knee_angle < DEADLIFT_DOWN_EXIT_KNEE_ANGLE): add_feedback(down_feedback, False)
                                elif current_exercise != "DEADLIFT" and logic_angle > down_exit: add_feedback(down_feedback, False)
                    else: add_feedback("Body Not Fully Visible", True); stage=None

                # --- Set Completion Logic ---
                if set_completed_this_frame:
                    add_feedback(f"Set {current_set_number} Complete!", False); finalize_last_exercise_duration(current_exercise)
                    if current_set_number < target_sets: current_set_number += 1; app_mode = "REST"; rest_start_time = time.time(); counter = 0; counter_left = 0; counter_right = 0; stage = None; stage_left = None; stage_right = None; print(f"Starting rest before set {current_set_number}")
                    else: feedback_list = ["Workout Complete! Well Done!"]; print("All sets completed."); end_session(); continue
            else: add_feedback("No Person Detected", True); stage=stage_left=stage_right=None # No landmarks
        except Exception as e_logic: print(f"!! Logic Error: {e_logic}"); traceback.print_exc(); add_feedback("Processing Error", True); stage=stage_left=stage_right=None

        if not feedback_list and stage is not None and app_mode == "TRACKING": (add_feedback("Keep Going...", False) if form_correct_overall else None)
        if app_mode == "TRACKING": draw_tracking_ui(canvas, frame, results) # Pass results, not landmarks_mp_obj

    if 'canvas' in locals() and isinstance(canvas, np.ndarray) and canvas.shape[0]>0 and canvas.shape[1]>0:
        try: cv2.imshow(window_name,canvas)
        except cv2.error as e:
            error_message_imshow = str(e).lower()
            if "null window" in error_message_imshow or "invalid window" in error_message_imshow:
                print(f"Window closed or invalid during imshow, exiting loop: {e}")
                break # Exit the loop
            else:
                print(f"cv2.imshow error: {e}")
                break # Also break on other imshow errors for safety
    else:
        try: err_canvas = np.zeros((default_win_height, default_win_width, 3), dtype=np.uint8); err_canvas[:] = COLORS['accent_red']; cv2.putText(err_canvas, "Canvas Error or Zero Size", (50,100), FONT, 1, (255,255,255), 2); cv2.imshow(window_name, err_canvas)
        except cv2.error: print("Failed to show error canvas, window likely closed."); break

    key = cv2.waitKey(1) & 0xFF
    if app_mode == "CHAT" and chat_input_active:
        if key == 8: chat_input_text = chat_input_text[:-1]
        elif key == 13:
            if chat_input_text.strip() and not is_llm_thinking: chat_messages.append({"role": "user", "content": chat_input_text.strip()}); is_llm_thinking = True; chat_input_text = ""; last_chat_error = None; chat_scroll_offset_y = 0
        elif key == 27: chat_input_active = False
        elif 32 <= key <= 126: chat_input_text += chr(key)
    elif key == ord('q'): print("Quit key pressed."); break

print("Releasing resources..."); (cap.release() if cap else None); print("Video capture released."); (pose.close() if pose else None); print("Pose model closed."); cv2.destroyAllWindows(); plt.close('all'); (tk_root_main.destroy() if tk_root_main else None); print("Application Closed.")
