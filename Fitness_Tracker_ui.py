# --- Imports ---
import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog # Added simpledialog for easier input initially
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

# --- Configuration Constants ---

# Rep Counting Thresholds (Hysteresis) - From test.py / fitness_tracker.py
BICEP_UP_ENTER_ANGLE = 55; BICEP_UP_EXIT_ANGLE = 70
BICEP_DOWN_ENTER_ANGLE = 155; BICEP_DOWN_EXIT_ANGLE = 140
SQUAT_UP_ENTER_ANGLE = 165; SQUAT_UP_EXIT_ANGLE = 155
SQUAT_DOWN_ENTER_ANGLE = 100; SQUAT_DOWN_EXIT_ANGLE = 110
PUSHUP_UP_ENTER_ANGLE = 155; PUSHUP_UP_EXIT_ANGLE = 145
PUSHUP_DOWN_ENTER_ANGLE = 95; PUSHUP_DOWN_EXIT_ANGLE = 105
PULLUP_UP_ENTER_ELBOW_ANGLE = 80; PULLUP_UP_EXIT_ELBOW_ANGLE = 95
PULLUP_DOWN_ENTER_ANGLE = 160; PULLUP_DOWN_EXIT_ANGLE = 150
PULLUP_CHIN_ABOVE_WRIST = True
DEADLIFT_UP_ENTER_ANGLE = 168; DEADLIFT_UP_EXIT_ANGLE = 158
DEADLIFT_DOWN_ENTER_HIP_ANGLE = 120; DEADLIFT_DOWN_ENTER_KNEE_ANGLE = 135
DEADLIFT_DOWN_EXIT_HIP_ANGLE = 130; DEADLIFT_DOWN_EXIT_KNEE_ANGLE = 145

# Form Correction Thresholds - From test.py / fitness_tracker.py
BACK_ANGLE_THRESHOLD_BICEP = 20; BACK_ANGLE_THRESHOLD_SQUAT = 45
BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT = 15; BACK_ANGLE_THRESHOLD_DEADLIFT_LIFT = 60
PUSHUP_BODY_STRAIGHT_MIN = 150; PUSHUP_BODY_STRAIGHT_MAX = 190
SQUAT_KNEE_VALGUS_THRESHOLD = 0.05; SQUAT_CHEST_FORWARD_THRESHOLD = 0.1
BICEP_UPPER_ARM_VERT_DEVIATION = 25

# EMA Smoothing - From test.py / fitness_tracker.py
EMA_ALPHA = 0.3

# Statistics Constants - From test.py
PROFILES_FILE = 'profiles.json'; STATS_FILE = 'stats.json'
MET_VALUES = {"BICEP CURL": 3.5, "SQUAT": 5.0, "PUSH UP": 4.0, "PULL UP": 8.0, "DEADLIFT": 6.0, "DEFAULT": 4.0}
# ** NEW ** Keys for saving set config in stats
STATS_SET_KEYS = ['last_config_sets', 'last_config_reps', 'last_config_rest']


# --- UI Constants --- (Adapted from fitness_tracker_ui.py, Profile colors added)
GIF_DIR = "GIFs" # Relative path to GIF directory
EXERCISES = ["BICEP CURL", "SQUAT", "PUSH UP", "PULL UP", "DEADLIFT"]
EXERCISE_GIF_MAP = {
    "BICEP CURL": "bicep.gif",
    "SQUAT": "squats.gif",
    "PUSH UP": "pushup.gif",
    "PULL UP": "pullup.gif",
    "DEADLIFT": "deadlift.gif"
}

# Enhanced Color Palette (BGR format)
COLORS = {
    "background": (245, 245, 245), # Light Gray background
    "primary_text": (20, 20, 20),     # Very Dark Gray / Almost Black
    "secondary_text": (100, 100, 100), # Medium Gray
    "accent_blue": (0, 122, 255),     # Apple Blue
    "accent_green": (52, 199, 89),    # Apple Green
    "accent_red": (255, 59, 48),      # Apple Red
    "accent_orange": (255, 149, 0),   # Apple Orange
    "accent_purple": (175, 82, 222),   # Apple Purple
    "button_bg_normal": (229, 229, 234), # Light Gray
    "button_bg_active": (0, 122, 255),   # Blue
    "button_bg_profile": (88, 86, 214),  # Indigo for Profile
    "button_bg_stats": (255, 149, 0),   # Orange for Stats
    "button_bg_freeplay": (255, 149, 0), # Orange for Free Play button
    "button_text_normal": (20, 20, 20),   # Dark Text
    "button_text_active": (255, 255, 255), # White Text
    "overlay_bg": (235, 235, 240, 230), # Semi-transparent light gray (Added Alpha)
    "landmark_vis": (52, 199, 89),     # Green
    "landmark_issue": (255, 59, 48),     # Red (for bad form highlight)
    "connection": (142, 142, 147),   # Medium Gray
    "profile_text": (88, 86, 214),   # Indigo for current user display
    "timer_text": (0, 122, 255),      # Blue for timers
}

# Font and Layout
FONT = cv2.FONT_HERSHEY_SIMPLEX # Use Simplex as it's widely available
TITLE_SCALE = 1.6
SELECT_TITLE_SCALE = 1.1
BUTTON_TEXT_SCALE = 0.65
STATUS_TEXT_SCALE = 0.6
REP_TEXT_SCALE = 1.4
FEEDBACK_TEXT_SCALE = 0.7
LARGE_TIMER_SCALE = 3.0 # For rest timer
STATS_TEXT_SCALE = 0.5 # For text list in stats screen
LINE_THICKNESS = 2
BUTTON_HEIGHT = 55
BUTTON_MARGIN = 20
CORNER_RADIUS = 15
OVERLAY_ALPHA = 0.85 # Transparency for overlays
PLUS_MINUS_BTN_SIZE = 40 # Size for +/- buttons

# --- Mediapipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Global Variables ---
# State & Source
app_mode = "HOME" # HOME, EXERCISE_SELECT, SET_SELECTION, GUIDE, TRACKING, REST, STATS
source_type = None # 'webcam' or 'video'
cap = None
video_source_selected = False
is_webcam_source = False

# Window & Canvas
try:
    tk_root_main = tk.Tk(); tk_root_main.withdraw() # Persistent root for dialogs
    target_win_width = tk_root_main.winfo_screenwidth()
    target_win_height = tk_root_main.winfo_screenheight()
except Exception: target_win_width, target_win_height = 1280, 720; tk_root_main = None
actual_win_width, actual_win_height = target_win_width, target_win_height
canvas = None # Main drawing canvas
last_frame_for_rest = None # Store the last frame for drawing rest UI over it

# Exercise & Tracking
current_exercise = EXERCISES[0]
counter, stage = 0, None
counter_left, counter_right = 0, 0
stage_left, stage_right = None, None
feedback_list = ["Select a profile or create a new one."]
last_feedback_display = ""
last_rep_time, last_rep_time_left, last_rep_time_right = 0, 0, 0
rep_cooldown = 0.5
form_correct_overall = True
form_issues_details = set()
ema_angles = {}

# Set/Rep/Rest Variables
target_sets = 3
target_reps_per_set = 10
target_rest_time = 30 # seconds
current_set_number = 1
rest_start_time = None
set_config_confirmed = False # Flag to track if set config is done for the CURRENT workout

# Profile & Stats
current_user = None
user_profiles = {}
user_stats = {}
session_start_time = None
session_reps = {} # Reps *during* the current session (across all sets)

# Guide Mode
guide_gif_frames = []
guide_gif_reader = None
guide_gif_index = 0
guide_last_frame_time = 0
guide_frame_delay = 0.1
guide_start_time = 0
guide_duration = 5 # Seconds to show GIF

# Stats Mode
stats_pie_image = None # To store the rendered pie chart image

# --- Helper Functions --- (Keep existing helpers: update_ema, calculate_angle, get_coords, etc.)
def update_ema(current_value, key, storage_dict):
    if not isinstance(current_value, (int, float)): return current_value
    if key not in storage_dict or storage_dict[key] is None: storage_dict[key] = float(current_value)
    else: storage_dict[key] = EMA_ALPHA * float(current_value) + (1 - EMA_ALPHA) * storage_dict[key]
    return storage_dict[key]
def calculate_angle(a, b, c, use_3d=False):
    if not all(coord[3] > 0.1 for coord in [a,b,c]): return 0
    a_np, b_np, c_np = np.array(a[:3]), np.array(b[:3]), np.array(c[:3]); dims = 3 if use_3d else 2
    a_np, b_np, c_np = a_np[:dims], b_np[:dims], c_np[:dims]; vec_ba = a_np - b_np; vec_bc = c_np - b_np
    norm_ba = np.linalg.norm(vec_ba); norm_bc = np.linalg.norm(vec_bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6: return 0
    dot_product = np.dot(vec_ba, vec_bc); cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle)); return int(angle)
def get_coords(landmarks, landmark_name):
    try: lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]; return [lm.x, lm.y, lm.z, lm.visibility]
    except Exception: return [0, 0, 0, 0]
def get_segment_vertical_angle(p1_coords, p2_coords):
     if p1_coords[3] < 0.5 or p2_coords[3] < 0.5: return None
     vec = np.array(p2_coords[:2]) - np.array(p1_coords[:2]); norm = np.linalg.norm(vec)
     if norm < 1e-6: return None
     vec_vert_down = np.array([0, 1]); dot_prod = np.dot(vec, vec_vert_down)
     angle_rad = np.arccos(np.clip(dot_prod / norm, -1.0, 1.0)); return np.degrees(angle_rad)
def add_feedback(new_msg, is_warning=False):
    global form_correct_overall, feedback_list
    prefix = "WARN: " if is_warning else "INFO: "; full_msg = prefix + new_msg
    if full_msg not in feedback_list: feedback_list.append(full_msg)
    if is_warning: form_correct_overall = False
def add_form_issue(part_name): form_issues_details.add(part_name)


# --- Data I/O Functions --- (No changes needed for adding simple fields)
def load_data():
    global user_profiles, user_stats
    try:
        if os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE, 'r') as f: user_profiles = json.load(f)
            print(f"Loaded {len(user_profiles)} profiles from {PROFILES_FILE}")
        else: user_profiles = {}; print(f"{PROFILES_FILE} not found, starting fresh.")
    except (json.JSONDecodeError, IOError) as e: print(f"Error loading profiles: {e}. Starting fresh."); user_profiles = {}
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f: user_stats = json.load(f)
            print(f"Loaded stats for {len(user_stats)} users from {STATS_FILE}")
        else: user_stats = {}; print(f"{STATS_FILE} not found, starting fresh.")
    except (json.JSONDecodeError, IOError) as e: print(f"Error loading stats: {e}. Starting fresh."); user_stats = {}

def save_data():
    try:
        with open(PROFILES_FILE, 'w') as f: json.dump(user_profiles, f, indent=4)
    except IOError as e: print(f"Error saving profiles: {e}")
    try:
        with open(STATS_FILE, 'w') as f: json.dump(user_stats, f, indent=4)
    except IOError as e: print(f"Error saving stats: {e}")


# --- Profile Management Popups --- (Keep existing)
def create_profile_popup():
    global user_profiles, user_stats, current_user, feedback_list
    if not tk_root_main: print("Error: Tkinter root not available for popup."); feedback_list = ["Error: Cannot open profile window."]; return
    popup = tk.Toplevel(tk_root_main); popup.title("Create New Profile"); popup.geometry("350x250"); popup.attributes('-topmost', True); popup.resizable(False, False)
    frame = tk.Frame(popup, padx=10, pady=10); frame.pack(expand=True, fill="both")
    tk.Label(frame, text="Username:", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew"); username_entry = tk.Entry(frame); username_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(frame, text="Age:", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew"); age_entry = tk.Entry(frame); age_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(frame, text="Height (cm):", anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="ew"); height_entry = tk.Entry(frame); height_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(frame, text="Weight (kg):", anchor="w").grid(row=3, column=0, padx=5, pady=5, sticky="ew"); weight_entry = tk.Entry(frame); weight_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
    frame.columnconfigure(1, weight=1)
    def submit_profile():
        global current_user, feedback_list
        username = username_entry.get().strip(); age_str = age_entry.get().strip(); height_str = height_entry.get().strip(); weight_str = weight_entry.get().strip()
        if not username: messagebox.showerror("Input Error", "Username cannot be empty.", parent=popup); return
        if username in user_profiles: messagebox.showerror("Input Error", "Username already exists.", parent=popup); return
        try:
            age = int(age_str) if age_str else 0; height = float(height_str) if height_str else 0.0; weight = float(weight_str) if weight_str else 0.0
            if age < 0 or height < 0 or weight < 0: raise ValueError("Values cannot be negative.")
        except ValueError: messagebox.showerror("Input Error", "Age, Height, Weight must be valid positive numbers.", parent=popup); return
        user_profiles[username] = {"age": age, "height": height, "weight": weight}
        if username not in user_stats: user_stats[username] = {}
        current_user = username; print(f"Profile created for {username}. Set as current user."); save_data()
        feedback_list = [f"Welcome, {current_user}! Select workout source."]
        popup.destroy()
    submit_button = tk.Button(frame, text="Create & Select", command=submit_profile, width=15); submit_button.grid(row=4, column=0, columnspan=2, pady=15)
    username_entry.focus_set()

def select_profile_popup():
    global current_user, feedback_list
    if not tk_root_main: print("Error: Tkinter root not available for popup."); feedback_list = ["Error: Cannot open profile window."]; return
    if not user_profiles: messagebox.showinfo("No Profiles", "No profiles found. Please create one first.", parent=tk_root_main); create_profile_popup(); return
    popup = tk.Toplevel(tk_root_main); popup.title("Select Profile"); popup.geometry("300x150"); popup.attributes('-topmost', True); popup.resizable(False, False)
    frame = tk.Frame(popup, padx=10, pady=10); frame.pack(expand=True, fill="both"); tk.Label(frame, text="Select User:").pack(pady=5)
    selected_user = tk.StringVar(popup); profile_options = list(user_profiles.keys())
    if current_user and current_user in profile_options: selected_user.set(current_user)
    elif profile_options: selected_user.set(profile_options[0])
    else: messagebox.showerror("Error", "No profiles available.", parent=popup); popup.destroy(); return
    profile_menu = tk.OptionMenu(frame, selected_user, *profile_options); profile_menu.pack(pady=10, fill="x")
    def submit_selection():
        global current_user, feedback_list
        chosen_user = selected_user.get()
        if chosen_user in user_profiles: current_user = chosen_user; print(f"Selected user: {current_user}"); feedback_list = [f"Welcome back, {current_user}! Select workout source."]; popup.destroy()
        else: messagebox.showerror("Selection Error", "Invalid user selected.", parent=popup)
    select_button = tk.Button(frame, text="Select", command=submit_selection, width=10); select_button.pack(pady=10)


# --- Statistics Generation --- (Keep existing: generate_stats_pie_image)
def generate_stats_pie_image(target_w, target_h):
    """Generates the stats pie chart as a BGR NumPy image."""
    global current_user, user_stats
    if not current_user or current_user not in user_stats: # No user / No stats check
        img = np.zeros((target_h, target_w, 3), dtype=np.uint8); img[:] = COLORS['background']; msg = "No User Selected or No Stats"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.8, 2); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.8, COLORS['primary_text'], 2, cv2.LINE_AA); return img
    stats = user_stats[current_user]
    if not stats: # User selected, but no stats recorded check
        img = np.zeros((target_h, target_w, 3), dtype=np.uint8); img[:] = COLORS['background']; msg = f"No Stats Recorded for {current_user}"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.8, 2); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.8, COLORS['primary_text'], 2, cv2.LINE_AA); return img
    labels = []; calories = []; total_calories = 0; total_reps_all = 0
    # Prepare data for pie chart (Calories)
    for exercise, data in stats.items():
        cal = data.get("total_calories", 0.0); reps = data.get("total_reps", 0); total_reps_all += reps
        if cal > 0 or reps > 0: # Include if *any* data exists
            # Use exercise name only for pie chart label, reps shown in text below
            labels.append(f"{exercise}")
            calories.append(cal if cal > 0 else 0.01) # Use small value for 0 cal entries
            total_calories += cal
    if not calories: # No calories recorded check
        img = np.zeros((target_h, target_w, 3), dtype=np.uint8); img[:] = COLORS['background']; msg = f"No Calories Recorded for {current_user}"; (tw, th), _ = cv2.getTextSize(msg, FONT, 0.8, 2); cv2.putText(img, msg, ((target_w - tw) // 2, (target_h + th) // 2 - 20), FONT, 0.8, COLORS['primary_text'], 2, cv2.LINE_AA); return img
    # Create Pie Chart
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except OSError: plt.style.use('default'); print("Seaborn style not found, using default.")
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100) # Slightly smaller figure for pie
    pie_colors = [tuple(np.array(COLORS[c_name]) / 255.0)[::-1] for c_name in ['accent_blue', 'accent_green', 'accent_orange', 'accent_purple', 'accent_red']]
    wedges, texts, autotexts = ax.pie(calories, autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=pie_colors[:len(calories)], textprops={'color':"w", 'fontsize': 9, 'weight': 'bold'})
    # Legend removed from pie chart itself to make space for text list
    # ax.legend(wedges, labels, title="Exercises", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize='small', frameon=True)
    title_str = (f'Calorie Distribution\nTotal: {total_calories:.1f} kcal') # Simpler title
    ax.set_title(title_str, fontsize=11, pad=15); ax.axis('equal'); plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect
    # Render to Buffer
    buf = io.BytesIO(); canvas_agg = FigureCanvas(fig); canvas_agg.print_png(buf); buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8); buf.close(); plt.close(fig)
    # Decode and Resize
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR); img_h, img_w = img_bgr.shape[:2]
    scale = min(target_w / img_w, target_h / img_h) if img_w > 0 and img_h > 0 else 1
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    if new_w > 0 and new_h > 0:
        resized_img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        final_img = np.zeros((target_h, target_w, 3), dtype=np.uint8); final_img[:] = COLORS['background'] # Use background color for padding
        off_x = (target_w - new_w) // 2; off_y = (target_h - new_h) // 2
        if off_y >= 0 and off_x >=0 and off_y+new_h <= target_h and off_x+new_w <= target_w: final_img[off_y:off_y + new_h, off_x:off_x + new_w] = resized_img
        else: print("Warning: Stats pie image centering calculation issue."); final_img = cv2.resize(img_bgr, (target_w, target_h))
        return final_img
    else: print("Warning: Stats pie image resize failed."); return img_bgr # Return original if resize fails

# --- Drawing Helper Functions --- (Keep existing)
def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1; x2, y2 = pt2; r = radius; r = max(0, min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2))
    x1_c, x2_c = min(x1, x2), max(x1, x2); y1_c, y2_c = min(y1, y2), max(y1, y2)
    x1_c = max(0, x1_c); y1_c = max(0, y1_c); x2_c = min(img.shape[1] - 1, x2_c); y2_c = min(img.shape[0] - 1, y2_c)
    if x1_c >= x2_c or y1_c >= y2_c: return
    if thickness < 0:
        cv2.rectangle(img, (x1_c + r, y1_c), (x2_c - r, y2_c), color, -1); cv2.rectangle(img, (x1_c, y1_c + r), (x2_c, y2_c - r), color, -1)
        cv2.circle(img, (x1_c + r, y1_c + r), r, color, -1); cv2.circle(img, (x2_c - r, y1_c + r), r, color, -1); cv2.circle(img, (x2_c - r, y2_c - r), r, color, -1); cv2.circle(img, (x1_c + r, y2_c - r), r, color, -1)
    elif thickness > 0:
        cv2.ellipse(img, (x1_c + r, y1_c + r), (r, r), 180, 0, 90, color, thickness); cv2.ellipse(img, (x2_c - r, y1_c + r), (r, r), 270, 0, 90, color, thickness); cv2.ellipse(img, (x2_c - r, y2_c - r), (r, r), 0, 0, 90, color, thickness); cv2.ellipse(img, (x1_c + r, y2_c - r), (r, r), 90, 0, 90, color, thickness)
        if x1_c + r < x2_c - r: cv2.line(img, (x1_c + r, y1_c), (x2_c - r, y1_c), color, thickness)
        if y1_c + r < y2_c - r: cv2.line(img, (x2_c, y1_c + r), (x2_c, y2_c - r), color, thickness)
        if x1_c + r < x2_c - r: cv2.line(img, (x2_c - r, y2_c), (x1_c + r, y2_c), color, thickness)
        if y1_c + r < y2_c - r: cv2.line(img, (x1_c, y2_c - r), (x1_c, y1_c + r), color, thickness)
def draw_semi_transparent_rect(img, pt1, pt2, color_bgr_alpha):
    x1, y1 = pt1; x2, y2 = pt2; x1, x2 = min(x1, x2), max(x1, x2); y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    if x1 >= x2 or y1 >= y2 : return
    try:
        sub_img = img[y1:y2, x1:x2];
        if sub_img.size == 0: return
        color = np.array(color_bgr_alpha[:3], dtype=np.uint8); alpha = color_bgr_alpha[3] / 255.0 if len(color_bgr_alpha) > 3 else OVERLAY_ALPHA
        rect = np.full(sub_img.shape, color, dtype=np.uint8); res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 0.0)
        img[y1:y2, x1:x2] = res
    except Exception as e: print(f"Error in draw_semi_transparent_rect: {e}")


# --- GIF Loading Function --- (Keep existing)
def load_guide_gif(exercise_name):
    global guide_gif_frames, guide_gif_reader, guide_gif_index, guide_frame_delay, feedback_list
    guide_gif_frames = []; guide_gif_index = 0
    if not os.path.exists(GIF_DIR):
        try: os.makedirs(GIF_DIR); print(f"Created GIF directory at: {GIF_DIR}"); feedback_list = [f"Created GIF directory. Please add GIFs."]
        except OSError as e: print(f"Error creating GIF directory '{GIF_DIR}': {e}"); feedback_list = [f"Error creating GIF folder."]; return False
    if exercise_name not in EXERCISE_GIF_MAP: print(f"Warning: No GIF mapping for {exercise_name}"); return False
    gif_filename = EXERCISE_GIF_MAP[exercise_name]; gif_path = os.path.join(GIF_DIR, gif_filename)
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
        guide_gif_reader.close()
        if not guide_gif_frames: print(f"Error: No frames loaded from GIF {gif_path}"); return False
        print(f"Loaded {len(guide_gif_frames)} frames for {exercise_name} (delay: {guide_frame_delay:.3f}s)")
        return True
    except FileNotFoundError: print(f"Error: GIF file not found at {gif_path}"); feedback_list = [f"Guide GIF not found: {gif_filename}"]; return False
    except Exception as e: print(f"Error loading GIF {gif_path}: {e}"); traceback.print_exc(); feedback_list = [f"Error loading guide: {gif_filename}"]; guide_gif_reader = None; return False


# --- Reset State Helper --- (Keep existing)
def reset_exercise_state():
    global counter, stage, counter_left, counter_right, stage_left, stage_right
    global last_rep_time, last_rep_time_left, last_rep_time_right
    global form_correct_overall, form_issues_details, ema_angles
    global current_set_number, set_config_confirmed # Reset set related states too
    counter = counter_left = counter_right = 0; stage = stage_left = stage_right = None; ct = time.time()
    last_rep_time = ct; last_rep_time_left = ct; last_rep_time_right = ct; form_correct_overall = True
    form_issues_details.clear(); ema_angles.clear(); current_set_number = 1
    set_config_confirmed = False # Require re-confirmation or free play selection


# --- End Session Helper --- (** Updated to save set config **)
def end_session():
    """Handles saving stats (including set config) and resetting session variables."""
    global session_start_time, session_reps, app_mode, cap, video_source_selected, is_webcam_source, feedback_list, current_user, source_type
    global target_sets, target_reps_per_set, target_rest_time, set_config_confirmed
    print("Ending session...")
    if current_user and session_start_time is not None and is_webcam_source:
        session_duration_sec = time.time() - session_start_time; session_duration_hr = session_duration_sec / 3600.0
        print(f"Session duration: {session_duration_sec:.1f}s")
        if set_config_confirmed: print(f"Configured: {target_sets} sets / {target_reps_per_set} reps / {target_rest_time}s rest")
        else: print("Mode: Free Play (No Set Structure)")

        if current_user not in user_stats: user_stats[current_user] = {}
        if current_user not in user_profiles: print(f"Warning: Profile data not found for {current_user}. Cannot calculate calories accurately."); weight_kg = 0
        else: weight_kg = user_profiles[current_user].get("weight", 0)

        print(f"Saving stats for user: {current_user} (Weight: {weight_kg}kg)")
        total_session_calories = 0.0; total_session_reps = 0

        for exercise, reps_in_session in session_reps.items():
            if reps_in_session <= 0: continue
            total_session_reps += reps_in_session
            exercise_key = exercise.upper()
            if exercise not in user_stats[current_user]: user_stats[current_user][exercise] = {"total_reps": 0, "total_calories": 0.0}
            current_total_reps = user_stats[current_user][exercise].get("total_reps", 0)
            user_stats[current_user][exercise]["total_reps"] = current_total_reps + reps_in_session
            session_calories_exercise = 0.0
            if weight_kg > 0:
                 met = MET_VALUES.get(exercise_key, MET_VALUES["DEFAULT"]); session_calories_exercise = met * weight_kg * session_duration_hr
                 current_total_cals = user_stats[current_user][exercise].get("total_calories", 0.0)
                 user_stats[current_user][exercise]["total_calories"] = current_total_cals + session_calories_exercise
                 total_session_calories += session_calories_exercise

            # ** NEW: Save set config if it was used **
            if set_config_confirmed:
                user_stats[current_user][exercise][STATS_SET_KEYS[0]] = target_sets # last_config_sets
                user_stats[current_user][exercise][STATS_SET_KEYS[1]] = target_reps_per_set # last_config_reps
                user_stats[current_user][exercise][STATS_SET_KEYS[2]] = target_rest_time # last_config_rest
            else:
                 # If free play, maybe clear old config or leave it? Let's clear it.
                 for key in STATS_SET_KEYS:
                     if key in user_stats[current_user][exercise]:
                         del user_stats[current_user][exercise][key]


            print(f"  Updated {exercise}: +{reps_in_session} reps, ~{session_calories_exercise:.2f} kcal")

        print(f"Total reps this session: {total_session_reps}")
        print(f"Approx total calories burned in session: {total_session_calories:.2f} kcal")
        save_data()

    elif not is_webcam_source and source_type == 'video': print("Video session ended. Stats not saved.")
    elif not current_user: print("No user logged in. Stats not saved.")
    else: print("No webcam session active or session start time missing. Stats not saved.")

    session_start_time = None; session_reps = {}
    app_mode = "HOME"; # Go back home
    if cap: cap.release(); cap = None
    video_source_selected = False; is_webcam_source = False; source_type = None
    feedback_list = ["Session Ended."]
    if current_user: feedback_list.append(f"Welcome back, {current_user}.")
    else: feedback_list.append("Select or create a profile.")
    reset_exercise_state()

# --- Mouse Callback (** Updated Set Selection Coords & Added Free Play **) ---
def mouse_callback(event, x, y, flags, param):
    global app_mode, current_exercise, feedback_list, video_source_selected, cap, source_type
    global guide_start_time, current_user, session_start_time, session_reps, is_webcam_source
    global stats_pie_image, target_sets, target_reps_per_set, target_rest_time, set_config_confirmed
    global current_set_number, rest_start_time

    canvas_w = param.get('canvas_w', actual_win_width)
    canvas_h = param.get('canvas_h', actual_win_height)

    if event != cv2.EVENT_LBUTTONDOWN: return

    # --- HOME Mode --- (No changes)
    if app_mode == "HOME":
        btn_w, btn_h = int(canvas_w * 0.25), BUTTON_HEIGHT; gap = BUTTON_MARGIN // 2; total_profile_width = btn_w * 3 + gap * 2; start_x_profile = (canvas_w - total_profile_width) // 2
        profile_y = int(canvas_h * 0.25); select_btn_x = start_x_profile; create_btn_x = select_btn_x + btn_w + gap; stats_btn_x = create_btn_x + btn_w + gap
        src_btn_w, src_btn_h = int(canvas_w * 0.35), int(BUTTON_HEIGHT * 1.2); src_btn_x = (canvas_w - src_btn_w) // 2; webcam_btn_y = profile_y + btn_h + BUTTON_MARGIN * 2; video_btn_y = webcam_btn_y + src_btn_h + BUTTON_MARGIN
        if profile_y <= y <= profile_y + btn_h:
            if select_btn_x <= x <= select_btn_x + btn_w: print("Select Profile button clicked"); select_profile_popup(); return
            elif create_btn_x <= x <= create_btn_x + btn_w: print("Create Profile button clicked"); create_profile_popup(); return
            elif stats_btn_x <= x <= stats_btn_x + btn_w: print("View Stats button clicked"); stats_pie_image = None; app_mode = "STATS"; feedback_list = ["Loading statistics..."]; return
        elif webcam_btn_y <= y <= webcam_btn_y + src_btn_h and src_btn_x <= x <= src_btn_x + src_btn_w:
            if current_user is None: feedback_list = ["Please select or create a profile first."]; messagebox.showwarning("Profile Needed", "Please select or create a profile first.", parent=tk_root_main); return
            print("Selecting Webcam..."); cap = cv2.VideoCapture(0);
            if not cap or not cap.isOpened(): cap = cv2.VideoCapture(1)
            if cap and cap.isOpened(): source_type = 'webcam'; is_webcam_source = True; session_start_time = time.time(); session_reps = {}; app_mode = "EXERCISE_SELECT"; feedback_list = ["Select an exercise"]; reset_exercise_state()
            else: feedback_list = ["Error: Webcam not found or busy."]; cap=None; is_webcam_source = False
            return
        elif video_btn_y <= y <= video_btn_y + src_btn_h and src_btn_x <= x <= src_btn_x + src_btn_w:
            if current_user is None: feedback_list = ["Please select or create a profile first."]; messagebox.showwarning("Profile Needed", "Please select or create a profile first.", parent=tk_root_main); return
            print("Selecting Video File...")
            video_path = filedialog.askopenfilename(parent=tk_root_main, title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
            if video_path:
                cap = cv2.VideoCapture(video_path)
                if cap and cap.isOpened(): source_type = 'video'; is_webcam_source = False; video_source_selected = True; session_start_time = None; session_reps = {}; app_mode = "EXERCISE_SELECT"; feedback_list = ["Select an exercise (Video Mode - No Sets/Stats)"]; print(f"Video loaded: {video_path}"); reset_exercise_state()
                else: feedback_list = [f"Error: Could not open video: {os.path.basename(video_path)}"]; cap=None
            else: feedback_list = ["Video selection cancelled."]
            return

    # --- EXERCISE_SELECT Mode (** Added Free Play Check **) ---
    elif app_mode == "EXERCISE_SELECT":
         h, w = canvas_h, canvas_w # Use canvas_h, canvas_w for consistency
         title_text = "Select Exercise"; (tw_title, th_title), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE, LINE_THICKNESS + 1)
         ty_title = BUTTON_MARGIN * 3; item_height = BUTTON_HEIGHT + BUTTON_MARGIN // 2
         list_h = len(EXERCISES) * item_height; start_y = ty_title + th_title + BUTTON_MARGIN * 2
         button_w = int(w * 0.4); button_x = w // 2 - button_w // 2
         # Calculate positions for main buttons
         start_btn_w, start_btn_h = 200, BUTTON_HEIGHT; start_btn_x = w // 2 - start_btn_w // 2
         start_btn_y = start_y + list_h + BUTTON_MARGIN
         # ** NEW: Calculate Free Play Button position **
         free_play_btn_w, free_play_btn_h = 200, BUTTON_HEIGHT
         free_play_btn_x = start_btn_x
         free_play_btn_y = start_btn_y + start_btn_h + BUTTON_MARGIN // 2
         # Back button position
         back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2
         back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2

         clicked_exercise = False
         for i, ex in enumerate(EXERCISES): # Check Exercise buttons
             btn_y = start_y + i * item_height
             if button_x <= x <= button_x + button_w and btn_y <= y <= btn_y + BUTTON_HEIGHT:
                 if current_exercise != ex: print(f"Selected: {ex}"); current_exercise = ex; reset_exercise_state()
                 clicked_exercise = True; break

         # Check "Configure Sets" / "Start" button
         if not clicked_exercise and start_btn_x <= x <= start_btn_x + start_btn_w and start_btn_y <= y <= start_btn_y + start_btn_h:
             print(f"Proceeding with {current_exercise}...")
             reset_exercise_state() # Ensure clean slate
             if source_type == 'webcam':
                 app_mode = "SET_SELECTION"; feedback_list = ["Configure sets and reps."]
                 target_sets = 3; target_reps_per_set = 10; target_rest_time = 30; set_config_confirmed = False
             else: # Video source - skip config
                 app_mode = "TRACKING"; feedback_list = [f"Start {current_exercise} (Video Mode)"]
             return

         # ** NEW: Check "Start Free Play" button **
         elif not clicked_exercise and free_play_btn_x <= x <= free_play_btn_x + free_play_btn_w and free_play_btn_y <= y <= free_play_btn_y + free_play_btn_h:
              print(f"Starting Free Play for {current_exercise}...")
              reset_exercise_state() # Resets counters, stage, set_config_confirmed=False
              set_config_confirmed = False # Explicitly ensure it's false for free play
              if source_type == 'webcam':
                  if load_guide_gif(current_exercise):
                      app_mode = "GUIDE"; guide_start_time = time.time()
                      feedback_list = [f"Guide: {current_exercise} (Free Play)"]
                  else:
                      app_mode = "TRACKING"; feedback_list = [f"Start {current_exercise} (Free Play)"]
              else: # Video mode inherently free play
                  app_mode = "TRACKING"; feedback_list = [f"Start {current_exercise} (Video Mode)"]
              return

         # Check Back Button
         elif not clicked_exercise and back_btn_x <= x <= back_btn_x + back_btn_w and back_btn_y <= y <= back_btn_y + back_btn_h:
              print("Going back to Home..."); app_mode = "HOME"
              if cap: cap.release(); cap = None
              source_type = None; video_source_selected = False; is_webcam_source = False
              feedback_list = ["Select profile or workout source."]; guide_gif_frames = []
              current_exercise = EXERCISES[0]; reset_exercise_state()
              return

    # --- SET_SELECTION Mode (** Fixed Coordinate Checks **) ---
    elif app_mode == "SET_SELECTION":
        h, w = canvas_h, canvas_w
        # ** Replicate coordinate calculations from draw_set_selection_ui **
        title_text = f"Configure: {current_exercise}"; (_, th_title), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE * 0.9, LINE_THICKNESS + 1)
        ty_title = int(h * 0.15)
        content_w = int(w * 0.5); content_x = (w - content_w) // 2
        item_y_start = ty_title + th_title + BUTTON_MARGIN * 2
        item_h = BUTTON_HEIGHT + 5 # Height for each row
        label_w = 180; value_w = 60; value_x = content_x + label_w + 10
        minus_btn_x = value_x + value_w + 10
        plus_btn_x = minus_btn_x + PLUS_MINUS_BTN_SIZE + 10
        btn_y_offset = (BUTTON_HEIGHT - PLUS_MINUS_BTN_SIZE) // 2 # Vertical offset for +/- buttons within the row height

        # Calculate Y coordinates for each row's buttons
        sets_btn_y = item_y_start + btn_y_offset
        reps_btn_y = item_y_start + item_h + btn_y_offset
        rest_btn_y = item_y_start + 2 * item_h + btn_y_offset

        # Define clickable areas using calculated coordinates
        sets_minus_rect = (minus_btn_x, sets_btn_y, PLUS_MINUS_BTN_SIZE, PLUS_MINUS_BTN_SIZE) # x, y, w, h
        sets_plus_rect = (plus_btn_x, sets_btn_y, PLUS_MINUS_BTN_SIZE, PLUS_MINUS_BTN_SIZE)
        reps_minus_rect = (minus_btn_x, reps_btn_y, PLUS_MINUS_BTN_SIZE, PLUS_MINUS_BTN_SIZE)
        reps_plus_rect = (plus_btn_x, reps_btn_y, PLUS_MINUS_BTN_SIZE, PLUS_MINUS_BTN_SIZE)
        rest_minus_rect = (minus_btn_x, rest_btn_y, PLUS_MINUS_BTN_SIZE, PLUS_MINUS_BTN_SIZE)
        rest_plus_rect = (plus_btn_x, rest_btn_y, PLUS_MINUS_BTN_SIZE, PLUS_MINUS_BTN_SIZE)

        # Check Sets +/- clicks using defined rectangles
        if sets_minus_rect[0] <= x < sets_minus_rect[0] + sets_minus_rect[2] and sets_minus_rect[1] <= y < sets_minus_rect[1] + sets_minus_rect[3]:
            target_sets = max(1, target_sets - 1); print(f"Sets: {target_sets}"); return
        if sets_plus_rect[0] <= x < sets_plus_rect[0] + sets_plus_rect[2] and sets_plus_rect[1] <= y < sets_plus_rect[1] + sets_plus_rect[3]:
            target_sets += 1; print(f"Sets: {target_sets}"); return

        # Check Reps +/- clicks
        if reps_minus_rect[0] <= x < reps_minus_rect[0] + reps_minus_rect[2] and reps_minus_rect[1] <= y < reps_minus_rect[1] + reps_minus_rect[3]:
            target_reps_per_set = max(1, target_reps_per_set - 1); print(f"Reps/Set: {target_reps_per_set}"); return
        if reps_plus_rect[0] <= x < reps_plus_rect[0] + reps_plus_rect[2] and reps_plus_rect[1] <= y < reps_plus_rect[1] + reps_plus_rect[3]:
            target_reps_per_set += 1; print(f"Reps/Set: {target_reps_per_set}"); return

        # Check Rest +/- clicks
        if rest_minus_rect[0] <= x < rest_minus_rect[0] + rest_minus_rect[2] and rest_minus_rect[1] <= y < rest_minus_rect[1] + rest_minus_rect[3]:
            target_rest_time = max(0, target_rest_time - 5); print(f"Rest: {target_rest_time}s"); return
        if rest_plus_rect[0] <= x < rest_plus_rect[0] + rest_plus_rect[2] and rest_plus_rect[1] <= y < rest_plus_rect[1] + rest_plus_rect[3]:
            target_rest_time += 5; print(f"Rest: {target_rest_time}s"); return

        # Check Confirm Button (using calculation from drawing function)
        confirm_btn_w, confirm_btn_h = 200, BUTTON_HEIGHT; confirm_btn_x = w // 2 - confirm_btn_w // 2
        confirm_btn_y = item_y_start + 3 * item_h + BUTTON_MARGIN # Adjusted Y based on drawing logic
        if confirm_btn_x <= x <= confirm_btn_x + confirm_btn_w and confirm_btn_y <= y <= confirm_btn_y + confirm_btn_h:
            print(f"Confirmed: {target_sets} sets, {target_reps_per_set} reps, {target_rest_time}s rest.")
            set_config_confirmed = True; current_set_number = 1
            if load_guide_gif(current_exercise):
                app_mode = "GUIDE"; guide_start_time = time.time()
                feedback_list = [f"Guide: {current_exercise} (Set {current_set_number}/{target_sets})"]
            else: app_mode = "TRACKING"; feedback_list = [f"Start Set {current_set_number}/{target_sets}"]
            return

        # Check Back Button (using calculation from drawing function)
        back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2
        back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2
        if back_btn_x <= x <= back_btn_x + back_btn_w and back_btn_y <= y <= back_btn_y + back_btn_h:
            print("Back to Exercise Selection from Set Config."); app_mode = "EXERCISE_SELECT"
            feedback_list = ["Select an exercise."]; reset_exercise_state()
            return

    # --- GUIDE Mode --- (No changes)
    elif app_mode == "GUIDE":
        start_btn_w, start_btn_h = 250, BUTTON_HEIGHT; start_btn_x = canvas_w // 2 - start_btn_w // 2; start_btn_y = canvas_h - start_btn_h - BUTTON_MARGIN * 2
        if start_btn_x <= x <= start_btn_x + start_btn_w and start_btn_y <= y <= start_btn_y + start_btn_h:
            print("Starting exercise tracking...")
            # Determine feedback based on whether sets were configured
            start_message = f"Start Set {current_set_number}/{target_sets}" if set_config_confirmed else f"Start {current_exercise} (Free Play)"
            app_mode = "TRACKING"; feedback_list = [start_message]
            return # Handled

    # --- TRACKING Mode --- (No changes needed in callback)
    elif app_mode == "TRACKING":
        try: total_button_width = canvas_w - 2 * BUTTON_MARGIN; btn_w = max(50, (total_button_width - (len(EXERCISES) - 1) * (BUTTON_MARGIN // 2)) // len(EXERCISES))
        except ZeroDivisionError: btn_w = 100
        home_btn_size = 50; home_btn_x = canvas_w - home_btn_size - BUTTON_MARGIN; home_btn_y = canvas_h - home_btn_size - BUTTON_MARGIN
        clicked_top_button = False
        for i, ex in enumerate(EXERCISES): # Check Exercise Buttons
            btn_x = BUTTON_MARGIN + i * (btn_w + BUTTON_MARGIN // 2)
            if btn_x <= x <= btn_x + btn_w and BUTTON_MARGIN <= y <= BUTTON_MARGIN + BUTTON_HEIGHT:
                clicked_top_button = True
                if current_exercise != ex:
                    print(f"Switching exercise to {ex}. Resetting workout structure.")
                    current_exercise = ex
                    reset_exercise_state() # Resets counters AND set structure/confirmation
                    if source_type == 'webcam': app_mode = "EXERCISE_SELECT"; feedback_list = ["Select an exercise or Free Play."] # Go back to choose config/freeplay
                    else: app_mode = "TRACKING"; feedback_list = [f"Start {current_exercise}"] # Video mode just continues
                break
        if not clicked_top_button and home_btn_x <= x <= home_btn_x + home_btn_size and home_btn_y <= y <= home_btn_y + home_btn_size: # Check Home Button
            end_session(); return

    # --- REST Mode --- (No changes)
    elif app_mode == "REST":
        skip_btn_w, skip_btn_h = 180, BUTTON_HEIGHT; skip_btn_x = canvas_w // 2 - skip_btn_w // 2; skip_btn_y = canvas_h // 2 + int(LARGE_TIMER_SCALE * 25)
        if skip_btn_x <= x <= skip_btn_x + skip_btn_w and skip_btn_y <= y <= skip_btn_y + skip_btn_h:
            print("Skipping rest."); app_mode = "TRACKING"; feedback_list = [f"Start Set {current_set_number}/{target_sets}"]; return
        home_btn_size = 50; home_btn_x = canvas_w - home_btn_size - BUTTON_MARGIN; home_btn_y = canvas_h - home_btn_size - BUTTON_MARGIN
        if home_btn_x <= x <= home_btn_x + home_btn_size and home_btn_y <= y <= home_btn_y + home_btn_size: end_session(); return

    # --- STATS Mode --- (No changes)
    elif app_mode == "STATS":
        back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = canvas_h - back_btn_h - BUTTON_MARGIN * 2
        if back_btn_x <= x <= back_btn_x + back_btn_w and back_btn_y <= y <= back_btn_y + back_btn_h:
             print("Going back to Home from Stats..."); app_mode = "HOME"; stats_pie_image = None; feedback_list = [f"Welcome back, {current_user}."] if current_user else ["Select profile or workout source."]; return

# --- UI Drawing Functions ---

# --- Home UI --- (Keep existing)
def draw_home_ui(canvas):
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]
    title_text = "Fitness Tracker Pro"; (tw, th), _ = cv2.getTextSize(title_text, FONT, TITLE_SCALE, LINE_THICKNESS + 1); tx = (w - tw) // 2; ty = int(h * 0.1) + th
    cv2.putText(canvas, title_text, (tx, ty), FONT, TITLE_SCALE, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)
    user_text = f"User: {current_user}" if current_user else "User: None Selected"; (tw_user, th_user), _ = cv2.getTextSize(user_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); tx_user = (w - tw_user) // 2; ty_user = ty + th + int(BUTTON_MARGIN * 0.75); cv2.putText(canvas, user_text, (tx_user, ty_user), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["profile_text"], LINE_THICKNESS, cv2.LINE_AA)
    btn_w, btn_h = int(w * 0.25), BUTTON_HEIGHT; gap = BUTTON_MARGIN // 2; total_profile_width = btn_w * 3 + gap * 2; start_x_profile = (w - total_profile_width) // 2; profile_y = ty_user + th_user + BUTTON_MARGIN // 2
    select_btn_x = start_x_profile; create_btn_x = select_btn_x + btn_w + gap; stats_btn_x = create_btn_x + btn_w + gap
    draw_rounded_rectangle(canvas, (select_btn_x, profile_y), (select_btn_x + btn_w, profile_y + btn_h), COLORS["button_bg_profile"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (select_btn_x, profile_y), (select_btn_x + btn_w, profile_y + btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS); btn_text = "Select Profile"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (select_btn_x + (btn_w - tw) // 2, profile_y + (btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    draw_rounded_rectangle(canvas, (create_btn_x, profile_y), (create_btn_x + btn_w, profile_y + btn_h), COLORS["button_bg_profile"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (create_btn_x, profile_y), (create_btn_x + btn_w, profile_y + btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS); btn_text = "Create Profile"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (create_btn_x + (btn_w - tw) // 2, profile_y + (btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    draw_rounded_rectangle(canvas, (stats_btn_x, profile_y), (stats_btn_x + btn_w, profile_y + btn_h), COLORS["button_bg_stats"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (stats_btn_x, profile_y), (stats_btn_x + btn_w, profile_y + btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS); btn_text = "View Stats"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (stats_btn_x + (btn_w - tw) // 2, profile_y + (btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    src_btn_w, src_btn_h = int(w * 0.35), int(BUTTON_HEIGHT * 1.2); src_btn_x = (w - src_btn_w) // 2; webcam_btn_y = profile_y + btn_h + BUTTON_MARGIN * 2; video_btn_y = webcam_btn_y + src_btn_h + BUTTON_MARGIN
    draw_rounded_rectangle(canvas, (src_btn_x, webcam_btn_y), (src_btn_x + src_btn_w, webcam_btn_y + src_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (src_btn_x, webcam_btn_y), (src_btn_x + src_btn_w, webcam_btn_y + src_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS); btn_text = "Start Webcam Workout"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); cv2.putText(canvas, btn_text, (src_btn_x + (src_btn_w - tw) // 2, webcam_btn_y + (src_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    draw_rounded_rectangle(canvas, (src_btn_x, video_btn_y), (src_btn_x + src_btn_w, video_btn_y + src_btn_h), COLORS["accent_blue"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (src_btn_x, video_btn_y), (src_btn_x + src_btn_w, video_btn_y + src_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS); btn_text = "Load Video (No Stats)"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); cv2.putText(canvas, btn_text, (src_btn_x + (src_btn_w - tw) // 2, video_btn_y + (src_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    feedback_str = " | ".join(feedback_list) if feedback_list else "Ready"; (tw, th), _ = cv2.getTextSize(feedback_str, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS); fx = (w - tw) // 2; fy = h - BUTTON_MARGIN * 3; feedback_color = COLORS["accent_red"] if "Error" in feedback_str or "Please select" in feedback_str else COLORS["secondary_text"]; cv2.putText(canvas, feedback_str, (fx, fy), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)
    quit_text = "Press 'Q' to Quit"; (tw, th), _ = cv2.getTextSize(quit_text, FONT, 0.6, 1); cv2.putText(canvas, quit_text, (w - tw - 20, h - th - 10), FONT, 0.6, COLORS["secondary_text"], 1, cv2.LINE_AA)

# --- Exercise Select UI (** Added Free Play Button **) ---
def draw_exercise_select_ui(canvas):
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]
    title_text = "Select Exercise"; (tw, th), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE, LINE_THICKNESS + 1)
    tx = (w - tw) // 2; ty = BUTTON_MARGIN * 3; cv2.putText(canvas, title_text, (tx, ty), FONT, SELECT_TITLE_SCALE, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)
    item_height = BUTTON_HEIGHT + BUTTON_MARGIN // 2; list_h = len(EXERCISES) * item_height; start_y = ty + th + BUTTON_MARGIN * 2
    button_w = int(w * 0.4); button_x = w // 2 - button_w // 2
    # Draw Exercise List
    for i, ex in enumerate(EXERCISES):
        btn_y = start_y + i * item_height; is_active = (ex == current_exercise)
        bg_color = COLORS["button_bg_active"] if is_active else COLORS["button_bg_normal"]; text_color = COLORS["button_text_active"] if is_active else COLORS["button_text_normal"]; border_color = COLORS["button_text_active"] if is_active else COLORS["secondary_text"]
        draw_rounded_rectangle(canvas, (button_x, btn_y), (button_x + button_w, btn_y + BUTTON_HEIGHT), bg_color, -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (button_x, btn_y), (button_x + button_w, btn_y + BUTTON_HEIGHT), border_color, 1, CORNER_RADIUS)
        (tw_ex, th_ex), _ = cv2.getTextSize(ex, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); tx_ex = button_x + max(0, (button_w - tw_ex) // 2); ty_ex = btn_y + (BUTTON_HEIGHT + th_ex) // 2
        cv2.putText(canvas, ex, (tx_ex, ty_ex), FONT, BUTTON_TEXT_SCALE * 1.1, text_color, LINE_THICKNESS, cv2.LINE_AA)

    # Configure Sets / Start Button
    start_btn_w, start_btn_h = 200, BUTTON_HEIGHT; start_btn_x = w // 2 - start_btn_w // 2
    start_btn_y = start_y + list_h + BUTTON_MARGIN # Y position for this button
    start_btn_text = "Configure Sets" if source_type == 'webcam' else "Start Video"
    draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS)
    (tw, th), _ = cv2.getTextSize(start_btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, start_btn_text, (start_btn_x + (start_btn_w - tw) // 2, start_btn_y + (start_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)

    # ** NEW: Free Play Button (only if webcam) **
    if source_type == 'webcam':
        free_play_btn_w, free_play_btn_h = 200, BUTTON_HEIGHT
        free_play_btn_x = start_btn_x
        free_play_btn_y = start_btn_y + start_btn_h + BUTTON_MARGIN // 2 # Position below previous button
        draw_rounded_rectangle(canvas, (free_play_btn_x, free_play_btn_y), (free_play_btn_x + free_play_btn_w, free_play_btn_y + free_play_btn_h), COLORS["button_bg_freeplay"], -1, CORNER_RADIUS)
        draw_rounded_rectangle(canvas, (free_play_btn_x, free_play_btn_y), (free_play_btn_x + free_play_btn_w, free_play_btn_y + free_play_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS)
        btn_text = "Start Free Play"
        (tw_fp, th_fp), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS)
        cv2.putText(canvas, btn_text, (free_play_btn_x + (free_play_btn_w - tw_fp) // 2, free_play_btn_y + (free_play_btn_h + th_fp) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)

    # Back Button
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2
    draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["secondary_text"], 1, CORNER_RADIUS)
    btn_text = "Back to Home"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (back_btn_x + (back_btn_w - tw) // 2, back_btn_y + (back_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)
    # Feedback Area
    feedback_str = " | ".join(feedback_list); (tw_fb, th_fb), _ = cv2.getTextSize(feedback_str, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS)
    fx = (w - tw_fb) // 2; fy = h - BUTTON_MARGIN - th_fb; feedback_color = COLORS["accent_red"] if "Error" in feedback_str else COLORS["secondary_text"]; cv2.putText(canvas, feedback_str, (fx, fy), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)


# --- Set Selection UI (** Fixed Coordinate Drawing **) ---
def draw_set_selection_ui(canvas):
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]
    title_text = f"Configure: {current_exercise}"; (tw, th_title), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE * 0.9, LINE_THICKNESS + 1)
    tx = (w - tw) // 2; ty_title = int(h * 0.15); cv2.putText(canvas, title_text, (tx, ty_title), FONT, SELECT_TITLE_SCALE * 0.9, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)

    # ** Use consistent coordinate calculations as in mouse_callback **
    content_w = int(w * 0.5); content_x = (w - content_w) // 2
    item_y_start = ty_title + th_title + BUTTON_MARGIN * 2
    item_h = BUTTON_HEIGHT + 5 # Height for each row
    label_w = 180; value_w = 60; value_x = content_x + label_w + 10
    minus_btn_x = value_x + value_w + 10
    plus_btn_x = minus_btn_x + PLUS_MINUS_BTN_SIZE + 10
    btn_y_offset = (BUTTON_HEIGHT - PLUS_MINUS_BTN_SIZE) // 2 # Vertical offset for +/- buttons

    # Draw Sets Row
    sets_y = item_y_start
    sets_btn_y = sets_y + btn_y_offset # Y coord for the buttons in this row
    cv2.putText(canvas, "Number of Sets:", (content_x, sets_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(canvas, str(target_sets), (value_x, sets_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["accent_blue"], LINE_THICKNESS + 1, cv2.LINE_AA)
    # Draw MINUS button using calculated coords
    draw_rounded_rectangle(canvas, (minus_btn_x, sets_btn_y), (minus_btn_x + PLUS_MINUS_BTN_SIZE, sets_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5)
    (tw_m, th_m), _ = cv2.getTextSize("-", FONT, 1.0, 2); cv2.putText(canvas, "-", (minus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_m)//2, sets_btn_y + (PLUS_MINUS_BTN_SIZE + th_m)//2), FONT, 1.0, COLORS["primary_text"], 2)
    # Draw PLUS button using calculated coords
    draw_rounded_rectangle(canvas, (plus_btn_x, sets_btn_y), (plus_btn_x + PLUS_MINUS_BTN_SIZE, sets_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5)
    (tw_p, th_p), _ = cv2.getTextSize("+", FONT, 1.0, 2); cv2.putText(canvas, "+", (plus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_p)//2, sets_btn_y + (PLUS_MINUS_BTN_SIZE + th_p)//2), FONT, 1.0, COLORS["primary_text"], 2)

    # Draw Reps Row
    reps_y = item_y_start + item_h
    reps_btn_y = reps_y + btn_y_offset
    cv2.putText(canvas, "Reps per Set:", (content_x, reps_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(canvas, str(target_reps_per_set), (value_x, reps_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["accent_blue"], LINE_THICKNESS + 1, cv2.LINE_AA)
    draw_rounded_rectangle(canvas, (minus_btn_x, reps_btn_y), (minus_btn_x + PLUS_MINUS_BTN_SIZE, reps_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5)
    (tw_m, th_m), _ = cv2.getTextSize("-", FONT, 1.0, 2); cv2.putText(canvas, "-", (minus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_m)//2, reps_btn_y + (PLUS_MINUS_BTN_SIZE + th_m)//2), FONT, 1.0, COLORS["primary_text"], 2)
    draw_rounded_rectangle(canvas, (plus_btn_x, reps_btn_y), (plus_btn_x + PLUS_MINUS_BTN_SIZE, reps_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5)
    (tw_p, th_p), _ = cv2.getTextSize("+", FONT, 1.0, 2); cv2.putText(canvas, "+", (plus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_p)//2, reps_btn_y + (PLUS_MINUS_BTN_SIZE + th_p)//2), FONT, 1.0, COLORS["primary_text"], 2)

    # Draw Rest Row
    rest_y = item_y_start + 2 * item_h
    rest_btn_y = rest_y + btn_y_offset
    cv2.putText(canvas, "Rest Time (sec):", (content_x, rest_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(canvas, str(target_rest_time), (value_x, rest_y + int(BUTTON_HEIGHT*0.7)), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["accent_blue"], LINE_THICKNESS + 1, cv2.LINE_AA)
    draw_rounded_rectangle(canvas, (minus_btn_x, rest_btn_y), (minus_btn_x + PLUS_MINUS_BTN_SIZE, rest_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5)
    (tw_m, th_m), _ = cv2.getTextSize("-", FONT, 1.0, 2); cv2.putText(canvas, "-", (minus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_m)//2, rest_btn_y + (PLUS_MINUS_BTN_SIZE + th_m)//2), FONT, 1.0, COLORS["primary_text"], 2)
    draw_rounded_rectangle(canvas, (plus_btn_x, rest_btn_y), (plus_btn_x + PLUS_MINUS_BTN_SIZE, rest_btn_y + PLUS_MINUS_BTN_SIZE), COLORS["button_bg_normal"], -1, 5)
    (tw_p, th_p), _ = cv2.getTextSize("+", FONT, 1.0, 2); cv2.putText(canvas, "+", (plus_btn_x + (PLUS_MINUS_BTN_SIZE - tw_p)//2, rest_btn_y + (PLUS_MINUS_BTN_SIZE + th_p)//2), FONT, 1.0, COLORS["primary_text"], 2)

    # Confirm Button
    confirm_btn_w, confirm_btn_h = 200, BUTTON_HEIGHT; confirm_btn_x = w // 2 - confirm_btn_w // 2
    confirm_btn_y = item_y_start + 3 * item_h + BUTTON_MARGIN # Adjusted Y calc
    draw_rounded_rectangle(canvas, (confirm_btn_x, confirm_btn_y), (confirm_btn_x + confirm_btn_w, confirm_btn_y + confirm_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (confirm_btn_x, confirm_btn_y), (confirm_btn_x + confirm_btn_w, confirm_btn_y + confirm_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS)
    btn_text = "Confirm & Start"; (tw_c, th_c), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); cv2.putText(canvas, btn_text, (confirm_btn_x + (confirm_btn_w - tw_c) // 2, confirm_btn_y + (confirm_btn_h + th_c) // 2), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    # Back Button
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2
    draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["secondary_text"], 1, CORNER_RADIUS)
    btn_text = "Back"; (tw_b, th_b), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (back_btn_x + (back_btn_w - tw_b) // 2, back_btn_y + (back_btn_h + th_b) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)
    # Feedback Area
    feedback_str = " | ".join(feedback_list); (tw_fb, th_fb), _ = cv2.getTextSize(feedback_str, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS); fx = (w - tw_fb) // 2; fy = h - BUTTON_MARGIN - th_fb; feedback_color = COLORS["accent_red"] if "Error" in feedback_str else COLORS["secondary_text"]; cv2.putText(canvas, feedback_str, (fx, fy), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)

# --- Guide UI --- (Updated title for free play)
def draw_guide_ui(canvas):
    global guide_gif_index, guide_last_frame_time
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]
    # Title includes set info or 'Free Play'
    mode_info = f"(Set {current_set_number}/{target_sets})" if set_config_confirmed else "(Free Play)"
    title = f"Guide: {current_exercise} {mode_info}"
    (tw_title, th_title), _ = cv2.getTextSize(title, FONT, TITLE_SCALE * 0.8, LINE_THICKNESS)
    cv2.putText(canvas, title, (BUTTON_MARGIN, BUTTON_MARGIN + th_title), FONT, TITLE_SCALE * 0.8, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA)
    gif_area_y_start = BUTTON_MARGIN * 2 + th_title; gif_area_h = h - gif_area_y_start - BUTTON_MARGIN * 4 - BUTTON_HEIGHT; gif_area_w = w - BUTTON_MARGIN * 2
    if guide_gif_frames:
        current_time = time.time()
        if current_time - guide_last_frame_time >= guide_frame_delay: guide_gif_index = (guide_gif_index + 1) % len(guide_gif_frames); guide_last_frame_time = current_time
        frame = guide_gif_frames[guide_gif_index]; frame_h, frame_w_orig = frame.shape[:2]
        if frame_w_orig > 0 and frame_h > 0:
            scale = min(gif_area_w / frame_w_orig, gif_area_h / frame_h); new_w, new_h = int(frame_w_orig * scale), int(frame_h * scale)
            if new_w > 0 and new_h > 0 :
                try:
                    display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    ox = BUTTON_MARGIN + (gif_area_w - new_w) // 2; oy = gif_area_y_start + (gif_area_h - new_h) // 2
                    if oy >= 0 and ox >= 0 and oy + new_h <= canvas.shape[0] and ox + new_w <= canvas.shape[1]: canvas[oy:oy + new_h, ox:ox + new_w] = display_frame
                    else: print(f"Warning: GIF display coords OOB."); frame = None
                except Exception as e: print(f"Error resizing/placing GIF frame: {e}"); frame = None
            else: print("Warning: GIF resize resulted in zero dimensions."); frame = None
        else: print("Warning: Invalid original GIF frame dimensions."); frame = None
        if frame is None: no_gif_text = "Error displaying guide"; (tw_ng, th_ng), _ = cv2.getTextSize(no_gif_text, FONT, 1.0, LINE_THICKNESS); tx = BUTTON_MARGIN + (gif_area_w - tw_ng) // 2; ty = gif_area_y_start + (gif_area_h + th_ng) // 2; cv2.putText(canvas, no_gif_text, (tx, ty), FONT, 1.0, COLORS["accent_red"], LINE_THICKNESS, cv2.LINE_AA)
    else: no_gif_text = "Guide unavailable"; (tw_ng, th_ng), _ = cv2.getTextSize(no_gif_text, FONT, 1.0, LINE_THICKNESS); tx = BUTTON_MARGIN + (gif_area_w - tw_ng) // 2; ty = gif_area_y_start + (gif_area_h + th_ng) // 2; cv2.putText(canvas, no_gif_text, (tx, ty), FONT, 1.0, COLORS["secondary_text"], LINE_THICKNESS, cv2.LINE_AA)
    start_btn_w, start_btn_h = 250, BUTTON_HEIGHT; start_btn_x = w // 2 - start_btn_w // 2; start_btn_y = h - start_btn_h - BUTTON_MARGIN * 2
    draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS)
    btn_text = "Start Exercise"; (tw_st, th_st), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (start_btn_x + (start_btn_w - tw_st) // 2, start_btn_y + (start_btn_h + th_st) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    elapsed = time.time() - guide_start_time; remaining = max(0, guide_duration - elapsed); skip_text = f"Starting in {remaining:.0f}s (Click Start)" if remaining > 0 else "Click Start Now"; (tw_skip, th_skip), _ = cv2.getTextSize(skip_text, FONT, FEEDBACK_TEXT_SCALE * 0.9, 1); cv2.putText(canvas, skip_text, (start_btn_x + (start_btn_w - tw_skip)//2, start_btn_y - th_skip - 5), FONT, FEEDBACK_TEXT_SCALE * 0.9, COLORS["secondary_text"], 1, cv2.LINE_AA)


# --- Tracking UI --- (Keep existing - displays set/rep conditionally)
def draw_tracking_ui(canvas, frame, results):
    global last_frame_for_rest
    h, w = canvas.shape[:2]; frame_h, frame_w = frame.shape[:2]; canvas[:] = (10,10,10)
    ox, oy, sw, sh = 0, 0, w, h
    if frame_w > 0 and frame_h > 0:
        scale = min(w / frame_w, h / frame_h); sw, sh = int(frame_w * scale), int(frame_h * scale); ox, oy = (w - sw) // 2, (h - sh) // 2; interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        try:
            if sw > 0 and sh > 0:
                resized_frame = cv2.resize(frame, (sw, sh), interpolation=interp); draw_pose_landmarks_on_frame(resized_frame, results.pose_landmarks if results else None, mp_pose.POSE_CONNECTIONS, form_issues_details)
                if oy >= 0 and ox >= 0 and oy + sh <= h and ox + sw <= w: canvas[oy:oy + sh, ox:ox + sw] = resized_frame
                else: print("Warning: Video ROI calc error.")
                last_frame_for_rest = canvas.copy()
            else: print("Warning: Invalid resize dimensions."); last_frame_for_rest = canvas.copy()
        except Exception as e: print(f"Error resizing/drawing landmarks/placing frame: {e}"); err_txt = "Video Display Error"; (tw, th),_ = cv2.getTextSize(err_txt, FONT, 1.0, 2); cv2.putText(canvas, err_txt, ((w-tw)//2, (h+th)//2), FONT, 1.0, COLORS['accent_red'], 2, cv2.LINE_AA); last_frame_for_rest = canvas.copy()
    else: err_txt = "Invalid Frame Input"; (tw, th),_ = cv2.getTextSize(err_txt, FONT, 1.0, 2); cv2.putText(canvas, err_txt, ((w-tw)//2, (h+th)//2), FONT, 1.0, COLORS['accent_red'], 2, cv2.LINE_AA); last_frame_for_rest = canvas.copy()
    overlay_canvas = np.zeros_like(canvas)
    try: total_button_width = w - 2 * BUTTON_MARGIN; btn_w = max(50, (total_button_width - (len(EXERCISES) - 1) * (BUTTON_MARGIN // 2)) // len(EXERCISES))
    except ZeroDivisionError: btn_w = 100
    for i, ex in enumerate(EXERCISES): # Top Buttons
        bx = BUTTON_MARGIN + i * (btn_w + BUTTON_MARGIN // 2); bxe = bx + btn_w; is_active = (ex == current_exercise); bg_color = COLORS["button_bg_active"] if is_active else COLORS["button_bg_normal"]; text_color = COLORS["button_text_active"] if is_active else COLORS["button_text_normal"]; border_color = COLORS["button_text_active"] if is_active else COLORS["secondary_text"]
        draw_rounded_rectangle(overlay_canvas, (bx, BUTTON_MARGIN), (bxe, BUTTON_MARGIN + BUTTON_HEIGHT), bg_color, -1, CORNER_RADIUS); draw_rounded_rectangle(overlay_canvas, (bx, BUTTON_MARGIN), (bxe, BUTTON_MARGIN + BUTTON_HEIGHT), border_color, 1, CORNER_RADIUS)
        (tw_ex, th_ex), _ = cv2.getTextSize(ex, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); tx = bx + max(0, (btn_w - tw_ex) // 2); ty = BUTTON_MARGIN + (BUTTON_HEIGHT + th_ex) // 2; cv2.putText(overlay_canvas, ex, (tx, ty), FONT, BUTTON_TEXT_SCALE, text_color, LINE_THICKNESS, cv2.LINE_AA)
    # Status Box
    is_bicep = current_exercise == "BICEP CURL"; sb_h = 200 if is_bicep else 170; sb_w = 400 if is_bicep else 320; sb_x, sb_y = BUTTON_MARGIN, BUTTON_MARGIN * 2 + BUTTON_HEIGHT; sb_xe, sb_ye = sb_x + sb_w, sb_y + sb_h; sb_xe = min(sb_xe, w - BUTTON_MARGIN); sb_ye = min(sb_ye, h - BUTTON_MARGIN); sb_w = sb_xe - sb_x; sb_h = sb_ye - sb_y
    if sb_w > 0 and sb_h > 0:
        draw_semi_transparent_rect(overlay_canvas, (sb_x, sb_y), (sb_xe, sb_ye), COLORS["overlay_bg"]); draw_rounded_rectangle(overlay_canvas, (sb_x, sb_y), (sb_xe, sb_ye), COLORS["secondary_text"], 1, CORNER_RADIUS)
        line_h = 25; label_color = COLORS["primary_text"]; value_color = COLORS["primary_text"]; rep_color = COLORS["accent_blue"]; stage_color = COLORS["primary_text"]; v_pad = 20
        user_display = f"User: {current_user}" if current_user else "User: -"; (tw_u, th_u), _ = cv2.getTextSize(user_display, FONT, STATUS_TEXT_SCALE * 0.9, 1); cv2.putText(overlay_canvas, user_display, (sb_x + 15, sb_y + v_pad), FONT, STATUS_TEXT_SCALE * 0.9, label_color, 1, cv2.LINE_AA); v_pad += line_h
        cv2.putText(overlay_canvas, 'EXERCISE:', (sb_x + 15, sb_y + v_pad), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA); cv2.putText(overlay_canvas, current_exercise, (sb_x + 110, sb_y + v_pad), FONT, STATUS_TEXT_SCALE, value_color, LINE_THICKNESS, cv2.LINE_AA); v_pad += line_h
        # Set Info / Mode
        mode_text = f"SET: {current_set_number}/{target_sets}" if set_config_confirmed else "MODE: Free Play"
        cv2.putText(overlay_canvas, mode_text, (sb_x + 15, sb_y + v_pad), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA); v_pad += line_h
        # Reps / Stage
        display_stage = stage if stage is not None else "INIT"; display_stage_l = stage_left if stage_left is not None else "INIT"; display_stage_r = stage_right if stage_right is not None else "INIT"
        rep_target_str = f"/{target_reps_per_set}" if set_config_confirmed else ""
        if is_bicep:
            rep_y = sb_y + v_pad; stage_y = rep_y + line_h + 5; col1_x = sb_x + 15; col2_x = sb_x + sb_w // 2 - 10
            cv2.putText(overlay_canvas, f'L REPS: {counter_left}{rep_target_str}', (col1_x, rep_y), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA)
            cv2.putText(overlay_canvas, 'L STAGE:', (col1_x, stage_y), FONT, STATUS_TEXT_SCALE * 0.9, label_color, 1, cv2.LINE_AA); cv2.putText(overlay_canvas, display_stage_l, (col1_x + 80, stage_y), FONT, STATUS_TEXT_SCALE, stage_color, LINE_THICKNESS, cv2.LINE_AA)
            cv2.putText(overlay_canvas, f'R REPS: {counter_right}{rep_target_str}', (col2_x, rep_y), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA)
            cv2.putText(overlay_canvas, 'R STAGE:', (col2_x, stage_y), FONT, STATUS_TEXT_SCALE * 0.9, label_color, 1, cv2.LINE_AA); cv2.putText(overlay_canvas, display_stage_r, (col2_x + 80, stage_y), FONT, STATUS_TEXT_SCALE, stage_color, LINE_THICKNESS, cv2.LINE_AA)
        else:
            rep_y = sb_y + v_pad; stage_y = rep_y + line_h + 5; rep_text = f"REPS: {counter}{rep_target_str}"
            cv2.putText(overlay_canvas, rep_text, (sb_x + 15, rep_y), FONT, STATUS_TEXT_SCALE * 1.1, label_color, 1, cv2.LINE_AA)
            cv2.putText(overlay_canvas, 'STAGE:', (sb_x + 15, stage_y), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA); cv2.putText(overlay_canvas, display_stage, (sb_x + 100, stage_y), FONT, STATUS_TEXT_SCALE * 1.1, stage_color, LINE_THICKNESS, cv2.LINE_AA)
    # Feedback Box
    fb_h = 65; home_btn_size = 50; fb_w = w - 2 * BUTTON_MARGIN - home_btn_size - BUTTON_MARGIN; fb_x, fb_y = BUTTON_MARGIN, h - fb_h - BUTTON_MARGIN; fb_xe, fb_ye = fb_x + fb_w, fb_y + fb_h; fb_xe = min(fb_xe, w - BUTTON_MARGIN); fb_ye = min(fb_ye, h - BUTTON_MARGIN); fb_w = fb_xe - fb_x; fb_h = fb_ye - fb_y
    if fb_w > 0 and fb_h > 0:
        draw_semi_transparent_rect(overlay_canvas, (fb_x, fb_y), (fb_xe, fb_ye), COLORS["overlay_bg"]); draw_rounded_rectangle(overlay_canvas, (fb_x, fb_y), (fb_xe, fb_ye), COLORS["secondary_text"], 1, CORNER_RADIUS)
        warnings = [f.replace("WARN: ", "") for f in feedback_list if "WARN:" in f]; infos = [f.replace("INFO: ", "") for f in feedback_list if "INFO:" in f and "WARN:" not in f]; display_feedback = ""; feedback_color = COLORS["accent_blue"]
        if warnings: display_feedback = "WARN: " + " | ".join(sorted(list(set(warnings)))); feedback_color = COLORS["accent_red"]
        elif infos: display_feedback = " | ".join(sorted(list(set(infos)))); feedback_color = COLORS["accent_blue"]
        elif stage is None and app_mode == "TRACKING": display_feedback = "Initializing..."; feedback_color = COLORS["secondary_text"]
        elif app_mode == "TRACKING": display_feedback = "Status OK"; feedback_color = COLORS["accent_green"]
        else: display_feedback = "..."; feedback_color = COLORS["secondary_text"]
        max_feedback_chars = int(fb_w / (FEEDBACK_TEXT_SCALE * 12));
        if len(display_feedback) > max_feedback_chars > 3 : display_feedback = display_feedback[:max_feedback_chars - 3] + "..."
        (tw_fb, th_fb), _ = cv2.getTextSize(display_feedback, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS); cv2.putText(overlay_canvas, display_feedback, (fb_x + 15, fb_y + (fb_h + th_fb) // 2), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)
    # Home Button
    hb_x = w - home_btn_size - BUTTON_MARGIN; hb_y = h - home_btn_size - BUTTON_MARGIN
    draw_rounded_rectangle(overlay_canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["accent_red"], -1, CORNER_RADIUS // 2); draw_rounded_rectangle(overlay_canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["button_text_active"], 1, CORNER_RADIUS // 2)
    icon_size = int(home_btn_size * 0.4); icon_x1 = hb_x + (home_btn_size - icon_size) // 2; icon_y1 = hb_y + (home_btn_size - icon_size) // 2; cv2.rectangle(overlay_canvas, (icon_x1, icon_y1), (icon_x1 + icon_size, icon_y1 + icon_size), COLORS["button_text_active"], -1)
    # Blend overlay
    try: gray_overlay = cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2GRAY); _, mask = cv2.threshold(gray_overlay, 5, 255, cv2.THRESH_BINARY); mask_inv = cv2.bitwise_not(mask); bg = cv2.bitwise_and(canvas, canvas, mask=mask_inv); fg = cv2.bitwise_and(overlay_canvas, overlay_canvas, mask=mask); cv2.add(bg, fg, dst=canvas)
    except Exception as e: print(f"Error blending overlay: {e}")


# --- Rest UI --- (Keep existing)
def draw_rest_ui(canvas):
    h, w = canvas.shape[:2]
    if last_frame_for_rest is not None and last_frame_for_rest.shape == canvas.shape: canvas[:] = last_frame_for_rest
    else: canvas[:] = (20, 20, 20)
    overlay_color = (30, 30, 30, 200); draw_semi_transparent_rect(canvas, (0, 0), (w, h), overlay_color)
    rest_text = "REST"; (tw_r, th_r), _ = cv2.getTextSize(rest_text, FONT, TITLE_SCALE * 0.8, LINE_THICKNESS + 1); tx_r = (w - tw_r) // 2; ty_r = int(h * 0.25); cv2.putText(canvas, rest_text, (tx_r, ty_r), FONT, TITLE_SCALE * 0.8, COLORS["background"], LINE_THICKNESS + 1, cv2.LINE_AA)
    time_elapsed = time.time() - rest_start_time if rest_start_time else 0; time_remaining = max(0, target_rest_time - time_elapsed); timer_text = f"{time_remaining:.0f}"; (tw_t, th_t), _ = cv2.getTextSize(timer_text, FONT, LARGE_TIMER_SCALE, LINE_THICKNESS + 2); tx_t = (w - tw_t) // 2; ty_t = h // 2 + th_t // 2; cv2.putText(canvas, timer_text, (tx_t, ty_t), FONT, LARGE_TIMER_SCALE, COLORS["timer_text"], LINE_THICKNESS + 2, cv2.LINE_AA)
    next_set_text = f"Next: Set {current_set_number}/{target_sets} - {current_exercise}"; (tw_n, th_n), _ = cv2.getTextSize(next_set_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS); tx_n = (w - tw_n) // 2; ty_n = ty_r + th_r + BUTTON_MARGIN; cv2.putText(canvas, next_set_text, (tx_n, ty_n), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["secondary_text"], LINE_THICKNESS, cv2.LINE_AA)
    skip_btn_w, skip_btn_h = 180, BUTTON_HEIGHT; skip_btn_x = w // 2 - skip_btn_w // 2; skip_btn_y = ty_t + th_t // 2 + BUTTON_MARGIN; draw_rounded_rectangle(canvas, (skip_btn_x, skip_btn_y), (skip_btn_x + skip_btn_w, skip_btn_y + skip_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (skip_btn_x, skip_btn_y), (skip_btn_x + skip_btn_w, skip_btn_y + skip_btn_h), COLORS["secondary_text"], 1, CORNER_RADIUS); btn_text = "Skip Rest"; (tw_s, th_s), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (skip_btn_x + (skip_btn_w - tw_s) // 2, skip_btn_y + (skip_btn_h + th_s) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)
    home_btn_size = 50; hb_x = w - home_btn_size - BUTTON_MARGIN; hb_y = h - home_btn_size - BUTTON_MARGIN; draw_rounded_rectangle(canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["accent_red"], -1, CORNER_RADIUS // 2); draw_rounded_rectangle(canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["button_text_active"], 1, CORNER_RADIUS // 2); icon_size = int(home_btn_size * 0.4); icon_x1 = hb_x + (home_btn_size - icon_size) // 2; icon_y1 = hb_y + (home_btn_size - icon_size) // 2; cv2.rectangle(canvas, (icon_x1, icon_y1), (icon_x1 + icon_size, icon_y1 + icon_size), COLORS["button_text_active"], -1)


# --- Stats UI (** Updated to show text list with set config **) ---
def draw_stats_ui(canvas):
    global stats_pie_image
    h, w = canvas.shape[:2]; canvas[:] = COLORS["background"]
    title_text = f"Statistics for {current_user}" if current_user else "Statistics"; (tw_title, th_title), _ = cv2.getTextSize(title_text, FONT, TITLE_SCALE * 0.9, LINE_THICKNESS + 1); tx = (w - tw_title) // 2; ty = BUTTON_MARGIN * 2 + th_title; cv2.putText(canvas, title_text, (tx, ty), FONT, TITLE_SCALE * 0.9, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)

    # Define areas: Smaller pie chart, space below for text list
    pie_chart_h = int(h * 0.40) # Height allocated for the pie chart
    pie_chart_w = int(w * 0.55) # Width for the pie chart
    pie_area_y_start = ty + th_title + BUTTON_MARGIN
    pie_area_x_start = (w - pie_chart_w) // 2 # Center the pie chart area horizontally

    text_list_y_start = pie_area_y_start + pie_chart_h + BUTTON_MARGIN * 2
    text_list_x_start = BUTTON_MARGIN * 3
    text_line_h = int(STATS_TEXT_SCALE * 45) # Vertical space per text line

    # Generate/Display Pie Chart in its allocated area
    if stats_pie_image is None:
        print("Generating stats pie image...")
        stats_pie_image = generate_stats_pie_image(pie_chart_w, pie_chart_h) # Generate for specific size
        if stats_pie_image is None: err_text = "Error generating stats image"; (tw_err, th_err), _ = cv2.getTextSize(err_text, FONT, 1.0, LINE_THICKNESS); cv2.putText(canvas, err_text, (pie_area_x_start + (pie_chart_w - tw_err) // 2, pie_area_y_start + (pie_chart_h + th_err) // 2), FONT, 1.0, COLORS["accent_red"], LINE_THICKNESS, cv2.LINE_AA)
    if stats_pie_image is not None:
         img_h, img_w_chart = stats_pie_image.shape[:2]
         if img_w_chart > 0 and img_h > 0:
             off_x = pie_area_x_start + (pie_chart_w - img_w_chart) // 2 # Center within pie area
             off_y = pie_area_y_start + (pie_chart_h - img_h) // 2
             target_roi = canvas[off_y:off_y + img_h, off_x:off_x + img_w_chart]
             if (off_y >= 0 and off_x >= 0 and off_y + img_h <= canvas.shape[0] and off_x + img_w_chart <= canvas.shape[1] and target_roi.shape == stats_pie_image.shape): canvas[off_y:off_y + img_h, off_x:off_x + img_w_chart] = stats_pie_image
             else: print(f"Warning: Stats image placement issue."); err_text = "Layout Error"; (tw_err, th_err), _ = cv2.getTextSize(err_text, FONT, 1.0, LINE_THICKNESS); cv2.putText(canvas, err_text, (pie_area_x_start + (pie_chart_w - tw_err) // 2, pie_area_y_start + (pie_chart_h + th_err) // 2), FONT, 1.0, COLORS["accent_red"], LINE_THICKNESS, cv2.LINE_AA)
         else: print(f"Warning: Stats pie image has invalid dimensions."); err_text = "Invalid Stats Image"; (tw_err, th_err), _ = cv2.getTextSize(err_text, FONT, 1.0, LINE_THICKNESS); cv2.putText(canvas, err_text, (pie_area_x_start + (pie_chart_w - tw_err) // 2, pie_area_y_start + (pie_chart_h + th_err) // 2), FONT, 1.0, COLORS["accent_red"], LINE_THICKNESS, cv2.LINE_AA)

    # ** NEW: Draw Text List of Stats Below Pie Chart **
    current_y = text_list_y_start
    if current_user and current_user in user_stats and user_stats[current_user]:
        stats_data = user_stats[current_user]
        for exercise, data in sorted(stats_data.items()): # Sort alphabetically
            total_reps = data.get("total_reps", 0)
            total_calories = data.get("total_calories", 0.0)
            # Retrieve last config data
            lcs = data.get(STATS_SET_KEYS[0]) # last_config_sets
            lcr = data.get(STATS_SET_KEYS[1]) # last_config_reps
            lcrt = data.get(STATS_SET_KEYS[2]) # last_config_rest

            # Format the display string
            stat_line_1 = f"- {exercise}: {total_reps} reps, {total_calories:.1f} kcal"
            stat_line_2 = ""
            if lcs is not None and lcr is not None and lcrt is not None:
                stat_line_2 = f"   (Last Config: {lcs} x {lcr}, {lcrt}s rest)"

            # Draw the text lines
            cv2.putText(canvas, stat_line_1, (text_list_x_start, current_y), FONT, STATS_TEXT_SCALE, COLORS['primary_text'], 1, cv2.LINE_AA)
            current_y += text_line_h
            if stat_line_2:
                cv2.putText(canvas, stat_line_2, (text_list_x_start, current_y), FONT, STATS_TEXT_SCALE * 0.9, COLORS['secondary_text'], 1, cv2.LINE_AA)
                current_y += text_line_h
            else:
                 current_y += int(text_line_h * 0.2) # Smaller gap if no config line

            if current_y > h - BUTTON_HEIGHT - BUTTON_MARGIN * 3: # Stop if running out of space
                 cv2.putText(canvas, "...", (text_list_x_start, current_y), FONT, STATS_TEXT_SCALE, COLORS['secondary_text'], 1, cv2.LINE_AA)
                 break

    else: # No stats data to display
        no_stats_text = "No exercise data recorded yet."
        (tw_ns, th_ns), _ = cv2.getTextSize(no_stats_text, FONT, STATS_TEXT_SCALE, 1)
        cv2.putText(canvas, no_stats_text, (text_list_x_start, current_y), FONT, STATS_TEXT_SCALE, COLORS['secondary_text'], 1, cv2.LINE_AA)


    # Back Button (Keep at bottom left)
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT; back_btn_x = BUTTON_MARGIN * 2; back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2
    draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS); draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["secondary_text"], 1, CORNER_RADIUS)
    btn_text = "Back to Home"; (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS); cv2.putText(canvas, btn_text, (back_btn_x + (back_btn_w - tw) // 2, back_btn_y + (back_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)


# --- Pose Landmark Drawing --- (Keep existing)
def draw_pose_landmarks_on_frame(target_image, landmarks_list, connections, form_issue_details):
    if not landmarks_list: return
    h, w = target_image.shape[:2];
    if h == 0 or w == 0: return
    default_landmark_spec = mp_drawing.DrawingSpec(color=COLORS["landmark_vis"], thickness=1, circle_radius=2)
    problem_landmark_spec = mp_drawing.DrawingSpec(color=COLORS["landmark_issue"], thickness=-1, circle_radius=4)
    default_connection_spec = mp_drawing.DrawingSpec(color=COLORS["connection"], thickness=1)
    problem_connection_spec = mp_drawing.DrawingSpec(color=COLORS["landmark_issue"], thickness=2)
    relevant_joint_indices = set(); mapping = {"BACK": [11, 12, 23, 24], "LEFT_KNEE": [25, 23, 27], "RIGHT_KNEE": [26, 24, 28], "LEFT_ELBOW": [13, 11, 15], "RIGHT_ELBOW": [14, 12, 16], "LEFT_UPPER_ARM": [11, 13], "RIGHT_UPPER_ARM": [12, 14], "BODY": [11, 12, 23, 24, 25, 26], "HIPS": [23, 24]}
    num_landmarks = len(landmarks_list.landmark) if landmarks_list else 0
    for part_name in form_issue_details:
        if part_name in mapping:
            for lm_index_value in mapping[part_name]:
                 if 0 <= lm_index_value < num_landmarks: relevant_joint_indices.add(lm_index_value)
    custom_connection_specs = {};
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if 0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks: spec = default_connection_spec;
            if start_idx in relevant_joint_indices or end_idx in relevant_joint_indices: spec = problem_connection_spec
            custom_connection_specs[connection] = spec
    try: mp_drawing.draw_landmarks(image=target_image, landmark_list=landmarks_list, connections=connections, landmark_drawing_spec=default_landmark_spec, connection_drawing_spec=custom_connection_specs)
    except Exception as draw_error: print(f"Error during mp_drawing.draw_landmarks: {draw_error}")
    try:
        for idx_value in relevant_joint_indices:
             if idx_value < num_landmarks:
                 lm = landmarks_list.landmark[idx_value]
                 if lm.visibility > 0.5: cx = int(lm.x * w); cy = int(lm.y * h); cx = np.clip(cx, 0, w - 1); cy = np.clip(cy, 0, h - 1); cv2.circle(target_image, (cx, cy), problem_landmark_spec.circle_radius, problem_landmark_spec.color, problem_landmark_spec.thickness)
    except Exception as redraw_error: print(f"Error during manual redraw of problematic landmarks: {redraw_error}")


# --- Main Application Loop ---
load_data() # Load profiles and stats at the start
window_name = 'Fitness Tracker Pro - Sets v2'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if platform.system() == "Windows": cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else: cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL); cv2.resizeWindow(window_name, actual_win_width, actual_win_height)

callback_param = {'canvas_w': actual_win_width, 'canvas_h': actual_win_height}
cv2.setMouseCallback(window_name, mouse_callback, callback_param)

pose = None
try: pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55, model_complexity=1)
except Exception as model_init_error: print(f"FATAL: Could not initialize pose model: {model_init_error}"); exit()

while True:
    try: # --- Window Size Update ---
        rect = cv2.getWindowImageRect(window_name)
        if isinstance(rect, (tuple, list)) and len(rect) == 4 and rect[2] > 1 and rect[3] > 1:
            if (rect[2] != actual_win_width or rect[3] != actual_win_height):
                 actual_win_width, actual_win_height = rect[2], rect[3]; callback_param['canvas_w'], callback_param['canvas_h'] = actual_win_width, actual_win_height
                 stats_pie_image = None; last_frame_for_rest = None
                 print(f"Window resized to: {actual_win_width}x{actual_win_height}")
    except Exception: pass
    if tk_root_main: # --- Update Tkinter Events ---
        try: tk_root_main.update_idletasks(); tk_root_main.update()
        except tk.TclError: tk_root_main = None

    # --- Create Canvas ---
    if app_mode == "REST" and last_frame_for_rest is not None and last_frame_for_rest.shape[0] == actual_win_height and last_frame_for_rest.shape[1] == actual_win_width:
         canvas = last_frame_for_rest.copy()
    else: canvas = np.zeros((actual_win_height, actual_win_width, 3), dtype=np.uint8); canvas[:] = COLORS['background']

    # === State Machine Logic ===
    if app_mode == "HOME": draw_home_ui(canvas)
    elif app_mode == "EXERCISE_SELECT": draw_exercise_select_ui(canvas)
    elif app_mode == "SET_SELECTION": draw_set_selection_ui(canvas)
    elif app_mode == "GUIDE": draw_guide_ui(canvas)
    elif app_mode == "STATS": draw_stats_ui(canvas)
    elif app_mode == "REST":
        if rest_start_time and time.time() - rest_start_time >= target_rest_time:
            app_mode = "TRACKING"; feedback_list = [f"Start Set {current_set_number}/{target_sets}"]
        else: draw_rest_ui(canvas)
    elif app_mode == "TRACKING":
        if not cap or not cap.isOpened(): feedback_list = ["Error: Video source lost."]; end_session(); continue
        ret, frame = cap.read()
        if not ret:
            is_video_file = (source_type == 'video');
            if is_video_file: print("Video file ended."); feedback_list = ["Video finished."]; end_session(); continue
            else: feedback_list = ["Error: Cannot read webcam frame."]; end_session(); continue
        frame_h, frame_w, _ = frame.shape
        if frame_h <= 0 or frame_w <= 0: continue
        results = None
        try: img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img_rgb.flags.writeable = False; results = pose.process(img_rgb)
        except Exception as pose_error: print(f"Error during pose processing: {pose_error}"); feedback_list = ["Error processing pose."]; draw_tracking_ui(canvas, frame, None); continue

        feedback_list = []; form_correct_overall = True; form_issues_details.clear()

        try: # --- Pose Logic ---
            if results and results.pose_landmarks:
                landmarks_raw = results.pose_landmarks.landmark
                # Get coords, calculate angles, apply EMA (Same as before)
                sh_l, sh_r=get_coords(landmarks_raw,'LEFT_SHOULDER'), get_coords(landmarks_raw,'RIGHT_SHOULDER'); hip_l, hip_r=get_coords(landmarks_raw,'LEFT_HIP'), get_coords(landmarks_raw,'RIGHT_HIP'); kn_l, kn_r=get_coords(landmarks_raw,'LEFT_KNEE'), get_coords(landmarks_raw,'RIGHT_KNEE'); an_l, an_r=get_coords(landmarks_raw,'LEFT_ANKLE'), get_coords(landmarks_raw,'RIGHT_ANKLE'); el_l, el_r=get_coords(landmarks_raw,'LEFT_ELBOW'), get_coords(landmarks_raw,'RIGHT_ELBOW'); wr_l, wr_r=get_coords(landmarks_raw,'LEFT_WRIST'), get_coords(landmarks_raw,'RIGHT_WRIST'); nose=get_coords(landmarks_raw, 'NOSE')
                angle_l_elbow=calculate_angle(sh_l, el_l, wr_l); angle_r_elbow=calculate_angle(sh_r, el_r, wr_r); angle_l_knee=calculate_angle(hip_l, kn_l, an_l); angle_r_knee=calculate_angle(hip_r, kn_r, an_r); angle_l_hip=calculate_angle(sh_l, hip_l, kn_l); angle_r_hip=calculate_angle(sh_r, hip_r, kn_r); angle_l_body=calculate_angle(sh_l, hip_l, kn_l); angle_r_body=calculate_angle(sh_r, hip_r, kn_r)
                l_elbow_logic=update_ema(angle_l_elbow,"LEFT_ELBOW",ema_angles); r_elbow_logic=update_ema(angle_r_elbow,"RIGHT_ELBOW",ema_angles); l_knee_logic=update_ema(angle_l_knee,"LEFT_KNEE",ema_angles); r_knee_logic=update_ema(angle_r_knee,"RIGHT_KNEE",ema_angles); l_hip_logic=update_ema(angle_l_hip,"LEFT_HIP",ema_angles); r_hip_logic=update_ema(angle_r_hip,"RIGHT_HIP",ema_angles); l_body_logic=update_ema(angle_l_body,"LEFT_BODY",ema_angles); r_body_logic=update_ema(angle_r_body,"RIGHT_BODY",ema_angles)
                avg_knee_angle=(l_knee_logic+r_knee_logic)/2 if(kn_l[3]>0.5 and kn_r[3]>0.5)else(l_knee_logic if kn_l[3]>0.5 else r_knee_logic if kn_r[3] > 0.5 else 90)
                avg_hip_angle=(l_hip_logic+r_hip_logic)/2 if(hip_l[3]>0.5 and hip_r[3]>0.5)else(l_hip_logic if hip_l[3]>0.5 else r_hip_logic if hip_r[3]>0.5 else 90)
                avg_elbow_angle=(l_elbow_logic+r_elbow_logic)/2 if(el_l[3]>0.5 and el_r[3]>0.5)else(l_elbow_logic if el_l[3]>0.5 else r_elbow_logic if el_r[3]>0.5 else 180)
                avg_body_angle=(l_body_logic+r_body_logic)/2 if(hip_l[3]>0.5 and hip_r[3]>0.5)else(l_body_logic if hip_l[3]>0.5 else r_body_logic if hip_r[3]>0.5 else 180)
                vertical_back_ok=True; back_angle_vertical=90; vis_sh=[s for s in[sh_l,sh_r]if s[3]>0.6]; vis_hip=[h for h in[hip_l,hip_r]if h[3]>0.6]
                if len(vis_sh)>0 and len(vis_hip)>0: sh_avg_pt=np.mean([s[:2]for s in vis_sh],axis=0); hip_avg_pt=np.mean([h[:2]for h in vis_hip],axis=0); vec_hs=sh_avg_pt-hip_avg_pt; vec_vert=np.array([0,-1]); norm_hs=np.linalg.norm(vec_hs)
                if norm_hs > 1e-6: back_angle_vertical=np.degrees(np.arccos(np.clip(np.dot(vec_hs,vec_vert)/norm_hs,-1.0,1.0)))
                thresh=90;
                if current_exercise == "BICEP CURL": thresh = BACK_ANGLE_THRESHOLD_BICEP
                elif current_exercise == "SQUAT": thresh = BACK_ANGLE_THRESHOLD_SQUAT
                if current_exercise != "DEADLIFT" and back_angle_vertical > thresh: vertical_back_ok=False; add_feedback(f"Back Angle ({back_angle_vertical:.0f}°>{thresh}°)",True); add_form_issue("BACK")

                ct = time.time(); rep_counted_this_frame = False; set_completed_this_frame = False

                # === BICEP CURL ===
                if current_exercise == "BICEP CURL":
                    angle_l = get_segment_vertical_angle(sh_l, el_l); angle_r = get_segment_vertical_angle(sh_r, el_r)
                    if angle_l is not None and abs(angle_l) > BICEP_UPPER_ARM_VERT_DEVIATION: add_feedback("L Arm Still", True); add_form_issue("LEFT_UPPER_ARM")
                    if angle_r is not None and abs(angle_r) > BICEP_UPPER_ARM_VERT_DEVIATION: add_feedback("R Arm Still", True); add_form_issue("RIGHT_UPPER_ARM")
                    # Left Rep
                    if vertical_back_ok and all(p[3] > 0.5 for p in [sh_l, el_l, wr_l]):
                        if stage_left is None: stage_left = "INIT"
                        if l_elbow_logic > BICEP_DOWN_ENTER_ANGLE and stage_left != "DOWN": stage_left = "DOWN"
                        elif l_elbow_logic < BICEP_UP_ENTER_ANGLE and stage_left == "DOWN":
                            stage_left = "UP";
                            if form_correct_overall and ct - last_rep_time_left > rep_cooldown:
                                counter_left += 1; last_rep_time_left = ct; rep_counted_this_frame = True
                                if is_webcam_source: session_reps[current_exercise] = session_reps.get(current_exercise, 0) + 1
                                add_feedback(f"L: Rep {counter_left}!", False)
                                if set_config_confirmed and max(counter_left, counter_right) >= target_reps_per_set: set_completed_this_frame = True
                            elif not form_correct_overall: add_feedback("L: Form?", True)
                            else: add_feedback("L: Cool", False)
                        if stage_left == "UP" and l_elbow_logic > BICEP_UP_EXIT_ANGLE: add_feedback("L: Lower", False)
                        if stage_left == "DOWN" and l_elbow_logic < BICEP_DOWN_EXIT_ANGLE: add_feedback("L: Curl", False)
                    # Right Rep
                    if vertical_back_ok and all(p[3] > 0.5 for p in [sh_r, el_r, wr_r]):
                        if stage_right is None: stage_right = "INIT"
                        if r_elbow_logic > BICEP_DOWN_ENTER_ANGLE and stage_right != "DOWN": stage_right = "DOWN"
                        elif r_elbow_logic < BICEP_UP_ENTER_ANGLE and stage_right == "DOWN":
                            stage_right = "UP";
                            if form_correct_overall and ct - last_rep_time_right > rep_cooldown:
                                counter_right += 1; last_rep_time_right = ct; # rep_counted_this_frame handled by left side
                                if is_webcam_source and not rep_counted_this_frame: session_reps[current_exercise] = session_reps.get(current_exercise, 0) + 1 # Count if right finishes first
                                add_feedback(f"R: Rep {counter_right}!", False)
                                if set_config_confirmed and max(counter_left, counter_right) >= target_reps_per_set: set_completed_this_frame = True
                            elif not form_correct_overall: add_feedback("R: Form?", True)
                            else: add_feedback("R: Cool", False)
                        if stage_right == "UP" and r_elbow_logic > BICEP_UP_EXIT_ANGLE: add_feedback("R: Lower", False)
                        if stage_right == "DOWN" and r_elbow_logic < BICEP_DOWN_EXIT_ANGLE: add_feedback("R: Curl", False)

                # === Other Exercises (Generic Logic) ===
                else:
                    # Form Checks specific to exercise
                    if current_exercise == "SQUAT":
                        if kn_l[3]>0.5 and an_l[3]>0.5 and kn_l[0] < an_l[0] - SQUAT_KNEE_VALGUS_THRESHOLD: add_feedback("L Knee In?", True); add_form_issue("LEFT_KNEE")
                        if kn_r[3]>0.5 and an_r[3]>0.5 and kn_r[0] > an_r[0] + SQUAT_KNEE_VALGUS_THRESHOLD: add_feedback("R Knee Out?", True); add_form_issue("RIGHT_KNEE")
                        if sh_l[3]>0.5 and kn_l[3]>0.5 and stage == "DOWN" and sh_l[0] < kn_l[0] - SQUAT_CHEST_FORWARD_THRESHOLD: add_feedback("Chest Up", True); add_form_issue("BACK")
                    elif current_exercise == "PUSH UP":
                        body_ok=True;
                        if avg_body_angle < PUSHUP_BODY_STRAIGHT_MIN or avg_body_angle > PUSHUP_BODY_STRAIGHT_MAX: body_ok = False; add_feedback(f"Body ({avg_body_angle:.0f})", True); add_form_issue("BODY")
                        form_correct_overall = form_correct_overall and body_ok
                    elif current_exercise == "DEADLIFT":
                        lockout_ok, lift_ok = True, True
                        if stage == "DOWN" or (stage == "INIT" and avg_hip_angle < 150):
                            if back_angle_vertical > BACK_ANGLE_THRESHOLD_DEADLIFT_LIFT: lift_ok = False; add_feedback(f"Back ({back_angle_vertical:.0f}deg)", True); add_form_issue("BACK")
                        is_near_up = avg_hip_angle > DEADLIFT_UP_EXIT_ANGLE and avg_knee_angle > DEADLIFT_UP_EXIT_ANGLE
                        if stage == "UP" or (stage != "UP" and is_near_up):
                            if back_angle_vertical > BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT: lockout_ok = False; add_feedback(f"Lock Back ({back_angle_vertical:.0f}deg)", True); add_form_issue("BACK")
                        form_correct_overall = form_correct_overall and lockout_ok and lift_ok and vertical_back_ok

                    # Landmark Visibility Check
                    landmarks_visible = False
                    if current_exercise == "SQUAT": landmarks_visible = kn_l[3] > 0.5 and kn_r[3] > 0.5 and vertical_back_ok
                    elif current_exercise == "PUSH UP": landmarks_visible = el_l[3] > 0.5 or el_r[3] > 0.5
                    elif current_exercise == "PULL UP": wrist_y = (wr_l[1] + wr_r[1]) / 2 if (wr_l[3] > 0.5 and wr_r[3] > 0.5) else (wr_l[1] if wr_l[3] > 0.5 else (wr_r[1] if wr_r[3] > 0.5 else 0)); nose_y = nose[1] if nose[3] > 0.5 else 1.0; landmarks_visible = (el_l[3] > 0.5 or el_r[3] > 0.5) and nose[3] > 0.5 and wrist_y > 0
                    elif current_exercise == "DEADLIFT": landmarks_visible = all(p[3] > 0.5 for p in [kn_l, kn_r, hip_l, hip_r])

                    # Rep Counting Logic
                    if landmarks_visible:
                        if stage is None: stage = "INIT"
                        up_condition, down_condition = False, False; up_exit_feedback, down_exit_feedback = "", ""
                        if current_exercise == "SQUAT": up_condition = avg_knee_angle > SQUAT_UP_ENTER_ANGLE; down_condition = avg_knee_angle < SQUAT_DOWN_ENTER_ANGLE; up_exit_feedback = "Stand" if avg_knee_angle < SQUAT_UP_EXIT_ANGLE else ""; down_exit_feedback = "Deeper" if avg_knee_angle > SQUAT_DOWN_EXIT_ANGLE else ""
                        elif current_exercise == "PUSH UP": up_condition = avg_elbow_angle > PUSHUP_UP_ENTER_ANGLE; down_condition = avg_elbow_angle < PUSHUP_DOWN_ENTER_ANGLE; up_exit_feedback = "Extend" if avg_elbow_angle < PUSHUP_UP_EXIT_ANGLE else ""; down_exit_feedback = "Lower" if avg_elbow_angle > PUSHUP_DOWN_EXIT_ANGLE else ""
                        elif current_exercise == "PULL UP": is_up_state = (avg_elbow_angle < PULLUP_UP_ENTER_ELBOW_ANGLE) and (nose_y < wrist_y if PULLUP_CHIN_ABOVE_WRIST else True); up_condition = avg_elbow_angle > PULLUP_DOWN_ENTER_ANGLE; down_condition = is_up_state; up_exit_feedback = "Hang" if avg_elbow_angle < PULLUP_DOWN_EXIT_ANGLE else ""; down_exit_feedback = "Higher" if avg_elbow_angle > PULLUP_UP_EXIT_ELBOW_ANGLE else ""
                        elif current_exercise == "DEADLIFT": up_condition = avg_hip_angle > DEADLIFT_UP_ENTER_ANGLE and avg_knee_angle > DEADLIFT_UP_ENTER_ANGLE; down_condition = avg_hip_angle < DEADLIFT_DOWN_ENTER_HIP_ANGLE and avg_knee_angle < DEADLIFT_DOWN_ENTER_KNEE_ANGLE; up_exit_feedback = "Lockout" if not (avg_hip_angle > DEADLIFT_UP_EXIT_ANGLE and avg_knee_angle > DEADLIFT_UP_EXIT_ANGLE) else ""; down_exit_feedback = "Lower" if not (avg_hip_angle < DEADLIFT_DOWN_EXIT_HIP_ANGLE and avg_knee_angle < DEADLIFT_DOWN_EXIT_KNEE_ANGLE) else ""

                        if up_condition and stage == "DOWN": # Transition to UP state
                            stage = "UP"
                            if form_correct_overall and ct - last_rep_time > rep_cooldown:
                                counter += 1; last_rep_time = ct; rep_counted_this_frame = True
                                if is_webcam_source: session_reps[current_exercise] = session_reps.get(current_exercise, 0) + 1
                                add_feedback(f"Rep {counter}!", False)
                                if set_config_confirmed and counter >= target_reps_per_set: set_completed_this_frame = True
                            elif not form_correct_overall: add_feedback("Form!", True)
                            else: add_feedback("Cool", False)
                        elif down_condition and stage != "DOWN": # Transition to DOWN state
                            stage = "DOWN"
                        if stage == "UP" and up_exit_feedback: add_feedback(up_exit_feedback, False)
                        if stage == "DOWN" and down_exit_feedback: add_feedback(down_exit_feedback, False)
                    else: add_feedback("Body Not Visible", True); stage=None

                # --- Set Completion Logic ---
                if set_completed_this_frame: # This flag is only True if set_config_confirmed is also True
                    add_feedback(f"Set {current_set_number} Complete!", False)
                    if current_set_number < target_sets:
                        current_set_number += 1; app_mode = "REST"; rest_start_time = time.time()
                        counter = 0; counter_left = 0; counter_right = 0; stage = None; stage_left = None; stage_right = None # Reset for next set
                        print(f"Starting rest before set {current_set_number}")
                    else: # Workout Complete!
                        feedback_list = ["Workout Complete! Well Done!"]; print("All sets completed.")
                        end_session(); continue # End session and skip drawing this frame
            else: # No landmarks detected
                add_feedback("No Person Detected", True); stage=stage_left=stage_right=None
        except Exception as e: print(f"!! Logic Error: {e}"); traceback.print_exc(); add_feedback("Processing Error", True); stage=stage_left=stage_right=None

        if not feedback_list and stage is not None and app_mode == "TRACKING":
             if form_correct_overall: add_feedback("Keep Going...", False)

        # --- Draw Tracking UI ---
        if app_mode == "TRACKING": draw_tracking_ui(canvas, frame, results)

    # --- Display Final Canvas ---
    if 'canvas' in locals() and isinstance(canvas, np.ndarray): cv2.imshow(window_name, canvas)
    else: canvas = np.zeros((actual_win_height, actual_win_width, 3), dtype=np.uint8); canvas[:] = COLORS['background']; cv2.imshow(window_name, canvas)

    # --- Quit Key ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): print("Quit key pressed."); break

# --- Cleanup ---
print("Releasing resources...")
if cap: cap.release(); print("Video capture released.")
if pose: pose.close(); print("Pose model closed.")
cv2.destroyAllWindows(); plt.close('all')
if tk_root_main:
    try: tk_root_main.destroy()
    except tk.TclError: pass
print("Application Closed.")