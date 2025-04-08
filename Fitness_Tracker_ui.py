import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog
import platform # To help with fullscreen adaptation
import traceback # For detailed error printing
import imageio # For GIF loading
import os # For path joining

# --- Constants ---
# Angle Thresholds (Tune these!) - Keep existing thresholds
BICEP_UP_ANGLE = 50
BICEP_DOWN_ANGLE = 160
SQUAT_UP_ANGLE = 170
SQUAT_DOWN_ANGLE = 95
PUSHUP_UP_ANGLE = 160
PUSHUP_DOWN_ANGLE = 90
PUSHUP_BODY_STRAIGHT_MIN = 155
PUSHUP_BODY_STRAIGHT_MAX = 195
PULLUP_DOWN_ANGLE = 165
PULLUP_UP_ELBOW_ANGLE = 70
PULLUP_CHIN_OVER_WRIST = True
DEADLIFT_UP_ANGLE_BODY = 170
DEADLIFT_DOWN_HIP_ANGLE = 115
DEADLIFT_DOWN_KNEE_ANGLE = 130
BACK_ANGLE_THRESHOLD_BICEP = 15
BACK_ANGLE_THRESHOLD_SQUAT = 40
BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT = 20

# --- UI Constants ---
GIF_DIR = "GIFs" # Relative path to GIF directory
EXERCISES = ["BICEP CURL", "SQUAT", "PUSH UP", "PULL UP", "DEADLIFT"]
EXERCISE_GIF_MAP = {
    "BICEP CURL": "bicep.gif",
    "SQUAT": "squats.gif",
    "PUSH UP": "pushup.gif",
    "PULL UP": "pullup.gif",
    "DEADLIFT": "deadlift.gif"
}

# Apple-inspired Color Palette (BGR format)
COLORS = {
    "background": (242, 242, 247), # Almost White
    "primary_text": (28, 28, 30),     # Almost Black
    "secondary_text": (90, 90, 95), # Darker Medium Gray for better contrast
    "accent_blue": (10, 132, 255),
    "accent_green": (52, 199, 89),
    "accent_red": (255, 69, 58),
    "button_bg_normal": (209, 209, 214), # Light Gray
    "button_bg_active": (10, 132, 255),   # Blue
    "button_text_normal": (28, 28, 30),   # Almost Black
    "button_text_active": (255, 255, 255), # White
    "overlay_bg": (229, 229, 234), # Base color for overlay
    "landmark_vis": (52, 199, 89),     # Green
    "landmark_low_vis": (255, 159, 10), # Orange
    "connection": (142, 142, 147),   # Medium Gray
}

# Font and Layout
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 1.8
SELECT_TITLE_SCALE = 1.2
BUTTON_TEXT_SCALE = 0.7
STATUS_TEXT_SCALE = 0.6
REP_TEXT_SCALE = 1.5
FEEDBACK_TEXT_SCALE = 0.7
LINE_THICKNESS = 2
BUTTON_HEIGHT = 55
BUTTON_MARGIN = 20
CORNER_RADIUS = 15
OVERLAY_ALPHA = 0.9 # Increased opacity for better text visibility

# --- Mediapipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Global Variables ---
app_mode = "HOME" # HOME, EXERCISE_SELECT, GUIDE, TRACKING
source_type = None # 'webcam' or 'video'
cap = None
video_source_selected = False # Kept for video looping logic, might be redundant
try:
    root_temp = tk.Tk(); root_temp.withdraw()
    target_win_width = root_temp.winfo_screenwidth()
    target_win_height = root_temp.winfo_screenheight()
    root_temp.destroy()
except: target_win_width, target_win_height = 1280, 720
actual_win_width, actual_win_height = target_win_width, target_win_height
canvas = None
current_exercise = EXERCISES[0] # Default selection
counter, stage = 0, "START"
counter_left, counter_right = 0, 0
stage_left, stage_right = "START", "START"
feedback_msg, last_feedback, back_posture_feedback = "Select Video Source", "", ""
last_rep_time, last_rep_time_left, last_rep_time_right = time.time(), time.time(), time.time()
rep_cooldown = 0.5
form_correct = True
guide_gif_frames = []
guide_gif_reader = None
guide_gif_index = 0
guide_last_frame_time = 0
guide_frame_delay = 0.1 # Seconds between GIF frames
guide_start_time = 0
guide_duration = 5 # Seconds to show GIF

# --- Helper Functions --- (calculate_angle, get_coords, set_feedback - Unchanged)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec_ab = b - a
    vec_cb = b - c
    norm_ab = np.linalg.norm(vec_ab)
    norm_cb = np.linalg.norm(vec_cb)
    if norm_ab == 0 or norm_cb == 0: return 0
    radians = np.arctan2(vec_cb[1], vec_cb[0]) - np.arctan2(vec_ab[1], vec_ab[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return int(angle)

def get_coords(landmarks, landmark_name):
    try:
        lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]
        return [lm.x, lm.y, lm.z, lm.visibility]
    except: return [0, 0, 0, 0]

def set_feedback(new_msg, is_warning=False):
    global feedback_msg, last_feedback
    prefix = "WARN: " if is_warning else "INFO: "
    full_msg = prefix + new_msg
    if new_msg != last_feedback:
        feedback_msg = full_msg
        last_feedback = new_msg

# --- Drawing Helper Functions ---
def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 0: r = 0
    if thickness > 0:
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.line(img, (x2 - r, y2), (x1 + r, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y1 + r), color, thickness)
    elif thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)

def draw_semi_transparent_rect(img, pt1, pt2, color, alpha):
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 >= x2 or y1 >= y2: return
    sub_img = img[y1:y2, x1:x2]
    if sub_img.size == 0: return
    rect = np.zeros(sub_img.shape, dtype=np.uint8)
    b, g, r = color[:3]
    cv2.rectangle(rect, (0, 0), (x2 - x1, y2 - y1), (b, g, r), -1)
    res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 1.0)
    img[y1:y2, x1:x2] = res

# --- GIF Loading Function --- (Only called when needed now)
def load_guide_gif(exercise_name):
    global guide_gif_frames, guide_gif_reader, guide_gif_index, guide_frame_delay
    guide_gif_frames = []
    guide_gif_index = 0
    if exercise_name not in EXERCISE_GIF_MAP:
        print(f"Warning: No GIF mapping for {exercise_name}")
        return False
    gif_filename = EXERCISE_GIF_MAP[exercise_name]
    gif_path = os.path.join(GIF_DIR, gif_filename)
    if not os.path.exists(gif_path):
        print(f"Error: GIF file not found at {gif_path}")
        set_feedback(f"Guide GIF not found: {gif_filename}", True)
        return False
    try:
        guide_gif_reader = imageio.get_reader(gif_path)
        try:
            meta = guide_gif_reader.get_meta_data()
            guide_frame_delay = meta.get('duration', 100) / 1000.0
            if guide_frame_delay < 0.02: guide_frame_delay = 0.1
        except: guide_frame_delay = 0.1
        for frame in guide_gif_reader:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR)
            guide_gif_frames.append(frame_bgr)
        guide_gif_reader.close()
        print(f"Loaded {len(guide_gif_frames)} frames for {exercise_name} with delay {guide_frame_delay:.3f}s")
        return len(guide_gif_frames) > 0
    except Exception as e:
        print(f"Error loading GIF {gif_path}: {e}"); traceback.print_exc()
        set_feedback(f"Error loading guide: {gif_filename}", True)
        guide_gif_reader = None
        return False

# --- Reset State Helper ---
def reset_exercise_state():
    global counter, stage, counter_left, counter_right, stage_left, stage_right
    global last_rep_time, last_rep_time_left, last_rep_time_right, form_correct, back_posture_feedback
    counter = counter_left = counter_right = 0
    stage = stage_left = stage_right = "START"
    ct = time.time()
    last_rep_time = ct; last_rep_time_left = ct; last_rep_time_right = ct
    form_correct = True
    back_posture_feedback = ""

# --- Mouse Callback ---
# --- Mouse Callback --- CORRECTED
def mouse_callback(event, x, y, flags, param):
    global app_mode, current_exercise, feedback_msg, video_source_selected, cap, source_type
    global guide_start_time

    canvas_w = param.get('canvas_w', actual_win_width)
    canvas_h = param.get('canvas_h', actual_win_height)

    if app_mode == "HOME":
        if event == cv2.EVENT_LBUTTONDOWN:
            # --- HOME Screen Button Calculations (Mirrors draw_home_ui) ---
            home_button_width = int(canvas_w * 0.3)
            home_button_height = int(BUTTON_HEIGHT * 1.2)
            button_x = canvas_w // 2 - home_button_width // 2
            webcam_btn_y = canvas_h // 2 - home_button_height - BUTTON_MARGIN // 2
            video_btn_y = canvas_h // 2 + BUTTON_MARGIN // 2

            # Check Webcam Button
            if button_x <= x <= button_x + home_button_width and webcam_btn_y <= y <= webcam_btn_y + home_button_height:
                print("Selecting Webcam...")
                cap = cv2.VideoCapture(0)
                if not cap or not cap.isOpened(): cap = cv2.VideoCapture(1)
                if cap and cap.isOpened():
                    app_mode = "EXERCISE_SELECT"; source_type = 'webcam'
                    video_source_selected = False; set_feedback("Select an exercise")
                    reset_exercise_state()
                else:
                    set_feedback("Error: Webcam not found or busy.", True)
                    if cap: cap.release(); cap = None; source_type = None

            # Check Video Button
            elif button_x <= x <= button_x + home_button_width and video_btn_y <= y <= video_btn_y + home_button_height:
                print("Selecting Video File...")
                root = tk.Tk(); root.withdraw()
                video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
                root.destroy()
                if video_path:
                    cap = cv2.VideoCapture(video_path)
                    if cap and cap.isOpened():
                        app_mode = "EXERCISE_SELECT"; source_type = 'video'
                        video_source_selected = True; set_feedback("Select an exercise")
                        print(f"Video loaded: {video_path}")
                        reset_exercise_state()
                    else:
                        set_feedback(f"Error: Could not open video: {os.path.basename(video_path)}", True)
                        if cap: cap.release(); cap = None; source_type = None
                else: set_feedback("Video selection cancelled.")

    elif app_mode == "EXERCISE_SELECT":
         if event == cv2.EVENT_LBUTTONDOWN:
             # --- EXERCISE SELECT Screen Button Calculations (Mirrors draw_exercise_select_ui) ---
             # 1. Calculate Title Position (needed for start_y)
             title_text = "Select Exercise"
             (tw_title, th_title), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE, LINE_THICKNESS + 1)
             ty_title = BUTTON_MARGIN * 3

             # 2. Calculate List Start Position
             item_height = BUTTON_HEIGHT + BUTTON_MARGIN // 2
             list_h = len(EXERCISES) * item_height
             start_y = ty_title + th_title + BUTTON_MARGIN * 2 # Use title height for accurate start_y
             button_w = int(canvas_w * 0.4)
             button_x = canvas_w // 2 - button_w // 2

             # 3. Check Exercise Buttons
             clicked_exercise = False
             for i, ex in enumerate(EXERCISES):
                 btn_y = start_y + i * item_height
                 if button_x <= x <= button_x + button_w and btn_y <= y <= btn_y + BUTTON_HEIGHT:
                     if current_exercise != ex:
                         print(f"Selected: {ex}")
                         current_exercise = ex
                     clicked_exercise = True # Mark that an exercise was clicked
                     break # Exit loop once clicked

             # 4. Calculate and Check Start Button
             start_btn_w, start_btn_h = 200, BUTTON_HEIGHT
             start_btn_x = canvas_w // 2 - start_btn_w // 2
             start_btn_y = start_y + list_h + BUTTON_MARGIN # Use accurate start_y and list_h
             if not clicked_exercise and start_btn_x <= x <= start_btn_x + start_btn_w and start_btn_y <= y <= start_btn_y + start_btn_h:
                 print(f"Starting {current_exercise}...")
                 reset_exercise_state()
                 if source_type == 'webcam':
                     if load_guide_gif(current_exercise):
                         app_mode = "GUIDE"; guide_start_time = time.time()
                         set_feedback(f"Showing guide for {current_exercise}")
                     else:
                         app_mode = "TRACKING"; set_feedback(f"Start {current_exercise} (Guide failed)")
                 elif source_type == 'video':
                     app_mode = "TRACKING"; set_feedback(f"Start {current_exercise}")

             # 5. Calculate and Check Back Button
             back_btn_w, back_btn_h = 150, BUTTON_HEIGHT
             back_btn_x = BUTTON_MARGIN * 2
             back_btn_y = canvas_h - back_btn_h - BUTTON_MARGIN * 2
             if not clicked_exercise and back_btn_x <= x <= back_btn_x + back_btn_w and back_btn_y <= y <= back_btn_y + back_btn_h:
                  print("Going back to Home...")
                  app_mode = "HOME"
                  if cap: cap.release(); cap = None
                  source_type = None; video_source_selected = False
                  set_feedback("Select Video Source")

    elif app_mode == "GUIDE":
         if event == cv2.EVENT_LBUTTONDOWN:
             # --- GUIDE Screen Button Calculations (Mirrors draw_guide_ui) ---
             start_btn_w, start_btn_h = 250, BUTTON_HEIGHT
             start_btn_x = canvas_w // 2 - start_btn_w // 2
             start_btn_y = canvas_h - start_btn_h - BUTTON_MARGIN * 2
             if start_btn_x <= x <= start_btn_x + start_btn_w and start_btn_y <= y <= start_btn_y + start_btn_h:
                 print("Starting exercise tracking...")
                 app_mode = "TRACKING"; set_feedback(f"Start {current_exercise}")

    elif app_mode == "TRACKING":
        if event == cv2.EVENT_LBUTTONDOWN:
            # --- TRACKING Screen Button Calculations (Mirrors draw_tracking_ui) ---
            # 1. Check Exercise Buttons (Top Bar)
            try:
                total_button_width = canvas_w - 2 * BUTTON_MARGIN
                btn_w = (total_button_width - (len(EXERCISES) - 1) * (BUTTON_MARGIN // 2)) // len(EXERCISES)
                btn_w = max(50, btn_w)
            except ZeroDivisionError: btn_w = 100

            clicked_top_button = False
            for i, ex in enumerate(EXERCISES):
                btn_x = BUTTON_MARGIN + i * (btn_w + BUTTON_MARGIN // 2)
                # Check within the button bounds
                if btn_x <= x <= btn_x + btn_w and BUTTON_MARGIN <= y <= BUTTON_MARGIN + BUTTON_HEIGHT:
                    clicked_top_button = True
                    if current_exercise != ex:
                        print(f"Switching to {ex}")
                        current_exercise = ex
                        reset_exercise_state()
                        if source_type == 'webcam':
                            if load_guide_gif(current_exercise):
                                app_mode = "GUIDE"; guide_start_time = time.time()
                                set_feedback(f"Showing guide for {current_exercise}")
                            else:
                                app_mode = "TRACKING"; set_feedback(f"Start {current_exercise} (Guide failed)")
                        else:
                            app_mode = "TRACKING"; set_feedback(f"Start {current_exercise}")
                    break # Exit loop once a button is handled

            # 2. Check Home Button (Bottom Right) - Only if a top button wasn't clicked
            if not clicked_top_button:
                home_btn_size = 50
                home_btn_x = canvas_w - home_btn_size - BUTTON_MARGIN
                home_btn_y = canvas_h - home_btn_size - BUTTON_MARGIN
                if home_btn_x <= x <= home_btn_x + home_btn_size and home_btn_y <= y <= home_btn_y + home_btn_size:
                     print("Returning Home...")
                     app_mode = "HOME"
                     if cap: cap.release(); cap = None
                     source_type = None; video_source_selected = False
                     set_feedback("Select Video Source")
                     guide_gif_frames = []


# --- UI Drawing Functions ---

def draw_home_ui(canvas):
    h, w = canvas.shape[:2]
    canvas[:] = COLORS["background"]
    # Title
    title_text = "Fitness Tracker Pro"
    (tw, th), _ = cv2.getTextSize(title_text, FONT, TITLE_SCALE, LINE_THICKNESS + 1)
    tx = (w - tw) // 2
    ty = h // 4 + th // 2
    cv2.putText(canvas, title_text, (tx, ty), FONT, TITLE_SCALE, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)
    # Buttons
    home_button_width = int(w * 0.3)
    home_button_height = int(BUTTON_HEIGHT * 1.2)
    button_x = w // 2 - home_button_width // 2
    # Webcam Button
    webcam_btn_y = h // 2 - home_button_height - BUTTON_MARGIN // 2
    draw_rounded_rectangle(canvas, (button_x, webcam_btn_y), (button_x + home_button_width, webcam_btn_y + home_button_height), COLORS["accent_green"], -1, CORNER_RADIUS)
    draw_rounded_rectangle(canvas, (button_x, webcam_btn_y), (button_x + home_button_width, webcam_btn_y + home_button_height), COLORS["button_text_active"], 1, CORNER_RADIUS)
    btn_text = "Use Webcam"
    (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS)
    cv2.putText(canvas, btn_text, (button_x + (home_button_width - tw) // 2, webcam_btn_y + (home_button_height + th) // 2), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    # Video Button
    video_btn_y = h // 2 + BUTTON_MARGIN // 2
    draw_rounded_rectangle(canvas, (button_x, video_btn_y), (button_x + home_button_width, video_btn_y + home_button_height), COLORS["accent_blue"], -1, CORNER_RADIUS)
    draw_rounded_rectangle(canvas, (button_x, video_btn_y), (button_x + home_button_width, video_btn_y + home_button_height), COLORS["button_text_active"], 1, CORNER_RADIUS)
    btn_text = "Load Video File"
    (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS)
    cv2.putText(canvas, btn_text, (button_x + (home_button_width - tw) // 2, video_btn_y + (home_button_height + th) // 2), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)
    # Feedback Area
    (tw, th), _ = cv2.getTextSize(feedback_msg, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS)
    fx = (w - tw) // 2
    fy = h * 3 // 4 + th // 2
    feedback_color = COLORS["accent_red"] if "Error" in feedback_msg or "WARN:" in feedback_msg else COLORS["secondary_text"]
    cv2.putText(canvas, feedback_msg, (fx, fy), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)
    # Quit Text
    quit_text = "Press 'Q' to Quit"
    (tw, th), _ = cv2.getTextSize(quit_text, FONT, 0.6, 1)
    cv2.putText(canvas, quit_text, (w - tw - 20, h - th - 10), FONT, 0.6, COLORS["secondary_text"], 1, cv2.LINE_AA)

def draw_exercise_select_ui(canvas):
    h, w = canvas.shape[:2]
    canvas[:] = COLORS["background"]

    # Title
    title_text = "Select Exercise"
    (tw, th), _ = cv2.getTextSize(title_text, FONT, SELECT_TITLE_SCALE, LINE_THICKNESS + 1)
    tx = (w - tw) // 2
    ty = BUTTON_MARGIN * 3 # Position title higher
    cv2.putText(canvas, title_text, (tx, ty), FONT, SELECT_TITLE_SCALE, COLORS["primary_text"], LINE_THICKNESS + 1, cv2.LINE_AA)

    # Exercise List Buttons (Vertical)
    item_height = BUTTON_HEIGHT + BUTTON_MARGIN // 2
    list_h = len(EXERCISES) * item_height
    start_y = ty + th + BUTTON_MARGIN * 2 # Start list below title
    button_w = int(w * 0.4)
    button_x = w // 2 - button_w // 2

    for i, ex in enumerate(EXERCISES):
        btn_y = start_y + i * item_height
        is_active = (ex == current_exercise)
        bg_color = COLORS["button_bg_active"] if is_active else COLORS["button_bg_normal"]
        text_color = COLORS["button_text_active"] if is_active else COLORS["button_text_normal"]
        border_color = COLORS["button_text_active"] if is_active else COLORS["secondary_text"]

        draw_rounded_rectangle(canvas, (button_x, btn_y), (button_x + button_w, btn_y + BUTTON_HEIGHT), bg_color, -1, CORNER_RADIUS)
        draw_rounded_rectangle(canvas, (button_x, btn_y), (button_x + button_w, btn_y + BUTTON_HEIGHT), border_color, 1, CORNER_RADIUS)

        (tw_ex, th_ex), _ = cv2.getTextSize(ex, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS)
        tx_ex = button_x + max(0, (button_w - tw_ex) // 2)
        ty_ex = btn_y + (BUTTON_HEIGHT + th_ex) // 2
        cv2.putText(canvas, ex, (tx_ex, ty_ex), FONT, BUTTON_TEXT_SCALE * 1.1, text_color, LINE_THICKNESS, cv2.LINE_AA)

    # Start Button
    start_btn_w, start_btn_h = 200, BUTTON_HEIGHT
    start_btn_x = w // 2 - start_btn_w // 2
    start_btn_y = start_y + list_h + BUTTON_MARGIN # Position below list
    draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS)
    draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS)
    btn_text = "Start"
    (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE * 1.1, LINE_THICKNESS)
    cv2.putText(canvas, btn_text, (start_btn_x + (start_btn_w - tw) // 2, start_btn_y + (start_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE * 1.1, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)

    # Back Button
    back_btn_w, back_btn_h = 150, BUTTON_HEIGHT
    back_btn_x = BUTTON_MARGIN * 2
    back_btn_y = h - back_btn_h - BUTTON_MARGIN * 2 # Bottom Left
    draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["button_bg_normal"], -1, CORNER_RADIUS)
    draw_rounded_rectangle(canvas, (back_btn_x, back_btn_y), (back_btn_x + back_btn_w, back_btn_y + back_btn_h), COLORS["secondary_text"], 1, CORNER_RADIUS)
    btn_text = "Back"
    (tw, th), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS)
    cv2.putText(canvas, btn_text, (back_btn_x + (back_btn_w - tw) // 2, back_btn_y + (back_btn_h + th) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_normal"], LINE_THICKNESS, cv2.LINE_AA)

    # Feedback Area (Optional, for errors like "Select an exercise first")
    (tw_fb, th_fb), _ = cv2.getTextSize(feedback_msg, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS)
    fx = (w - tw_fb) // 2
    fy = h - BUTTON_MARGIN * 2 # Above Back button? Or near top? Let's put near top
    fy = ty + th + BUTTON_MARGIN // 2
    feedback_color = COLORS["accent_red"] if "Error" in feedback_msg or "WARN:" in feedback_msg else COLORS["secondary_text"]
    cv2.putText(canvas, feedback_msg, (fx, fy), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)


def draw_guide_ui(canvas):
    global guide_gif_index, guide_last_frame_time
    h, w = canvas.shape[:2]
    canvas[:] = COLORS["background"]
    # Title
    title = f"Guide: {current_exercise}"
    (tw, th), _ = cv2.getTextSize(title, FONT, TITLE_SCALE * 0.8, LINE_THICKNESS)
    cv2.putText(canvas, title, (BUTTON_MARGIN, BUTTON_MARGIN + th), FONT, TITLE_SCALE * 0.8, COLORS["primary_text"], LINE_THICKNESS, cv2.LINE_AA)
    # GIF Display Area
    gif_area_y_start = BUTTON_MARGIN * 2 + th
    gif_area_h = h - gif_area_y_start - BUTTON_MARGIN * 4 - BUTTON_HEIGHT
    gif_area_w = w - BUTTON_MARGIN * 2
    if guide_gif_frames:
        current_time = time.time()
        if current_time - guide_last_frame_time >= guide_frame_delay:
            guide_gif_index = (guide_gif_index + 1) % len(guide_gif_frames)
            guide_last_frame_time = current_time
        frame = guide_gif_frames[guide_gif_index]
        frame_h, frame_w = frame.shape[:2]
        scale = min(gif_area_w / frame_w, gif_area_h / frame_h) if frame_w > 0 and frame_h > 0 else 1
        if scale < 1:
            new_w, new_h = int(frame_w * scale), int(frame_h * scale)
            if new_w > 0 and new_h > 0: display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else: display_frame = frame; new_w, new_h = frame_w, frame_h
        else: display_frame = frame; new_w, new_h = frame_w, frame_h
        ox = BUTTON_MARGIN + (gif_area_w - new_w) // 2
        oy = gif_area_y_start + (gif_area_h - new_h) // 2
        if oy >= 0 and ox >= 0 and oy + new_h <= h and ox + new_w <= w:
             canvas[oy:oy + new_h, ox:ox + new_w] = display_frame
        else: print(f"Warning: GIF display coordinates out of bounds")
    else:
        no_gif_text = "Guide unavailable"
        (tw_ng, th_ng), _ = cv2.getTextSize(no_gif_text, FONT, 1.0, LINE_THICKNESS)
        tx = BUTTON_MARGIN + (gif_area_w - tw_ng) // 2
        ty = gif_area_y_start + (gif_area_h + th_ng) // 2
        cv2.putText(canvas, no_gif_text, (tx, ty), FONT, 1.0, COLORS["secondary_text"], LINE_THICKNESS, cv2.LINE_AA)
    # Start Button
    start_btn_w, start_btn_h = 250, BUTTON_HEIGHT
    start_btn_x = w // 2 - start_btn_w // 2
    start_btn_y = h - start_btn_h - BUTTON_MARGIN * 2
    draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["accent_green"], -1, CORNER_RADIUS)
    draw_rounded_rectangle(canvas, (start_btn_x, start_btn_y), (start_btn_x + start_btn_w, start_btn_y + start_btn_h), COLORS["button_text_active"], 1, CORNER_RADIUS)
    btn_text = "Start Exercise"
    (tw_st, th_st), _ = cv2.getTextSize(btn_text, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS)
    cv2.putText(canvas, btn_text, (start_btn_x + (start_btn_w - tw_st) // 2, start_btn_y + (start_btn_h + th_st) // 2), FONT, BUTTON_TEXT_SCALE, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)

def draw_tracking_ui(canvas, frame):
    h, w = canvas.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    canvas[:] = (0,0,0) # Black background

    # --- Letterboxing & Video Display ---
    ox, oy, sw, sh = 0, 0, w, h
    if frame_w > 0 and frame_h > 0:
        scale = min(w / frame_w, h / frame_h)
        sw, sh = int(frame_w * scale), int(frame_h * scale)
        ox, oy = (w - sw) // 2, (h - sh) // 2
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        try:
            if sw > 0 and sh > 0:
                resized_frame = cv2.resize(frame, (sw, sh), interpolation=interp)
                if oy >= 0 and ox >= 0 and oy + sh <= h and ox + sw <= w:
                    canvas[oy:oy + sh, ox:ox + sw] = resized_frame
                else: print("Warning: Video ROI calculation error.")
            else: print("Warning: Invalid resize dimensions.")
        except Exception as e:
            print(f"Error resizing/placing frame: {e}")
            err_txt = "Video Error"; (tw, th),_ = cv2.getTextSize(err_txt, FONT, 1.0, 2)
            cv2.putText(canvas, err_txt, ((w-tw)//2, (h+th)//2), FONT, 1.0, COLORS['accent_red'], 2, cv2.LINE_AA)
    else:
        err_txt = "Invalid Frame"; (tw, th),_ = cv2.getTextSize(err_txt, FONT, 1.0, 2)
        cv2.putText(canvas, err_txt, ((w-tw)//2, (h+th)//2), FONT, 1.0, COLORS['accent_red'], 2, cv2.LINE_AA)

    # --- Draw UI Elements on Top using Overlay ---
    overlay_canvas = np.zeros_like(canvas) # Transparent overlay

    # Top Exercise Buttons
    try:
        total_button_width = w - 2 * BUTTON_MARGIN
        btn_w = (total_button_width - (len(EXERCISES) - 1) * (BUTTON_MARGIN // 2)) // len(EXERCISES)
        btn_w = max(50, btn_w)
    except ZeroDivisionError: btn_w = 100

    for i, ex in enumerate(EXERCISES):
        bx = BUTTON_MARGIN + i * (btn_w + BUTTON_MARGIN // 2)
        bxe = bx + btn_w
        is_active = (ex == current_exercise)
        bg_color = COLORS["button_bg_active"] if is_active else COLORS["button_bg_normal"]
        text_color = COLORS["button_text_active"] if is_active else COLORS["button_text_normal"]
        border_color = COLORS["button_text_active"] if is_active else COLORS["secondary_text"]
        draw_rounded_rectangle(overlay_canvas, (bx, BUTTON_MARGIN), (bxe, BUTTON_MARGIN + BUTTON_HEIGHT), bg_color, -1, CORNER_RADIUS)
        draw_rounded_rectangle(overlay_canvas, (bx, BUTTON_MARGIN), (bxe, BUTTON_MARGIN + BUTTON_HEIGHT), border_color, 1, CORNER_RADIUS)
        (tw_ex, th_ex), _ = cv2.getTextSize(ex, FONT, BUTTON_TEXT_SCALE, LINE_THICKNESS)
        tx = bx + max(0, (btn_w - tw_ex) // 2)
        ty = BUTTON_MARGIN + (BUTTON_HEIGHT + th_ex) // 2
        cv2.putText(overlay_canvas, ex, (tx, ty), FONT, BUTTON_TEXT_SCALE, text_color, LINE_THICKNESS, cv2.LINE_AA)

    # Status Box (Top-Left)
    is_bicep = current_exercise == "BICEP CURL"
    sb_h = 160 if is_bicep else 130
    sb_w = 380 if is_bicep else 280
    sb_x, sb_y = BUTTON_MARGIN, BUTTON_MARGIN * 2 + BUTTON_HEIGHT
    sb_xe, sb_ye = sb_x + sb_w, sb_y + sb_h
    # Draw semi-transparent background and border on overlay
    draw_semi_transparent_rect(overlay_canvas, (sb_x, sb_y), (sb_xe, sb_ye), COLORS["overlay_bg"], OVERLAY_ALPHA) # Use overlay_bg color
    draw_rounded_rectangle(overlay_canvas, (sb_x, sb_y), (sb_xe, sb_ye), COLORS["secondary_text"], 1, CORNER_RADIUS)

    # Status Box Content (use primary_text for labels for better contrast)
    line_h = 30
    label_color = COLORS["primary_text"] # Use primary text for labels now
    value_color = COLORS["primary_text"] # Keep values as primary text
    rep_color = COLORS["primary_text"] # Rep count color
    stage_color = COLORS["primary_text"] # Stage text color

    cv2.putText(overlay_canvas, 'EXERCISE:', (sb_x + 15, sb_y + line_h), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA)
    cv2.putText(overlay_canvas, current_exercise, (sb_x + 110, sb_y + line_h), FONT, STATUS_TEXT_SCALE, value_color, LINE_THICKNESS, cv2.LINE_AA)

    if is_bicep:
        rep_y = sb_y + line_h * 2 + 15
        stage_y = sb_y + line_h * 3 + 25
        col1_x = sb_x + 15
        col2_x = sb_x + sb_w // 2
        cv2.putText(overlay_canvas, 'L REPS:', (col1_x, rep_y), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, str(counter_left), (col1_x + 80, rep_y + 5), FONT, REP_TEXT_SCALE * 0.8, rep_color, LINE_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, 'L STAGE:', (col1_x, stage_y), FONT, STATUS_TEXT_SCALE * 0.9, label_color, 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, stage_left, (col1_x + 85, stage_y), FONT, STATUS_TEXT_SCALE, stage_color, LINE_THICKNESS, cv2.LINE_AA)
        cv2.putText(overlay_canvas, 'R REPS:', (col2_x, rep_y), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, str(counter_right), (col2_x + 80, rep_y + 5), FONT, REP_TEXT_SCALE * 0.8, rep_color, LINE_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, 'R STAGE:', (col2_x, stage_y), FONT, STATUS_TEXT_SCALE * 0.9, label_color, 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, stage_right, (col2_x + 85, stage_y), FONT, STATUS_TEXT_SCALE, stage_color, LINE_THICKNESS, cv2.LINE_AA)
    else:
        rep_y = sb_y + line_h * 2 + 15
        stage_y = sb_y + line_h * 3 + 15
        cv2.putText(overlay_canvas, 'REPS:', (sb_x + 15, rep_y), FONT, STATUS_TEXT_SCALE, label_color, 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, str(counter), (sb_x + 90, rep_y + 10), FONT, REP_TEXT_SCALE, rep_color, LINE_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, 'STAGE:', (sb_x + 15, stage_y), FONT, STATUS_TEXT_SCALE * 0.9, label_color, 1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, stage, (sb_x + 90, stage_y), FONT, STATUS_TEXT_SCALE, stage_color, LINE_THICKNESS, cv2.LINE_AA)

    # Feedback Box (Bottom Left)
    fb_h = 65
    home_btn_size = 50
    fb_w = w - 2 * BUTTON_MARGIN - home_btn_size - BUTTON_MARGIN
    fb_x, fb_y = BUTTON_MARGIN, h - fb_h - BUTTON_MARGIN
    fb_xe, fb_ye = fb_x + fb_w, fb_y + fb_h
    # Draw background and border on overlay
    draw_semi_transparent_rect(overlay_canvas, (fb_x, fb_y), (fb_xe, fb_ye), COLORS["overlay_bg"], OVERLAY_ALPHA)
    draw_rounded_rectangle(overlay_canvas, (fb_x, fb_y), (fb_xe, fb_ye), COLORS["secondary_text"], 1, CORNER_RADIUS)
    # Feedback Text
    feedback_color = COLORS["accent_red"] if "WARN:" in feedback_msg else COLORS["accent_blue"]
    display_feedback = feedback_msg.replace("WARN: ", "").replace("INFO: ", "")
    max_feedback_chars = int(fb_w / (FEEDBACK_TEXT_SCALE * 15))
    if len(display_feedback) > max_feedback_chars > 3 :
        display_feedback = display_feedback[:max_feedback_chars - 3] + "..."
    (tw_fb, th_fb), _ = cv2.getTextSize(display_feedback, FONT, FEEDBACK_TEXT_SCALE, LINE_THICKNESS)
    cv2.putText(overlay_canvas, display_feedback, (fb_x + 15, fb_y + (fb_h + th_fb) // 2), FONT, FEEDBACK_TEXT_SCALE, feedback_color, LINE_THICKNESS, cv2.LINE_AA)

    # Home Button (Bottom Right)
    hb_x = w - home_btn_size - BUTTON_MARGIN
    hb_y = h - home_btn_size - BUTTON_MARGIN
    draw_rounded_rectangle(overlay_canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["accent_red"], -1, CORNER_RADIUS // 2)
    draw_rounded_rectangle(overlay_canvas, (hb_x, hb_y), (hb_x + home_btn_size, hb_y + home_btn_size), COLORS["button_text_active"], 1, CORNER_RADIUS // 2)
    (tw_h, th_h), _ = cv2.getTextSize("H", FONT, 0.8, LINE_THICKNESS)
    cv2.putText(overlay_canvas, "H", (hb_x + (home_btn_size - tw_h) // 2, hb_y + (home_btn_size + th_h) // 2), FONT, 0.8, COLORS["button_text_active"], LINE_THICKNESS, cv2.LINE_AA)

    # Blend the overlay onto the main canvas using masking
    gray_overlay = cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
    fg = cv2.bitwise_and(overlay_canvas, overlay_canvas, mask=mask)
    # Add fg and bg, saving the result back into canvas
    cv2.add(bg, fg, dst=canvas)

    return ox, oy, sw, sh # Return letterbox offsets and dimensions


# --- Main Application Loop ---
window_name = 'Fitness Tracker Pro'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Attempt fullscreen setup
if platform.system() == "Windows":
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_win_width, actual_win_height)

callback_param = {'canvas_w': actual_win_width, 'canvas_h': actual_win_height}
cv2.setMouseCallback(window_name, mouse_callback, callback_param)

pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

while True:
    # --- Check for window resize ---
    try:
        win_rect = cv2.getWindowImageRect(window_name)
        if win_rect[2] > 1 and win_rect[3] > 1 and \
           (win_rect[2] != actual_win_width or win_rect[3] != actual_win_height):
            actual_win_width, actual_win_height = win_rect[2], win_rect[3]
            callback_param['canvas_w'] = actual_win_width
            callback_param['canvas_h'] = actual_win_height
            print(f"Window resized to: {actual_win_width}x{actual_win_height}")
    except Exception as e: pass

    # Create canvas for the current frame
    current_canvas = np.zeros((actual_win_height, actual_win_width, 3), dtype=np.uint8)

    # === State Machine Logic ===
    if app_mode == "HOME":
        draw_home_ui(current_canvas)

    elif app_mode == "EXERCISE_SELECT":
        draw_exercise_select_ui(current_canvas)

    elif app_mode == "GUIDE":
        draw_guide_ui(current_canvas)
        # Optional: Auto-transition after duration
        # if time.time() - guide_start_time >= guide_duration:
        #    app_mode = "TRACKING"; set_feedback(f"Start {current_exercise}")

    elif app_mode == "TRACKING":
        if not cap or not cap.isOpened():
            set_feedback("Error: Video source lost.", True)
            app_mode = "HOME"; source_type = None; video_source_selected = False; cap = None; guide_gif_frames=[]
            continue # Skip rest of the loop iteration

        ret, frame = cap.read()
        if not ret:
            # Video looping/error handling
            if video_source_selected and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0: # Check if it's a video file
                 current_frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                 total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                 if total_frames > 0 and current_frame_num >= total_frames - 1:
                     print("Video ended, restarting...")
                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read()
                     if not ret:
                         set_feedback("Error restarting video.", True); app_mode="HOME"; source_type = None; video_source_selected=False; cap.release(); cap=None; guide_gif_frames=[]
                         continue
                 else:
                     set_feedback("Error reading video frame.", True); app_mode = "HOME"; source_type = None; video_source_selected = False; cap.release(); cap = None; guide_gif_frames=[]
                     continue
            else: # Webcam error
                set_feedback("Error reading frame (Webcam?).", True); app_mode = "HOME"; source_type = None; video_source_selected = False; cap.release(); cap = None; guide_gif_frames=[]
                continue

        # --- Draw Base UI and Video ---
        ox, oy, sw, sh = draw_tracking_ui(current_canvas, frame) # Modifies current_canvas

        # --- Image Processing for Pose (using original frame) ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = pose.process(img_rgb)

        # --- Pose Logic ---
        form_correct = True
        current_feedback_parts = []

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # --- Back Posture Check ---
                sh_l, sh_r = get_coords(landmarks,'LEFT_SHOULDER'), get_coords(landmarks,'RIGHT_SHOULDER')
                hip_l, hip_r = get_coords(landmarks,'LEFT_HIP'), get_coords(landmarks,'RIGHT_HIP')
                vis_sh = [s for s in [sh_l, sh_r] if s[3] > 0.6]
                vis_hip = [h for h in [hip_l, hip_r] if h[3] > 0.6]
                back_posture_feedback = ""
                vertical_back_ok = True
                back_angle_vertical = 0.0
                if len(vis_sh) > 0 and len(vis_hip) > 0:
                    sh_ax = np.mean([s[0] for s in vis_sh]); sh_ay = np.mean([s[1] for s in vis_sh])
                    hip_ax = np.mean([h[0] for h in vis_hip]); hip_ay = np.mean([h[1] for h in vis_hip])
                    vec_sh_hp_x = hip_ax - sh_ax; vec_sh_hp_y = hip_ay - sh_ay
                    mag_sh_hp = np.sqrt(vec_sh_hp_x**2 + vec_sh_hp_y**2)
                    if mag_sh_hp > 1e-6:
                        v_vert_x, v_vert_y = 0, 1; dot_product = vec_sh_hp_y
                        cos_theta = np.clip(dot_product / mag_sh_hp, -1.0, 1.0)
                        back_angle_vertical = np.degrees(np.arccos(cos_theta))
                        current_vert_thresh = 90
                        if current_exercise == "BICEP CURL": current_vert_thresh = BACK_ANGLE_THRESHOLD_BICEP
                        elif current_exercise == "SQUAT": current_vert_thresh = BACK_ANGLE_THRESHOLD_SQUAT
                        elif current_exercise == "DEADLIFT": current_vert_thresh = BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT
                        if back_angle_vertical > current_vert_thresh:
                            vertical_back_ok = False
                            back_posture_feedback = f"Fix Back Angle! ({back_angle_vertical:.0f}>{current_vert_thresh})"
                        else: back_posture_feedback = "Back Angle OK"
                    else: back_posture_feedback = "Calc Back Angle..."
                else: back_posture_feedback = "Back Vis Low"

                # --- Get Other Landmarks ---
                el_l, wr_l = get_coords(landmarks,'LEFT_ELBOW'), get_coords(landmarks,'LEFT_WRIST')
                el_r, wr_r = get_coords(landmarks,'RIGHT_ELBOW'), get_coords(landmarks,'RIGHT_WRIST')
                kn_l, an_l = get_coords(landmarks,'LEFT_KNEE'), get_coords(landmarks,'LEFT_ANKLE')
                kn_r, an_r = get_coords(landmarks,'RIGHT_KNEE'), get_coords(landmarks,'RIGHT_ANKLE')
                nose = get_coords(landmarks, 'NOSE')

                # --- Coordinate Scaling Func ---
                def scale_coords_tracking(coords_norm):
                    if coords_norm is None or len(coords_norm) < 2 or any(c is None for c in coords_norm[:2]): return None
                    scaled_x = int(coords_norm[0] * sw + ox)
                    scaled_y = int(coords_norm[1] * sh + oy)
                    scaled_x = max(0, min(actual_win_width - 1, scaled_x))
                    scaled_y = max(0, min(actual_win_height - 1, scaled_y))
                    return (scaled_x, scaled_y)

                # --- EXERCISE SPECIFIC LOGIC ---
                form_correct = vertical_back_ok # Start with back check result
                is_up = False # Define before DL check

                # === BICEP CURL ===
                if current_exercise == "BICEP CURL":
                    ct = time.time()
                    # Left Arm
                    if all(c[3] > 0.5 for c in [sh_l, el_l, wr_l]):
                        a_l = calculate_angle(sh_l[:2], el_l[:2], wr_l[:2])
                        if form_correct:
                            if a_l > BICEP_DOWN_ANGLE: stage_left = "DOWN"
                            if a_l < BICEP_UP_ANGLE and stage_left == 'DOWN':
                                stage_left = "UP";
                                if ct - last_rep_time_left > rep_cooldown: counter_left += 1; last_rep_time_left = ct; current_feedback_parts.append("L: Rep!")
                                else: current_feedback_parts.append("L: Cool")
                        elif stage_left != "START": stage_left = "HOLD"; current_feedback_parts.append("L: Fix Form!")
                    else: stage_left = "START"; current_feedback_parts.append("L Arm?")
                    # Right Arm
                    if all(c[3] > 0.5 for c in [sh_r, el_r, wr_r]):
                        a_r = calculate_angle(sh_r[:2], el_r[:2], wr_r[:2])
                        if form_correct:
                            if a_r > BICEP_DOWN_ANGLE: stage_right = "DOWN"
                            if a_r < BICEP_UP_ANGLE and stage_right == 'DOWN':
                                stage_right = "UP";
                                if ct - last_rep_time_right > rep_cooldown: counter_right += 1; last_rep_time_right = ct; current_feedback_parts.append("R: Rep!")
                                else: current_feedback_parts.append("R: Cool")
                        elif stage_right != "START": stage_right = "HOLD"; current_feedback_parts.append("R: Fix Form!")
                    else: stage_right = "START"; current_feedback_parts.append("R Arm?")

                # === SQUAT ===
                elif current_exercise == "SQUAT":
                    if all(c[3] > 0.5 for c in [hip_l, kn_l, an_l]):
                        ka = calculate_angle(hip_l[:2], kn_l[:2], an_l[:2])
                        if form_correct:
                            if ka > SQUAT_UP_ANGLE:
                                if stage == 'DOWN':
                                    stage = "UP"; ct = time.time()
                                    if ct - last_rep_time > rep_cooldown: counter += 1; last_rep_time = ct; current_feedback_parts.append("Rep!")
                                    else: current_feedback_parts.append("Cooldown")
                                elif stage != 'UP': stage = 'UP'
                            elif ka < SQUAT_DOWN_ANGLE:
                                if stage == 'UP': current_feedback_parts.append("Squat Deeper")
                                stage = "DOWN"
                            if stage == 'UP' and ka < SQUAT_UP_ANGLE * 0.95: current_feedback_parts.append("Stand Fully")
                        elif stage != "START": stage = "HOLD"; current_feedback_parts.append("Fix Back!")
                    else: stage = "START"; current_feedback_parts.append("Leg?")

                # === PUSH UP ===
                elif current_exercise == "PUSH UP":
                    body_form_ok = True
                    if all(c[3] > 0.5 for c in [sh_l, el_l, wr_l, hip_l, kn_l]):
                        ea = calculate_angle(sh_l[:2], el_l[:2], wr_l[:2])
                        ba = calculate_angle(sh_l[:2], hip_l[:2], kn_l[:2])
                        if not (PUSHUP_BODY_STRAIGHT_MIN < ba < PUSHUP_BODY_STRAIGHT_MAX):
                            body_form_ok = False; form_correct = False; current_feedback_parts.append("Keep Body Straight!")
                        if form_correct:
                            if ea > PUSHUP_UP_ANGLE:
                                if stage == 'DOWN':
                                    stage = "UP"; ct = time.time()
                                    if ct - last_rep_time > rep_cooldown: counter += 1; last_rep_time = ct; current_feedback_parts.append("Rep!")
                                    else: current_feedback_parts.append("Cooldown")
                                elif stage != 'UP': stage = 'UP'
                            elif ea < PUSHUP_DOWN_ANGLE:
                                 if stage == 'UP': current_feedback_parts.append("Lower Chest")
                                 stage = "DOWN"
                        elif stage != "START": stage = "HOLD"
                    else: stage = "START"; current_feedback_parts.append("Body?")

                # === PULL UP ===
                elif current_exercise == "PULL UP":
                     form_correct = True
                     if all(c[3] > 0.5 for c in [sh_l, el_l, wr_l, nose]):
                         ea = calculate_angle(sh_l[:2], el_l[:2], wr_l[:2])
                         ny, wy = nose[1], wr_l[1]
                         is_up_pos = ny < wy and ea < PULLUP_UP_ELBOW_ANGLE
                         is_down_pos = ea > PULLUP_DOWN_ANGLE
                         if is_down_pos:
                              if stage == 'UP':
                                   stage = "DOWN"; ct = time.time()
                                   if ct - last_rep_time > rep_cooldown: counter += 1; last_rep_time = ct; current_feedback_parts.append("Rep!")
                                   else: current_feedback_parts.append("Cooldown")
                              elif stage != 'DOWN': stage = 'DOWN'
                         elif is_up_pos:
                              if stage == 'DOWN': current_feedback_parts.append("Pull Higher!")
                              stage = "UP"
                     else: stage = "START"; current_feedback_parts.append("Body/Head?")

                # === DEADLIFT ===
                elif current_exercise == "DEADLIFT":
                    deadlift_lockout_form_ok = True
                    full_body_visible = all(c[3] > 0.5 for c in [sh_l, hip_l, kn_l, an_l, sh_r, hip_r, kn_r, an_r])
                    if full_body_visible:
                        hip_avg = np.mean([hip_l[:2], hip_r[:2]], axis=0); kn_avg = np.mean([kn_l[:2], kn_r[:2]], axis=0)
                        an_avg = np.mean([an_l[:2], an_r[:2]], axis=0); sh_avg = np.mean([sh_l[:2], sh_r[:2]], axis=0)
                        hip_a = calculate_angle(sh_avg, hip_avg, kn_avg); knee_a = calculate_angle(hip_avg, kn_avg, an_avg)
                        is_up = hip_a > DEADLIFT_UP_ANGLE_BODY and knee_a > DEADLIFT_UP_ANGLE_BODY
                        is_down = hip_a < DEADLIFT_DOWN_HIP_ANGLE
                        if is_up and not vertical_back_ok:
                            deadlift_lockout_form_ok = False; form_correct = False; current_feedback_parts.append(f"Bad Lockout Angle! ({back_angle_vertical:.0f}>{BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT})")
                        elif is_up and vertical_back_ok: form_correct = True
                        else: form_correct = True;
                        if back_posture_feedback == "Back Angle OK" and not (is_up and form_correct): back_posture_feedback = ""
                        if is_up:
                            if stage == 'DOWN':
                                if deadlift_lockout_form_ok:
                                    stage = "UP"; ct = time.time()
                                    if ct - last_rep_time > rep_cooldown: counter += 1; last_rep_time = ct; current_feedback_parts.append("Rep!")
                                    else: current_feedback_parts.append("Cooldown")
                                else: stage = "HOLD"
                            elif stage == 'HOLD' and deadlift_lockout_form_ok: stage = 'UP'; current_feedback_parts.append("Lockout Fixed!")
                            elif stage != 'UP' and stage != 'HOLD' and deadlift_lockout_form_ok: stage = 'UP'
                        if is_down:
                            if stage == 'UP' or stage == 'HOLD': stage = "DOWN"; current_feedback_parts.append("Lift Up!")
                            elif stage != 'DOWN': stage = 'DOWN'
                    else: stage = "START"; current_feedback_parts.append("Full Body?")


                # --- Update Combined Feedback ---
                final_feedback_parts = []
                has_warning = False
                # Use stage variable which is updated by exercise logic
                current_stage_is_up = (stage == 'UP' or stage_left == 'UP' or stage_right == 'UP')

                if current_feedback_parts:
                     form_warnings = [f for f in current_feedback_parts if "Form" in f or "Angle" in f or "Straight" in f or "Fix" in f or "Bad" in f]
                     if form_warnings: final_feedback_parts.extend(list(dict.fromkeys(form_warnings))); has_warning = True
                     else: final_feedback_parts.extend(list(dict.fromkeys(current_feedback_parts)))

                show_back_feedback = False
                if back_posture_feedback:
                    if "OK" not in back_posture_feedback: show_back_feedback = True; has_warning = True
                    elif not final_feedback_parts:
                         if current_exercise != "DEADLIFT" or (current_stage_is_up and form_correct): show_back_feedback = True
                if show_back_feedback: final_feedback_parts.insert(0, back_posture_feedback)

                if final_feedback_parts: set_feedback(" | ".join(final_feedback_parts), is_warning=has_warning)
                elif stage != "START": set_feedback("Keep Going...")
                else: set_feedback("Analyzing...")

                # --- Draw Landmarks ---
                drawing_spec_landmarks = mp_drawing.DrawingSpec(color=COLORS["landmark_vis"], thickness=1, circle_radius=2)
                drawing_spec_connections = mp_drawing.DrawingSpec(color=COLORS["connection"], thickness=1)
                if sw > 0 and sh > 0:
                    # Draw connections
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if results.pose_landmarks.landmark[start_idx].visibility > 0.5 and results.pose_landmarks.landmark[end_idx].visibility > 0.5:
                             start_pt = scale_coords_tracking(get_coords(landmarks, mp_pose.PoseLandmark(start_idx).name)[:2])
                             end_pt = scale_coords_tracking(get_coords(landmarks, mp_pose.PoseLandmark(end_idx).name)[:2])
                             if start_pt and end_pt: cv2.line(current_canvas, start_pt, end_pt, drawing_spec_connections.color, drawing_spec_connections.thickness)
                    # Draw landmarks
                    for i, landmark in enumerate(landmarks):
                         if landmark.visibility > 0.5:
                              point = scale_coords_tracking([landmark.x, landmark.y])
                              if point: cv2.circle(current_canvas, point, drawing_spec_landmarks.circle_radius, drawing_spec_landmarks.color, -1)

            else: # No pose detected
                set_feedback("No Person Detected", True)
                reset_exercise_state() # Reset stages if no person

        except Exception as e:
            print(f"!! Error during pose logic/drawing: {e}"); traceback.print_exc()
            set_feedback("Error processing frame.", True)
            reset_exercise_state() # Reset stages on error


    # --- Display Final Canvas ---
    cv2.imshow(window_name, current_canvas)

    # --- Quit Key ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quit key pressed.")
        break

# --- Cleanup ---
print("Releasing resources...")
if pose: pose.close()
if cap: cap.release(); print("Video capture released.")
cv2.destroyAllWindows()
print("Application Closed.")