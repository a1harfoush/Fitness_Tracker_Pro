import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog
import platform # To help with fullscreen adaptation
import traceback # For detailed error printing

# --- Configuration Constants ---

# Rep Counting Thresholds (Incorporating Hysteresis)
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

# --- Form Correction Thresholds ---
# Back Posture (Deviation from Vertical UP - degrees)
BACK_ANGLE_THRESHOLD_BICEP = 20       # Stricter for Bicep Curls
BACK_ANGLE_THRESHOLD_SQUAT = 45       # Max lean during squat
BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT = 15 # Very strict at lockout
BACK_ANGLE_THRESHOLD_DEADLIFT_LIFT = 60   # Max lean during the lift itself (approx)

# Pushup Specific
PUSHUP_BODY_STRAIGHT_MIN = 150 # Min angle Shoulder-Hip-Knee
PUSHUP_BODY_STRAIGHT_MAX = 190 # Max angle Shoulder-Hip-Knee

# Squat Specific
SQUAT_KNEE_VALGUS_THRESHOLD = 0.05 # Max normalized horizontal distance diff (knee vs ankle) - adjust!
SQUAT_CHEST_FORWARD_THRESHOLD = 0.1 # Max normalized horizontal distance (shoulder ahead of knee) - adjust!

# Bicep Specific
BICEP_UPPER_ARM_VERT_DEVIATION = 25 # Max degrees deviation from vertical down

# --- EMA Smoothing ---
EMA_ALPHA = 0.3 # Smoothing factor (higher means less smoothing, faster response)

# --- Visuals ---
COLOR_INFO_TEXT = (0, 255, 255) # Cyan
COLOR_WARN_TEXT = (0, 0, 255)   # Red
COLOR_GOOD_FORM = (0, 255, 0)   # Green
COLOR_BAD_FORM = (0, 0, 255)    # Red
COLOR_NEUTRAL_POSE = (255, 100, 255) # Magenta
COLOR_HIGHLIGHT_POSE = (80, 200, 80) # Light Green (default landmark)
COLOR_FEEDBACK_BG = (50, 50, 50)
COLOR_FEEDBACK_BORDER = (200, 200, 200)
COLOR_BUTTON_INACTIVE = (100, 100, 100)
COLOR_BUTTON_ACTIVE = (0, 220, 0)
COLOR_BUTTON_TEXT = (255, 255, 255)
COLOR_HOME_BUTTON = (0, 0, 200)

# --- Mediapipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Global Variables ---
app_mode = "HOME"
cap = None
video_source_selected = False
try: # Get screen dimensions safely
    root_temp = tk.Tk(); root_temp.withdraw()
    target_win_width = root_temp.winfo_screenwidth()
    target_win_height = root_temp.winfo_screenheight()
    root_temp.destroy()
except: target_win_width, target_win_height = 1280, 720
actual_win_width, actual_win_height = target_win_width, target_win_height
canvas = None
EXERCISES = ["BICEP CURL", "SQUAT", "PUSH UP", "PULL UP", "DEADLIFT"]
current_exercise = EXERCISES[0]
button_height, button_margin = 50, 15
counter, stage = 0, None # Use None for initial state
counter_left, counter_right = 0, 0
stage_left, stage_right = None, None # Use None for initial state
feedback_list = ["Select Video Source"] # Store feedback messages as a list
last_feedback_display = ""
last_rep_time, last_rep_time_left, last_rep_time_right = 0, 0, 0
rep_cooldown = 0.5 # Seconds
form_correct_overall = True
form_issues_details = set() # Store specific joints/parts with issues
home_button_width, home_button_height = 300, 60

# --- EMA State Variables --- (Store smoothed values)
ema_angles = {} # Dictionary to store EMA values for angles used in logic

# --- Helper Functions ---

# EMA Calculation
def update_ema(current_value, key, storage_dict):
    if not isinstance(current_value, (int, float)):
        return current_value # Cannot smooth non-numeric types

    if key not in storage_dict or storage_dict[key] is None:
        storage_dict[key] = float(current_value) # Initialize as float
    else:
        prev_ema = storage_dict[key]
        storage_dict[key] = EMA_ALPHA * float(current_value) + (1 - EMA_ALPHA) * prev_ema
    return storage_dict[key]

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

    if norm_ba == 0 or norm_bc == 0: return 0 # Avoid division by zero

    dot_product = np.dot(vec_ba, vec_bc)
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return int(angle)

def get_coords(landmarks, landmark_name):
    """Safely retrieves landmark coordinates and visibility. Uses RAW coordinates."""
    try:
        lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]
        # Return RAW coordinates for drawing accuracy & logic needing positions
        return [lm.x, lm.y, lm.z, lm.visibility]
    except Exception:
        return [0, 0, 0, 0] # Return neutral value with 0 visibility

def get_segment_vertical_angle(p1_coords, p2_coords):
     """Calculates the angle of the segment p1->p2 relative to vertical DOWN (degrees)."""
     if p1_coords[3] < 0.5 or p2_coords[3] < 0.5: return None # Need visible points
     # Use only x, y for 2D angle calculation
     vec = np.array(p2_coords[:2]) - np.array(p1_coords[:2]) # Vector p1 -> p2
     norm = np.linalg.norm(vec)
     if norm == 0: return None

     # Vertical vector pointing DOWN on screen is (0, 1)
     vec_vert_down = np.array([0, 1])
     dot_prod = np.dot(vec, vec_vert_down)

     # Calculate angle in degrees
     angle_rad = np.arccos(np.clip(dot_prod / norm, -1.0, 1.0))
     angle_deg = np.degrees(angle_rad)
     # Angle near 0 means pointing down, near 180 means pointing up
     return angle_deg

def add_feedback(new_msg, is_warning=False):
    """Adds feedback message to the list, avoiding duplicates for this frame."""
    prefix = "WARN: " if is_warning else "INFO: "
    full_msg = prefix + new_msg
    # Only add if it's not already the *exact same* message in the list
    if full_msg not in feedback_list:
        feedback_list.append(full_msg)
    if is_warning:
        global form_correct_overall
        form_correct_overall = False # Mark overall form as incorrect if any warning

def add_form_issue(part_name):
    """Adds the name of the body part with a form issue."""
    form_issues_details.add(part_name)

# --- Mouse Callback ---
def mouse_callback(event, x, y, flags, param):
    global app_mode, current_exercise, counter, stage, feedback_list, video_source_selected, cap
    global counter_left, counter_right, stage_left, stage_right
    global last_rep_time, last_rep_time_left, last_rep_time_right, ema_angles

    canvas_w = param.get('canvas_w', actual_win_width)
    canvas_h = param.get('canvas_h', actual_win_height)

    if app_mode == "HOME":
        if event == cv2.EVENT_LBUTTONDOWN:
            webcam_btn_x, webcam_btn_y = canvas_w//2-home_button_width//2, canvas_h//2-home_button_height-button_margin//2
            if webcam_btn_x <= x <= webcam_btn_x+home_button_width and webcam_btn_y <= y <= webcam_btn_y+home_button_height:
                print("Selecting Webcam...")
                cap = cv2.VideoCapture(0) # Try default first
                if not cap or not cap.isOpened(): cap = cv2.VideoCapture(1) # Fallback
                if cap and cap.isOpened():
                    app_mode="TRACKING"; video_source_selected=True; feedback_list = [f"Starting {current_exercise}"]
                    counter=counter_left=counter_right=0; stage=stage_left=stage_right=None
                    ct=time.time(); last_rep_time=ct; last_rep_time_left=ct; last_rep_time_right=ct
                    ema_angles.clear() # Clear EMA state
                else: feedback_list = ["Error: Webcam not found or busy."]; cap=None
            video_btn_x, video_btn_y = canvas_w//2-home_button_width//2, canvas_h//2+button_margin//2
            if video_btn_x <= x <= video_btn_x+home_button_width and video_btn_y <= y <= video_btn_y+home_button_height:
                print("Selecting Video File...")
                root = tk.Tk(); root.withdraw()
                vp = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
                root.destroy()
                if vp:
                    cap = cv2.VideoCapture(vp)
                    if cap and cap.isOpened():
                        app_mode="TRACKING"; video_source_selected=True; feedback_list = [f"Starting {current_exercise}"]; print(f"Loaded: {vp}")
                        counter=counter_left=counter_right=0; stage=stage_left=stage_right=None
                        ct=time.time(); last_rep_time=ct; last_rep_time_left=ct; last_rep_time_right=ct
                        ema_angles.clear() # Clear EMA state
                    else: feedback_list = [f"Error: Could not open video: {vp}"]; cap=None
                else: feedback_list = ["Video selection cancelled."]
    elif app_mode == "TRACKING":
        if event == cv2.EVENT_LBUTTONDOWN:
            # Exercise Buttons
            try: btn_w = (canvas_w - (len(EXERCISES)+1)*button_margin)//len(EXERCISES)
            except ZeroDivisionError: btn_w = 50
            btn_w = max(10, btn_w)
            for i, ex in enumerate(EXERCISES):
                btn_x = button_margin + i*(btn_w+button_margin)
                btn_xe = min(btn_x+btn_w, canvas_w - button_margin)
                if btn_x <= x <= btn_xe and button_margin <= y <= button_margin+button_height:
                    if current_exercise != ex:
                        print(f"Switching to {ex}"); current_exercise = ex
                        counter=counter_left=counter_right=0; stage=stage_left=stage_right=None
                        feedback_list = [f"Start {ex}"]
                        ct=time.time(); last_rep_time=ct; last_rep_time_left=ct; last_rep_time_right=ct
                        form_correct_overall=True; form_issues_details.clear()
                        ema_angles.clear() # Clear EMA state when switching
                    break
            # Home Button
            home_btn_size=50; home_btn_x=canvas_w-home_btn_size-button_margin; home_btn_y=canvas_h-home_btn_size-button_margin
            if home_btn_x <= x <= home_btn_x+home_btn_size and home_btn_y <= y <= home_btn_y+home_btn_size:
                 print("Returning Home..."); app_mode="HOME"
                 if cap: cap.release(); cap=None; video_source_selected=False; feedback_list = ["Select Video Source"]

# --- Drawing Functions ---
def draw_buttons(canvas_img, w, h):
    try: btn_w = (w - (len(EXERCISES)+1)*button_margin)//len(EXERCISES)
    except ZeroDivisionError: btn_w = 50
    btn_w = max(10, btn_w)
    for i, ex in enumerate(EXERCISES):
        bx=button_margin+i*(btn_w+button_margin); bxe=min(bx+btn_w, w-button_margin)
        clr = COLOR_BUTTON_ACTIVE if ex==current_exercise else COLOR_BUTTON_INACTIVE
        cv2.rectangle(canvas_img,(bx,button_margin),(bxe,button_margin+button_height),clr,-1)
        cv2.rectangle(canvas_img,(bx,button_margin),(bxe,button_margin+button_height),COLOR_FEEDBACK_BORDER,1)
        fs=0.6; (tw,th),_=cv2.getTextSize(ex,0,fs,2); cbw=bxe-bx
        tx=bx+max(0,(cbw-tw)//2); ty=button_margin+(button_height+th)//2
        cv2.putText(canvas_img,ex,(tx,ty),0,fs,COLOR_BUTTON_TEXT,2,cv2.LINE_AA)

def draw_status_box(canvas_img, w, h):
    sby=button_margin*2+button_height; is_bicep=current_exercise=="BICEP CURL"
    sbw=420 if is_bicep else 350; sbh=150 if is_bicep else 120
    sbxe=min(button_margin+sbw, w-button_margin); sbye=min(sby+sbh, h-button_margin)
    cv2.rectangle(canvas_img,(button_margin,sby),(sbxe,sbye),COLOR_FEEDBACK_BG,-1)
    cv2.rectangle(canvas_img,(button_margin,sby),(sbxe,sbye),COLOR_FEEDBACK_BORDER,1)
    cv2.putText(canvas_img,'EXERCISE:',(button_margin+15,sby+25),0,0.6,COLOR_BUTTON_TEXT,1,cv2.LINE_AA)
    cv2.putText(canvas_img,current_exercise,(button_margin+120,sby+25),0,0.6,COLOR_GOOD_FORM,2,cv2.LINE_AA)
    # Display Stage safely (handle None)
    display_stage = stage if stage is not None else "INIT"
    display_stage_l = stage_left if stage_left is not None else "INIT"
    display_stage_r = stage_right if stage_right is not None else "INIT"

    if is_bicep:
        cv2.putText(canvas_img,'L REPS:',(button_margin+15,sby+60),0,0.7,COLOR_BUTTON_TEXT,1); cv2.putText(canvas_img,str(counter_left),(button_margin+110,sby+65),0,1.2,COLOR_BUTTON_TEXT,2)
        cv2.putText(canvas_img,'L STAGE:',(button_margin+15,sby+105),0,0.5,COLOR_BUTTON_TEXT,1); cv2.putText(canvas_img,display_stage_l,(button_margin+90,sby+105),0,0.6,COLOR_BUTTON_TEXT,2)
        cv2.putText(canvas_img,'R REPS:',(button_margin+215,sby+60),0,0.7,COLOR_BUTTON_TEXT,1); cv2.putText(canvas_img,str(counter_right),(button_margin+310,sby+65),0,1.2,COLOR_BUTTON_TEXT,2)
        cv2.putText(canvas_img,'R STAGE:',(button_margin+215,sby+105),0,0.5,COLOR_BUTTON_TEXT,1); cv2.putText(canvas_img,display_stage_r,(button_margin+290,sby+105),0,0.6,COLOR_BUTTON_TEXT,2)
    else:
        cv2.putText(canvas_img,'REPS:',(button_margin+15,sby+60),0,1.2,COLOR_BUTTON_TEXT,2); cv2.putText(canvas_img,str(counter),(button_margin+130,sby+65),0,1.5,COLOR_BUTTON_TEXT,3)
        cv2.putText(canvas_img,'STAGE:',(button_margin+15,sby+100),0,0.6,COLOR_BUTTON_TEXT,1); cv2.putText(canvas_img,display_stage,(button_margin+90,sby+100),0,0.7,COLOR_BUTTON_TEXT,2)

def draw_feedback_box(canvas_img, w, h):
    global last_feedback_display
    hbs=50; fbh=60; fby=h-fbh-button_margin; fbw=w-2*button_margin-hbs-button_margin
    fbxe=min(button_margin+fbw, w-button_margin); fbye=min(fby+fbh, h-button_margin)
    cv2.rectangle(canvas_img,(button_margin,fby),(fbxe,fbye),COLOR_FEEDBACK_BG,-1)
    cv2.rectangle(canvas_img,(button_margin,fby),(fbxe,fbye),COLOR_FEEDBACK_BORDER,1)

    # Combine feedback messages
    warnings = [f.replace("WARN: ", "") for f in feedback_list if "WARN:" in f]
    infos = [f.replace("INFO: ", "") for f in feedback_list if "INFO:" in f and "WARN:" not in f]

    display_text = ""
    if warnings:
        display_text = "WARN: " + " | ".join(sorted(list(set(warnings))))
        fc = COLOR_WARN_TEXT
    elif infos:
        display_text = " | ".join(sorted(list(set(infos))))
        fc = COLOR_INFO_TEXT
    elif stage is None and app_mode == "TRACKING": # Handle initial state explicitly
        display_text = "Initializing Tracker..."
        fc = COLOR_INFO_TEXT
    else: # Fallback if list is empty unexpectedly
        display_text = "Status OK"
        fc = COLOR_INFO_TEXT


    # Truncate if too long
    mfl=int(fbw / 9) # Estimate max chars based on width (adjust divisor as needed)
    if len(display_text) > mfl: display_text = display_text[:mfl-3] + "..."

    if display_text != last_feedback_display:
         last_feedback_display = display_text

    (tw,th),_=cv2.getTextSize(display_text,0,0.7,2); fty=fby+(fbh+th)//2
    cv2.putText(canvas_img,display_text,(button_margin+15,fty),0,0.7,fc,2,cv2.LINE_AA)

    # Home Button
    hbx=w-hbs-button_margin; hby=h-hbs-button_margin
    cv2.rectangle(canvas_img,(hbx,hby),(hbx+hbs,hby+hbs),COLOR_HOME_BUTTON,-1); cv2.rectangle(canvas_img,(hbx,hby),(hbx+hbs,hby+hbs),COLOR_BUTTON_TEXT,2)
    (tw,th),_=cv2.getTextSize("H",0,1,2); cv2.putText(canvas_img,"H",(hbx+(hbs-tw)//2,hby+(hbs+th)//2),0,1,COLOR_BUTTON_TEXT,2,cv2.LINE_AA)

def draw_pose_landmarks(target_image, landmarks_list, connections, form_issue_details):
    """Draws pose landmarks on the target_image, highlighting issues."""
    if not landmarks_list: # Check if landmark list exists
        return

    # --- Define Drawing Specs ---
    default_landmark_spec = mp_drawing.DrawingSpec(color=COLOR_HIGHLIGHT_POSE, thickness=2, circle_radius=3) # Smaller radius
    default_connection_spec = mp_drawing.DrawingSpec(color=COLOR_NEUTRAL_POSE, thickness=2)
    problem_landmark_spec = mp_drawing.DrawingSpec(color=COLOR_BAD_FORM, thickness=-1, circle_radius=5) # Filled, larger radius
    problem_connection_spec = mp_drawing.DrawingSpec(color=COLOR_BAD_FORM, thickness=3)

    # --- Prepare Connection Specs ---
    custom_connection_specs = {}
    if connections:
        for connection in connections:
            custom_connection_specs[connection] = default_connection_spec # Start with default

    # --- Identify Problematic Joints (Indices) ---
    relevant_joint_indices = set()
    # Mapping uses landmark ENUM value (integer)
    mapping = {
        "BACK": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
        "LEFT_KNEE": [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
        "RIGHT_KNEE": [mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        "LEFT_ELBOW": [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
        "RIGHT_ELBOW": [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
        "LEFT_UPPER_ARM": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value],
        "RIGHT_UPPER_ARM": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        "BODY": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
        "HIPS": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    }
    for part_name in form_issue_details:
        if part_name in mapping:
            for lm_index_value in mapping[part_name]:
                 # Check if landmark index is valid before adding
                 if 0 <= lm_index_value < len(landmarks_list.landmark):
                    relevant_joint_indices.add(lm_index_value)

    # --- Update Connection Specs for Problematic Joints ---
    if connections:
        for joint_index_value in relevant_joint_indices:
            for connection in connections:
                # A connection is a tuple (start_idx, end_idx)
                if joint_index_value in connection:
                    custom_connection_specs[connection] = problem_connection_spec

    # --- Draw Base Landmarks and Connections ---
    try:
        mp_drawing.draw_landmarks(
            image=target_image,
            landmark_list=landmarks_list, # Pass the actual NormalizedLandmarkList
            connections=connections,
            landmark_drawing_spec=default_landmark_spec, # Use default spec here
            connection_drawing_spec=custom_connection_specs # Use custom dict for connections
        )
    except Exception as draw_error:
        print(f"Error during mp_drawing.draw_landmarks (base): {draw_error}")
        # traceback.print_exc() # Optional detailed traceback

    # --- Redraw Problematic Landmarks On Top ---
    img_h, img_w = target_image.shape[:2]
    if img_h == 0 or img_w == 0: return # Cannot draw on empty image

    try:
        for idx_value in relevant_joint_indices:
             # Access landmark data safely
             if idx_value < len(landmarks_list.landmark):
                 lm = landmarks_list.landmark[idx_value]
                 if lm.visibility > 0.5: # Only redraw visible problematic landmarks
                     # Denormalize coordinates
                     cx = int(lm.x * img_w)
                     cy = int(lm.y * img_h)
                     # Ensure coordinates are within bounds
                     cx = np.clip(cx, 0, img_w - 1)
                     cy = np.clip(cy, 0, img_h - 1)

                     radius = problem_landmark_spec.circle_radius
                     color = problem_landmark_spec.color
                     thickness = problem_landmark_spec.thickness # Use spec's thickness (-1 for filled)
                     cv2.circle(target_image, (cx, cy), radius, color, thickness)
             # else: # Optional: Warn about invalid index if debugging needed
             #    print(f"Warning: Invalid landmark index {idx_value} requested for redraw.")

    except Exception as redraw_error:
        print(f"Error during manual redraw of problematic landmarks: {redraw_error}")
        traceback.print_exc()

# --- Main Application Loop ---
window_name = 'Fitness Tracker Pro v7 - Corrected'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# Attempt fullscreen based on platform
if platform.system() == "Windows":
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
elif platform.system() == "Darwin": # macOS might need normal window
     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
     cv2.resizeWindow(window_name, actual_win_width, actual_win_height) # Resize manually if needed
else: # Linux/Other - Try normal first, might work
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_win_width, actual_win_height)

callback_param = {'canvas_w': actual_win_width, 'canvas_h': actual_win_height}
cv2.setMouseCallback(window_name, mouse_callback, callback_param)

# Initialize Pose model
pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55, model_complexity=1) # Slightly adjusted confidence

while True:
    # --- Window Size Update ---
    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect[2] > 1 and rect[3] > 1 and (rect[2] != actual_win_width or rect[3] != actual_win_height):
            actual_win_width, actual_win_height = rect[2], rect[3]
            callback_param['canvas_w'], callback_param['canvas_h'] = actual_win_width, actual_win_height
            # print(f"Window resized to: {actual_win_width}x{actual_win_height}") # Optional debug
    except Exception: pass

    # --- Create Canvas ---
    canvas = np.zeros((actual_win_height, actual_win_width, 3), dtype=np.uint8)

    # --- Mode Handling ---
    if app_mode == "HOME":
        # --- Home Screen Drawing ---
        title_text = window_name
        (tw,th),_ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        tx,ty=(actual_win_width-tw)//2, actual_win_height//4+th//2
        cv2.putText(canvas,title_text,(tx,ty), cv2.FONT_HERSHEY_SIMPLEX,1.5,COLOR_INFO_TEXT,3,cv2.LINE_AA)
        # Webcam Btn
        wbx,wby=actual_win_width//2-home_button_width//2, actual_win_height//2-home_button_height-button_margin//2
        cv2.rectangle(canvas,(wbx,wby),(wbx+home_button_width,wby+home_button_height),(0,200,0),-1)
        cv2.rectangle(canvas,(wbx,wby),(wbx+home_button_width,wby+home_button_height),COLOR_BUTTON_TEXT,2)
        (tw,th),_ = cv2.getTextSize("Use Webcam", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas,"Use Webcam",(wbx+(home_button_width-tw)//2, wby+(home_button_height+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,COLOR_BUTTON_TEXT,2,cv2.LINE_AA)
        # Video Btn
        vbx,vby=actual_win_width//2-home_button_width//2, actual_win_height//2+button_margin//2
        cv2.rectangle(canvas,(vbx,vby),(vbx+home_button_width,vby+home_button_height),(200,0,0),-1)
        cv2.rectangle(canvas,(vbx,vby),(vbx+home_button_width,vby+home_button_height),COLOR_BUTTON_TEXT,2)
        (tw,th),_ = cv2.getTextSize("Load Video File", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas,"Load Video File",(vbx+(home_button_width-tw)//2, vby+(home_button_height+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,COLOR_BUTTON_TEXT,2,cv2.LINE_AA)
        # Feedback
        feedback_str = feedback_list[0] if feedback_list else ""
        (tw,th),_ = cv2.getTextSize(feedback_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        fx,fy = (actual_win_width-tw)//2, actual_win_height*3//4+th//2
        fc = COLOR_WARN_TEXT if "Error" in feedback_str else COLOR_INFO_TEXT
        cv2.putText(canvas,feedback_str,(fx,fy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,fc,2,cv2.LINE_AA)
        # Quit Text
        qt="Press 'Q' to Quit"; (tw,th),_ = cv2.getTextSize(qt,cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(canvas,qt,(actual_win_width-tw-20, actual_win_height-th-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(150,150,150),1,cv2.LINE_AA)

    elif app_mode == "TRACKING":
        # --- Video Reading ---
        if not cap or not cap.isOpened():
             feedback_list = ["Error: Video source lost."]; app_mode="HOME"; cap=None; continue
        ret, frame = cap.read()
        if not ret:
            is_video_file = False
            try: # Check if it's a file by checking frame count
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                is_video_file = frame_count > 0
            except: pass # Webcam might not support frame count

            if is_video_file:
                 print("Video ended, looping...")
                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read()
                 if not ret: feedback_list = ["Error reading video after loop."]; app_mode="HOME"; cap=None; continue
            else: # Webcam error or file read error
                 feedback_list = ["Error: Cannot read frame."]; app_mode="HOME"; cap=None; continue

        # --- Pose Processing on ORIGINAL frame ---
        frame_h, frame_w, _ = frame.shape
        if frame_h == 0 or frame_w == 0: continue # Skip empty frames

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = pose.process(img_rgb)
        # No need to make img_rgb writable again unless modifying it directly

        # --- Letterboxing & Resizing the ORIGINAL frame for display ---
        scale = min(actual_win_width/frame_w, actual_win_height/frame_h) if frame_w > 0 and frame_h > 0 else 1
        sw, sh = int(frame_w*scale), int(frame_h*scale)
        ox, oy = (actual_win_width-sw)//2, (actual_win_height-sh)//2
        interp = cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR

        # Ensure target dimensions are valid before resizing
        if sw > 0 and sh > 0 :
            rf = cv2.resize(frame,(sw,sh),interpolation=interp)
        else:
             # print(f"Warning: Invalid target dimensions for scaling ({sw}x{sh}). Skipping frame.")
             rf = np.zeros((max(sh,1), max(sw,1), 3), dtype=np.uint8) # Create minimal black image

        # --- Pose Logic (Uses 'results' from ORIGINAL frame processing) ---
        feedback_list = []
        form_correct_overall = True
        form_issues_details.clear()
        rep_counted_this_frame = False
        landmarks = None # Initialize landmarks to None

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks # Keep the actual object

                # --- Get RAW Coords (for logic that needs positions) ---
                sh_l, sh_r = get_coords(landmarks.landmark,'LEFT_SHOULDER'), get_coords(landmarks.landmark,'RIGHT_SHOULDER')
                hip_l, hip_r = get_coords(landmarks.landmark,'LEFT_HIP'), get_coords(landmarks.landmark,'RIGHT_HIP')
                kn_l, kn_r = get_coords(landmarks.landmark,'LEFT_KNEE'), get_coords(landmarks.landmark,'RIGHT_KNEE')
                an_l, an_r = get_coords(landmarks.landmark,'LEFT_ANKLE'), get_coords(landmarks.landmark,'RIGHT_ANKLE')
                el_l, el_r = get_coords(landmarks.landmark,'LEFT_ELBOW'), get_coords(landmarks.landmark,'RIGHT_ELBOW')
                wr_l, wr_r = get_coords(landmarks.landmark,'LEFT_WRIST'), get_coords(landmarks.landmark,'RIGHT_WRIST')
                nose = get_coords(landmarks.landmark, 'NOSE')

                # --- Basic Visibility Check ---
                # (Add more sophisticated checks if needed)
                critical_joints_visible = True # Assume true initially
                # Example: check if hips and shoulders are visible for back angle calc
                if not all(c[3] > 0.3 for c in [sh_l, sh_r, hip_l, hip_r]):
                    critical_joints_visible = False
                    add_feedback("Torso not fully visible", True)

                # --- Calculate Angles using RAW coordinates ---
                if critical_joints_visible: # Only calculate if needed parts are somewhat visible
                    angle_l_elbow = calculate_angle(sh_l, el_l, wr_l)
                    angle_r_elbow = calculate_angle(sh_r, el_r, wr_r)
                    angle_l_knee = calculate_angle(hip_l, kn_l, an_l)
                    angle_r_knee = calculate_angle(hip_r, kn_r, an_r)
                    angle_l_hip = calculate_angle(sh_l, hip_l, kn_l)
                    angle_r_hip = calculate_angle(sh_r, hip_r, kn_r)
                    angle_l_body = calculate_angle(sh_l, hip_l, kn_l)
                    angle_r_body = calculate_angle(sh_r, hip_r, kn_r)

                    # --- Apply EMA Smoothing (Only for angles used in STATE logic) ---
                    ema_l_elbow = update_ema(angle_l_elbow, "LEFT_ELBOW", ema_angles)
                    ema_r_elbow = update_ema(angle_r_elbow, "RIGHT_ELBOW", ema_angles)
                    ema_l_knee = update_ema(angle_l_knee, "LEFT_KNEE", ema_angles)
                    ema_r_knee = update_ema(angle_r_knee, "RIGHT_KNEE", ema_angles)
                    ema_l_hip = update_ema(angle_l_hip, "LEFT_HIP", ema_angles)
                    ema_r_hip = update_ema(angle_r_hip, "RIGHT_HIP", ema_angles)
                    ema_l_body = update_ema(angle_l_body, "LEFT_BODY", ema_angles)
                    ema_r_body = update_ema(angle_r_body, "RIGHT_BODY", ema_angles)

                    # --- Use Averaged EMA Angles for Logic ---
                    # Use EMA angles if available, otherwise fallback to raw (or skip logic)
                    # Need to handle potential None from EMA init
                    l_knee_logic = ema_angles.get("LEFT_KNEE", angle_l_knee)
                    r_knee_logic = ema_angles.get("RIGHT_KNEE", angle_r_knee)
                    l_hip_logic = ema_angles.get("LEFT_HIP", angle_l_hip)
                    r_hip_logic = ema_angles.get("RIGHT_HIP", angle_r_hip)
                    l_elbow_logic = ema_angles.get("LEFT_ELBOW", angle_l_elbow)
                    r_elbow_logic = ema_angles.get("RIGHT_ELBOW", angle_r_elbow)
                    l_body_logic = ema_angles.get("LEFT_BODY", angle_l_body)
                    r_body_logic = ema_angles.get("RIGHT_BODY", angle_r_body)

                    avg_knee_angle = (l_knee_logic + r_knee_logic) / 2 if (kn_l[3] > 0.5 and kn_r[3] > 0.5) else (l_knee_logic if kn_l[3] > 0.5 else r_knee_logic)
                    avg_hip_angle = (l_hip_logic + r_hip_logic) / 2 if (hip_l[3] > 0.5 and hip_r[3] > 0.5) else (l_hip_logic if hip_l[3] > 0.5 else r_hip_logic)
                    avg_elbow_angle = (l_elbow_logic + r_elbow_logic) / 2 if (el_l[3] > 0.5 and el_r[3] > 0.5) else (l_elbow_logic if el_l[3] > 0.5 else r_elbow_logic)
                    avg_body_angle = (l_body_logic + r_body_logic) / 2 if (hip_l[3] > 0.5 and hip_r[3] > 0.5) else (l_body_logic if hip_l[3] > 0.5 else r_body_logic)

                    # --- Master Form Check: Back Angle (Vertical) ---
                    vertical_back_ok = True
                    back_angle_vertical = 90 # Neutral value
                    vis_sh = [s for s in [sh_l, sh_r] if s[3]>0.6]
                    vis_hip = [h for h in [hip_l, hip_r] if h[3]>0.6]
                    if len(vis_sh)>0 and len(vis_hip)>0:
                        sh_avg_pt = np.mean([s[:2] for s in vis_sh], axis=0)
                        hip_avg_pt = np.mean([h[:2] for h in vis_hip], axis=0)
                        vec_hs = sh_avg_pt - hip_avg_pt
                        vec_vert = np.array([0, -1]) # Vertical UP in image coords
                        norm_hs = np.linalg.norm(vec_hs)
                        if norm_hs > 1e-6: # Avoid division by zero
                            dot_prod = np.dot(vec_hs, vec_vert)
                            cos_theta = np.clip(dot_prod / norm_hs, -1.0, 1.0)
                            back_angle_vertical = np.degrees(np.arccos(cos_theta))

                        # Exercise-Specific Vertical Threshold Check (excluding Deadlift here)
                        if current_exercise != "DEADLIFT":
                            current_vert_thresh = 90 # Default lenient
                            if current_exercise == "BICEP CURL": current_vert_thresh = BACK_ANGLE_THRESHOLD_BICEP
                            elif current_exercise == "SQUAT": current_vert_thresh = BACK_ANGLE_THRESHOLD_SQUAT
                            if back_angle_vertical > current_vert_thresh:
                                vertical_back_ok = False
                                add_feedback(f"Back Angle ({back_angle_vertical:.0f}° > {current_vert_thresh}°)", True)
                                add_form_issue("BACK")

                    # --- EXERCISE SPECIFIC LOGIC ---
                    ct = time.time()

                    # === BICEP CURL ===
                    if current_exercise == "BICEP CURL":
                        # Form Check: Upper Arm Stability
                        upper_arm_vert_angle_l = get_segment_vertical_angle(sh_l, el_l)
                        upper_arm_vert_angle_r = get_segment_vertical_angle(sh_r, el_r)
                        if upper_arm_vert_angle_l is not None and abs(upper_arm_vert_angle_l) > BICEP_UPPER_ARM_VERT_DEVIATION:
                             add_feedback("Keep L Upper Arm Still", True); add_form_issue("LEFT_UPPER_ARM")
                        if upper_arm_vert_angle_r is not None and abs(upper_arm_vert_angle_r) > BICEP_UPPER_ARM_VERT_DEVIATION:
                             add_feedback("Keep R Upper Arm Still", True); add_form_issue("RIGHT_UPPER_ARM")

                        # Rep Counting (Left - uses EMA angles)
                        if vertical_back_ok and sh_l[3] > 0.5 and el_l[3] > 0.5 and wr_l[3] > 0.5:
                            if stage_left is None: stage_left = "INIT"
                            if l_elbow_logic > BICEP_DOWN_ENTER_ANGLE and stage_left != "DOWN":
                                if stage_left == "UP": add_feedback("L: Extend", False)
                                stage_left = "DOWN"
                            elif l_elbow_logic < BICEP_UP_ENTER_ANGLE and stage_left == "DOWN":
                                stage_left = "UP"
                                if form_correct_overall and ct - last_rep_time_left > rep_cooldown:
                                    counter_left += 1; last_rep_time_left = ct; print(f"L Curl: {counter_left}")
                                    add_feedback("L: Rep!", False); rep_counted_this_frame = True
                                elif not form_correct_overall: add_feedback("L: Fix Form", True)
                                else: add_feedback("L: Too Fast!", False)
                            if stage_left == "UP" and l_elbow_logic > BICEP_UP_EXIT_ANGLE: add_feedback("L: Lower", False)
                            if stage_left == "DOWN" and l_elbow_logic < BICEP_DOWN_EXIT_ANGLE: add_feedback("L: Curl", False)

                        # Rep Counting (Right - Mirror Logic)
                        if vertical_back_ok and sh_r[3] > 0.5 and el_r[3] > 0.5 and wr_r[3] > 0.5:
                           if stage_right is None: stage_right = "INIT"
                           if r_elbow_logic > BICEP_DOWN_ENTER_ANGLE and stage_right != "DOWN":
                               if stage_right == "UP": add_feedback("R: Extend", False)
                               stage_right = "DOWN"
                           elif r_elbow_logic < BICEP_UP_ENTER_ANGLE and stage_right == "DOWN":
                               stage_right = "UP"
                               if form_correct_overall and ct - last_rep_time_right > rep_cooldown:
                                   counter_right += 1; last_rep_time_right = ct; print(f"R Curl: {counter_right}")
                                   add_feedback("R: Rep!", False); rep_counted_this_frame = True
                               elif not form_correct_overall: add_feedback("R: Fix Form", True)
                               else: add_feedback("R: Too Fast!", False)
                           if stage_right == "UP" and r_elbow_logic > BICEP_UP_EXIT_ANGLE: add_feedback("R: Lower", False)
                           if stage_right == "DOWN" and r_elbow_logic < BICEP_DOWN_EXIT_ANGLE: add_feedback("R: Curl", False)


                    # === SQUAT ===
                    elif current_exercise == "SQUAT":
                        # Form Check: Knee Valgus (Basic - check if knee X is inside ankle X)
                        if kn_l[3]>0.5 and an_l[3]>0.5 and kn_l[0] < an_l[0] - SQUAT_KNEE_VALGUS_THRESHOLD:
                             add_feedback("L Knee In?", True); add_form_issue("LEFT_KNEE")
                        if kn_r[3]>0.5 and an_r[3]>0.5 and kn_r[0] > an_r[0] + SQUAT_KNEE_VALGUS_THRESHOLD: # Assuming right knee shouldn't be outside right ankle X
                             add_feedback("R Knee Out?", True); add_form_issue("RIGHT_KNEE")
                        # Form Check: Chest Forward
                        if sh_l[3]>0.5 and kn_l[3]>0.5 and stage == "DOWN" and sh_l[0] < kn_l[0] - SQUAT_CHEST_FORWARD_THRESHOLD:
                             add_feedback("Chest Up", True); add_form_issue("BACK")

                        # Rep Counting (uses EMA avg_knee_angle)
                        if vertical_back_ok and kn_l[3] > 0.5 and kn_r[3] > 0.5:
                            if stage is None: stage = "INIT"
                            if avg_knee_angle > SQUAT_UP_ENTER_ANGLE and stage == "DOWN":
                                stage = "UP"
                                if form_correct_overall and (ct - last_rep_time > rep_cooldown):
                                    counter += 1; last_rep_time = ct; print(f"Squat: {counter}")
                                    add_feedback("Rep!", False); rep_counted_this_frame = True
                                elif not form_correct_overall: add_feedback("Fix Form!", True)
                                else: add_feedback("Too Fast!", False)
                            elif avg_knee_angle < SQUAT_DOWN_ENTER_ANGLE and stage != "DOWN":
                                if stage == "UP" or stage == "INIT": add_feedback("Squat", False)
                                stage = "DOWN"; form_correct_overall = True # Reset form for down phase
                            if stage == "UP" and avg_knee_angle < SQUAT_UP_EXIT_ANGLE: add_feedback("Stand Up", False)
                            if stage == "DOWN" and avg_knee_angle > SQUAT_DOWN_EXIT_ANGLE: add_feedback("Deeper", False)

                    # === PUSH UP ===
                    elif current_exercise == "PUSH UP":
                         # Form Check: Body Straightness
                         body_form_ok = True
                         if avg_body_angle < PUSHUP_BODY_STRAIGHT_MIN or avg_body_angle > PUSHUP_BODY_STRAIGHT_MAX:
                              body_form_ok = False
                              add_feedback(f"Body Straight ({avg_body_angle:.0f}deg)", True); add_form_issue("BODY")
                         # Rep Counting (uses EMA avg_elbow_angle)
                         if el_l[3] > 0.5 or el_r[3] > 0.5:
                             if stage is None: stage = "INIT"
                             if avg_elbow_angle > PUSHUP_UP_ENTER_ANGLE and stage == "DOWN":
                                 stage = "UP"
                                 if body_form_ok and (ct - last_rep_time > rep_cooldown):
                                     counter += 1; last_rep_time = ct; print(f"Push Up: {counter}")
                                     add_feedback("Rep!", False); rep_counted_this_frame = True
                                 elif not body_form_ok: add_feedback("Body Line!", True)
                                 else: add_feedback("Too Fast!", False)
                             elif avg_elbow_angle < PUSHUP_DOWN_ENTER_ANGLE and stage != "DOWN":
                                 if stage == "UP" or stage == "INIT": add_feedback("Down", False)
                                 stage = "DOWN"; form_correct_overall = True # Reset form
                             if stage == "UP" and avg_elbow_angle < PUSHUP_UP_EXIT_ANGLE: add_feedback("Extend", False)
                             if stage == "DOWN" and avg_elbow_angle > PUSHUP_DOWN_EXIT_ANGLE: add_feedback("Lower", False)

                    # === PULL UP ===
                    elif current_exercise == "PULL UP":
                        # Logic uses EMA avg_elbow_angle and RAW nose_y, avg_wrist_y
                        avg_wrist_y = (wr_l[1] + wr_r[1]) / 2 if (wr_l[3] > 0.5 and wr_r[3] > 0.5) else (wr_l[1] if wr_l[3] > 0.5 else wr_r[1])
                        nose_y = nose[1]
                        if (el_l[3] > 0.5 or el_r[3] > 0.5) and nose[3] > 0.5:
                            if stage is None: stage = "INIT"
                            is_up_criteria = (avg_elbow_angle < PULLUP_UP_ENTER_ELBOW_ANGLE) and (nose_y < avg_wrist_y if PULLUP_CHIN_ABOVE_WRIST else True)
                            is_down_criteria = avg_elbow_angle > PULLUP_DOWN_ENTER_ANGLE
                            if is_down_criteria and stage == "UP":
                                 stage = "DOWN"
                                 if form_correct_overall and ct - last_rep_time > rep_cooldown: # Assuming form check needed?
                                     counter += 1; last_rep_time = ct; print(f"Pull Up: {counter}")
                                     add_feedback("Rep!", False); rep_counted_this_frame = True
                                 elif not form_correct_overall: add_feedback("Form?", True)
                                 else: add_feedback("Too Fast!", False)
                            elif is_up_criteria and stage != "UP":
                                  if stage == "DOWN" or stage == "INIT": add_feedback("Pull", False)
                                  stage = "UP"; form_correct_overall = True # Reset form
                            if stage == "UP" and avg_elbow_angle > PULLUP_UP_EXIT_ELBOW_ANGLE: add_feedback("Higher", False)
                            if stage == "DOWN" and avg_elbow_angle < PULLUP_DOWN_EXIT_ANGLE: add_feedback("Hang", False)

                    # === DEADLIFT ===
                    elif current_exercise == "DEADLIFT":
                        deadlift_lockout_form_ok = True
                        deadlift_lift_form_ok = True
                        # Form Check: Back Rounding during lift
                        if stage == "DOWN" or (stage == "INIT" and avg_hip_angle < 150): # Check during lift/setup
                            if back_angle_vertical > BACK_ANGLE_THRESHOLD_DEADLIFT_LIFT:
                                 add_feedback(f"Back Round? ({back_angle_vertical:.0f}deg)", True); add_form_issue("BACK"); deadlift_lift_form_ok = False
                        # Form Check: Lockout Back Angle
                        is_potentially_up = avg_hip_angle > DEADLIFT_UP_EXIT_ANGLE and avg_knee_angle > DEADLIFT_UP_EXIT_ANGLE
                        if stage == "UP" or (stage == "DOWN" and is_potentially_up): # Check near/at lockout
                            if back_angle_vertical > BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT:
                                 deadlift_lockout_form_ok = False
                                 add_feedback(f"Lockout Back ({back_angle_vertical:.0f}deg)", True); add_form_issue("BACK")
                        # Rep Counting
                        if kn_l[3]>0.5 and kn_r[3]>0.5 and hip_l[3]>0.5 and hip_r[3]>0.5:
                            if stage is None: stage = "INIT"
                            is_up = avg_hip_angle > DEADLIFT_UP_ENTER_ANGLE and avg_knee_angle > DEADLIFT_UP_ENTER_ANGLE
                            is_down = avg_hip_angle < DEADLIFT_DOWN_ENTER_HIP_ANGLE and avg_knee_angle < DEADLIFT_DOWN_ENTER_KNEE_ANGLE
                            if is_up and stage == "DOWN":
                                stage = "UP"
                                if deadlift_lockout_form_ok and deadlift_lift_form_ok and (ct - last_rep_time > rep_cooldown):
                                    counter += 1; last_rep_time = ct; print(f"Deadlift Rep: {counter}")
                                    add_feedback("Rep!", False); rep_counted_this_frame = True
                                elif not deadlift_lockout_form_ok: add_feedback("Lockout Form!", True)
                                elif not deadlift_lift_form_ok: add_feedback("Lift Form!", True)
                                else: add_feedback("Too Fast!", False)
                            elif is_down and stage != "DOWN":
                                 if stage == "UP" or stage == "INIT": add_feedback("Lower", False)
                                 stage = "DOWN"; form_correct_overall = True
                            if stage == "UP" and not is_up: add_feedback("Lockout", False) # If exited UP state
                            if stage == "DOWN" and not is_down: add_feedback("Lift", False) # If exited DOWN state

            else: # No landmarks detected
                add_feedback("No Person Detected", True)
                stage=stage_left=stage_right=None

        except Exception as e:
            print(f"!! Error during pose logic: {e}")
            traceback.print_exc()
            add_feedback("Error processing frame.", True)
            stage=stage_left=stage_right=None

        # --- Final Feedback Aggregation ---
        if not feedback_list and stage is not None :
             if form_correct_overall: add_feedback("Analyzing..." if stage != "INIT" else "Ready", False)
             # else: warnings already added by form checks

        # --- Draw Pose Landmarks onto Resized Frame (rf) ---
        # Pass the results.pose_landmarks object directly if it exists
        draw_pose_landmarks(
            target_image=rf,
            landmarks_list=results.pose_landmarks if results else None,
            connections=mp_pose.POSE_CONNECTIONS,
            form_issue_details=form_issues_details
        )

        # --- Place Resized Frame with Landmarks onto Canvas ---
        if oy >= 0 and ox >= 0 and oy+sh <= canvas.shape[0] and ox+sw <= canvas.shape[1] and rf.shape[0] == sh and rf.shape[1] == sw :
            canvas[oy:oy+sh, ox:ox+sw] = rf
        else:
            # print("Warning: rf dimensions/offset issue during placement.") # Debug if needed
            pass # Avoid placing if dimensions mismatch

        # --- Draw UI Elements ---
        draw_buttons(canvas, actual_win_width, actual_win_height)
        draw_status_box(canvas, actual_win_width, actual_win_height)
        draw_feedback_box(canvas, actual_win_width, actual_win_height)

    # --- Display Final Canvas ---
    cv2.imshow(window_name, canvas)

    # --- Quit Key ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quit key pressed.")
        break

# --- Cleanup ---
print("Releasing resources...")
if cap:
    cap.release()
    print("Video capture released.")
if 'pose' in locals() and pose is not None: # Safely close pose model
    pose.close()
    print("Pose model closed.")
cv2.destroyAllWindows()
print("Application Closed.")