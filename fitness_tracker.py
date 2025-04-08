import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog
import platform # To help with fullscreen adaptation
import traceback # For detailed error printing

# --- Constants ---
# Angle Thresholds (Tune these!)

# Bicep Curl
BICEP_UP_ANGLE = 50
BICEP_DOWN_ANGLE = 160

# Squat
SQUAT_UP_ANGLE = 170
SQUAT_DOWN_ANGLE = 95

# Push Up
PUSHUP_UP_ANGLE = 160
PUSHUP_DOWN_ANGLE = 90
PUSHUP_BODY_STRAIGHT_MIN = 155 # Slightly more tolerant min body angle
PUSHUP_BODY_STRAIGHT_MAX = 195

# Pull Up
PULLUP_DOWN_ANGLE = 165
PULLUP_UP_ELBOW_ANGLE = 70
PULLUP_CHIN_OVER_WRIST = True

# Deadlift
DEADLIFT_UP_ANGLE_BODY = 170
DEADLIFT_DOWN_HIP_ANGLE = 115 # Allow a bit more hinge based on previous value
DEADLIFT_DOWN_KNEE_ANGLE = 130 # Knees don't always bend below 120

# Back Posture Checks (Deviation from Vertical)
BACK_ANGLE_THRESHOLD_BICEP = 15   # *** Made stricter for Bicep Curls ***
BACK_ANGLE_THRESHOLD_SQUAT = 40
BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT = 20 # *** Specific threshold for lockout ***
# BACK_ANGLE_THRESHOLD_GENERAL = 25 # Less relevant now, using Bicep specific

# --- Mediapipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Global Variables ---
app_mode = "HOME"
cap = None
video_source_selected = False
try:
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
counter, stage = 0, "START"
counter_left, counter_right = 0, 0
stage_left, stage_right = "START", "START"
feedback_msg, last_feedback, back_posture_feedback = "Select Video Source", "", ""
last_rep_time, last_rep_time_left, last_rep_time_right = time.time(), time.time(), time.time()
rep_cooldown = 0.4
form_correct = True
home_button_width, home_button_height = 300, 60

# --- Helper Functions --- (calculate_angle, get_coords, set_feedback - Unchanged)
def calculate_angle(a, b, c):
    a,b,c = np.array(a), np.array(b), np.array(c)
    if np.linalg.norm(a-b) == 0 or np.linalg.norm(c-b) == 0: return 0
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rad*180.0/np.pi)
    return int(360-ang if ang > 180.0 else ang)

def get_coords(landmarks, landmark_name):
    try: lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]; return [lm.x, lm.y, lm.z, lm.visibility]
    except: return [0,0,0,0]

def set_feedback(new_msg, is_warning=False):
    global feedback_msg, last_feedback
    prefix = "WARN: " if is_warning else "INFO: "
    full_msg = prefix + new_msg
    if new_msg != last_feedback: feedback_msg, last_feedback = full_msg, new_msg

# --- Mouse Callback --- (Unchanged - uses global timers correctly)
def mouse_callback(event, x, y, flags, param):
    global app_mode, current_exercise, counter, stage, feedback_msg, video_source_selected, cap
    global counter_left, counter_right, stage_left, stage_right
    global last_rep_time, last_rep_time_left, last_rep_time_right

    canvas_w = param.get('canvas_w', actual_win_width)
    canvas_h = param.get('canvas_h', actual_win_height)

    if app_mode == "HOME":
        if event == cv2.EVENT_LBUTTONDOWN:
            webcam_btn_x, webcam_btn_y = canvas_w//2-home_button_width//2, canvas_h//2-home_button_height-button_margin//2
            if webcam_btn_x <= x <= webcam_btn_x+home_button_width and webcam_btn_y <= y <= webcam_btn_y+home_button_height:
                print("Selecting Webcam..."); cap = cv2.VideoCapture(0)
                if not cap or not cap.isOpened(): cap = cv2.VideoCapture(1)
                if cap and cap.isOpened():
                    app_mode="TRACKING"; video_source_selected=True; set_feedback(f"Starting {current_exercise}")
                    counter=counter_left=counter_right=0; stage=stage_left=stage_right="START"
                    ct=time.time(); last_rep_time=ct; last_rep_time_left=ct; last_rep_time_right=ct
                else: set_feedback("Error: Webcam not found or busy.", True); cap=None
            video_btn_x, video_btn_y = canvas_w//2-home_button_width//2, canvas_h//2+button_margin//2
            if video_btn_x <= x <= video_btn_x+home_button_width and video_btn_y <= y <= video_btn_y+home_button_height:
                print("Selecting Video File..."); root = tk.Tk(); root.withdraw()
                vp = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
                root.destroy()
                if vp:
                    cap = cv2.VideoCapture(vp)
                    if cap and cap.isOpened():
                        app_mode="TRACKING"; video_source_selected=True; set_feedback(f"Starting {current_exercise}"); print(f"Loaded: {vp}")
                        counter=counter_left=counter_right=0; stage=stage_left=stage_right="START"
                        ct=time.time(); last_rep_time=ct; last_rep_time_left=ct; last_rep_time_right=ct
                    else: set_feedback(f"Error: Could not open video: {vp}", True); cap=None
                else: set_feedback("Video selection cancelled.")
    elif app_mode == "TRACKING":
        if event == cv2.EVENT_LBUTTONDOWN:
            try: btn_w = (canvas_w - (len(EXERCISES)+1)*button_margin)//len(EXERCISES)
            except: btn_w = 50
            for i, ex in enumerate(EXERCISES):
                btn_x = button_margin + i*(btn_w+button_margin)
                if btn_x <= x <= btn_x+btn_w and button_margin <= y <= button_margin+button_height:
                    if current_exercise != ex:
                        print(f"Switching to {ex}"); current_exercise = ex
                        counter=counter_left=counter_right=0; stage=stage_left=stage_right="START"; set_feedback(f"Start {ex}")
                        ct=time.time(); last_rep_time=ct; last_rep_time_left=ct; last_rep_time_right=ct
                        form_correct=True; back_posture_feedback=""
                    break
            home_btn_size=50; home_btn_x=canvas_w-home_btn_size-button_margin; home_btn_y=canvas_h-home_btn_size-button_margin
            if home_btn_x <= x <= home_btn_x+home_btn_size and home_btn_y <= y <= home_btn_y+home_btn_size:
                 print("Returning Home..."); app_mode="HOME"
                 if cap: cap.release(); cap=None; video_source_selected=False; set_feedback("Select Video Source")

# --- Main Application ---
window_name = 'Fitness Tracker Pro v5'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
if platform.system()=="Windows": cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else: cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL); cv2.resizeWindow(window_name, actual_win_width, actual_win_height)
callback_param = {'canvas_w': actual_win_width, 'canvas_h': actual_win_height}
cv2.setMouseCallback(window_name, mouse_callback, callback_param)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        try: # Window size update
            wp=cv2.getWindowImageRect(window_name)
            if wp[2]>1 and wp[3]>1 and (wp[2]!=actual_win_width or wp[3]!=actual_win_height):
                actual_win_width,actual_win_height = wp[2],wp[3]
                callback_param['canvas_w'], callback_param['canvas_h'] = actual_win_width, actual_win_height
        except: pass
        canvas = np.zeros((actual_win_height, actual_win_width, 3), dtype=np.uint8)

        if app_mode == "HOME": # Home Screen Drawing (Unchanged)
            title_text = "FITNESS TRACKER PRO v5"
            (tw,th),_ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            tx,ty=(actual_win_width-tw)//2, actual_win_height//4+th//2
            cv2.putText(canvas,title_text,(tx,ty), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),3,cv2.LINE_AA)
            # Webcam Btn
            wbx,wby=actual_win_width//2-home_button_width//2, actual_win_height//2-home_button_height-button_margin//2
            cv2.rectangle(canvas,(wbx,wby),(wbx+home_button_width,wby+home_button_height),(0,200,0),-1)
            cv2.rectangle(canvas,(wbx,wby),(wbx+home_button_width,wby+home_button_height),(255,255,255),2)
            (tw,th),_ = cv2.getTextSize("Use Webcam", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(canvas,"Use Webcam",(wbx+(home_button_width-tw)//2, wby+(home_button_height+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
            # Video Btn
            vbx,vby=actual_win_width//2-home_button_width//2, actual_win_height//2+button_margin//2
            cv2.rectangle(canvas,(vbx,vby),(vbx+home_button_width,vby+home_button_height),(200,0,0),-1)
            cv2.rectangle(canvas,(vbx,vby),(vbx+home_button_width,vby+home_button_height),(255,255,255),2)
            (tw,th),_ = cv2.getTextSize("Load Video File", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(canvas,"Load Video File",(vbx+(home_button_width-tw)//2, vby+(home_button_height+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
            # Feedback
            (tw,th),_ = cv2.getTextSize(feedback_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            fx,fy = (actual_win_width-tw)//2, actual_win_height*3//4+th//2
            fc = (0,0,255) if "Error" in feedback_msg else (0,255,255)
            cv2.putText(canvas,feedback_msg,(fx,fy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,fc,2,cv2.LINE_AA)
            # Quit Text
            qt="Press 'Q' to Quit"; (tw,th),_ = cv2.getTextSize(qt,cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(canvas,qt,(actual_win_width-tw-20, actual_win_height-th-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(150,150,150),1,cv2.LINE_AA)
            cv2.imshow(window_name, canvas)

        elif app_mode == "TRACKING": # Tracking Mode
            # --- Video Reading & Processing ---
            if not cap or not cap.isOpened():
                 set_feedback("Error: Video source lost.", True); app_mode="HOME"; cap=None; continue
            ret, frame = cap.read()
            if not ret: # Handle video end/error
                if cap.get(cv2.CAP_PROP_POS_FRAMES)>0 and cap.get(cv2.CAP_PROP_FRAME_COUNT)>0:
                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read()
                     if not ret: set_feedback("Error reading video.", True); app_mode="HOME"; cap=None; continue
                else: set_feedback("Cannot read frame.", True); app_mode="HOME"; cap=None; continue
            frame_h, frame_w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img_rgb.flags.writeable=False
            results = pose.process(img_rgb); img_rgb.flags.writeable=True

            # --- Letterboxing & Display ---
            scale = min(actual_win_width/frame_w, actual_win_height/frame_h)
            sw, sh = int(frame_w*scale), int(frame_h*scale)
            ox, oy = (actual_win_width-sw)//2, (actual_win_height-sh)//2
            interp = cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR
            rf = cv2.resize(frame,(sw,sh),interpolation=interp)
            canvas[oy:oy+sh, ox:ox+sw] = rf

            # --- Pose Logic ---
            form_correct = True # Assume good unless specific check fails
            current_feedback_parts = []

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # --- Back Posture Check (Vertical) ---
                    sh_l, sh_r = get_coords(landmarks,'LEFT_SHOULDER'), get_coords(landmarks,'RIGHT_SHOULDER')
                    hip_l, hip_r = get_coords(landmarks,'LEFT_HIP'), get_coords(landmarks,'RIGHT_HIP')
                    vis_sh = [s for s in [sh_l, sh_r] if s[3]>0.6]
                    vis_hip = [h for h in [hip_l, hip_r] if h[3]>0.6]
                    back_posture_feedback = ""
                    vertical_back_ok = True # Assume OK

                    if len(vis_sh)>0 and len(vis_hip)>0:
                        sh_ax, sh_ay = np.mean([s[0] for s in vis_sh]), np.mean([s[1] for s in vis_sh])
                        hip_ax, hip_ay = np.mean([h[0] for h in vis_hip]), np.mean([h[1] for h in vis_hip])
                        vsh_x, vsh_y = hip_ax-sh_ax, hip_ay-sh_ay; mag_sh = np.sqrt(vsh_x**2 + vsh_y**2)
                        if mag_sh > 0:
                            v_vert_y = 1; dot = vsh_y * v_vert_y # Simplified dot product with vertical
                            cos_theta = np.clip(dot/mag_sh, -1.0, 1.0)
                            back_angle_vertical = np.degrees(np.arccos(cos_theta))

                            # --- Determine Exercise-Specific Vertical Threshold ---
                            if current_exercise == "BICEP CURL":
                                current_vert_thresh = BACK_ANGLE_THRESHOLD_BICEP
                            elif current_exercise == "SQUAT":
                                current_vert_thresh = BACK_ANGLE_THRESHOLD_SQUAT
                            elif current_exercise == "DEADLIFT":
                                # Check only applied at lockout later
                                current_vert_thresh = BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT
                            elif current_exercise == "PUSH UP":
                                current_vert_thresh = 90 # Effectively disabled for pushup counting
                            else: # General case (Pullups etc.)
                                current_vert_thresh = 90 # Default to lenient if not specified

                            # Set initial vertical_back_ok flag (can be overridden by DL/PU logic)
                            if back_angle_vertical > current_vert_thresh:
                                vertical_back_ok = False
                                back_posture_feedback = f"Fix Back Angle! ({back_angle_vertical:.0f}° > {current_vert_thresh}°)"
                            else:
                                back_posture_feedback = "Back Angle OK."
                        else: back_posture_feedback = "Calc Back..."
                    else: back_posture_feedback = "Back Vis Low"

                    # --- Get Other Landmarks ---
                    el_l, wr_l = get_coords(landmarks,'LEFT_ELBOW'), get_coords(landmarks,'LEFT_WRIST')
                    el_r, wr_r = get_coords(landmarks,'RIGHT_ELBOW'), get_coords(landmarks,'RIGHT_WRIST')
                    kn_l, an_l = get_coords(landmarks,'LEFT_KNEE'), get_coords(landmarks,'LEFT_ANKLE')
                    kn_r, an_r = get_coords(landmarks,'RIGHT_KNEE'), get_coords(landmarks,'RIGHT_ANKLE')
                    nose = get_coords(landmarks, 'NOSE')

                    # --- Coordinate Scaling Func ---
                    def scale_coords(coords_norm):
                        if coords_norm is None or len(coords_norm)<2: return None
                        return (int(coords_norm[0]*sw+ox), int(coords_norm[1]*sh+oy))

                    # --- EXERCISE SPECIFIC LOGIC ---
                    form_correct = vertical_back_ok # Start with vertical check

                    # === BICEP CURL ===
                    if current_exercise == "BICEP CURL":
                        ct = time.time()
                        # Left Arm
                        if all(c[3]>0.5 for c in [sh_l, el_l, wr_l]):
                            a_l = calculate_angle(sh_l[:2], el_l[:2], wr_l[:2])
                            ed_l = scale_coords(el_l)
                            if ed_l:
                                cv2.putText(canvas, f"L:{a_l}", (ed_l[0]-40, ed_l[1]), 0, 0.6, (255, 255, 0), 2)
                            if form_correct: # Requires strict vertical back for biceps
                                if a_l > BICEP_DOWN_ANGLE:
                                    if stage_left=='UP': stage_left="DOWN"; current_feedback_parts.append("L:Extend OK")
                                    elif stage_left!='DOWN': stage_left="DOWN"
                                if a_l < BICEP_UP_ANGLE:
                                    if stage_left=='DOWN':
                                        stage_left="UP"
                                        if ct-last_rep_time_left > rep_cooldown:
                                            counter_left+=1; last_rep_time_left=ct; print(f"L Curl: {counter_left}"); current_feedback_parts.append("L:Rep!")
                                        else: current_feedback_parts.append("L:Cool")
                            elif stage_left != "START": stage_left="HOLD"; current_feedback_parts.append("L:Hold(Form)")
                        else: stage_left="START"; current_feedback_parts.append("L Arm?")
                        # Right Arm (Mirror)
                        if all(c[3]>0.5 for c in [sh_r, el_r, wr_r]):
                            a_r = calculate_angle(sh_r[:2], el_r[:2], wr_r[:2])
                            ed_r = scale_coords(el_r)
                            if ed_r:
                                cv2.putText(canvas, f"R:{a_r}", (ed_r[0]+10, ed_r[1]), 0, 0.6, (255, 255, 0), 2)
                            if form_correct:
                                if a_r > BICEP_DOWN_ANGLE:
                                    if stage_right=='UP': stage_right="DOWN"; current_feedback_parts.append("R:Extend OK")
                                    elif stage_right!='DOWN': stage_right="DOWN"
                                if a_r < BICEP_UP_ANGLE:
                                    if stage_right=='DOWN':
                                        stage_right="UP"
                                        if ct-last_rep_time_right > rep_cooldown:
                                            counter_right+=1; last_rep_time_right=ct; print(f"R Curl: {counter_right}"); current_feedback_parts.append("R:Rep!")
                                        else: current_feedback_parts.append("R:Cool")
                            elif stage_right != "START": stage_right="HOLD"; current_feedback_parts.append("R:Hold(Form)")
                        else: stage_right="START"; current_feedback_parts.append("R Arm?")

                    # === SQUAT === (Seems OK - Keep as is)
                    elif current_exercise == "SQUAT":
                        if all(c[3] > 0.5 for c in [hip_l, kn_l, an_l]):
                            ka = calculate_angle(hip_l[:2], kn_l[:2], an_l[:2])
                            kd = scale_coords(kn_l)
                            if kd:
                                cv2.putText(canvas, f"K:{ka}", (kd[0]-50, kd[1]-10), 0, 0.6, (255, 255, 0), 2)
                            if form_correct: # Uses SQUAT threshold for vertical_back_ok
                                if ka > SQUAT_UP_ANGLE:
                                    if stage=='DOWN':
                                         stage="UP"; ct=time.time()
                                         if ct-last_rep_time > rep_cooldown: counter+=1; last_rep_time=ct; print(f"Squat: {counter}"); current_feedback_parts.append("Rep!")
                                         else: current_feedback_parts.append("Cooldown")
                                    elif stage!='UP': stage='UP'
                                if ka < SQUAT_DOWN_ANGLE:
                                    if stage=='UP': stage="DOWN"; current_feedback_parts.append("Depth OK - Stand")
                                    elif stage!='DOWN': stage='DOWN'
                                if stage=='UP' and ka<SQUAT_UP_ANGLE*0.95: current_feedback_parts.append("Lower Deeper")
                                if stage=='DOWN' and ka>SQUAT_DOWN_ANGLE*1.05: current_feedback_parts.append("Stand Fully")
                            elif stage!="START": stage="HOLD"; current_feedback_parts.append("Hold(Form)")
                        else: stage="START"; current_feedback_parts.append("Leg?")

                    # === PUSH UP === (Revised logic and debugging)
                    elif current_exercise == "PUSH UP":
                        body_form_ok = True # Body line check flag
                        if all(c[3] > 0.5 for c in [sh_l, el_l, wr_l, hip_l, kn_l]):
                            ea = calculate_angle(sh_l[:2], el_l[:2], wr_l[:2])
                            ba = calculate_angle(sh_l[:2], hip_l[:2], kn_l[:2])
                            ed = scale_coords(el_l); hd = scale_coords(hip_l)
                            if ed: cv2.putText(canvas,f"E:{ea}",(ed[0]-40,ed[1]),0,0.6,(255,255,0),2)
                            if hd: cv2.putText(canvas,f"Body:{ba}",(hd[0]-60,hd[1]-10),0,0.6,(0,255,255),2)

                            # Pushup Form Check: Body Straightness
                            if not (PUSHUP_BODY_STRAIGHT_MIN < ba < PUSHUP_BODY_STRAIGHT_MAX):
                                body_form_ok = False
                                current_feedback_parts.append("Keep Body Straight!")
                            else: current_feedback_parts.append("Body OK.")

                            form_correct = body_form_ok # Prioritize body line for pushups

                            # Debug Print for Pushups
                            # print(f"Pushup - Elbow: {ea}, Body: {ba}, Stage: {stage}, BodyFormOK: {body_form_ok}")

                            if form_correct:
                                # Detect UP state (Arms Extended)
                                if ea > PUSHUP_UP_ANGLE:
                                    if stage == 'DOWN': # Just finished pushing up
                                        stage = "UP"
                                        ct = time.time()
                                        if ct - last_rep_time > rep_cooldown:
                                            counter += 1
                                            last_rep_time = ct
                                            print(f"Push Up Rep: {counter}") # *** Make sure this prints ***
                                            current_feedback_parts.append("Rep!")
                                        else: current_feedback_parts.append("Cooldown")
                                    elif stage != 'UP': # If not already UP, set it
                                        stage = 'UP'
                                        # current_feedback_parts.append("Ready/Up") # Can be noisy

                                # Detect DOWN state (Arms Bent)
                                if ea < PUSHUP_DOWN_ANGLE:
                                    if stage == 'UP': # Starting to go down
                                        stage = "DOWN"
                                        current_feedback_parts.append("Push Up!")
                                    elif stage != 'DOWN': # Ensure stage is DOWN if low
                                        stage = 'DOWN'

                                # Intermediate feedback (Optional)
                                # if stage=='UP' and ea<PUSHUP_UP_ANGLE*0.95: current_feedback_parts.append("Lower Chest")
                                # if stage=='DOWN' and ea>PUSHUP_DOWN_ANGLE*1.05: current_feedback_parts.append("Extend Fully")

                            elif stage != "START": # If body form is bad
                                stage = "HOLD"; current_feedback_parts.append("Hold(Body Form)")
                        else:
                            stage = "START"; current_feedback_parts.append("Body?")

                    # === PULL UP === (Seems OK - Keep as is)
                    elif current_exercise == "PULL UP":
                         if all(c[3] > 0.5 for c in [sh_l, el_l, wr_l, nose]):
                             ea = calculate_angle(sh_l[:2], el_l[:2], wr_l[:2])
                             ny, wy = nose[1], wr_l[1]
                             ed=scale_coords(el_l); nd=scale_coords(nose); wd=scale_coords(wr_l)
                             if ed: cv2.putText(canvas,f"E:{ea}",(ed[0]-40,ed[1]),0,0.6,(255,255,0),2)
                             if nd: cv2.line(canvas,(ox,nd[1]),(ox+sw,nd[1]),(0,255,0),1)
                             if wd: cv2.line(canvas,(ox,wd[1]),(ox+sw,wd[1]),(0,0,255),1)
                             is_up_pos = ny < wy and ea < PULLUP_UP_ELBOW_ANGLE
                             is_down_pos = ea > PULLUP_DOWN_ANGLE
                             if is_down_pos:
                                 if stage=='UP':
                                     stage="DOWN"; ct=time.time()
                                     if ct-last_rep_time > rep_cooldown: counter+=1; last_rep_time=ct; print(f"Pull Up: {counter}"); current_feedback_parts.append("Rep!")
                                     else: current_feedback_parts.append("Cooldown")
                                 elif stage!='DOWN': stage='DOWN' # ; current_feedback_parts.append("Hang/Ready")
                             if is_up_pos:
                                 if stage=='DOWN': stage="UP"; current_feedback_parts.append("Lower Fully")
                                 elif stage!='UP': stage='UP'
                             #if stage=='DOWN' and not is_down_pos and ea < PULLUP_DOWN_ANGLE*0.98: current_feedback_parts.append("Extend More")
                             #if stage=='UP' and not is_up_pos: current_feedback_parts.append("Pull Higher / Lower")
                         else: stage="START"; current_feedback_parts.append("Body/Head?")

                    # === DEADLIFT === (Revised back check and debugging)
                    elif current_exercise == "DEADLIFT":
                        form_correct = True # Reset specific DL form check
                        deadlift_lockout_form_ok = True # Flag for checking form *at lockout*

                        full_body_visible = all(c[3] > 0.5 for c in [
                            sh_l, hip_l, kn_l, an_l, sh_r, hip_r, kn_r, an_r])

                        if full_body_visible:
                            hip_l_2d, hip_r_2d = hip_l[:2], hip_r[:2]; kn_l_2d, kn_r_2d = kn_l[:2], kn_r[:2]
                            an_l_2d, an_r_2d = an_l[:2], an_r[:2]; sh_l_2d, sh_r_2d = sh_l[:2], sh_r[:2]
                            hip_avg = np.mean([hip_l_2d, hip_r_2d], axis=0); kn_avg = np.mean([kn_l_2d, kn_r_2d], axis=0)
                            an_avg = np.mean([an_l_2d, an_r_2d], axis=0); sh_avg = np.mean([sh_l_2d, sh_r_2d], axis=0)
                            hip_a = calculate_angle(sh_avg, hip_avg, kn_avg)
                            knee_a = calculate_angle(hip_avg, kn_avg, an_avg)

                            hd=scale_coords(hip_avg.tolist()); kd=scale_coords(kn_avg.tolist())
                            if hd: cv2.putText(canvas,f"H:{hip_a}",(hd[0]-60,hd[1]),0,0.6,(255,255,0),2)
                            if kd: cv2.putText(canvas,f"K:{knee_a}",(kd[0]-50,kd[1]+20),0,0.6,(0,255,255),2)

                            # Define states
                            is_up = hip_a > DEADLIFT_UP_ANGLE_BODY and knee_a > DEADLIFT_UP_ANGLE_BODY
                            is_down = hip_a < DEADLIFT_DOWN_HIP_ANGLE and knee_a < DEADLIFT_DOWN_KNEE_ANGLE

                            # *** Deadlift Specific Back Check ***
                            # Only apply the strict vertical check if the lifter *is* in the UP position
                            if is_up and not vertical_back_ok: # vertical_back_ok uses DEADLIFT_LOCKOUT threshold
                                deadlift_lockout_form_ok = False
                                form_correct = False # Overall form is bad if lockout posture fails
                                current_feedback_parts.append(f"Bad Lockout Angle! ({back_angle_vertical:.0f}°) ")
                            elif not is_up:
                                # If not in UP position, ignore the vertical back check for form correctness
                                # We still might want feedback if the general back angle seems excessive, but don't fail the rep count based on it
                                form_correct = True # Allow rep counting based on angles unless lockout fails
                                if back_posture_feedback == "Back Angle OK.": # Avoid showing OK during lift
                                      back_posture_feedback = "" # Clear neutral feedback during motion


                            # Debug Print for Deadlifts
                            # print(f"DL - H:{hip_a} K:{knee_a} Stage:{stage} isUP:{is_up} isDN:{is_down} LockoutOK:{deadlift_lockout_form_ok} VertOK:{vertical_back_ok} ({back_angle_vertical:.1f})")


                            # State Machine & Counting
                            if is_up:
                                if stage == 'DOWN': # Just completed the lift
                                     if deadlift_lockout_form_ok: # Count only if lockout form was good
                                        stage = "UP"
                                        ct=time.time()
                                        if ct-last_rep_time > rep_cooldown:
                                            counter+=1; last_rep_time=ct; print(f"Deadlift Rep: {counter}"); current_feedback_parts.append("Rep!")
                                        else: current_feedback_parts.append("Cooldown")
                                     else: # Reached UP position but form was bad
                                         stage = "HOLD" # Hold stage until form corrects at top
                                         current_feedback_parts.append("Fix Lockout Form!")
                                elif stage != 'UP' and stage != 'HOLD': # Ensure stage is UP if locked out and form is good
                                     if deadlift_lockout_form_ok:
                                          stage = 'UP'
                                          current_feedback_parts.append("Lockout OK")

                            if is_down:
                                if stage == 'UP' or stage == 'HOLD': # Starting to lower (from good OR bad lockout)
                                    stage = "DOWN"
                                    current_feedback_parts.append("Lift: Stand Tall!")
                                elif stage != 'DOWN': # Ensure stage is DOWN if low
                                    stage = 'DOWN'

                            # Handle HOLD state - require getting back to good UP state
                            if stage == 'HOLD' and is_up and deadlift_lockout_form_ok:
                                stage = 'UP' # Recovered from bad lockout form
                                current_feedback_parts.append("Lockout OK")


                        else: stage="START"; current_feedback_parts.append("Full Body?")


                    # --- Update Combined Feedback ---
                    final_feedback = ""
                    if current_feedback_parts:
                         # Prioritize form warnings if any exist
                         form_warnings = [f for f in current_feedback_parts if "Form" in f or "Angle" in f or "Straight" in f or "Fix" in f]
                         if form_warnings:
                             final_feedback = " | ".join(list(dict.fromkeys(form_warnings)))
                             set_feedback(final_feedback, is_warning=True)
                         else:
                             # Show unique non-warning feedback parts
                             final_feedback = " | ".join(list(dict.fromkeys(current_feedback_parts)))
                             set_feedback(final_feedback)
                    elif back_posture_feedback and "OK" not in back_posture_feedback:
                         # Show back posture warning if no other feedback and back is bad
                         set_feedback(back_posture_feedback, is_warning=True)
                    elif back_posture_feedback and current_exercise != "DEADLIFT": # Don't show "Back OK" during DL motion
                         set_feedback(back_posture_feedback)
                    else:
                         set_feedback("Analyzing...")


                    # --- Draw Landmarks --- (Unchanged)
                    slm = [scale_coords([lm.x,lm.y]) for lm in landmarks] # Scaled Landmarks
                    con = mp_pose.POSE_CONNECTIONS
                    if con:
                        for c in con:
                            s,e=c[0],c[1]
                            if s<len(slm) and e<len(slm):
                                sp,ep=slm[s],slm[e]
                                if sp and ep:
                                    vs,ve=landmarks[s].visibility, landmarks[e].visibility
                                    if vs>0.4 and ve>0.4: cv2.line(canvas,sp,ep,(255,100,255),2)
                        for i,p in enumerate(slm):
                             if p:
                                v=landmarks[i].visibility
                                if v>0.5: cv2.circle(canvas,p,4,(80,200,80),-1)
                                elif v>0.2: cv2.circle(canvas,p,3,(80,100,80),-1)
                else:
                    set_feedback("No Person Detected", True)
                    stage=stage_left=stage_right="START"

            except Exception as e:
                print(f"!! Error during pose logic/drawing: {e}"); traceback.print_exc()
                set_feedback("Error processing frame.", True); stage=stage_left=stage_right="START"


            # --- Draw UI Elements --- (Unchanged - using latest version)
            try: btn_w = max(10, (actual_win_width - (len(EXERCISES)+1)*button_margin)//len(EXERCISES))
            except: btn_w = 50
            for i, ex in enumerate(EXERCISES): # Buttons
                bx=button_margin+i*(btn_w+button_margin); bxe=min(bx+btn_w, actual_win_width-button_margin)
                clr = (0,220,0) if ex==current_exercise else (100,100,100)
                cv2.rectangle(canvas,(bx,button_margin),(bxe,button_margin+button_height),clr,-1)
                cv2.rectangle(canvas,(bx,button_margin),(bxe,button_margin+button_height),(255,255,255),1)
                fs=0.6; (tw,th),_=cv2.getTextSize(ex,0,fs,2); cbw=bxe-bx
                tx=bx+max(0,(cbw-tw)//2); ty=button_margin+(button_height+th)//2
                cv2.putText(canvas,ex,(tx,ty),0,fs,(255,255,255),2,cv2.LINE_AA)
            # Status Box
            sby=button_margin*2+button_height; isb=current_exercise=="BICEP CURL"
            sbw=420 if isb else 350; sbh=150 if isb else 120
            sbxe=min(button_margin+sbw, actual_win_width-button_margin); sbye=min(sby+sbh, actual_win_height-button_margin)
            cv2.rectangle(canvas,(button_margin,sby),(sbxe,sbye),(50,50,50),-1)
            cv2.rectangle(canvas,(button_margin,sby),(sbxe,sbye),(200,200,200),1)
            cv2.putText(canvas,'EXERCISE:',(button_margin+15,sby+25),0,0.6,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,current_exercise,(button_margin+120,sby+25),0,0.6,(0,255,0),2,cv2.LINE_AA)
            if isb:
                cv2.putText(canvas,'L REPS:',(button_margin+15,sby+60),0,0.7,(255,255,255),1); cv2.putText(canvas,str(counter_left),(button_margin+110,sby+65),0,1.2,(255,255,255),2)
                cv2.putText(canvas,'L STAGE:',(button_margin+15,sby+105),0,0.5,(255,255,255),1); cv2.putText(canvas,stage_left,(button_margin+90,sby+105),0,0.6,(255,255,255),2)
                cv2.putText(canvas,'R REPS:',(button_margin+215,sby+60),0,0.7,(255,255,255),1); cv2.putText(canvas,str(counter_right),(button_margin+310,sby+65),0,1.2,(255,255,255),2)
                cv2.putText(canvas,'R STAGE:',(button_margin+215,sby+105),0,0.5,(255,255,255),1); cv2.putText(canvas,stage_right,(button_margin+290,sby+105),0,0.6,(255,255,255),2)
            else:
                cv2.putText(canvas,'REPS:',(button_margin+15,sby+60),0,1.2,(255,255,255),2); cv2.putText(canvas,str(counter),(button_margin+130,sby+65),0,1.5,(255,255,255),3)
                cv2.putText(canvas,'STAGE:',(button_margin+15,sby+100),0,0.6,(255,255,255),1); cv2.putText(canvas,stage,(button_margin+90,sby+100),0,0.7,(255,255,255),2)
            # Feedback Box
            hbs=50; fbh=60; fby=actual_win_height-fbh-button_margin; fbw=actual_win_width-2*button_margin-hbs-button_margin
            fbxe=min(button_margin+fbw, actual_win_width-button_margin); fbye=min(fby+fbh, actual_win_height-button_margin)
            cv2.rectangle(canvas,(button_margin,fby),(fbxe,fbye),(50,50,50),-1)
            cv2.rectangle(canvas,(button_margin,fby),(fbxe,fbye),(200,200,200),1)
            fc = (0,0,255) if "WARN:" in feedback_msg else (0,255,255)
            mfl=70; dfb=feedback_msg;
            if len(dfb)>mfl: dfb=dfb[:mfl-3]+"..."
            (tw,th),_=cv2.getTextSize(dfb,0,0.7,2); fty=fby+(fbh+th)//2
            cv2.putText(canvas,dfb,(button_margin+15,fty),0,0.7,fc,2,cv2.LINE_AA)
            # Home Button
            hbx=actual_win_width-hbs-button_margin; hby=actual_win_height-hbs-button_margin
            cv2.rectangle(canvas,(hbx,hby),(hbx+hbs,hby+hbs),(0,0,200),-1); cv2.rectangle(canvas,(hbx,hby),(hbx+hbs,hby+hbs),(255,255,255),2)
            (tw,th),_=cv2.getTextSize("H",0,1,2); cv2.putText(canvas,"H",(hbx+(hbs-tw)//2,hby+(hbs+th)//2),0,1,(255,255,255),2,cv2.LINE_AA)

            # --- Display ---
            cv2.imshow(window_name, canvas)

        # --- Quit Key ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Quit key pressed."); break

# --- Cleanup ---
print("Releasing resources...")
if cap: cap.release(); print("Video capture released.")
cv2.destroyAllWindows()
print("Application Closed.")