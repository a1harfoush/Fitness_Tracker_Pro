# Fitness Tracker Pro - Pose Estimation Details

This document details how the Fitness Tracker Pro application utilizes MediaPipe Pose estimation to count exercise repetitions and provide real-time form correction feedback.

## 1. Core Pose Estimation Engine: MediaPipe Pose

The foundation of the tracker is Google's [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) solution.

*   **Function:** Takes an image (video frame) as input and predicts the location of 33 key body landmarks (joints and key points) in 3D space (x, y, z coordinates) along with a visibility score for each landmark.
*   **Usage:** In this application, `mediapipe.solutions.pose.Pose` is initialized (`min_detection_confidence=0.55`, `min_tracking_confidence=0.55`, `model_complexity=1`). It processes each frame (`pose.process(image)`) to get the landmark data.
*   **Key Landmarks Used:** While MediaPipe provides 33 landmarks, this application primarily relies on:
    *   `LEFT_SHOULDER`, `RIGHT_SHOULDER`
    *   `LEFT_ELBOW`, `RIGHT_ELBOW`
    *   `LEFT_WRIST`, `RIGHT_WRIST`
    *   `LEFT_HIP`, `RIGHT_HIP`
    *   `LEFT_KNEE`, `RIGHT_KNEE`
    *   `LEFT_ANKLE`, `RIGHT_ANKLE`
    *   `NOSE` (especially for Pull Ups)

## 2. Fundamental Calculations

Several core calculations are performed using the extracted landmark coordinates:

*   **Coordinate Extraction (`get_coords`):** Retrieves the `x`, `y`, `z`, and `visibility` for a specific landmark by its name (e.g., `LEFT_ELBOW`). Handles cases where landmarks might not be detected.
*   **Angle Calculation (`calculate_angle`):**
    *   Calculates the angle (in degrees) formed by three landmarks (e.g., Shoulder-Elbow-Wrist for elbow angle).
    *   Uses vector geometry: Finds vectors `BA` and `BC`, calculates the dot product, and uses the arccosine formula (`arccos(dot(BA, BC) / (norm(BA) * norm(BC)))`).
    *   Operates primarily in 2D (`x`, `y`) for robustness against minor depth variations unless `use_3d=True`.
    *   Includes a check for landmark visibility (`visibility > 0.1`) to avoid calculations with unreliable points.
*   **Segment Vertical Angle (`get_segment_vertical_angle`):**
    *   Calculates the angle between a body segment (defined by two landmarks, e.g., Shoulder-Elbow for upper arm) and the vertical axis (downwards).
    *   Used primarily for form checks like ensuring the upper arm stays relatively still during bicep curls or checking back straightness relative to vertical.
    *   Uses vector dot product between the segment vector and a vertical vector `[0, 1]`.
*   **Exponential Moving Average (EMA) Smoothing (`update_ema`):**
    *   Applies EMA with a factor `EMA_ALPHA = 0.3` to the calculated joint angles.
    *   **Purpose:** Reduces jitter and noise inherent in pose estimation, providing a smoother, more stable angle value for state transitions and rep counting. A higher alpha reacts faster, a lower alpha smooths more.

## 3. General Repetition Counting Strategy

A state machine approach combined with angle thresholds (using hysteresis) is used:

*   **States:** Each tracked joint (or the primary movement focus) typically has states like:
    *   `INIT`: The initial state before a full cycle is detected.
    *   `DOWN`: The "bottom" or starting phase of the movement (e.g., arm extended in bicep curl, squatted down).
    *   `UP`: The "top" or contracted phase of the movement (e.g., arm curled up, standing up from squat).
*   **Hysteresis Thresholds:** To prevent counting jitter or small movements as reps, two thresholds are used for each state transition:
    *   **Enter Threshold:** The angle must cross this value to *enter* a new state (e.g., angle must go *below* `BICEP_UP_ENTER_ANGLE` to enter the `UP` state).
    *   **Exit Threshold:** The angle must cross this (less strict) value in the *opposite* direction *after* entering a state before it can transition back (e.g., angle must go *above* `BICEP_UP_EXIT_ANGLE` after being `UP` before it can potentially go `DOWN` again). This confirms the movement went far enough.
*   **Counting Logic:** A rep is typically counted when transitioning from the `DOWN` state to the `UP` state (or vice-versa depending on the exercise logic, e.g., Pull Ups count on the way *down* physically, which corresponds to the elbow angle *increasing* past the 'down' threshold).
*   **Cooldown (`rep_cooldown = 0.5` seconds):** A minimum time must pass between counted reps to prevent overly fast or accidental counts.
*   **Form Check Integration:** A rep is only counted if `form_correct_overall` is `True` at the moment the state transition occurs.

## 4. General Form Correction Strategy

Form correction relies on checking specific angles or landmark positions against predefined thresholds:

*   **Thresholds:** Constants define acceptable ranges or limits (e.g., `BACK_ANGLE_THRESHOLD_SQUAT`, `PUSHUP_BODY_STRAIGHT_MIN`).
*   **Detection:** During each frame, relevant angles/positions are calculated (e.g., average back angle relative to vertical, knee X-position relative to ankle X-position).
*   **Comparison:** These calculated values are compared against the corresponding thresholds.
*   **Feedback:**
    *   If a threshold is breached, `form_correct_overall` is set to `False`.
    *   A specific warning message is added to `feedback_list` (e.g., "WARN: Back Angle (...)").
    *   The body part(s) associated with the issue are added to `form_issues_details` (e.g., `"BACK"`, `"LEFT_KNEE"`).
    *   The `draw_pose_landmarks_on_frame` function uses `form_issues_details` to highlight the problematic joints/connections in a distinct color (Red).

## 5. Exercise-Specific Details

Here's how counting and form correction are applied to each exercise:

---

### A. Bicep Curl

*   **Primary Angle(s) for Counting:** Left Elbow Angle (`angle_l_elbow`), Right Elbow Angle (`angle_r_elbow`). Counted independently for each arm using `ema_angles["LEFT_ELBOW"]` and `ema_angles["RIGHT_ELBOW"]`.
*   **Rep Counting Logic (Per Arm):**
    *   **State `DOWN`:** Entered when Elbow Angle > `BICEP_DOWN_ENTER_ANGLE` (155°). Must decrease below `BICEP_DOWN_EXIT_ANGLE` (140°) to be ready for UP transition. Feedback: "Curl".
    *   **State `UP`:** Entered when Elbow Angle < `BICEP_UP_ENTER_ANGLE` (55°) *from* the `DOWN` state. A rep is counted here if form is good and cooldown met. Must increase above `BICEP_UP_EXIT_ANGLE` (70°) to be ready for DOWN transition. Feedback: "Lower".
*   **Form Correction Checks:**
    *   **Back Angle:** Average back angle relative to vertical (calculated using shoulders/hips) must be less than `BACK_ANGLE_THRESHOLD_BICEP` (20°). Issue: `"BACK"`. Feedback: "Back Angle (...)".
    *   **Upper Arm Movement:** Vertical angle of the Left Upper Arm (Shoulder-Elbow) and Right Upper Arm must deviate less than `BICEP_UPPER_ARM_VERT_DEVIATION` (25°) from vertical. Issue: `"LEFT_UPPER_ARM"`, `"RIGHT_UPPER_ARM"`. Feedback: "L/R Arm Still".

---

### B. Squat

*   **Primary Angle(s) for Counting:** Average Knee Angle (average of `ema_angles["LEFT_KNEE"]` and `ema_angles["RIGHT_KNEE"]` if both visible, otherwise uses the visible one).
*   **Rep Counting Logic:**
    *   **State `DOWN`:** Entered when Average Knee Angle < `SQUAT_DOWN_ENTER_ANGLE` (100°). Must increase above `SQUAT_DOWN_EXIT_ANGLE` (110°) to be ready for UP transition. Feedback: "Deeper".
    *   **State `UP`:** Entered when Average Knee Angle > `SQUAT_UP_ENTER_ANGLE` (165°) *from* the `DOWN` state. A rep is counted here. Must decrease below `SQUAT_UP_EXIT_ANGLE` (155°) to be ready for DOWN transition. Feedback: "Stand".
*   **Form Correction Checks:**
    *   **Back Angle:** Average back angle relative to vertical must be less than `BACK_ANGLE_THRESHOLD_SQUAT` (45°). Issue: `"BACK"`. Feedback: "Back Angle (...)".
    *   **Knee Valgus (Knees Caving In):**
        *   Left Knee X-coord should not be significantly less than Left Ankle X-coord (`< ankle.x - SQUAT_KNEE_VALGUS_THRESHOLD` (0.05 relative units)). Issue: `"LEFT_KNEE"`. Feedback: "L Knee In?".
        *   Right Knee X-coord should not be significantly more than Right Ankle X-coord (`> ankle.x + SQUAT_KNEE_VALGUS_THRESHOLD` (0.05 relative units)). Issue: `"RIGHT_KNEE"`. Feedback: "R Knee Out?". *(Note: The feedback "R Knee Out?" might be slightly counter-intuitive for valgus, but checks the relative position)*.
    *   **Chest Forward Lean (During Down Phase):** When `stage == "DOWN"`, Left Shoulder X-coord should not be significantly less than Left Knee X-coord (`< knee.x - SQUAT_CHEST_FORWARD_THRESHOLD` (0.1 relative units)). Issue: `"BACK"`. Feedback: "Chest Up".

---

### C. Push Up

*   **Primary Angle(s) for Counting:** Average Elbow Angle (average of `ema_angles["LEFT_ELBOW"]` and `ema_angles["RIGHT_ELBOW"]`).
*   **Rep Counting Logic:**
    *   **State `DOWN`:** Entered when Average Elbow Angle < `PUSHUP_DOWN_ENTER_ANGLE` (95°). Must increase above `PUSHUP_DOWN_EXIT_ANGLE` (105°) to be ready for UP transition. Feedback: "Lower".
    *   **State `UP`:** Entered when Average Elbow Angle > `PUSHUP_UP_ENTER_ANGLE` (155°) *from* the `DOWN` state. A rep is counted here. Must decrease below `PUSHUP_UP_EXIT_ANGLE` (145°) to be ready for DOWN transition. Feedback: "Extend".
*   **Form Correction Checks:**
    *   **Body Straightness:** Average Body Angle (Shoulder-Hip-Knee angle, averaged for left/right) must be within `PUSHUP_BODY_STRAIGHT_MIN` (150°) and `PUSHUP_BODY_STRAIGHT_MAX` (190°). Issue: `"BODY"`. Feedback: "Body (...)".

---

### D. Pull Up

*   **Primary Angle(s) for Counting:** Average Elbow Angle, combined with relative Nose/Wrist position.
*   **Rep Counting Logic:** *(Note: The physical "up" motion corresponds to the elbow angle *decreasing*, so the state names reflect the angle range)*
    *   **State `UP` (Arms Extended / Bottom):** Entered when Average Elbow Angle > `PULLUP_DOWN_ENTER_ANGLE` (160°) *from* the `DOWN` state. A rep is counted here (completion of the downward phase). Must decrease below `PULLUP_DOWN_EXIT_ANGLE` (150°) to be ready for DOWN transition. Feedback: "Hang".
    *   **State `DOWN` (Arms Flexed / Top):** Entered when Average Elbow Angle < `PULLUP_UP_ENTER_ELBOW_ANGLE` (80°) **AND** Nose Y-coord < Average Wrist Y-coord (if `PULLUP_CHIN_ABOVE_WRIST` is True). Must increase above `PULLUP_UP_EXIT_ELBOW_ANGLE` (95°) to be ready for UP transition. Feedback: "Higher".
*   **Form Correction Checks:** Primarily implicit in reaching the required angle and chin/wrist position thresholds. No separate explicit form checks are defined beyond the rep counting criteria.

---

### E. Deadlift

*   **Primary Angle(s) for Counting:** Average Hip Angle AND Average Knee Angle (Both must meet criteria).
*   **Rep Counting Logic:**
    *   **State `DOWN`:** Entered when Average Hip Angle < `DEADLIFT_DOWN_ENTER_HIP_ANGLE` (120°) **AND** Average Knee Angle < `DEADLIFT_DOWN_ENTER_KNEE_ANGLE` (135°). Must increase above `DEADLIFT_DOWN_EXIT_HIP_ANGLE` (130°) AND `DEADLIFT_DOWN_EXIT_KNEE_ANGLE` (145°) to be ready for UP transition. Feedback: "Lower".
    *   **State `UP` (Lockout):** Entered when Average Hip Angle > `DEADLIFT_UP_ENTER_ANGLE` (168°) **AND** Average Knee Angle > `DEADLIFT_UP_ENTER_ANGLE` (168°) *from* the `DOWN` state. A rep is counted here. Must decrease below `DEADLIFT_UP_EXIT_ANGLE` (158°) for *both* hip and knee to be ready for DOWN transition. Feedback: "Lockout".
*   **Form Correction Checks:**
    *   **Back Angle (During Lift):** When `stage == "DOWN"` or nearly down, average back angle relative to vertical must be less than `BACK_ANGLE_THRESHOLD_DEADLIFT_LIFT` (60°). Issue: `"BACK"`. Feedback: "Back (...)deg".
    *   **Back Angle (During Lockout):** When `stage == "UP"` or nearly up, average back angle relative to vertical must be less than `BACK_ANGLE_THRESHOLD_DEADLIFT_LOCKOUT` (15°). Issue: `"BACK"`. Feedback: "Lock Back (...)deg".

---

## 6. Set/Rep/Rest Integration (Webcam Mode)

*   The rep counting logic (`counter`, `counter_left`, `counter_right`) increments within the current set.
*   When the counter reaches `target_reps_per_set`, the `set_completed_this_frame` flag is set.
*   If it's not the final set (`current_set_number < target_sets`):
    *   The application transitions to the `REST` mode.
    *   `rest_start_time` is recorded.
    *   Counters and stages are reset (`counter = 0`, `stage = None`, etc.).
    *   The rest timer counts down from `target_rest_time`.
    *   After rest (or skipping), `current_set_number` increments, and the app returns to `TRACKING` mode for the next set.
*   If it *is* the final set, the workout completes, and the session ends (saving stats).

## Conclusion

The Fitness Tracker Pro combines MediaPipe's robust pose estimation with exercise-specific angle calculations, state machines, hysteresis thresholds, EMA smoothing, and form validation rules. This allows for automated rep counting and real-time feedback, guided by configurable set and rest structures for a comprehensive workout tracking experience in webcam mode.