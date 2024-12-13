import cv2
import numpy as np
from scipy.spatial import distance as dist
import time

MAX_FRAMES = 20  
RECENT_FRAMES = int(MAX_FRAMES / 10) 
EYE_BLINK_HEIGHT = 0.15 


FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

EPOCH = time.time()


blinks = [False] * MAX_FRAMES
hr_times = list(range(0, MAX_FRAMES))
hr_values = [400] * MAX_FRAMES
avg_bpms = [0] * MAX_FRAMES
gaze_values = [0] * MAX_FRAMES

def get_aspect_ratio(top, bottom, right, left):
    height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
    width = dist.euclidean([right.x, right.y], [left.x, left.y])
    return height / width

def get_area(image, draw, topL, topR, bottomR, bottomL):
    topY = int((topR.y + topL.y) / 2 * image.shape[0])
    botY = int((bottomR.y + bottomL.y) / 2 * image.shape[0])
    leftX = int((topL.x + bottomL.x) / 2 * image.shape[1])
    rightX = int((topR.x + bottomR.x) / 2 * image.shape[1])

    return image[topY:botY, rightX:leftX]


def is_blinking(face):
    eyeR = [face[p] for p in [159, 145, 133, 33]]
    eyeR_ar = get_aspect_ratio(*eyeR)

    eyeL = [face[p] for p in [386, 374, 362, 263]]
    eyeL_ar = get_aspect_ratio(*eyeL)

    eyeA_ar = (eyeR_ar + eyeL_ar) / 2
    return eyeA_ar < EYE_BLINK_HEIGHT

def check_hand_on_face(hands_landmarks, face):
    if hands_landmarks:
        face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
        face_points = [[[p.x, p.y] for p in face_landmarks]]
        face_contours = np.array(face_points).astype(np.single)

        for hand_landmarks in hands_landmarks:
            hand = [(point.x, point.y) for point in hand_landmarks.landmark]

            for finger in [4, 8, 20]:
                overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
                if overlap != -1:
                    return True
    return False

def get_avg_gaze(face):
    gaze_left = get_gaze(face, 476, 474, 263, 362)
    gaze_right = get_gaze(face, 471, 469, 33, 133)
    return round((gaze_left + gaze_right) / 2, 1)

def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
    iris = (
        face[iris_L_side].x + face[iris_R_side].x,
        face[iris_L_side].y + face[iris_R_side].y,
    )
    eye_center = (
        face[eye_L_corner].x + face[eye_R_corner].x,
        face[eye_L_corner].y + face[eye_R_corner].y,
    )

    gaze_dist = dist.euclidean(iris, eye_center)
    eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
    gaze_relative = gaze_dist / eye_width

    if (eye_center[0] - iris[0]) < 0: 
        gaze_relative *= -1

    return gaze_relative


def calculate_gaze_score(gaze_data):
    if not gaze_data or len(gaze_data) < 2:
        return 0

    absolute_differences = np.abs(np.diff(gaze_data))
    mean_absolute_difference = np.mean(absolute_differences)

    threshold = 0.3
    jumps_above_threshold = np.sum(absolute_differences > threshold)

    direction_changes = np.sum(np.diff(np.sign(gaze_data)) != 0)

    variance = np.var(gaze_data)
    std_deviation = np.std(gaze_data)

    max_diff = 1.0
    max_jumps = 10
    max_direction_changes = 10
    max_variance = 0.25
    max_std_deviation = 0.5

    normalized_diff = mean_absolute_difference / max_diff
    normalized_jumps = jumps_above_threshold / max_jumps
    normalized_direction_changes = direction_changes / max_direction_changes
    normalized_variance = variance / max_variance
    normalized_std_deviation = std_deviation / max_std_deviation

    weights = {
        "diff": 0.3,
        "jumps": 0.2,
        "direction_changes": 0.3,
        "variance": 0.1,
        "std_deviation": 0.1
    }

    aggregate_score = (
        weights["diff"] * normalized_diff +
        weights["jumps"] * normalized_jumps +
        weights["direction_changes"] * normalized_direction_changes +
        weights["variance"] * normalized_variance +
        weights["std_deviation"] * normalized_std_deviation
    )

    final_score = aggregate_score * 0.8

    return round(final_score, 4)

def detect_gaze_change(avg_gaze, gaze_values):

    gaze_values = gaze_values[1:] + [avg_gaze]
    gaze_relative_matches = gaze_values.count(avg_gaze) / MAX_FRAMES
    return gaze_relative_matches if gaze_relative_matches < 0.01 else 0

def get_lip_ratio(face):
    return get_aspect_ratio(face[0], face[17], face[61], face[291])

def get_face_relative_area(face):
    face_width = abs(max(face[454].x, 0) - max(face[234].x, 0))
    face_height = abs(max(face[152].y, 0) - max(face[10].y, 0))
    return face_width * face_height

def find_face_and_hands(image_original, face_mesh, hands):
    image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 

    faces = face_mesh.process(image)
    hands_landmarks = hands.process(image).multi_hand_landmarks

    face_landmarks = None
    x,y,w,h = None, None, None, None
    if faces.multi_face_landmarks and len(faces.multi_face_landmarks) > 0:
        face_landmarks = faces.multi_face_landmarks[0]
        
        h, w = image_original.shape[:2]
        landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
        x_min, y_min = min(landmarks, key=lambda p: p[0])[0], min(landmarks, key=lambda p: p[1])[1]
        x_max, y_max = max(landmarks, key=lambda p: p[0])[0], max(landmarks, key=lambda p: p[1])[1]
        
        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

    return face_landmarks, hands_landmarks, (x,y,w,h)
