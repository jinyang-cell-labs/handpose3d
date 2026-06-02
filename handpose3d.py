import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

frame_shape = [720, 1280]

#path to the MediaPipe Tasks hand landmark model bundle (auto-downloaded by run.sh)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'hand_landmarker.task')

#hand skeleton connections (21 landmarks), formerly mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          #thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          #index
    (5, 9), (9, 10), (10, 11), (11, 12),     #middle
    (9, 13), (13, 14), (14, 15), (15, 16),   #ring
    (13, 17), (17, 18), (18, 19), (19, 20),  #pinky
    (0, 17),                                 #palm base
]


def make_hand_landmarker():
    #create a HandLandmarker in VIDEO running mode (new MediaPipe Tasks API)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Hand landmark model not found at {MODEL_PATH}. "
            "Run ./run.sh (it auto-downloads it) or fetch hand_landmarker.task manually."
        )
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(options)


def draw_hand_landmarks(frame, hand_landmarks):
    #draw the 21-point hand skeleton on a BGR frame (replaces mp.solutions drawing_utils)
    h, w = frame.shape[:2]
    pts = [(int(round(lm.x * w)), int(round(lm.y * h))) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv.circle(frame, p, 3, (0, 0, 255), -1)


def run_mp(input_stream1, input_stream2, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    #create hand keypoints detector object (one per camera so VIDEO-mode timestamps stay independent).
    hands0 = make_hand_landmarker()
    hands1 = make_hand_landmarker()

    #containers for detected keypoints for each camera
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    #VIDEO running mode requires a monotonically increasing timestamp (ms) per frame
    frame_idx = 0
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        #wrap each RGB frame as a MediaPipe Image and run detection (VIDEO mode + timestamp)
        timestamp_ms = frame_idx * 33  #~30 fps; only needs to be monotonically increasing
        mp_image0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame0))
        mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame1))
        results0 = hands0.detect_for_video(mp_image0, timestamp_ms)
        results1 = hands1.detect_for_video(mp_image1, timestamp_ms)
        frame_idx += 1

        #prepare list of hand keypoints of this frame
        #frame0 kpts
        frame0_keypoints = []
        if results0.hand_landmarks:
            for hand_landmarks in results0.hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks[p].x, hand_landmarks[p].y)
                    pxl_x = int(round(frame0.shape[1]*hand_landmarks[p].x))
                    pxl_y = int(round(frame0.shape[0]*hand_landmarks[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame0_keypoints.append(kpts)

        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*21

        kpts_cam0.append(frame0_keypoints)

        #frame1 kpts
        frame1_keypoints = []
        if results1.hand_landmarks:
            for hand_landmarks in results1.hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks[p].x, hand_landmarks[p].y)
                    pxl_x = int(round(frame1.shape[1]*hand_landmarks[p].x))
                    pxl_y = int(round(frame1.shape[0]*hand_landmarks[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame1_keypoints.append(kpts)

        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]]*21

        #update keypoints container
        kpts_cam1.append(frame1_keypoints)


        #calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        kpts_3d.append(frame_p3ds)

        # Draw the hand annotations on the image.
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        if results0.hand_landmarks:
          for hand_landmarks in results0.hand_landmarks:
            draw_hand_landmarks(frame0, hand_landmarks)

        if results1.hand_landmarks:
          for hand_landmarks in results1.hand_landmarks:
            draw_hand_landmarks(frame1, hand_landmarks)
        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()
    hands0.close()
    hands1.close()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':

    input_stream1 = 'media/cam0_test.mp4'
    input_stream2 = 'media/cam1_test.mp4'

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    #projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    #this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)
