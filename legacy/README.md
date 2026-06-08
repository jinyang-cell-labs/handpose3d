**Real time 3D hand pose estimation using MediaPipe**

This is a demo on how to obtain 3D coordinates of hand keypoints using MediaPipe and two calibrated cameras. Two cameras are required as there is no way to obtain 3D coordinates from a single camera. Check here: [stereo calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate) for a calibration package. Also my blog post on how to stereo calibrate two cameras: [link](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html). Alternatively, follow the camera calibration at Opencv documentations: [link](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html). If you want to know some details on how this code works, take a look at my accompanying blog post here: [link](https://temugeb.github.io/python/computer_vision/2021/06/27/handpose3d.html).

![input1](media/output_kpts.gif "input1") ![input2](media/output2_kpts.gif "input2") 
![output](media/fig_0.gif "output")

**Requirements**
```
Python 3.11+        (tested on 3.12)
mediapipe >= 0.10.30   (modern MediaPipe Tasks API)
opencv-contrib-python
numpy
matplotlib
```

> **Note on MediaPipe versions:** this project was updated to the modern **MediaPipe Tasks** API
> (`mediapipe.tasks.python.vision.HandLandmarker`). The old `mp.solutions.hands` ("Solutions")
> API was removed by Google in the `0.10.3x` releases, so any recent mediapipe works here — but
> code written against `mp.solutions` will not. The Tasks API loads a downloaded model bundle
> (`hand_landmarker.task`) instead of bundling the pipeline. See the
> [Hand landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

**Installation (quickstart)**
A helper script, `run.sh`, creates a virtual environment, installs the requirements, and
downloads the `hand_landmarker.task` model bundle automatically:
```
source run.sh        # sets up + activates .venv, downloads the model, keeps it active in your shell
```
Or run the demo in one shot (sets things up, then runs the app):
```
./run.sh             # uses the bundled sample videos
./run.sh 0 1         # uses webcams 0 and 1
```

**Manual installation**
If you prefer to set things up yourself:
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the hand landmark model bundle into models/
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
The model is expected at `models/hand_landmarker.task` (this path is git-ignored).

**Usage: Getting real time 3D coordinates**  
As a demo, I've included two short video clips and corresponding camera calibration parameters. Simply run as:
```
python handpose3d.py
```
If you want to use webcam, call the program with camera ids. For example, cameras registered to 0 and 1:
```
python handpose3d.py 0 1
```
Make sure the corresponding camera parameters are also updated for your cameras.

The 3D coordinate in each video frame is recorded in ```frame_p3ds``` parameter. Use this for real time application. If keypoints are not found, then the keypoints are recorded as (-1, -1, -1). **Warning**: The code also saves keypoints for all previous frames. If you run the code for long periods, then you will run out of memory. To fix this, remove append calls to: ```kpts_3d, kpts_cam0. kpts_cam1```. When you press the ESC key, hand keypoints detection will stop and three files will be saved to disk. These contain recorded 2D and 3D coordinates. 

**Usage: Viewing 3D coordinates**  
The ```handpose3d.py``` program creates a 3D coordinates file: ```kpts_3d.dat```. To view the recorded 3D coordinates, simply call:
```
python show_3d_hands.py
```
This renders the animated 3D hand and saves each frame as a PNG into a ```figs/``` directory
(created automatically; git-ignored). If matplotlib falls back to the non-interactive ```Agg```
backend it will still write the frames but cannot pop up a live window — install a GUI backend
(e.g. ```pip install PyQt5```) if you want the interactive view.

