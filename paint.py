import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

import warnings
warnings.filterwarnings("ignore")

import cv2, mediapipe as mp, numpy as np, torch, random, time, pyttsx3, threading, urllib.request
import torch.nn as nn
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

softmax = nn.Softmax(dim=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def thread_speak(text):
    def _speak():
        try:
            e = pyttsx3.init()
            e.setProperty('rate', 150)
            e.setProperty('voice', e.getProperty('voices')[1].id)
            e.say(text)
            e.runAndWait()
            e.stop()
        except: pass
    threading.Thread(target=_speak, daemon=True).start()

# ── Wrapper thay mp.solutions.hands ──────────────────────────────────
_FINGER_TIP = 8
_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

_latest_lm, _lm_lock = None, threading.Lock()
_latest_frame, _frame_lock = None, threading.Lock()

def _callback(result, _, __):
    global _latest_lm
    with _lm_lock:
        _latest_lm = result.hand_landmarks or None

def _download_model():
    path = "hand_landmarker.task"
    if not os.path.exists(path):
        print("Dang tai hand_landmarker.task...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", path)
    return path

class _Lm:
    def __init__(self, lm): self._lm = lm
    @property
    def x(self): return self._lm.x
    @property
    def y(self): return self._lm.y

class _Hand:
    def __init__(self, lm_list): self.landmark = [_Lm(l) for l in lm_list]

class _Results:
    def __init__(self, lms): self.multi_hand_landmarks = [_Hand(l) for l in lms] if lms else None

class _Hands:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_download_model()),
            num_hands=1, min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.LIVE_STREAM, result_callback=_callback)
        self._lm = mp_vision.HandLandmarker.create_from_options(opts)
        self._ts = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            with _frame_lock: f = _latest_frame
            if f is not None:
                self._ts += 1
                self._lm.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), self._ts)
            time.sleep(0.015)

    def process(self, _):
        with _lm_lock: return _Results(_latest_lm)

    def __enter__(self): return self
    def __exit__(self, *a): self._lm.close()

class _FakeHands:
    Hands = _Hands
    HAND_CONNECTIONS = _CONNECTIONS
    class HandLandmark:
        INDEX_FINGER_TIP = _FINGER_TIP

class _FakeDrawing:
    @staticmethod
    def draw_landmarks(frame, hand_landmark, connections):
        h, w = frame.shape[:2]
        for s, e in connections:
            cv2.line(frame,
                     (int(hand_landmark.landmark[s].x*w), int(hand_landmark.landmark[s].y*h)),
                     (int(hand_landmark.landmark[e].x*w), int(hand_landmark.landmark[e].y*h)),
                     (0,255,0), 2)
        for lm in hand_landmark.landmark:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255,0,0), -1)
# ─────────────────────────────────────────────────────────────────────

def load_model(model_path):
    model = torch.load(model_path, map_location=device, weights_only=False)
    return model.eval()

def load_icon(image_path):
    icons = {}
    if not os.path.exists(image_path): return icons
    for f in os.listdir(image_path):
        if f.endswith("png"):
            icon = cv2.imread(os.path.join(image_path, f), cv2.IMREAD_UNCHANGED)
            if icon is None: continue
            if icon.shape[2] == 3:
                b,g,r = cv2.split(icon)
                icon = cv2.merge((b,g,r, np.ones(b.shape,dtype=b.dtype)*255))
            icons[f.split(".")[0]] = cv2.resize(icon,(50,50))
    return icons

def predict(model, canvas):
    mask = cv2.GaussianBlur(cv2.medianBlur(cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY),9),(5,5),0)
    x, y = np.nonzero(mask)
    if not len(x): return None
    crop = cv2.resize(mask[x.min():x.max(), y.min():y.max()], (28,28))
    t = torch.from_numpy(crop.astype(np.float32)[None,None]/255.0).to(device)
    with torch.no_grad():
        prob = softmax(model(t))
    v, i = torch.max(prob, dim=1)
    return v.item(), i.item()

def overlay_icon(combined, icon, x, y):
    h, w, _ = icon.shape
    if y+h > combined.shape[0] or x+w > combined.shape[1]: return
    b,g,r,a = cv2.split(icon)
    alpha = a/255.0
    roi = combined[y:y+h, x:x+w]
    for c,(ch) in enumerate(cv2.split(cv2.merge((b,g,r)))):
        roi[:,:,c] = (1-alpha)*roi[:,:,c] + alpha*ch
    combined[y:y+h, x:x+w] = roi

def paint(model, classes):
    global _latest_frame
    mp_hands = _FakeHands()
    mp_drawing = _FakeDrawing()
    icons = load_icon("./Icon_image")
    canvas = None
    is_drawing = False
    start_time = None
    text = ""
    cap = cv2.VideoCapture(0)
    aim = classes[random.randint(0, len(classes)-1)]
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            if canvas is None: canvas = np.zeros_like(frame)
            with _frame_lock: _latest_frame = frame.copy()
            results = hands.process(None)
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    x_index = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    y_index = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                    if is_drawing:
                        cv2.circle(canvas, (x_index,y_index), 5, (255,255,255), -1)
                    mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            combined = cv2.add(frame, canvas)
            cv2.putText(combined, "Your challenge is:  {}".format(aim), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(combined, "Guide:", (0,410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(combined, "Press [D] to delete your paint",    (0,426), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (207,207,207), 1, cv2.LINE_AA)
            cv2.putText(combined, "Press [P] to predict your paint",   (0,443), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (207,207,207), 1, cv2.LINE_AA)
            cv2.putText(combined, "Press [S] to start/stop painting",  (0,460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (207,207,207), 1, cv2.LINE_AA)
            cv2.putText(combined, "Press [C] to change your challenge",(0,477), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (207,207,207), 1, cv2.LINE_AA)
            if text and start_time:
                if time.time() - start_time <= 3:
                    cv2.putText(combined, text, (50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
                    icon_text = text.split(":")[-1].strip()
                    if icon_text in icons:
                        icon = icons[class_name]
                        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0][0]
                        overlay_icon(combined, icon, 50+text_width+10, 40)
                else:
                    text = ""; start_time = None; canvas = np.zeros_like(frame)
            cv2.imshow("Draw", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                is_drawing = not is_drawing
            elif key == ord("p"):
                is_drawing = False
                result_p = predict(model, canvas)
                if result_p:
                    max_value_p, max_index_p = result_p
                    class_name = classes[max_index_p]
                    text = f"You are drawing:  {class_name}"
                    start_time = time.time()
                    print("You are drawing: {} with {} %".format(class_name, max_value_p*100))
                    if class_name == aim:
                        thread_speak("Oh no, It's {} , Correct!, You Win".format(class_name))
                        is_drawing = False
                        aim = classes[random.randint(0, len(classes)-1)]
                        thread_speak("Your new challenge is: {}".format(aim))
            elif key == ord("d"):
                canvas = np.zeros_like(frame)
            elif key == 27:
                break
            elif key == ord("c"):
                aim = classes[random.randint(0, len(classes)-1)]
                thread_speak("Your new challenge is {}".format(aim))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = load_model("./train_model_QuickDraw/quickdraw")
    classes = ["Airplane","Angel","Apple","Axe","Bat","Book","Boomerang","Camera","Cup","Fish","Flower","Mushroom","Radio","Sun","Sword"]
    paint(model, classes)