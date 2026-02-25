import os, cv2, time, torch, requests
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from collections import deque
from multiprocessing import Process, Queue
import random

import clip
from ResEmoteNet import ResEmoteNet
from batch_face import RetinaFace
from utils import play_emotions

# =====================
# SYSTEM CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

EMOTIONS = ["neutral", "happy", "sad", "surprise", "anger"]

CNN_LABELS = ["happy", "surprise", "sad", "anger", "fear", "disgust", "neutral"]
CNN_MAP = {
    "happy": "happy",
    "surprise": "surprise",
    "sad": "sad",
    "anger": "anger",
    "fear": "neutral",
    "disgust": "neutral",
    "neutral": "neutral"
}

CNN_STRONG = 0.80
CLIP_STRONG = 0.75
SMOOTHING = 7

print("Device:", DEVICE)

# =====================
# VOCAL SUPPORT (FAST & NATURAL)
# =====================
VOCAL_PHRASES = {
    "happy": ["Nice.", "That’s good.", "I like that."],
    "sad": ["It’s okay.", "I’m here.", "Take your time."],
    "anger": ["Hmm.", "Let’s slow down.", "Alright."],
    "surprise": ["Oh!", "Interesting.", "Wow."],
    "neutral": ["Okay.", "Alright.", "Let’s continue."]
}

_last_spoken = None

def speak(emotion):
    global _last_spoken
    if emotion == _last_spoken:
        return
    phrase = random.choice(VOCAL_PHRASES[emotion])
    print("[TTS]", phrase)  # replace with robot TTS later
    _last_spoken = emotion

# =====================
# LOAD CNN
# =====================
cnn = ResEmoteNet().to(DEVICE)
ckpt = torch.load("backup_model.pth", map_location=DEVICE)
cnn.load_state_dict(ckpt["model_state_dict"])
cnn.eval()

cnn_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def cnn_predict(pil):
    t = cnn_tf(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = F.softmax(cnn(t), dim=1).cpu().numpy()[0]
    idx = np.argmax(p)
    return CNN_MAP[CNN_LABELS[idx]], float(p[idx])

# =====================
# LOAD CLIP
# =====================
clip_model, clip_pre = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

clip_texts = ["happy face", "sad face", "surprised face", "angry face", "neutral face"]
clip_emotions = ["happy", "sad", "surprise", "anger", "neutral"]

with torch.no_grad():
    txt = clip.tokenize(clip_texts).to(DEVICE)
    text_feat = clip_model.encode_text(txt)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

def clip_predict(pil):
    img = clip_pre(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
        sims = (feat @ text_feat.T).squeeze(0)
    idx = sims.argmax().item()
    return clip_emotions[idx], float(sims[idx])

# =====================
# FACE DETECTOR + ROBOT
# =====================
detector = RetinaFace()
robot_q = Queue(maxsize=1)

emotion_frames = {}
for e in EMOTIONS:
    p = f"feelings/{e}"
    if os.path.isdir(p):
        imgs = [cv2.imread(os.path.join(p,f)) for f in sorted(os.listdir(p)) if f.endswith(".jpg")]
        if imgs:
            emotion_frames[e] = imgs

Process(target=play_emotions, args=(emotion_frames, robot_q), daemon=True).start()

# =====================
# VIDEO
# =====================
cap = cv2.VideoCapture(0)

stable = "neutral"
history = deque(maxlen=SMOOTHING)
state_id = 0

print("[EVENT] System online")

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = detector(rgb, threshold=0.85, return_dict=True)

    if det:
        x1,y1,x2,y2 = map(int, det[0]["box"])
        face = frame[y1:y2, x1:x2]
        pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        cnn_l, cnn_c = cnn_predict(pil)
        clip_l, clip_c = clip_predict(pil)

        # =====================
        # FUSION LOGIC (UNCHANGED)
        # =====================
        if clip_c >= CLIP_STRONG:
            final = clip_l
            src = "CLIP"
        elif cnn_c >= CNN_STRONG:
            final = cnn_l
            src = "CNN"
        elif cnn_l == clip_l:
            final = cnn_l
            src = "agreement"
        else:
            final = stable
            src = "hold"

        history.append(final)
        smooth = max(set(history), key=history.count)

        if smooth != stable:
            state_id += 1
            stable = smooth
            print(f"\n[STATE {state_id}] EMOTION CHANGE")
            print(f"  New:        {stable}")
            print(f"  Source:     {src}")
            print(f"  CNN conf:   {cnn_c:.3f}")
            print(f"  CLIP conf:  {clip_c:.3f}")

            if robot_q.empty():
                robot_q.put(stable)

            speak(stable)

        # =====================
        # DRAW
        # =====================
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"CNN: {cnn_l} {cnn_c:.2f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
        cv2.putText(frame,f"CLIP: {clip_l} {clip_c:.2f}",(10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,255,150),2)
        cv2.putText(frame,f"FINAL: {stable}",(10,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,0),2)

    cv2.imshow("Emotion System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
