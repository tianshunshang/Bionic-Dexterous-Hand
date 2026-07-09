"""
gesture_servo.py
功能：关键点 → 手指弯曲角度 → 舵机PWM值
在现有 live_test.py 基础上增加角度计算和串口输出

硬件连接（STM32N6侧另写C代码接收）：
  PC USB-串口 → STM32 USART
  发送格式：$T1,T2,T3,T4,T5\n
  T1~T5 = 拇指~小拇指的舵机角度(0~180)
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import serial  # pip install pyserial
import serial.tools.list_ports

#  串口配置（不用串口就把 USE_SERIAL 改成 False）

USE_SERIAL = True  # True=连接STM32  False=只在屏幕显示
SERIAL_PORT = "COM4"  # Windows: "COM3"  Linux: "/dev/ttyUSB0"
SERIAL_BAUD = 115200
SEND_HZ = 20  # 每秒发送次数，避免串口过载

#  关键点索引定义（和训练标注顺序一致）
#  0=wrist
#  1-4=拇指  5-8=食指  9-12=中指  13-16=无名指  17-20=小拇指

FINGER_JOINTS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
}

# 舵机角度范围（根据你的仿生手机械结构调整）
SERVO_MIN = 0  # 对应手指完全伸直
SERVO_MAX = 180  # 对应手指完全弯曲


def calc_angle(p1, p2, p3):
    """
    计算 p1→p2→p3 的夹角（度）
    p1, p2, p3: [x, y] 像素坐标
    返回：0~180度，角度越小表示弯曲越大
    """
    v1 = np.array(p1) - np.array(p2)  # 向量 p2→p1
    v2 = np.array(p3) - np.array(p2)  # 向量 p2→p3
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def keypoints_to_angles(pts_abs):
    """
    输入：21个关键点的绝对像素坐标列表 [(x0,y0), (x1,y1), ...]
    输出：5根手指的平均弯曲角度 dict {'thumb':..., 'index':..., ...}

    逻辑：取每根手指所有相邻三点的夹角均值作为该手指的弯曲程度
    """
    angles = {}
    for finger, joints in FINGER_JOINTS.items():
        finger_angles = []
        for i in range(len(joints) - 2):  # 三点一组
            j0 = joints[i]
            j1 = joints[i + 1]
            j2 = joints[i + 2]
            a = calc_angle(pts_abs[j0], pts_abs[j1], pts_abs[j2])
            finger_angles.append(a)
        # 取最小角（最弯曲的关节代表整根手指状态）
        angles[finger] = float(np.min(finger_angles))
    return angles


def angle_to_servo(finger_angle, straight_angle=170, bent_angle=60):
    """
    将关节夹角映射到舵机角度

    finger_angle: 关节夹角（度），接近180=伸直，接近0=弯曲
    straight_angle: 手指伸直时的典型夹角（可通过打印实测调整）
    bent_angle:    手指完全弯曲时的典型夹角

    返回：舵机角度 0~180
    """
    # 线性映射，夹角大→舵机角小（伸直），夹角小→舵机角大（弯曲）
    ratio = 1.0 - (finger_angle - bent_angle) / (straight_angle - bent_angle)
    servo = SERVO_MIN + ratio * (SERVO_MAX - SERVO_MIN)
    return int(np.clip(servo, SERVO_MIN, SERVO_MAX))


def build_serial_cmd(servo_angles: dict) -> bytes:
    """
    构建串口数据包
    格式：$90,45,120,30,60\n  （拇指,食指,中指,无名指,小拇指）
    STM32侧解析这个字符串，直接设置TIM PWM
    """
    order = ['thumb', 'index', 'middle', 'ring', 'pinky']
    vals = [str(servo_angles[f]) for f in order]
    cmd = '$' + ','.join(vals) + '\n'
    return cmd.encode('ascii')


# ══════════════════════════════════════════════════════════
#  正方形裁剪 & 模型（和 live_test.py 完全一致）
# ══════════════════════════════════════════════════════════
def square_crop_frame(frame, x1, y1, x2, y2):
    crop_w, crop_h = x2 - x1, y2 - y1
    side = max(crop_w, crop_h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    H, W = frame.shape[:2]
    sq_x1 = max(0, cx - side // 2)
    sq_y1 = max(0, cy - side // 2)
    sq_x2 = min(W, sq_x1 + side)
    sq_y2 = min(H, sq_y1 + side)
    crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]
    ah, aw = crop.shape[:2]
    if ah != aw:
        sq = np.zeros((side, side, 3), dtype=np.uint8)
        sq[:ah, :aw] = crop
        crop = sq
    return crop, sq_x1, sq_y1, side


class HandNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v2
        backbone = mobilenet_v2(weights=None)
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone.last_channel, 42),
            nn.Sigmoid()
        )
        self.model = backbone

    def forward(self, x): return self.model(x)


# ══════════════════════════════════════════════════════════
#  初始化
# ══════════════════════════════════════════════════════════
MODEL_PATH = "model_a_best.pth"
if not os.path.exists(MODEL_PATH):
    print(f"错误：找不到 '{MODEL_PATH}'");
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device);
model.eval()
print(f"模型加载成功，使用 {device}")

# 串口初始化
ser = None
if USE_SERIAL:
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        print(f"串口已连接: {SERIAL_PORT} @ {SERIAL_BAUD}")
    except Exception as e:
        print(f"串口连接失败: {e}")
        print("可用串口:", [p.device for p in serial.tools.list_ports.comports()])
        USE_SERIAL = False

mp_hands = mp.solutions.hands
detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                          min_detection_confidence=0.6, min_tracking_confidence=0.5)

COLORS = [
    (128, 128, 128),
    (0, 0, 255), (0, 0, 200), (0, 0, 160), (0, 0, 120),
    (0, 255, 0), (0, 200, 0), (0, 160, 0), (0, 120, 0),
    (255, 0, 0), (200, 0, 0), (160, 0, 0), (120, 0, 0),
    (255, 255, 0), (200, 200, 0), (160, 160, 0), (120, 120, 0),
    (255, 0, 255), (200, 0, 200), (160, 0, 160), (120, 0, 120),
]
CONNECTIONS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]

INPUT_SIZE, PAD = 224, 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0;
fps = 0.0;
last_time = time.time()
last_send_time = 0
coords_global = None;
bbox_global = None
servo_angles = {'thumb': 90, 'index': 90, 'middle': 90, 'ring': 90, 'pinky': 90}

print("手势→舵机控制 | 按 ESC 退出 | 按 C 打印当前角度（用于标定）")

# ══════════════════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    frame_count += 1

    if frame_count % 30 == 0:
        fps = 30.0 / (time.time() - last_time)
        last_time = time.time()

    result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        xs = [p.x * W for p in lm.landmark]
        ys = [p.y * H for p in lm.landmark]
        x1 = max(0, int(min(xs)) - PAD)
        y1 = max(0, int(min(ys)) - PAD)
        x2 = min(W - 1, int(max(xs)) + PAD)
        y2 = min(H - 1, int(max(ys)) + PAD)
        bbox_global = (x1, y1, x2, y2)

        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            crop, sq_x1, sq_y1, sq_side = square_crop_frame(frame, x1, y1, x2, y2)
            inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
            inp = inp.astype(np.float32) / 255.0
            inp_t = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(inp_t).squeeze().cpu().numpy()

            # 映射回全图绝对像素
            coords_full = np.zeros(42, dtype=np.float32)
            for i in range(21):
                coords_full[i * 2] = np.clip(pred[i * 2] * sq_side + sq_x1, 0, W - 1)
                coords_full[i * 2 + 1] = np.clip(pred[i * 2 + 1] * sq_side + sq_y1, 0, H - 1)
            coords_global = coords_full

            # ★ 计算角度 → 舵机值
            pts_abs = [(int(coords_global[i * 2]), int(coords_global[i * 2 + 1]))
                       for i in range(21)]
            finger_angles = keypoints_to_angles(pts_abs)
            servo_angles = {f: angle_to_servo(a) for f, a in finger_angles.items()}

            # ★ 限频发送串口
            now = time.time()
            if USE_SERIAL and ser and (now - last_send_time) > 1.0 / SEND_HZ:
                try:
                    cmd = build_serial_cmd(servo_angles)
                    ser.write(cmd)
                    last_send_time = now
                except Exception as e:
                    print(f"串口发送失败: {e}")
    else:
        bbox_global = None
        coords_global = None

    # ── 绘制关键点 ────────────────────────────────────────
    if bbox_global:
        x1, y1, x2, y2 = bbox_global
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if coords_global is not None:
        pts = [(int(coords_global[i * 2]), int(coords_global[i * 2 + 1])) for i in range(21)]
        for finger in CONNECTIONS:
            for j in range(len(finger) - 1):
                cv2.line(frame, pts[finger[j]], pts[finger[j + 1]], COLORS[finger[j]], 2)
        for i, (px, py) in enumerate(pts):
            cv2.circle(frame, (px, py), 5, COLORS[i], -1)

    # ── 显示舵机角度 ──────────────────────────────────────
    finger_labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    finger_keys = ['thumb', 'index', 'middle', 'ring', 'pinky']
    bar_x = W - 160
    for i, (label, key) in enumerate(zip(finger_labels, finger_keys)):
        angle = servo_angles[key]
        # 进度条
        bar_len = int(angle / 180.0 * 100)
        cv2.rectangle(frame, (bar_x, 80 + i * 28), (bar_x + 100, 96 + i * 28), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, 80 + i * 28), (bar_x + bar_len, 96 + i * 28), COLORS[i * 4 + 1], -1)
        cv2.putText(frame, f"{label}:{angle:3d}", (bar_x - 10, 93 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[i * 4 + 1], 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    serial_status = f"Serial: {'ON ' + SERIAL_PORT if USE_SERIAL else 'OFF'}"
    cv2.putText(frame, serial_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if USE_SERIAL else (100, 100, 100), 1)

    cv2.imshow("Gesture → Servo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c') or key == ord('C'):
        # 按C打印当前原始角度，用于标定 straight_angle 和 bent_angle
        if coords_global is not None:
            pts_abs = [(int(coords_global[i * 2]), int(coords_global[i * 2 + 1])) for i in range(21)]
            raw = keypoints_to_angles(pts_abs)
            print("当前关节角度（用于标定）:")
            for f, a in raw.items():
                print(f"  {f:8s}: {a:.1f}°  → servo={servo_angles[f]}")

cap.release()
cv2.destroyAllWindows()
detector.close()
if ser: ser.close()
print("程序已退出")