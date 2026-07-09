import cv2
import os

os.makedirs("hand_images", exist_ok=True)

# 统计已有图片数，续拍不覆盖
existing = [f for f in os.listdir("hand_images") if f.endswith('.jpg')]
count = len(existing)

cap = cv2.VideoCapture(0)
print("=" * 40)
print("手部图像采集")
print("空格键 = 拍照保存")
print("ESC键  = 退出")
print(f"已有 {count} 张，从 {count+1} 开始续拍")
print("=" * 40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ★ 正方形采集框（和训练裁剪一致）
    box_size = int(min(w, h) * 0.75)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    display = frame.copy()
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 中心十字
    cx, cy = w // 2, h // 2
    cv2.line(display, (cx-20, cy), (cx+20, cy), (0,255,0), 1)
    cv2.line(display, (cx, cy-20), (cx, cy+20), (0,255,0), 1)

    cv2.putText(display, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(display, "SPACE=Save  ESC=Exit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(display, "Put hand in green box", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Hand Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        count += 1
        cropped = frame[y1:y2, x1:x2]
        # ★ 直接存 224×224 正方形
        cropped = cv2.resize(cropped, (224, 224))
        filename = f"hand_images/hand_{count:04d}.jpg"
        cv2.imwrite(filename, cropped)
        print(f"[{count}] 已保存: {filename}")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n采集完成！共 {count} 张")