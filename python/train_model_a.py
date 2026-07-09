"""
新增功能：
  --kfold N     : N折交叉验证（默认不开启，正常训练）
  TensorBoard   : 自动记录loss、PCK@0.05、学习率曲线
  PCK指标       : Percentage of Correct Keypoints
  可视化保存    : 每10个epoch保存验证集预测图到 E:\PythonProject\visualizations

用法：
  正常训练：    python train_model_a.py
  5折交叉验证： python train_model_a.py --kfold 5

TensorBoard查看：
  tensorboard --logdir E:\PythonProject\runs
  浏览器打开 http://localhost:6006
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import albumentations as A
import os
import shutil


# ── 正方形填充 ────────────────────────────────────────────────────
def square_pad(img, coords_norm=None):
    h, w = img.shape[:2]
    side = max(h, w)
    pad_top  = (side - h) // 2
    pad_bot  = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right= side - w - pad_left
    img_sq = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=0)
    if coords_norm is not None:
        c = coords_norm.copy()
        for i in range(21):
            c[i*2]   = np.clip((coords_norm[i*2]  *w + pad_left) / side, 0, 1)
            c[i*2+1] = np.clip((coords_norm[i*2+1]*h + pad_top)  / side, 0, 1)
        return img_sq, c
    return img_sq


# ── Dataset ───────────────────────────────────────────────────────
class HandDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data    = data
        self.augment = augment
        if augment:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
                A.Affine(translate_percent={"x":(-0.05,0.05),"y":(-0.05,0.05)},
                         scale=(0.85,1.15), rotate=(-15,15), p=0.5),
                A.GaussNoise(var_limit=(10,50), p=0.3),
                A.HueSaturationValue(p=0.2),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img  = cv2.imread(item['image'])
        if img is None:
            img = np.zeros((224,224,3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        coords = item['coords'].copy().astype(np.float32)

        img, coords = square_pad(img, coords)

        if self.augment and hasattr(self, 'transform'):
            S   = img.shape[0]
            kps = [(coords[i*2]*S, coords[i*2+1]*S) for i in range(21)]
            img = cv2.resize(img, (224,224))
            out = self.transform(image=img, keypoints=kps)
            img = out['image']
            for i,(kx,ky) in enumerate(out['keypoints']):
                coords[i*2]   = np.clip(kx/224, 0, 1)
                coords[i*2+1] = np.clip(ky/224, 0, 1)
        else:
            img = cv2.resize(img, (224,224))

        img = img.astype(np.float32)/255.0
        img = np.transpose(img, (2,0,1))
        return torch.from_numpy(img), torch.from_numpy(coords)


# ── 模型 ──────────────────────────────────────────────────────────
class HandNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone.last_channel, 42),
            nn.Sigmoid()
        )
        self.model = backbone
    def forward(self, x): return self.model(x)


# ── 损失函数 ──────────────────────────────────────────────────────
def wing_loss(pred, target, w=10.0, eps=2.0):
    diff = torch.abs(pred - target)
    C    = w - w * torch.log(torch.tensor(1.0 + w/eps, device=pred.device))
    return torch.where(diff < w, w * torch.log(1.0 + diff/eps), diff - C).mean()


def structure_loss(pred, target):
    pred   = pred.view(-1,21,2)
    target = target.view(-1,21,2)
    fingers = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    loss = sum(
        torch.mean((torch.norm(pred[:,f[i+1]]-pred[:,f[i]],dim=1) -
                    torch.norm(target[:,f[i+1]]-target[:,f[i]],dim=1))**2)
        for f in fingers for i in range(len(f)-1)
    )
    return loss / 20.0


# ── PCK 指标 ─────────────────────────────────────────────────────
def compute_pck(pred, target, threshold=0.05):
    """
    pred, target: (N, 42) 归一化坐标
    threshold: 距离阈值（相对于图像尺寸的比例）
    返回：0~1 之间的 PCK 值
    """
    pred   = pred.view(-1, 21, 2)
    target = target.view(-1, 21, 2)
    dist   = torch.norm(pred - target, dim=2)   # (N, 21)
    correct = (dist < threshold).float()
    return correct.mean().item()


# ── 保存可视化图片 ────────────────────────────────────────────────
def save_visualization(model, val_data, device, epoch, save_dir, writer=None):
    """
    在验证集上随机选一张图片，预测并保存可视化结果
    同时写入TensorBoard IMAGES
    """
    if len(val_data) == 0:
        return

    # 随机选一张
    idx = np.random.randint(0, len(val_data))
    item = val_data[idx]

    img = cv2.imread(item['image'])
    if img is None:
        return

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    coords_gt = item['coords'].copy().astype(np.float32)

    # 预处理（和训练一致）
    img_sq, coords_sq = square_pad(img_rgb, coords_gt)
    img_resized = cv2.resize(img_sq, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_norm, (2,0,1))).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor).squeeze().cpu().numpy()

    # 映射回原图尺寸
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_left = (side - w) // 2

    # 绘制
    display = img_rgb.copy()
    colors = [
        (128,128,128),
        (0,0,255),(0,0,200),(0,0,160),(0,0,120),
        (0,255,0),(0,200,0),(0,160,0),(0,120,0),
        (255,0,0),(200,0,0),(160,0,0),(120,0,0),
        (255,255,0),(200,200,0),(160,160,0),(120,120,0),
        (255,0,255),(200,0,200),(160,0,160),(120,0,120),
    ]

    # 画真实点（绿色）
    for i in range(21):
        x = int(coords_gt[i*2] * w)
        y = int(coords_gt[i*2+1] * h)
        cv2.circle(display, (x, y), 4, (0, 255, 0), -1)

    # 画预测点（红色）
    for i in range(21):
        px = pred[i*2] * side - pad_left
        py = pred[i*2+1] * side - pad_top
        px = int(np.clip(px, 0, w-1))
        py = int(np.clip(py, 0, h-1))
        cv2.circle(display, (px, py), 4, (255, 0, 0), -1)
        # 连线（预测与真实）
        gx = int(coords_gt[i*2] * w)
        gy = int(coords_gt[i*2+1] * h)
        cv2.line(display, (gx, gy), (px, py), (255, 255, 255), 1)

    # 保存到 visualizations 文件夹
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

    # 写入TensorBoard
    if writer is not None:
        # (H, W, C) -> (C, H, W) 且归一化到0-1
        img_tensorboard = np.transpose(display, (2,0,1)) / 255.0
        writer.add_image('Validation/Prediction', img_tensorboard, epoch)

    return save_path


# ── 单折训练函数 ──────────────────────────────────────────────────
def train_fold(train_data, val_data, fold_name, args, device):
    """
    训练一折，返回最佳val_loss和最佳PCK
    """
    writer = SummaryWriter(log_dir=f"runs/{fold_name}")

    train_loader = DataLoader(
        HandDataset(train_data, augment=True),
        batch_size=32, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        HandDataset(val_data, augment=False),
        batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    model     = HandNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    save_path  = f"checkpoints/model_a_best_{fold_name}.pth"
    best_loss  = float('inf')
    best_pck   = 0.0
    patience   = 0
    MAX_P      = args.patience

    for epoch in range(args.epochs):
        # ── 训练 ──
        model.train()
        tl = 0.0
        for imgs, coords in train_loader:
            imgs, coords = imgs.to(device, non_blocking=True), \
                           coords.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    pred = model(imgs)
                    loss = wing_loss(pred, coords) + 0.1*structure_loss(pred, coords)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(imgs)
                loss = wing_loss(pred, coords) + 0.1*structure_loss(pred, coords)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)
        scheduler.step()

        # ── 验证 ──
        model.eval()
        vl = 0.0
        pck_05_list = []
        pck_10_list = []
        with torch.no_grad():
            for imgs, coords in val_loader:
                imgs, coords = imgs.to(device, non_blocking=True), \
                               coords.to(device, non_blocking=True)
                pred  = model(imgs)
                vl   += wing_loss(pred, coords).item()
                pck_05_list.append(compute_pck(pred, coords, threshold=0.05))
                pck_10_list.append(compute_pck(pred, coords, threshold=0.10))
        vl     /= len(val_loader)
        pck_05  = float(np.mean(pck_05_list))
        pck_10  = float(np.mean(pck_10_list))
        lr      = optimizer.param_groups[0]['lr']

        # ── TensorBoard 写入 ──
        writer.add_scalars('Loss', {'train': tl, 'val': vl}, epoch+1)
        writer.add_scalar('PCK/PCK@0.05', pck_05, epoch+1)
        writer.add_scalar('PCK/PCK@0.10', pck_10, epoch+1)
        writer.add_scalar('LearningRate',  lr,     epoch+1)

        # ── 每10个epoch保存可视化 ──
        if (epoch + 1) % 10 == 0:
            save_visualization(model, val_data, device, epoch+1,
                             "visualizations", writer)

        # ── 保存最佳 ──
        if vl < best_loss:
            best_loss = vl
            best_pck  = pck_05
            patience  = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[{fold_name}] Epoch {epoch+1:3d} | ★ "
                  f"train={tl:.5f} val={vl:.5f} "
                  f"PCK@5%={pck_05:.3f} PCK@10%={pck_10:.3f} "
                  f"lr={lr:.6f}")
        else:
            patience += 1
            if (epoch+1) % 5 == 0:
                print(f"[{fold_name}] Epoch {epoch+1:3d} |   "
                      f"train={tl:.5f} val={vl:.5f} "
                      f"PCK@5%={pck_05:.3f} PCK@10%={pck_10:.3f} "
                      f"({patience}/{MAX_P})")

        if patience >= MAX_P:
            print(f"[{fold_name}] 早停 @ epoch {epoch+1}")
            break

    writer.close()
    print(f"[{fold_name}] 完成 | best_val={best_loss:.5f} best_PCK@5%={best_pck:.3f}")
    print(f"[{fold_name}] 权重保存: {save_path}")
    return best_loss, best_pck


# ── 主程序 ────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold',   type=int, default=0,
                        help='K折交叉验证折数，0=不使用（默认正常训练）')
    parser.add_argument('--epochs',  type=int, default=300)
    parser.add_argument('--patience',type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ""))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if not os.path.exists("train_data.npz"):
        print("错误: 找不到 train_data.npz"); exit(1)

    loaded = np.load("train_data.npz", allow_pickle=True)
    data   = [{'image': str(p), 'coords': c}
               for p, c in zip(loaded['image_paths'], loaded['coords'])]
    print(f"总样本数: {len(data)}")

    np.random.seed(42)
    np.random.shuffle(data)

    os.makedirs("checkpoints",    exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("runs",           exist_ok=True)

    # ════════════════════════════════════════════════════
    #  模式A：普通训练（默认）
    # ════════════════════════════════════════════════════
    if args.kfold <= 1:
        split     = int(len(data) * 0.8)
        train_set = data[:split]
        val_set   = data[split:]
        print(f"训练集: {len(train_set)}  验证集: {len(val_set)}")
        print(f"TensorBoard: tensorboard --logdir runs  然后打开 http://localhost:6006")

        best_loss, best_pck = train_fold(train_set, val_set, "final", args, device)

        # final 模型同时复制一份到标准名称
        if os.path.exists("checkpoints/model_a_best_final.pth"):
            shutil.copy("checkpoints/model_a_best_final.pth", "model_a_best.pth")
            print("已同步到 model_a_best.pth")

    # ════════════════════════════════════════════════════
    #  模式B：K-Fold 交叉验证
    # ════════════════════════════════════════════════════
    else:
        K = args.kfold
        print(f"\n{'='*50}")
        print(f"  {K}折交叉验证开始")
        print(f"{'='*50}")
        print(f"TensorBoard: tensorboard --logdir runs  然后打开 http://localhost:6006")
        fold_size = len(data) // K
        results   = []

        for k in range(K):
            print(f"\n{'='*50}")
            print(f"  Fold {k+1}/{K}")
            print(f"{'='*50}")

            val_start = k * fold_size
            val_end   = val_start + fold_size if k < K-1 else len(data)

            val_data   = data[val_start:val_end]
            train_data = data[:val_start] + data[val_end:]

            print(f"  训练: {len(train_data)}  验证: {len(val_data)}")

            bl, bp = train_fold(train_data, val_data, f"fold{k+1}", args, device)
            results.append((bl, bp))

        # ── K-Fold 汇总 ──
        print(f"\n{'='*50}")
        print(f"  {K}折交叉验证汇总")
        print(f"{'='*50}")
        losses = [r[0] for r in results]
        pcks   = [r[1] for r in results]
        for k,(bl,bp) in enumerate(results):
            print(f"  Fold {k+1}: val_loss={bl:.5f}  PCK@5%={bp:.3f}")
        print(f"  平均 val_loss = {np.mean(losses):.5f} ± {np.std(losses):.5f}")
        print(f"  平均 PCK@5%  = {np.mean(pcks):.3f} ± {np.std(pcks):.3f}")
        print(f"\n  选最佳fold的权重作为最终模型：")
        best_fold = int(np.argmin(losses)) + 1
        print(f"  Fold {best_fold} (val_loss={losses[best_fold-1]:.5f})")

        shutil.copy(f"checkpoints/model_a_best_fold{best_fold}.pth", "model_a_best.pth")
        print(f"  已复制 fold{best_fold} 权重 → model_a_best.pth")