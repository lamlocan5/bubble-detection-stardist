# 🫧 Bubble Detection với StarDist Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![StarDist](https://img.shields.io/badge/StarDist-0.9.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Mô hình phát hiện và phân đoạn bubble sử dụng StarDist 2D**

[Giới thiệu](#-giới-thiệu) • [Cài đặt](#-cài-đặt) • [Lấy dữ liệu](#-lấy-dữ-liệu) • [Sử dụng](#-sử-dụng)

</div>

---

## 📖 Giới thiệu

Dự án này sử dụng mô hình **StarDist 2D** để phát hiện và phân đoạn các bubble trong ảnh. StarDist là một mô hình deep learning chuyên dụng cho instance segmentation, đặc biệt hiệu quả với các đối tượng có hình dạng tương tự như tế bào hoặc bubble.

### ✨ Tính năng

- 🔍 **Phát hiện bubble tự động**: Phát hiện và đếm số lượng bubble trong ảnh
- 🎯 **Phân đoạn chính xác**: Tách biệt từng bubble riêng lẻ với độ chính xác cao
- 📊 **Phân tích chi tiết**: Tính toán diện tích, tọa độ centroid cho từng bubble
- 🖼️ **Xử lý batch**: Xử lý nhiều ảnh cùng lúc một cách hiệu quả
- 📈 **Visualization**: Hiển thị kết quả với visualization trực quan

### 🏗️ Kiến trúc Model

- **Backbone**: U-Net với 4 tầng (depth=4)
- **Base filters**: 64 filters
- **Rays**: 64 rays cho star-convex polygon
- **Grid**: (2,2) cho tối ưu tốc độ
- **Patch size**: 256x256 pixels
- **Batch size**: 4
- **Learning rate**: 2e-4

### 📊 Kết quả Training

Model đã được train trên dataset với 886 ảnh, chia thành:
- **Training set**: 80% (708 ảnh)
- **Validation set**: 20% (178 ảnh)

**Metrics đạt được:**
- Validation IoU: ~0.9
- Validation Loss: ~0.87

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- CUDA (khuyến nghị cho GPU) - tùy chọn
- RAM: Tối thiểu 8GB (khuyến nghị 16GB+)

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd btl
```

### Bước 2: Tạo virtual environment (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý quan trọng:**
- StarDist yêu cầu `numpy < 2.0.0`. Nếu bạn đã cài numpy 2.x, hãy downgrade:
  ```bash
  pip install "numpy<2" --force-reinstall
  ```

### Bước 4: Kiểm tra cài đặt

```python
import numpy as np
import stardist

print(f"NumPy version: {np.__version__}")  # Phải < 2.0.0
print(f"StarDist version: {stardist.__version__}")
```

---

## 📦 Lấy dữ liệu

Để sử dụng model, bạn cần có:
1. **Model weights** đã được train sẵn
2. **Dataset ảnh** để inference

### 📧 Liên hệ để lấy dữ liệu

Vui lòng liên hệ qua email để nhận:
- Model weights đã train (`weights_best.h5`, `config.json`)
- Dataset ảnh mẫu
- Hướng dẫn chi tiết

**Email:** 📧 [dolam.work@gmail.com](mailto:dolam.work@gmail.com)

---

## 💻 Sử dụng

### 1. Load model đã train sẵn

```python
from stardist.models import StarDist2D
import numpy as np
import imageio.v2 as imageio
from csbdeep.utils import normalize

# Đường dẫn đến thư mục chứa model
MODEL_DIR = "path/to/model/directory"
MODEL_NAME = "stardist_model"  # Tên model của bạn

# Load model
model = StarDist2D(
    config=None,
    name=MODEL_NAME,
    basedir=MODEL_DIR
)

print("✅ Model loaded successfully!")
```

### 2. Inference trên một ảnh

```python
# Đọc ảnh
img_path = "path/to/your/image.png"
img = imageio.imread(img_path)

# Chuyển sang grayscale nếu cần
if img.ndim == 3:
    img = img[..., :3]
    img = np.mean(img, axis=-1)

# Normalize ảnh
img_normalized = normalize(img, 1, 99.8)

# Predict
labels, details = model.predict_instances(img_normalized)

# Số lượng bubble phát hiện được
num_bubbles = labels.max()
print(f"🔍 Phát hiện {num_bubbles} bubble")
```

### 3. Phân tích chi tiết từng bubble

```python
from skimage.measure import regionprops
import matplotlib.pyplot as plt

# Phân tích properties của từng bubble
regions = regionprops(labels)

print("📊 Thông tin từng bubble:")
for i, region in enumerate(regions, start=1):
    area = region.area
    cy, cx = region.centroid
    print(f"Bubble {i}: area={area} px, centroid=({cx:.1f}, {cy:.1f})")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(labels, cmap='jet')
axes[1].set_title(f'Detected Bubbles ({num_bubbles})')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 4. Xử lý batch nhiều ảnh

```python
import os
import glob
import pandas as pd
from tqdm import tqdm

# Thư mục chứa ảnh
IMAGE_DIR = "path/to/images"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lấy danh sách ảnh
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))

# List lưu kết quả
results = []

# Xử lý từng ảnh
for img_path in tqdm(image_paths):
    filename = os.path.basename(img_path)
    
    # Load và normalize
    img = imageio.imread(img_path)
    if img.ndim == 3:
        img = np.mean(img[..., :3], axis=-1)
    img_n = normalize(img, 1, 99.8)
    
    # Predict
    labels, _ = model.predict_instances(img_n)
    
    # Phân tích
    regions = regionprops(labels)
    for region in regions:
        results.append({
            'filename': filename,
            'bubble_id': region.label,
            'area_px': region.area,
            'centroid_x': region.centroid[1],
            'centroid_y': region.centroid[0]
        })
    
    # Lưu mask
    mask_path = os.path.join(OUTPUT_DIR, f"{filename}_mask.png")
    imageio.imwrite(mask_path, labels.astype(np.uint16))

# Lưu kết quả CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
print(f"✅ Đã xử lý {len(image_paths)} ảnh, phát hiện {len(results)} bubble")
```

### 5. Tùy chỉnh threshold (nếu cần)

```python
# Tối ưu threshold cho dataset của bạn
model.optimize_thresholds(
    X_val,  # Validation images
    Y_val   # Validation masks
)

# Hoặc set thủ công
labels, details = model.predict_instances(
    img_normalized,
    prob_thresh=0.5,  # Probability threshold
    nms_thresh=0.4    # Non-maximum suppression threshold
)
```

---

## 📁 Cấu trúc thư mục

```
btl/
├── README.md
├── requirements.txt
├── model1_bubbles.ipynb           # Notebook train model 1 (bubbles dataset)
├── model2_DSB2018.ipynb           # Notebook train model 2 (DSB2018 dataset)
├── model1_bubbles/                # Model 1 weights
│   ├── config.json
│   ├── weights_best.weights.h5
│   └── weights_last.h5
├── model2_DSB20018/               # Model 2 weights
│   └── stardist_model/
│       ├── config.json
│       ├── weights_best.h5
│       └── weights_last.h5
└── result_model1_bubbles/         # Kết quả inference
```

---

## 🔧 Troubleshooting

### Lỗi: NumPy version conflict

```bash
# Giải pháp: Downgrade numpy
pip install "numpy<2" --force-reinstall
```

### Lỗi: CUDA out of memory

- Giảm `train_batch_size` trong config
- Giảm `train_patch_size` (ví dụ: từ 256x256 xuống 128x128)
- Sử dụng CPU nếu GPU không đủ memory

### Lỗi: Model không load được

- Kiểm tra đường dẫn đến thư mục model
- Đảm bảo có đủ các file: `config.json`, `weights_best.h5`
- Kiểm tra tên model (`name`) phải khớp với tên thư mục

---

## 👤 Tác giả

**Nguyễn Thạc Anh**

**Lại Quốc Đạt**

**Trần Đức Lợi**

**Đỗ Ngọc Lâm**
- 📧 Email: [dolam.work@gmail.com](mailto:dolam.work@gmail.com) / [lamdn.b22cn476@stu.ptit.edu.vn](mailto:lamdn.b22cn476@stu.ptit.edu.vn)
- 🏫 Trường: PTIT
- 📚 Môn học: Xử lý ảnh

---

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

<div align="center">

**⭐ Nếu project này hữu ích, hãy star repo này! ⭐**

Made with ❤️ by Nguyễn Thạc Anh, Lại Quốc Đạt, Trần Đức Lợi, Đỗ Ngọc Lâm

</div>

