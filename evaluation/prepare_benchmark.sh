#!/bin/bash
# Prepare benchmark datasets for OpenCustom evaluation
set -e

DATA_DIR=${1:-./benchmark_data}
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  OpenCustom Benchmark Data Preparation"
echo "============================================"

# --- COCO val2017 ---
echo ""
echo "[1/3] Downloading COCO val2017..."
if [ ! -d "$DATA_DIR/coco/val2017" ]; then
    mkdir -p "$DATA_DIR/coco"
    wget -q --show-progress -O "$DATA_DIR/coco/val2017.zip" \
        http://images.cocodataset.org/zips/val2017.zip
    unzip -q "$DATA_DIR/coco/val2017.zip" -d "$DATA_DIR/coco/"
    rm "$DATA_DIR/coco/val2017.zip"
    echo "COCO val2017 downloaded to $DATA_DIR/coco/val2017/"
else
    echo "COCO val2017 already exists, skipping."
fi

# --- DreamBooth ---
echo ""
echo "[2/3] Downloading DreamBooth dataset..."
if [ ! -d "$DATA_DIR/dreambooth" ]; then
    git clone https://github.com/google/dreambooth.git "$DATA_DIR/dreambooth_repo"
    mkdir -p "$DATA_DIR/dreambooth"
    # Copy dataset images
    if [ -d "$DATA_DIR/dreambooth_repo/dataset" ]; then
        cp -r "$DATA_DIR/dreambooth_repo/dataset/"* "$DATA_DIR/dreambooth/"
    fi
    echo "DreamBooth dataset downloaded to $DATA_DIR/dreambooth/"
else
    echo "DreamBooth dataset already exists, skipping."
fi

# --- ImageNet ---
echo ""
echo "[3/3] ImageNet (ILSVRC2012) validation set"
echo "  ImageNet requires manual download with academic credentials."
echo "  1. Register at https://image-net.org/download-images.php"
echo "  2. Download ILSVRC2012_img_val.tar"
echo "  3. Extract to: $DATA_DIR/imagenet/val/"
echo "     mkdir -p $DATA_DIR/imagenet/val"
echo "     tar -xf ILSVRC2012_img_val.tar -C $DATA_DIR/imagenet/val/"

echo ""
echo "============================================"
echo "  Done! Dataset locations:"
echo "    COCO:      $DATA_DIR/coco/val2017/"
echo "    DreamBooth: $DATA_DIR/dreambooth/"
echo "    ImageNet:   $DATA_DIR/imagenet/val/ (manual)"
echo "============================================"
