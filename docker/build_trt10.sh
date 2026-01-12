#!/bin/bash
# BEVFormer TensorRT 10 Docker build and run script

set -e

# Configuration
IMAGE_NAME="bevformer-trt10"
CONTAINER_NAME="bevformer_trt10"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}BEVFormer TensorRT 10 build script${NC}"
echo "Workspace: $WORKSPACE_DIR"

# Check for TensorRT 10 deb packages
TENSORRT_DEB_DIR="/var/nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-12.8"
if [ ! -d "$TENSORRT_DEB_DIR" ]; then
    echo -e "${YELLOW}Warning: TensorRT 10 deb directory not found${NC}"
    echo "Please download TensorRT 10.x for Ubuntu 22.04 + CUDA 12.x from NVIDIA"
    echo "Download: https://developer.nvidia.com/tensorrt"
    echo "Or set a custom path: export TENSORRT_DEB_DIR=/path/to/tensorrt"
fi

# Build the Docker image
echo -e "${GREEN}Step 1: Building Docker image${NC}"
docker build \
    -t $IMAGE_NAME:latest \
    -f docker/Dockerfile.trt10 \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully${NC}"
else
    echo -e "${RED}✗ Image build failed${NC}"
    exit 1
fi

# Stop and remove any previous container
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo -e "${YELLOW}Stopping and removing existing container: $CONTAINER_NAME${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Start the container
echo -e "${GREEN}Step 2: Starting container${NC}"
docker run -itd \
    --name $CONTAINER_NAME \
    --gpus all \
    --shm-size=16g \
    -v $WORKSPACE_DIR:/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -w /workspace \
    $IMAGE_NAME:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Container started successfully${NC}"
    echo ""
    echo "Container name: $CONTAINER_NAME"
    echo "Enter container: docker exec -it $CONTAINER_NAME bash"
    echo ""
    
    # Install TensorRT inside container if deb packages are available
    if [ -d "$TENSORRT_DEB_DIR" ]; then
        echo -e "${GREEN}Step 3: Installing TensorRT 10 inside container${NC}"
        docker cp $TENSORRT_DEB_DIR $CONTAINER_NAME:/tmp/tensorrt_debs
        docker exec $CONTAINER_NAME bash -c "cd /tmp/tensorrt_debs && dpkg -i *.deb 2>&1 | grep -E '(Setting up|Errors)'"
        echo -e "${GREEN}✓ TensorRT installation completed${NC}"
    else
        echo -e "${YELLOW}Skipping TensorRT installation (deb packages not found)${NC}"
        echo "To install manually: docker exec -it $CONTAINER_NAME bash"
        echo "Then run: apt-get install tensorrt"
    fi
    
    # Build TensorRT plugins
    echo -e "${GREEN}Step 4: Building TensorRT plugins inside container${NC}"
    docker exec $CONTAINER_NAME bash -c "cd /workspace/TensorRT && rm -rf build && mkdir -p build && cd build && cmake .. -DCMAKE_TENSORRT_PATH=/usr/src/tensorrt -DCUDA_VERSION=12.1 && make -j\$(nproc)"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ TensorRT plugins built successfully${NC}"
    else
        echo -e "${YELLOW}! TensorRT plugin build failed, please check manually${NC}"
    fi
    
    # Verify environment
    echo -e "${GREEN}Step 5: Verifying environment${NC}"
    docker exec $CONTAINER_NAME python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
    docker exec $CONTAINER_NAME python3 -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')" 2>/dev/null || echo -e "${YELLOW}TensorRT Python bindings not installed${NC}"
    docker exec $CONTAINER_NAME nvcc --version | grep "release"
    
else
    echo -e "${RED}✗ Failed to start container${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Environment ready${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Enter container: docker exec -it $CONTAINER_NAME bash"
echo "2. Export ONNX: cd /workspace && bash samples/bevformer/tiny/pth2onnx.sh"
echo "3. Convert to TRT: cd /workspace && bash samples/bevformer/tiny/onnx2trt.sh"
echo ""
