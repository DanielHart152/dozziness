#!/bin/bash
# Fixed installation script for Raspberry Pi

echo "=========================================="
echo "Driver Drowsiness Detection - Installation"
echo "=========================================="
echo ""

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-dev cmake build-essential
sudo apt install -y libopencv-dev python3-opencv

# Try to install BLAS/LAPACK (may have different names)
echo "Installing BLAS/LAPACK libraries..."
sudo apt install -y liblapack-dev libblas-dev || echo "Note: Some packages not available"
sudo apt install -y libatlas-base-dev 2>/dev/null || sudo apt install -y libopenblas-dev 2>/dev/null || echo "Note: Using system BLAS"

# Install Python packages
echo "Installing Python packages..."
pip3 install numpy opencv-python RPi.GPIO

# Try to install dlib (may take a while)
echo ""
echo "Installing dlib (this may take 10-15 minutes)..."
echo "If this fails, the system will use OpenCV fallback."
pip3 install dlib || echo "Warning: dlib installation failed. System will use OpenCV fallback."

# Download shape predictor (optional)
echo ""
read -p "Download dlib shape predictor file? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Downloading shape predictor..."
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
    echo "Shape predictor downloaded!"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect USB cameras"
echo "2. Check camera indices: ls -l /dev/video*"
echo "3. Update config.json with camera indices"
echo "4. Connect buzzer to GPIO 18 (optional)"
echo "5. Run: python3 main.py"
echo ""

