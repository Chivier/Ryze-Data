#!/bin/bash

set -e  # Exit on error

echo "=== Ryze-Data Installation Script ==="
echo ""

# Step 1: Check Python version & venv
echo "Step 1: Checking Python version and virtual environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

echo "Found Python version: $PYTHON_VERSION"

# Check minimum Python version (3.10)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not running in a virtual environment."
    echo "It's recommended to use a virtual environment for this installation."
    read -p "Do you want to continue without a virtual environment? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled. Please activate a virtual environment and run this script again."
        echo "To create a virtual environment: python3 -m venv venv"
        echo "To activate it: source venv/bin/activate"
        exit 1
    fi
else
    echo "Virtual environment detected: $VIRTUAL_ENV"
fi

# Confirm Python executable
PYTHON_EXEC=$(which python3)
echo "Using Python executable: $PYTHON_EXEC"
read -p "Is this the correct Python installation? (Y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Installation cancelled. Please ensure the correct Python is in your PATH."
    exit 1
fi

echo " Python version and environment check complete"
echo ""

# Step 2: Install marker
echo "Step 2: Installing marker..."

# Check if pip is installed
if ! python3 -m pip --version &> /dev/null; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

# Check Python version is 3.10+
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10 or higher is required for marker. Current version: $PYTHON_VERSION"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch (required dependency)
echo "Installing PyTorch..."
python3 -m pip install torch torchvision

# Install marker
echo "Installing marker-pdf..."
read -p "Do you need support for documents other than PDFs (DOCX, PPTX, etc.)? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing marker with full document support..."
    python3 -m pip install "marker-pdf[full]"
else
    echo "Installing marker for PDF conversion only..."
    python3 -m pip install marker-pdf
fi

echo " Marker installation complete"
echo ""

# Step 3: Check CLI command
echo "Step 3: Verifying marker CLI command..."

# Check if marker command is available
if command -v marker &> /dev/null; then
    echo " marker command is available"
    marker --version 2>/dev/null || echo "Note: marker may not have a --version flag"
else
    echo "Warning: marker command not found in PATH"
    echo "Checking if it's installed as a Python module..."
    
    if python3 -c "import marker" 2>/dev/null; then
        echo " marker is installed as a Python module"
        echo "You may need to run it using: python3 -m marker"
    else
        echo "Error: marker installation verification failed"
        echo "Please check the installation logs above for errors"
        exit 1
    fi
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. If not already done, activate your virtual environment"
echo "2. Run 'marker' command to start using the tool"
echo "3. Check the documentation at https://github.com/datalab-to/marker for usage"