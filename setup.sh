#!/usr/bin/env bash

set -e

echo "----------------------------------------"
echo "Setting up FAIR-Checker API Integration Pipeline environment"
echo "----------------------------------------"

# -----------------------------
# Optional clean flag
# -----------------------------
CLEAN=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=true
    echo "Clean flag detected. Previous run artifacts will be removed."
fi

# -----------------------------
# Check Python installation
# -----------------------------
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Enforce minimum Python version (3.10)
PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "Python 3.10+ required. Detected version: $PY_VERSION"
    exit 1
fi

echo "Using Python $PY_VERSION"

# -----------------------------
# Clean previous run artifacts
# -----------------------------
if [ "$CLEAN" = true ]; then
    echo "Resetting Outbound run directories (removing child folders/files)..."

    for d in "Outbound/Logging" \
             "Outbound/Sampling" \
             "Outbound/Raw Data" \
             "Outbound/Results" \
             "Outbound/Error Record Reports"
    do
        if [ -d "$d" ]; then
            echo "Cleaning: $d"
            find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        else
            echo "Creating: $d"
            mkdir -p "$d"
        fi
    done

    echo "Outbound reset complete."
fi

# -----------------------------
# Create virtual environment
# -----------------------------
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# Activate environment
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# -----------------------------
# Install dependencies
# -----------------------------
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
    exit 1
fi

# -----------------------------
# Create expected directory structure
# -----------------------------
echo "Ensuring expected directory structure exists..."

mkdir -p "Inbound"
mkdir -p "Outbound/Logging"
mkdir -p "Outbound/Sampling"
mkdir -p "Outbound/Raw Data"
mkdir -p "Outbound/Results"
mkdir -p "Outbound/Error Record Reports"

# -----------------------------
# Make run script executable
# -----------------------------
if [ -f "run.sh" ]; then
    chmod +x run.sh
    echo "run.sh made executable."
fi

chmod +x setup.sh

echo ""
echo "----------------------------------------"
echo "Setup complete."
echo ""
echo "Next steps:"
echo "1. Place your Excel input file inside:"
echo "   Inbound/"
echo ""
echo "2. Run the pipeline:"
echo "   ./run.sh"
echo "----------------------------------------"