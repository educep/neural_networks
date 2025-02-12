#!/bin/bash

# Usage:
# ./ws_install_venv.sh -e nn

# Parse command-line arguments
while getopts e: flag
do
    case "${flag}" in
        e) envName=${OPTARG};;
    esac
done

# Check if environment name is provided
if [ -z "$envName" ]; then
    echo "Please provide an environment name using -e <name>"
    exit 1
fi

# Check if 'uv' is installed
if command -v uv &> /dev/null; then
    echo "uv is already installed (Version: $(uv --version))."
else
    echo "uv is not installed. Installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Verify installation
    if command -v uv &> /dev/null; then
        echo "uv installed successfully (Version: $(uv --version))."
    else
        echo "uv installation failed. Please install manually."
        exit 1
    fi
fi

# Get the current username
username=$(whoami)

# Define the path to the virtual environment based on the current user
envPath="/home/$username/envs/$envName"

# Check if the virtual environment folder exists
if [ -d "$envPath" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    # Create a virtual environment with uv
    echo "Creating virtual environment '$envName'..."
    uv venv "$envPath"
fi

# Activate the virtual environment
source "$envPath/bin/activate"

# Install dependencies using uv
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Virtual environment '$envName' has been set up with uv."
