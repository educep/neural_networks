# run it:
# .\ws_install_venv.ps1 -envName nn

# Get the environment name from the command line
param (
    [string]$envName
)

# Check if environment name is provided
if (-not $envName) {
    Write-Output "Please provide an environment name using --env <name>"
    exit
}


# Get the current username
$username = $env:USERNAME

# Define the path to the virtual environment based on the current user
$envPath = "C:/Users/$username/envs/$envName"

# Check if the virtual environment folder exists
if (Test-Path $envPath) {
    Write-Output "Virtual environment already exists. Skipping creation."
} else {
    # Create a virtual environment with uv
    uv venv $envPath
}

# Activate the virtual environment
& "$envPath/Scripts/Activate.ps1"

# Install dependencies using uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

Write-Output "Virtual environment '$envName' has been set up with uv."
