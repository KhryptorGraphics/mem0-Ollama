# mem0-Ollama Installation Script for Windows

# Ensure script runs with admin privileges
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "Please run this script as Administrator!"
    exit
}

Write-Host "=== mem0-Ollama Installation Script for Windows ===" -ForegroundColor Green
Write-Host "This script will install mem0-Ollama and its dependencies." -ForegroundColor Cyan
Write-Host ""

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { return $true } }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Check for Python installation
if (-Not (Test-CommandExists python)) {
    Write-Host "Python not found. Installing Python..." -ForegroundColor Yellow
    
    # Download Python installer
    $pythonUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller
        
        # Install Python with pip and add to PATH
        Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0" -Wait
        
        # Remove installer
        Remove-Item $pythonInstaller
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        Write-Host "Python installed successfully." -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to install Python. Please install it manually from https://www.python.org/downloads/" -ForegroundColor Red
        exit
    }
}
else {
    Write-Host "Python is already installed." -ForegroundColor Green
}

# Check for Git installation
if (-Not (Test-CommandExists git)) {
    Write-Host "Git not found. Installing Git..." -ForegroundColor Yellow
    
    # Download Git installer
    $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/Git-2.44.0-64-bit.exe"
    $gitInstaller = "$env:TEMP\git-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $gitUrl -OutFile $gitInstaller
        
        # Install Git
        Start-Process -FilePath $gitInstaller -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
        
        # Remove installer
        Remove-Item $gitInstaller
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        Write-Host "Git installed successfully." -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to install Git. Please install it manually from https://git-scm.com/download/win" -ForegroundColor Red
        exit
    }
}
else {
    Write-Host "Git is already installed." -ForegroundColor Green
}

# Check for Ollama installation
if (-Not (Test-CommandExists ollama)) {
    Write-Host "Ollama not found. Please install Ollama..." -ForegroundColor Yellow
    Write-Host "You can download it from: https://ollama.com/download/windows" -ForegroundColor Cyan
    $installOllama = Read-Host "Do you want to continue without installing Ollama? (y/n)"
    if ($installOllama -ne "y") {
        exit
    }
}
else {
    Write-Host "Ollama is already installed." -ForegroundColor Green
}

# Check for Docker Desktop installation
if (-Not (Test-CommandExists docker)) {
    Write-Host "Docker not found. Please install Docker Desktop..." -ForegroundColor Yellow
    Write-Host "You can download it from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
    $installDocker = Read-Host "Do you want to continue without installing Docker? (y/n)"
    if ($installDocker -ne "y") {
        exit
    }
}
else {
    Write-Host "Docker is already installed." -ForegroundColor Green
}

# Create project directory
$projectDir = "$env:USERPROFILE\mem0-ollama"
if (-Not (Test-Path $projectDir)) {
    Write-Host "Creating project directory at $projectDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $projectDir | Out-Null
}

# Clone the repository
Write-Host "Cloning mem0-Ollama repository..." -ForegroundColor Yellow
if (Test-Path "$projectDir\.git") {
    Write-Host "Repository already exists. Pulling latest changes..." -ForegroundColor Cyan
    Set-Location -Path $projectDir
    git pull
}
else {
    git clone https://github.com/KhryptorGraphics/mem0-ollama.git $projectDir
    Set-Location -Path $projectDir
}

# Create a Python virtual environment
Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "$projectDir\venv")) {
    python -m venv "$projectDir\venv"
}

# Activate the virtual environment and install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
$venvActivate = "$projectDir\venv\Scripts\Activate.ps1"
. $venvActivate
pip install -r "$projectDir\requirements.txt"

# Create a startup script for direct mode
Write-Host "Creating startup scripts..." -ForegroundColor Yellow
$startScript = @"
# mem0-Ollama Startup Script
`$scriptPath = Split-Path -Parent `$MyInvocation.MyCommand.Path
Set-Location -Path `$scriptPath
. .\venv\Scripts\Activate.ps1
python main.py
"@

Set-Content -Path "$projectDir\start.ps1" -Value $startScript

# Create a startup script for Docker
$dockerScript = @"
# mem0-Ollama Docker Startup Script
`$scriptPath = Split-Path -Parent `$MyInvocation.MyCommand.Path
Set-Location -Path `$scriptPath
docker-compose up -d
"@

Set-Content -Path "$projectDir\start_docker.ps1" -Value $dockerScript

# Create a batch file for easy startup
$batchScript = @"
@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"
"@

Set-Content -Path "$projectDir\start.bat" -Value $batchScript

# Create a batch file for Docker startup
$dockerBatchScript = @"
@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0start_docker.ps1"
"@

Set-Content -Path "$projectDir\start_docker.bat" -Value $dockerBatchScript

# Create desktop shortcuts
Write-Host "Creating desktop shortcuts..." -ForegroundColor Yellow
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\mem0-Ollama.lnk")
$Shortcut.TargetPath = "$projectDir\start.bat"
$Shortcut.WorkingDirectory = $projectDir
$Shortcut.Description = "Start mem0-Ollama"
$Shortcut.Save()

$DockerShortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\mem0-Ollama (Docker).lnk")
$DockerShortcut.TargetPath = "$projectDir\start_docker.bat"
$DockerShortcut.WorkingDirectory = $projectDir
$DockerShortcut.Description = "Start mem0-Ollama with Docker"
$DockerShortcut.Save()

Write-Host ""
Write-Host "=== Installation Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start mem0-Ollama:" -ForegroundColor Cyan
Write-Host "1. Use the desktop shortcuts" -ForegroundColor White
Write-Host "2. Direct mode: $projectDir\start.bat" -ForegroundColor White
Write-Host "3. Docker mode: $projectDir\start_docker.bat" -ForegroundColor White
Write-Host ""
Write-Host "Then open your browser at http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: If using Docker, make sure Ollama is running before starting the container." -ForegroundColor Yellow
Write-Host "You can start Ollama by searching for it in the Start menu." -ForegroundColor Yellow
Write-Host ""
