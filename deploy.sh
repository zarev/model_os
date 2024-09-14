#!/bin/bash

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command succeeded
check_command() {
    if [ $? -ne 0 ]; then
        log "Error: $1"
        exit 1
    fi
}

# Update and upgrade the system
log "Updating and upgrading the system..."
sudo apt-get update
check_command "Failed to update package list"
sudo apt-get upgrade -y
check_command "Failed to upgrade packages"

# Install Python and pip
log "Installing Python and pip..."
sudo apt-get install -y python3 python3-pip
check_command "Failed to install Python and pip"

# Install system dependencies
log "Installing system dependencies..."
sudo apt-get install -y libsndfile1-dev ffmpeg
check_command "Failed to install system dependencies"

# Clone the repository (assuming it's hosted on GitHub)
log "Cloning the repository..."
git clone https://github.com/your-username/your-repo-name.git
check_command "Failed to clone the repository"
cd your-repo-name
check_command "Failed to change directory"

# Install Python dependencies
log "Installing Python dependencies..."
pip3 install -r requirements.txt
check_command "Failed to install Python dependencies"

# Check for required environment variables
log "Checking environment variables..."
required_vars=("PGHOST" "PGUSER" "PGPASSWORD" "PGDATABASE" "PGPORT")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        log "Error: $var is not set. Please set it before running this script."
        exit 1
    fi
done

# Set up systemd service
log "Setting up systemd service..."
sudo tee /etc/systemd/system/fastapi-app.service > /dev/null <<EOT
[Unit]
Description=FastAPI Application
After=network.target

[Service]
User=$USER
WorkingDirectory=$PWD
ExecStart=$(which uvicorn) main:app --host 0.0.0.0 --port 8000
Restart=always
Environment="PGHOST=${PGHOST}"
Environment="PGUSER=${PGUSER}"
Environment="PGPASSWORD=${PGPASSWORD}"
Environment="PGDATABASE=${PGDATABASE}"
Environment="PGPORT=${PGPORT}"
Environment="DATABASE_URL=postgresql://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}"

[Install]
WantedBy=multi-user.target
EOT
check_command "Failed to create systemd service file"

# Reload systemd and start the service
log "Starting the FastAPI application..."
sudo systemctl daemon-reload
check_command "Failed to reload systemd"
sudo systemctl enable fastapi-app
check_command "Failed to enable FastAPI application service"
sudo systemctl start fastapi-app
check_command "Failed to start FastAPI application service"

log "Deployment completed successfully!"
