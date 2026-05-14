#!/bin/bash
# Setup script for Ubuntu/Debian EC2 instances

echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv libgl1 libglib2.0-0 libgles2 curl

echo "Installing Node.js and PM2..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2

echo "Creating Python virtual environment..."
python3 -m venv venv
./venv/bin/pip install --upgrade pip

echo "Starting the automation script via PM2..."
# Start the script using the ecosystem configuration
pm2 start ecosystem.config.js

echo "Saving PM2 process list to restart automatically on EC2 reboot..."
pm2 save
sudo pm2 startup | grep "sudo env" | bash

echo "=========================================================="
echo "Setup Complete!"
echo "To view live logs, run: pm2 logs gopro-s3-automation"
echo "To check status, run: pm2 status"
echo "=========================================================="
