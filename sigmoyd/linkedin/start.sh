#!/bin/bash
# Start virtual framebuffer
Xvfb :0 -screen 0 1920x1080x24 &
export DISPLAY=:0
sleep 2

# Start a simple window manager (xfwm4 is part of xfce4)
/usr/bin/xfwm4 &
sleep 1

# Start xfce4-panel for a basic desktop feel
/usr/bin/xfce4-panel &
sleep 1

# Create VNC password if it doesn't exist
if [ ! -f /root/.vnc/passwd ]; then
    mkdir -p /root/.vnc
    echo "1234" | x11vnc -storepasswd /root/.vnc/passwd
fi

# Start x11vnc server (or use -passwd for direct password)
x11vnc -forever -passwd 1234 -display :0 -shared &
sleep 2

# Start websockify to proxy VNC to web (noVNC)
echo "Starting websockify with noVNC at: /opt/novnc"
python3 -m websockify --web=/opt/novnc 6080 localhost:5900
