#!/bin/bash
# Start virtual framebuffer
Xvfb :0 -screen 0 1920x1080x24 &
export DISPLAY=:0
sleep 2

# Start XFCE as the scraper user
su - scraper -c "export DISPLAY=:0 && /usr/bin/xfwm4 &"
sleep 1

su - scraper -c "export DISPLAY=:0 && /usr/bin/xfce4-panel &"
sleep 1

# Start x11vnc server with password
x11vnc -forever -passwd 312 -display :0 -shared &
sleep 2

# Start websockify to proxy VNC to web (noVNC)
echo "Starting websockify with noVNC at: /opt/novnc"
python3 -m websockify --web=/opt/novnc 6080 localhost:5900
