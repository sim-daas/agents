#!/bin/bash
set -eux
apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools \
    firefox-esr \
    xfce4 xfce4-goodies xfce4-session \
    x11vnc xvfb \
    wget ca-certificates \
    supervisor \
    dbus-x11 \
    fonts-dejavu \
    novnc websockify \
    git

apt-get remove -y python3-urllib3 || true
apt-get clean
rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir --break-system-packages --ignore-installed selenium requests beautifulsoup4 websockify fastapi uvicorn

mkdir -p /home/admin/.vnc
echo "312" | x11vnc -storepasswd - /home/admin/.vnc/passwd
chown -R admin:admin /home/admin/.vnc

mkdir -p /opt/novnc
wget -qO- https://github.com/novnc/noVNC/archive/refs/tags/v1.4.0.tar.gz | tar xz --strip-components=1 -C /opt/novnc
ln -sf /opt/novnc/vnc.html /opt/novnc/index.html

git clone https://github.com/sim-daas/agents /home/admin/agents || true
