# Use Debian stable as base image
FROM debian:stable-slim

# Install system dependencies
RUN apt-get update && \
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
    && apt-get remove -y python3-urllib3 \
    && rm -rf /var/lib/apt/lists/*

# Install Selenium and other Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed selenium requests beautifulsoup4 websockify fastapi uvicorn

# Create non-root user with password
RUN useradd -m -s /bin/bash -G sudo admin && \
    echo "admin:312" | chpasswd

# Set up VNC password for the user
RUN mkdir -p /home/admin/.vnc && \
    echo "312" | x11vnc -storepasswd - /home/admin/.vnc/passwd && \
    chown -R admin:admin /home/admin/.vnc

# Download and install noVNC
RUN mkdir -p /opt/novnc && \
    wget -qO- https://github.com/novnc/noVNC/archive/refs/tags/v1.4.0.tar.gz | tar xz --strip-components=1 -C /opt/novnc && \
    ln -s /opt/novnc/vnc.html /opt/novnc/index.html

# Expose VNC and web VNC ports
EXPOSE 5900 6080

# Make start.sh executable and set as entrypoint
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Switch to admin user
USER admin
WORKDIR /home/admin

ENTRYPOINT ["/start.sh"]
