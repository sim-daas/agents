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
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed selenium requests beautifulsoup4 websockify

# Create non-root user with password
RUN useradd -m -s /bin/bash -G sudo scraper && \
    echo "scraper:312" | chpasswd

# Set up VNC password for the user
RUN mkdir -p /home/scraper/.vnc && \
    echo "312" | x11vnc -storepasswd - /home/scraper/.vnc/passwd && \
    chown -R scraper:scraper /home/scraper/.vnc

# Download and install noVNC
RUN mkdir -p /opt/novnc && \
    wget -qO- https://github.com/novnc/noVNC/archive/refs/tags/v1.4.0.tar.gz | tar xz --strip-components=1 -C /opt/novnc && \
    ln -s /opt/novnc/vnc.html /opt/novnc/index.html

# Expose VNC and web VNC ports
EXPOSE 5900 6080

# Make start.sh executable and set as entrypoint
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Switch to scraper user
USER scraper
WORKDIR /home/scraper

ENTRYPOINT ["/start.sh"]
