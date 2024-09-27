# Use an Ubuntu base image
FROM ubuntu:latest

# Install required packages and Termux
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    build-essential \
    cmake \
    unzip \
    openssh-server \
    termux \
    && rm -rf /var/lib/apt/lists/*

# Install additional packages as per the documentation
RUN apt-get update && apt-get install -y \
    x11-repo \
    termux-x11-nightly \
    pulseaudio \
    xfce4 \
    tur-repo \
    && rm -rf /var/lib/apt/lists/*

# Clone the Termux Desktop repository
RUN git clone https://github.com/LinuxDroidMaster/Termux-Desktops.git /termux-desktop

# Setup Termux Desktop
WORKDIR /termux-desktop
RUN bash native_install.sh

# Expose the ports for SSH
EXPOSE 33
EXPOSE 3333

# Start SSH service
CMD ["/usr/sbin/sshd", "-D"]