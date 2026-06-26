#!/bin/bash

sudo swapoff -a
sudo modprobe zram
sudo zramctl --size 4G /dev/zram0
sudo mkswap /dev/zram0
sudo swapon /dev/zram0
sudo ip addr add 192.168.0.$1/24 dev eth0
sudo ip link set eth0 up
