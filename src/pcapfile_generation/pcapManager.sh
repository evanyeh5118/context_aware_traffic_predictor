#!/bin/bash 

# Teleoperation (Audio) using UE1 port 5001
python3 modify_pcap.py \
--sensation audio \
--application teleoperation \
--new_src_ip 192.168.72.135 \
--new_dst_ip 12.1.1.4 \
--new_src_mac 6e:59:49:05:29:9d \
--new_dst_mac 2a:04:5a:16:fa:7d \
--new_src_port 1234 \
--new_dst_port 5001

# Teleoperation (Video) using UE2 port 5002
python3 modify_pcap.py \
--sensation video \
--application teleoperation \
--new_src_ip 192.168.72.135 \
--new_dst_ip 12.1.1.3 \
--new_src_mac 6e:59:49:05:29:9d \
--new_dst_mac 2a:04:5a:16:fa:7d \
--new_src_port 1235 \
--new_dst_port 5002

# Teleoperation (Haptic) using UE3 port 5003
python3 modify_pcap.py \
--sensation haptic \
--application teleoperation \
--new_src_ip 192.168.72.135 \
--new_dst_ip 12.1.1.2 \
--new_src_mac 6e:59:49:05:29:9d \
--new_dst_mac 2a:04:5a:16:fa:7d \
--new_src_port 1236 \
--new_dst_port 5003
