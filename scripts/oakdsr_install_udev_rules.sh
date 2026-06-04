#!/usr/bin/env bash
set -euo pipefail

rule_file="/etc/udev/rules.d/80-oak-d-sr-depthai.rules"

sudo tee "$rule_file" >/dev/null <<'RULE'
SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"
RULE

sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Installed $rule_file"
echo "Unplug and reconnect the OAK-D SR, then run scripts/oakdsr_connect_check.py again."
