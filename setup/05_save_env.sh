#!/usr/bin/env bash
# Step 0.5 - make every new shell in this container load setup/env.sh,
# then (in PowerShell, on Windows) freeze the container as an image:
#     docker commit soc-accel soc-accel-ready
#   bash setup/05_save_env.sh
set -e
ENV_FILE="$(cd "$(dirname "$0")" && pwd)/env.sh"
LINE="source \"$ENV_FILE\""
grep -qxF "$LINE" ~/.bashrc 2>/dev/null || echo "$LINE" >> ~/.bashrc
echo "added to ~/.bashrc: $LINE"
echo "now, in PowerShell on Windows:  docker commit soc-accel soc-accel-ready"
