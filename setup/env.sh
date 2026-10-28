# Source this in every new container shell:   source setup/env.sh
# (setup/05_save_env.sh can add it to ~/.bashrc for you)

# where this project lives (the folder that contains setup/, accel/, sw/ ...)
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# third-party sources and builds live in the container filesystem, not on the
# Windows-mounted /workspace (faster builds, executable bits preserved)
export EXT=${EXT:-/opt/soc-ext}
export STONNE_DIR=$EXT/stonne/stonne
export VPPP_DIR=$EXT/riscv-vp-plusplus
export TINY_DIR=$EXT/tiny

# the course SystemC (already set in the systemc-env image; repeated for safety)
export SYSTEMC_HOME=/opt/systemc
export CPLUS_INCLUDE_PATH=$SYSTEMC_HOME/include
export LIBRARY_PATH=$SYSTEMC_HOME/lib
export LD_LIBRARY_PATH=$SYSTEMC_HOME/lib

export PATH=$VPPP_DIR/vp/build/bin:$STONNE_DIR:$PATH
