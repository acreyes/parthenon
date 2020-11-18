#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/.bashrc

# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

# Ensure the submodules are pulled in
git submodule update --init

cmake -S. -Bbuild

cmake --build build

cd build

ctest 

exit 0
