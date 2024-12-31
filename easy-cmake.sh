#! /bin/bash
# 是否启用代码中的计时
WITH_CLOCKING=$1    
cd build
cmake .. -D WITH_CLOCKING=${WITH_CLOCKING}
make -j12 
make install
