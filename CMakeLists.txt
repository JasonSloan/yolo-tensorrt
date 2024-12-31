cmake_minimum_required(VERSION 3.0)
project(mainproject)
set(lib_name obdet)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

# 是否启用代码中的计时功能
if(WITH_CLOCKING STREQUAL "ON")
    add_definitions(-DWITH_CLOCKING)
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
set(CPP_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64 -fPIC -fopenmp -pthread -fpermissive)
set(NVCC_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64)

# 设置OpenCV_HOME以及要链接的OpenCV库名
set(OpenCV_HOME opencv-build)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)

# 设置SPDLOG_HOME(纯头文件格式)
set(SPDLOG_HOME include/spdlog-build)

# 设置EIGEN_HOME(纯头文件格式)
set(EIGEN_HOME include/eigen-build)

# 设置CUDA_HOME以及要链接的CUDA的库名
set(CUDA_HOME /usr/local/cuda-12)
set(CUDA_LIBS cudart cudnn nvinfer nvinfer_plugin)

# 设置nvcc路径
set(NVCC_PATH ${CUDA_HOME}/bin)
# 设置cu文件编译出的静态库的名字
set(nvcc_target_lib nvtmp)       # nvtmp也就是静态库libnvtmp.a中间的名字
set(nvcc_target_lib_full_name libnvtmp.a)

# 头文件寻找路径
include_directories(include ${OpenCV_HOME}/include/opencv4 ${EIGEN_HOME}/include/eigen3 ${CUDA_HOME}/include ${SPDLOG_HOME})
# 给cmake指定include_directories并不会让nvcc也去这里找, 所以还要再指定一遍
set(NVCC_INCLUDE_DIRS -I${OpenCV_HOME}/include/opencv4 -I${CUDA_HOME}/include)
set(NVCC_CXX_FLAGS ${NVCC_CXX_FLAGS} ${NVCC_INCLUDE_DIRS})

# 库文件寻找路径
# 指定库文件寻找路径包括${CMAKE_BINARY_DIR}是因为nvcc编译生成的临时静态库libnvtmp.a会产生在这里
link_directories(${OpenCV_HOME}/lib ${CUDA_HOME}/lib64 ${CMAKE_BINARY_DIR}) # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake ..命令的路径(也就是build文件夹)

# 添加源文件
file(GLOB_RECURSE SRC "src/*.cpp")
file(GLOB_RECURSE cu_files "src/*.cu")

# 设置输出库文件
add_library(${lib_name} SHARED ${SRC})
target_link_libraries(${lib_name} ${OpenCV_LIBS} ${CUDA_LIBS} ${nvcc_target_lib})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 设置库文件安装路径
install(DIRECTORY include/ DESTINATION /home/ai/include CONFIGURATIONS ${CMAKE_BUILD_TYPE})
install(DIRECTORY lib/ DESTINATION /home/ai/lib CONFIGURATIONS ${CMAKE_BUILD_TYPE})
install(DIRECTORY lib/ DESTINATION /root/test-model-infer/lib CONFIGURATIONS ${CMAKE_BUILD_TYPE})

# 添加pre-build和post-build，使用dlcc并指定相应的编译选项编译出obj文件并在之后删除这些obj文件
foreach (cu_file ${cu_files})
    set(tmp ${cu_file}.o)
    # CMAKE_SOURCE_DIR: cmake中的预置变量, 存储最顶层的CMakeLists.txt文件的目录的路径
    # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake命令的路径
    # 该行命令实现的是将tmp变量对应的string中的CMAKE_SOURCE_DIR的路径替换为CMAKE_BINARY_DIR
    string(REPLACE ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} cu_obj "${tmp}")
    string(REGEX MATCH "/.*/" cu_dir ${cu_obj}) 
    # message("tmp: " ${tmp})             # /path/to/your/kernelfunction/dir/kernelfunction.cu.o
    # message("cu_obj: " ${cu_obj})       # /path/to/your/build/dir/kernelfunction.cu.o
    # message("cu_dir: " ${cu_dir})       # /path/to/your/build/dir
    set(cu_objs ${cu_objs} ${cu_obj})
    add_custom_command(TARGET ${lib_name} PRE_BUILD
                   COMMAND mkdir -p ${cu_dir})
    add_custom_command(TARGET ${lib_name} PRE_BUILD
                   COMMAND ${NVCC_PATH}/nvcc ${NVCC_CXX_FLAGS} -Xcompiler -fPIC -o ${cu_obj} -c ${cu_file})
    add_custom_command(TARGET ${lib_name} POST_BUILD
                   COMMAND rm ${cu_obj})
endforeach()

# 将nvcc编译出来的obj文件打包成临时静态库，编译出可执行程序后删除
# ar是编译静态库的命令
add_custom_command(TARGET ${lib_name} PRE_BUILD
                   COMMAND ar cqs ${nvcc_target_lib_full_name} ${cu_objs})
add_custom_command(TARGET ${lib_name} POST_BUILD
                   COMMAND rm ${nvcc_target_lib_full_name})

