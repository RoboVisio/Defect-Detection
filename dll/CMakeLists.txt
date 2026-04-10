# =============================================================================
#  defect_dll —— CMake 构建脚本
#
#  用法 (Windows + VS 2022):
#      cd dll && mkdir build && cd build
#      cmake .. -G "Visual Studio 17 2022" -A x64 ^
#               -DHALCON_ROOT="C:/Program Files/MVTec/HALCON-25.11" ^
#               -DONNXRUNTIME_ROOT="C:/onnxruntime-win-x64-1.18.0"
#      cmake --build . --config Release
#  产物: build/Release/defect_dll.dll  +  defect_dll.lib
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(defect_dll LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----- 依赖路径 (按需改) -----
set(HALCON_ROOT      "C:/Users/11154/AppData/Local/Programs/MVTec/HALCON-25.11-Progress"        CACHE PATH "HALCON 安装根目录")
set(ONNXRUNTIME_ROOT "C:/onnxruntime-win-x64-gpu" CACHE PATH "ONNX Runtime GPU 包解压目录 (必须用 onnxruntime-win-x64-gpu-*.zip, 不是 CPU 版)")

if(NOT EXISTS "${HALCON_ROOT}/include/HalconCpp.h")
    message(FATAL_ERROR "找不到 HALCON: 请设置 -DHALCON_ROOT=...")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "找不到 ONNX Runtime: 请设置 -DONNXRUNTIME_ROOT=...")
endif()

# ----- 源文件 -----
# 注意: 把项目根目录的 main.cpp 也编进来, 但定义 NO_EXPORT_MAIN 以剥离 main()/action()
set(SRC
    defect_dll.cpp
    preprocess_with_roi.cpp
    ../main.cpp
)

add_library(defect_dll SHARED ${SRC})

target_compile_definitions(defect_dll PRIVATE
    DEFECT_DLL_EXPORTS
    NO_EXPORT_MAIN
    NO_EXPORT_APP_MAIN
    _CRT_SECURE_NO_WARNINGS
)

target_include_directories(defect_dll PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HALCON_ROOT}/include
    ${HALCON_ROOT}/include/halconcpp
    ${ONNXRUNTIME_ROOT}/include
)

# ----- 库文件 -----
if(WIN32)
    set(HALCON_LIB    ${HALCON_ROOT}/lib/x64-win64/halcon.lib)
    set(HALCONCPP_LIB ${HALCON_ROOT}/lib/x64-win64/halconcpp.lib)
    set(ORT_LIB       ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
else()
    set(HALCON_LIB    halcon)
    set(HALCONCPP_LIB halconcpp)
    set(ORT_LIB       onnxruntime)
endif()

target_link_libraries(defect_dll PRIVATE
    ${HALCONCPP_LIB}
    ${HALCON_LIB}
    ${ORT_LIB}
)

# ----- 运行时 DLL 拷贝 (方便直接跑) -----
if(WIN32)
    add_custom_command(TARGET defect_dll POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:defect_dll>/onnxruntime.dll")
endif()
