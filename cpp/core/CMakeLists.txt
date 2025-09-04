# cpp/core/CMakeLists.txt

# 收集源文件
# cpp/core/CMakeLists.txt

# 收集源文件
set(CORE_SOURCES
    FRICP.cpp
    spare.cpp
)

set(CORE_HEADERS
    Registration.h
    FRICP.h
    median.h
    spare.h
    Types.h

)

# 生成静态库 core
add_library(core STATIC ${CORE_SOURCES} ${CORE_HEADERS})

# 指定 core 库头文件搜索路径
target_include_directories(core 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}           # core 自己的头文件
        ${CMAKE_SOURCE_DIR}/tools             # tools 头文件
        ${THIRDPARTY_DIR}                     # 第三方库
        ${THIRDPARTY_DIR}/eigen               # eigen
        ${OpenMesh_INCLUDE_DIRS}              # OpenMesh
)

# 链接依赖库
target_link_libraries(core
    PUBLIC
        io                      # core 依赖 io
        OpenMeshCore
        OpenMeshTools   # core 依赖 OpenMesh
)

