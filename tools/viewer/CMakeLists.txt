# Vendored (FetchContent) GLFW + Dear ImGui for the interactive viewer.
include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glfw)

# Dear ImGui ships no CMakeLists; populate the sources and build them here.
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.5
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)

# Vendored third-party sources: silence warnings. The root CMakeLists sets
# -Wall -Wextra with add_compile_options, which DOES propagate into
# FetchContent subdirectories -- so without this the project's "zero warnings"
# claim would be carrying an upstream ImGui diagnostic that is not ours to fix.
target_compile_options(imgui PRIVATE -w)
if(TARGET glfw)
  target_compile_options(glfw PRIVATE -w)
endif()
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw)

find_package(OpenGL REQUIRED)

# Shared viewer support: vertex batch, camera, flat shader program, CPU-shaded
# mesh primitives, RSS readout -- linked by both apps so nothing is duplicated.
add_library(viewer_support STATIC viewer_support.cpp)
target_include_directories(viewer_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(viewer_support PUBLIC bochner_core OpenGL::GL)
target_compile_definitions(viewer_support PUBLIC GL_SILENCE_DEPRECATION)

add_executable(bochner_viewer main.cpp)
target_link_libraries(bochner_viewer PRIVATE bochner_core viewer_support imgui glfw OpenGL::GL)
target_compile_definitions(bochner_viewer PRIVATE GL_SILENCE_DEPRECATION)

# Flow-past-obstacle viewer (the vortex-shedding demo app).
add_executable(bochner_obstacle_viewer obstacle_main.cpp)
target_link_libraries(bochner_obstacle_viewer PRIVATE bochner_core viewer_support imgui glfw OpenGL::GL)
target_compile_definitions(bochner_obstacle_viewer PRIVATE GL_SILENCE_DEPRECATION)
