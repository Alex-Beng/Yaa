message(STATUS "Third-party: creating target 'interception'")

include(FetchContent)
FetchContent_Declare(
    interception
    GIT_REPOSITORY https://github.com/Alex-Beng/interception-cmake.git
    GIT_TAG c9c281f929dd3c2fb3a0b28938dc47f58a149a54
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(interception)

# 把dll拷贝到bin目录
file(
    COPY
    ${interception_BINARY_DIR}/library/Debug/interception.dll
    DESTINATION
    ${CMAKE_BINARY_DIR}/bin
)