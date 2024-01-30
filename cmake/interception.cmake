message(STATUS "Third-party: creating target 'interception'")

include(FetchContent)
FetchContent_Declare(
    interception
    GIT_REPOSITORY https://github.com/Alex-Beng/interception-cmake.git
    GIT_TAG 304dadd76fd30efe00976e08f83e18ba29bf28a9
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(interception)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
