message(STATUS "Third-party: creating target 'CLI11::CLI11'")

include(FetchContent)
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.2.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cli11)
