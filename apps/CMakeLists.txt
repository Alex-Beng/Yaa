cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE SOURCES "*.cpp")

# add the executables
foreach(sourcefile ${SOURCES})
    get_filename_component(name ${sourcefile} NAME_WE)
    add_executable(${name} ${sourcefile})
    target_link_libraries(
        ${name} 
        interception 
        nlohmann_json::nlohmann_json 
        ${OpenCV_LIBS}
        CLI11::CLI11
    )
endforeach(sourcefile ${SOURCES})
