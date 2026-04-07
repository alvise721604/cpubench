cmake_minimum_required(VERSION 3.21)
project(PiCalculatorQtCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(PiCalculatorQtCpp
    src/main.cpp
    src/picalculatorwindow.cpp
    src/calc.cpp
)

target_include_directories(PiCalculatorQtCpp PRIVATE src)
target_link_libraries(PiCalculatorQtCpp PRIVATE Qt6::Widgets)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(PiCalculatorQtCpp PRIVATE -O2)
endif()

if(APPLE)
    set_target_properties(PiCalculatorQtCpp PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "PiCalculatorQtCpp"
    )
endif()