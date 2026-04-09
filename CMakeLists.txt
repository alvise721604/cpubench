cmake_minimum_required(VERSION 3.21)
project(PiCalculatorQtCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Prova prima il find_package standard
find_package(OpenMP)

# Fallback per Apple Clang + libomp installata con Homebrew
if(APPLE AND NOT OpenMP_CXX_FOUND)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(LIBOMP_PREFIX)
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
            INTERFACE_INCLUDE_DIRECTORIES "${LIBOMP_PREFIX}/include"
            INTERFACE_LINK_LIBRARIES "${LIBOMP_PREFIX}/lib/libomp.dylib"
        )
        set(OpenMP_CXX_FOUND TRUE)
    endif()
endif()

add_executable(PiCalculatorQtCpp
    src/main.cpp
    src/picalculatorwindow.cpp
    src/calc.cpp
    src/mem.cpp
)

target_include_directories(PiCalculatorQtCpp PRIVATE src)

target_link_libraries(PiCalculatorQtCpp PRIVATE
    Qt6::Widgets
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(PiCalculatorQtCpp PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(PiCalculatorQtCpp PRIVATE -O2)
endif()

if(APPLE)
    set_target_properties(PiCalculatorQtCpp PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "PiCalculatorQtCpp"
    )
endif()
    src/main.cpp
    src/picalculatorwindow.cpp
    src/calc.cpp
    src/mem.cpp
)

target_include_directories(PiCalculatorQtCpp PRIVATE src)

target_link_libraries(PiCalculatorQtCpp PRIVATE
    Qt6::Widgets
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(PiCalculatorQtCpp PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(PiCalculatorQtCpp PRIVATE -O2)
endif()

if(APPLE)
    set_target_properties(PiCalculatorQtCpp PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "PiCalculatorQtCpp"
    )
endif()
