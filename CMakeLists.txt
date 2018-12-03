cmake_minimum_required(VERSION 3.1)


set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
include_directories(/home/zqp/github/trackers_v2/include)
include_directories(/home/zqp/github/IFaceDetect/include)
include_directories(./include)
include_directories(/usr/include/python2.7)

#file(GLOB src src/VehicleAnalyze.cpp)

find_library(track_lib TrackDP /home/zqp/github/trackers_v2/lib)
find_library(detect_lib mtcnn /home/zqp/github/IFaceDetect/lib)
find_library(boost_python_LIB boost_python)
message(STATUS ${boost_python_LIB})
find_package(OpenCV)

add_library(_IFaceAnalyze SHARED src/PyFaceAnalyze.cpp)
target_link_libraries(_IFaceAnalyze IFaceAnalyze ${boost_python_LIB})

add_library(IFaceAnalyze SHARED src/FaceAnalyze.cpp)
target_link_libraries(IFaceAnalyze ${track_lib} ${detect_lib} ${OpenCV_LIBS})
