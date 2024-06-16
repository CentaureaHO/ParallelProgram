#!/bin/bash
nvcc -c kernels.cu -o kernels.o -std=c++17 -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_photo
g++ -c main.cpp -o main.o -std=c++17 -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_photo
g++ kernels.o main.o -o Canny_CUDA -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_photo -lcudart