cmake_minimum_required(VERSION 3.25)
project(Computational_mathematics C)

set(CMAKE_C_STANDARD 11)
add_link_options(-fopenmp)
add_compile_options(-fopenmp)
add_compile_options(-O3)

add_executable(Computational_mathematics main.c)
