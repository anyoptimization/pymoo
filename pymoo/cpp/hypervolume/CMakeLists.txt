cmake_minimum_required(VERSION 2.8.12)
project(hypervolume)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

find_package(pybind11)
pybind11_add_module(hypervolume src/hypervolume.cpp)
