FROM ros:noetic
MAINTAINER Lexi Scott aas220@case.edu
RUN sudo apt-get -y update
RUN sudo apt-get -y upgrade
RUN sudo apt-get install -y libfcl-dev
RUN sudo apt-get install -y libeigen3-dev
RUN sudo apt-get install -y ros-noetic-tf
RUN sudo apt-get install -y libyaml-cpp-dev
RUN sudo apt-get install -y libhdf5-serial-dev
RUN sudo apt-get install -y mpich
RUN sudo apt-get install libopenmpi-dev

RUN sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/libhdf5.so
RUN sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so /usr/lib/libhdf5_hl.so
# commented out copy step
# COPY . . 
WORKDIR "/catkin_localhost"
RUN /bin/bash -c ' . /opt/ros/noetic/setup.bash ; catkin_make'
