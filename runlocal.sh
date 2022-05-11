#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source catkin_localhost/devel/setup.bash
rosrun active_sensing_continuous_local peg_hole_2d_sim_local catkin_localhost/src/active_sensing_local/active_sensing_yml/peg_hole_2d/peg_hole_2d.yml outputfile.txt
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
