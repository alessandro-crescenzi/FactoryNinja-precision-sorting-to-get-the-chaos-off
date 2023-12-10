<p align="center">
  <h2 align="center">FactoryNinja: precision sorting to get the chaos off</h2>

  
</p>
<br>

<img src="https://github.com/pietrolechthaler/UR5-Pick-and-Place-Simulation/blob/main/main.png">

## Table of contents
- [Description](#description)
- [Requirement](#requirements)
- [Folder](#folder)
- [Setup](#setup)
- [Usage](#usage)
- [Contributors](#contributors)

### Description
This repository demonstrates UR5 pick-and-place in ROS and Gazebo. The UR5 uses a Xbox Kinect cam to detect eleven types of Lego Bricks, and publish its position and angolation. 

The goals of this project are:
- simulate the iteration of a UR5 robot with Lego bricks
- The robotic arm must be able to move a block from position A to B and construct a castle by assembling different bricks

<img src="https://github.com/pietrolechthaler/UR5-Pick-and-Place-Simulation/blob/main/intro.gif">

### Folder
```
FactoryNinja-precision-sorting-to-get-the-chaos-off/catkin_ws/src
├── ninja_manager
├── vision
├── motion_planning
├── gazebo_ros_link_attacher
├── robot
```
- `ninja_manager:` the task of this package is to launch the world and spawn the different screws and nuts
- `vision:` the task of this package is to recognize the object type (nuts and screws) and localize it
- `motion_planning:` the task is to move the robot and pick and place the factory element
- `gazebo_ros_link_attacher:` A gazebo plugin definable from URDF to inform a client of a collision with an object
- `robot:` the task is to define the robot model with appropriate PID settings


### Requirements

For running each sample code:
- `Ros Noetic:` http://wiki.ros.org/noetic/Installation
- `Gazebo:` https://classic.gazebosim.org/tutorials?tut=ros_installing&cat=connect_ros
- `Catkin` https://catkin-tools.readthedocs.io/en/latest/

### Setup

After installing the libraries needed to run the project. Clone this repo:
```
git clone https://github.com/alessandro-crescenzi/FactoryNinja-precision-sorting-to-get-the-chaos-off/
```

Setup the project:
```
cd FactoryNinja-precision-sorting-to-get-the-chaos-off/catkin_ws
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
echo "source $PWD/devel/setup.bash" >> $HOME/.bashrc
```

### Usage

Launch the world
```
roslaunch ninja_manager ninja_world.launch
```
Choose how many screws and nuts to spawn (up to 8, if you try to spawn more than 8 objects, it is possible that they are
not created due to insufficient space in working area):
```
rosrun ninja_manager ninja-manager.py -n 5
```
Start the kinematics process
```
rosrun motion_planning motion_planning.py
```
Start the localization process
```
rosrun vision ninja-vision.py -show
```
- `-show` : show the results of the recognition and localization process with an image

### Contributor

| Name                 | Github                                                          |
|----------------------|-----------------------------------------------------------------|
| Alessandro Crescenzi | [alessandro-crescenzi](https://github.com/alessandro-crescenzi) |
