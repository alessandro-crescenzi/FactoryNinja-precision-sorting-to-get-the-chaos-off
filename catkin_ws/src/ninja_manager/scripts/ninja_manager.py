#!/usr/bin/python3
import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import rospkg
import rospy
import rosservice
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import *
from tf.transformations import quaternion_from_euler
import random

path = rospkg.RosPack().get_path("ninja_manager")

# DEFAULT PARAMETERS
package_name = "ninja_manager"
spawn_name = '_spawn'
level = None
selectBrick = None
maxLego = 11
spawn_pos = (-0.38, -0.35, 0.74)  # center of spawn area
spawn_dim = (0.25, 0.2)  # spawning area
min_space = 0.010  # min space between screws and nuts
min_distance = 0.15  # min distance between screws and nuts

screwDict = {
    'screw_25x8': (0, (0.0277128, 0.06, 0.024)),
    'nut_6x9': (1, (0.0277128, 0.024, 0.012))
}

# color bricks
colorList = ['Gazebo/Indigo', 'Gazebo/Orange',
             'Gazebo/Red', 'Gazebo/Purple',
             'Gazebo/DarkYellow', 'Gazebo/Green']

screwList = list(screwDict.keys())
counters = [0 for screw in screwList]

spawned_screws = []  # screw_or_nut = [[name, type, pose, radius], ...]


def change_model_color(model_xml, color):
    root = ET.XML(model_xml)
    root.find('.//material/script/name').text = color
    return ET.tostring(root, encoding='unicode')


# get model path
def get_model_path(model):
    pkg_path = rospkg.RosPack().get_path(package_name)
    return f'{pkg_path}/screws_and_nuts_models/{model}/model.sdf'


# set position brick
def random_pose(screw_type):
    _, dim, = screwDict[screw_type]
    spawn_x = spawn_dim[0]
    spawn_y = spawn_dim[1]
    rot_x = 0
    rot_y = 0
    rot_z = random.uniform(-3.14, 3.14)
    point_x = random.uniform(-spawn_x, spawn_x)
    point_y = random.uniform(-spawn_y, spawn_y)
    point_z = dim[2] / 2
    screw_dimx = dim[0]
    screw_dimy = dim[1]

    rot = Quaternion(*quaternion_from_euler(rot_x, rot_y, rot_z))
    point = Point(point_x, point_y, point_z)
    return Pose(point, rot), screw_dimx, screw_dimy


class PoseError(Exception):
    pass


# function to get a valid pose
def get_valid_pose(screwtype):
    trys = 1000
    valid = False
    pos = None
    radius = None
    while not valid:
        pos, dim1, dim2 = random_pose(screwtype)
        radius = np.sqrt((dim1 ** 2 + dim2 ** 2)) / 2
        valid = True
        for screw in spawned_screws:
            point = screw[2].position
            r2 = screw[3]
            min_dist = max(radius + r2 + min_space, min_distance)
            if (point.x - pos.position.x) ** 2 + (point.y - pos.position.y) ** 2 < min_dist ** 2:
                valid = False
                trys -= 1
                if trys == 0:
                    raise PoseError("No space available in spawn area")
                break
    return pos, radius


# function to spawn model
def spawn_model(model, pos, name=None, ref_frame='world', color=None):
    if name is None:
        name = model

    model_xml = open(get_model_path(model), 'r').read()
    if color is not None:
        model_xml = change_model_color(model_xml, color)

    spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    return spawn_model_client(model_name=name,
                              model_xml=model_xml,
                              robot_namespace='/foo',
                              initial_pose=pos,
                              reference_frame=ref_frame)


# support function delete bricks on table
def delete_model(name):
    delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    return delete_model_client(model_name=name)


# support functon spawn bricks
def spawn_screw():
    screw_type = random.choice(screwList)

    screw_idx = screwDict[screw_type][0]
    name = f'{screw_type}_{str(time.time_ns())[-9:]}'
    pos, radius = get_valid_pose(screw_type)
    assert pos is not None, "pos is None"
    assert radius is not None, "radius is None"
    color = random.choice(colorList)

    spawn_model(screw_type, pos, name, spawn_name, color)
    spawned_screws.append((name, screw_type, pos, radius))
    counters[screw_idx] += 1


# main function setup area and level manager
def set_up_area(screws_number=None):
    rospy.init_node('element_spawner')

    # delete all nuts and screws on the table
    models = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=None)
    environment_elements = ["ground_plane", "modern_table", "kinect", "robot", "_spawn"]
    spawn_area_exists = False
    for model in models.name:
        if model.startswith('_spawn'):
            spawn_area_exists = True
        if model not in environment_elements and (model.startswith('screw') or model.startswith('nut')):
            delete_model(model)

    if not spawn_area_exists:
        spawn_model(spawn_name, Pose(Point(*spawn_pos), None))

    try:
        for _ in range(screws_number):
            spawn_screw()
    except PoseError as _:
        print("[Error]: no space in spawning area")
        pass

    print(f"Added {len(spawned_screws)} screw(s) or nut(s)")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Spawn a specified number of screws.')
    parser.add_argument('-n', '--number', type=int, required=True, help='Number of screws to spawn')

    args = parser.parse_args()
    number_of_screws = args.number

    try:
        if '/gazebo/spawn_sdf_model' not in rosservice.get_service_list():
            print("Waining gazebo service..")
            rospy.wait_for_service('/gazebo/spawn_sdf_model')

        spawn_model(spawn_name, Pose(Point(*spawn_pos), None))
        set_up_area(number_of_screws)
        print("All done. Ready to start.")
    except rosservice.ROSServiceIOException as err:
        print("No ROS master execution")
        pass
    except rospy.ROSInterruptException as err:
        print(err)
        pass
    except rospy.service.ServiceException:
        print("No Gazebo services in execution")
        pass
    except rospkg.common.ResourceNotFound:
        print(f"Package not found: '{package_name}'")
        pass
    except FileNotFoundError as err:
        print(f"Model not found: \n{err}")
        print(f"Check model in folder'{package_name}/lego_models'")
        pass
