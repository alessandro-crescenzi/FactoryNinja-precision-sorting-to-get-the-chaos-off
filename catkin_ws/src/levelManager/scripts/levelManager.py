#!/usr/bin/python3
import argparse
import os
import random
import time
import xml.etree.ElementTree as ET

import cv2
import message_filters
import numpy as np
import rospkg
import rospy
import rosservice
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
import random

path = rospkg.RosPack().get_path("levelManager")

costruzioni = ['costruzione-1', 'costruzione-2']

save_path = os.path.join(os.path.expanduser('~'), 'Desktop/screw_and_nuts')
last_received_image = None


def randomCostruzione():
    return random.choice(costruzioni)


def getPose(modelEl):
    strpose = modelEl.find('pose').text
    return [float(x) for x in strpose.split(" ")]


def get_Name_Type(modelEl):
    if modelEl.tag == 'model':
        name = modelEl.attrib['name']
    else:
        name = modelEl.find('name').text
    return name, name.split('_')[0]


def get_Parent_Child(jointEl):
    parent = jointEl.find('parent').text.split('::')[0]
    child = jointEl.find('child').text.split('::')[0]
    return parent, child


def getLego4Costruzione(select=None):
    nome_cost = randomCostruzione()
    if select is not None: nome_cost = costruzioni[select]
    print("spawning", nome_cost)

    tree = ET.parse(f'{path}/lego_models/{nome_cost}/model.sdf')
    root = tree.getroot()
    costruzioneEl = root.find('model')

    brickEls = []
    for modEl in costruzioneEl:
        if modEl.tag in ['model', 'include']:
            brickEls.append(modEl)

    models = ModelStates()
    for bEl in brickEls:
        pose = getPose(bEl)
        models.name.append(get_Name_Type(bEl)[1])
        rot = Quaternion(*quaternion_from_euler(*pose[3:]))
        models.pose.append(Pose(Point(*pose[:3]), rot))

    rospy.init_node("levelManager")
    istruzioni = rospy.Publisher("costruzioneIstruzioni", ModelStates, queue_size=1)
    istruzioni.publish(models)

    return models


def change_model_color(model_xml, color):
    root = ET.XML(model_xml)
    root.find('.//material/script/name').text = color
    return ET.tostring(root, encoding='unicode')


# DEFAULT PARAMETERS
package_name = "levelManager"
spawn_name = '_spawn'
level = None
selectBrick = None
maxLego = 11
spawn_pos = (-0.55, -0.5, 0.74)  # center of spawn area
spawn_dim = (0.40, 0.30)  # spawning area
min_space = 0.010  # min space between lego
min_distance = 0.15  # min distance between leg

screwDict = {
    # 'X1-Y1-Z2': (0, (0.031, 0.031, 0.057)),
    # 'X1-Y2-Z1': (1, (0.031, 0.063, 0.038)),
    # 'X1-Y2-Z2': (2, (0.031, 0.063, 0.057)),
    # 'X1-Y2-Z2-CHAMFER': (3, (0.031, 0.063, 0.057)),
    # 'X1-Y2-Z2-TWINFILLET': (4, (0.031, 0.063, 0.057)),
    # 'X1-Y3-Z2': (5, (0.031, 0.095, 0.057)),
    # 'X1-Y3-Z2-FILLET': (6, (0.031, 0.095, 0.057)),
    # 'X1-Y4-Z1': (7, (0.031, 0.127, 0.038)),
    # 'X1-Y4-Z2': (8, (0.031, 0.127, 0.057)),
    # 'X2-Y2-Z2': (9, (0.063, 0.063, 0.057)),
    # 'X2-Y2-Z2-FILLET': (10, (0.063, 0.063, 0.057))
    'bolt_25x8': (0, (0.020, 0.064, 0.012)),
    'nut_6x9': (1, (0.014, 0.0126, 0.008))
}
#
# brickOrientations = {
#     'X1-Y2-Z1': (((1, 1), (1, 3)), -1.715224, 0.031098),
#     'X1-Y2-Z2-CHAMFER': (((1, 1), (1, 2), (0, 2)), 2.359515, 0.015460),
#     'X1-Y2-Z2-TWINFILLET': (((1, 1), (1, 3)), 2.145295, 0.024437),
#     'X1-Y3-Z2-FILLET': (((1, 1), (1, 2), (0, 2)), 2.645291, 0.014227),
#     'X1-Y4-Z1': (((1, 1), (1, 3)), 3.14, 0.019),
#     'X2-Y2-Z2-FILLET': (((1, 1), (1, 2), (0, 2)), 2.496793, 0.018718)
# }  # brickOrientations = (((side, roll), ...), rotX, height)

# color bricks
colorList = ['Gazebo/Indigo', 'Gazebo/Orange',
             'Gazebo/Red', 'Gazebo/Purple',
             'Gazebo/DarkYellow', 'Gazebo/Green']

screwList = list(screwDict.keys())
counters = [0 for screw in screwList]

spawned_screws = []  # lego = [[name, type, pose, radius], ...]


# get model path
def get_model_path(model):
    pkg_path = rospkg.RosPack().get_path(package_name)
    return f'{pkg_path}/lego_models/{model}/model.sdf'


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
    name = f'{screw_type}_{counters[screw_idx] + 1}'
    pos, radius = get_valid_pose(screw_type)
    assert pos is not None, "pos is None"
    assert radius is not None, "radius is None"
    color = random.choice(colorList)

    spawn_model(screw_type, pos, name, spawn_name, color)
    spawned_screws.append((name, screw_type, pos, radius))
    counters[screw_idx] += 1


# main function setup area and level manager
def set_up_area(screws_number=None):
    # delete all nuts and screws on the table
    # for screwType in screwList:
    #     count = 1
    #     while delete_model(f'{screwType}_{count}').success:
    #         count += 1

    for screw in spawned_screws:
        delete_model(screw[0])
        spawned_screws.remove(screw)

    # # screating spawn area
    # spawn_model(spawn_name, Pose(Point(*spawn_pos), None))

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

        # starting position bricks
        for _ in range(100):
            random_number = random.randint(1, 15)
            set_up_area(random_number)
            rospy.sleep(5)
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
