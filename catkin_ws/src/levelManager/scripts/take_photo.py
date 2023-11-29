import time

import message_filters
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

last_received_image = None

def process_CB(image_rgb):
    global last_received_image
    t_start = time.time_ns()
    # from standard message image to opencv image
    rgb = CvBridge().imgmsg_to_cv2(image_rgb, "bgr8")
    # depth = CvBridge().imgmsg_to_cv2(image_depth, "32FC1")

    last_received_image = rgb.copy()


# init node function
def start_node_camera():
    global pub
    global last_received_image

    print("Starting Node Vision 1.0")

    rospy.init_node('take_photo_node')

    print("Subscribing to camera images")
    # topics subscription
    rgb = message_filters.Subscriber("/camera/color/image_raw", Image)

    # images synchronization
    syncro = message_filters.TimeSynchronizer([rgb], 100, reset=False)
    syncro.registerCallback(process_CB)

    # keep node always alive
    rospy.spin()

    return last_received_image


if __name__ == '__main__':

    # load_models()
    try:
        start_node_camera()
    except rospy.ROSInterruptException:
        pass