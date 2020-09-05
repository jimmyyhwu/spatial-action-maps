# Adapted from https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/detect_markers.cpp
import argparse
import cv2
import utils

def main(args):
    # Read camera parameters
    camera_params_file_path = utils.get_camera_params_file_path(args.camera_name)
    image_width, image_height, camera_matrix, dist_coeffs = utils.get_camera_params(camera_params_file_path)

    # Set up webcam
    cap = utils.get_video_cap(image_width, image_height, args.camera_id)

    # Set up aruco dict
    params = utils.get_marker_parameters()
    aruco_dict = cv2.aruco.Dictionary_get(params['dict_id'])

    # Enable corner refinement
    #detector_params = cv2.aruco.DetectorParameters_create()
    #detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    while True:
        if cv2.waitKey(1) == 27:  # Esc key
            break

        _, image = cap.read()
        if image is None:
            continue

        # Undistort image and detect markers
        image = cv2.undistort(image, camera_matrix, dist_coeffs)
        #corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=detector_params)
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)

        # Show detections
        image_copy = image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
        cv2.imshow('out', image_copy)

    cap.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument('--camera-id', type=int, default=0)
parser.add_argument('--camera-name', default='logitech-c930e')
main(parser.parse_args())
