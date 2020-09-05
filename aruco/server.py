import argparse
from multiprocessing.connection import Listener
import cv2
import numpy as np
import utils

def main(args):
    # Read camera parameters
    camera_params_file_path = utils.get_camera_params_file_path(args.camera_name)
    image_width, image_height, camera_matrix, dist_coeffs = utils.get_camera_params(camera_params_file_path)

    # Set up webcam
    cap = utils.get_video_cap(image_width, image_height, args.camera_id)

    # Board and marker params
    boards = [{'name': 'robots', 'corner_offset_mm': 36}, {'name': 'cubes', 'corner_offset_mm': 12}]
    marker_params = utils.get_marker_parameters()
    room_length_mm = 1000 * args.room_length
    room_width_mm = 1000 * args.room_width
    room_length_pixels = int(room_length_mm * marker_params['pixels_per_mm'])
    room_width_pixels = int(room_width_mm * marker_params['pixels_per_mm'])

    # Set up aruco dicts
    for board in boards:
        aruco_dict = cv2.aruco.Dictionary_get(marker_params['dict_id'])
        aruco_dict.bytesList = aruco_dict.bytesList[utils.get_marker_ids('corners_{}'.format(board['name']))]
        board['board_dict'] = aruco_dict
        aruco_dict = cv2.aruco.Dictionary_get(marker_params['dict_id'])
        aruco_dict.bytesList = aruco_dict.bytesList[utils.get_marker_ids(board['name'])]
        board['marker_dict'] = aruco_dict

    # Board warping
    for board in boards:
        corner_offset_pixels = board['corner_offset_mm'] * marker_params['pixels_per_mm']
        board['src_points'] = None
        board['dst_points'] = [
            [-corner_offset_pixels, -corner_offset_pixels],
            [room_length_pixels + corner_offset_pixels, -corner_offset_pixels],
            [room_length_pixels + corner_offset_pixels, room_width_pixels + corner_offset_pixels],
            [-corner_offset_pixels, room_width_pixels + corner_offset_pixels]
        ]

    # Enable corner refinement
    detector_params = cv2.aruco.DetectorParameters_create()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Set up server
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'secret password')
    conn = None

    def process_image(image):
        # Undistort image
        image = cv2.undistort(image, camera_matrix, dist_coeffs)

        data = {}
        for board in boards:
            board_name = board['name']
            data[board_name] = {}

            if board['src_points'] is None:
                # Detect board markers (assumes board won't move since this is only done once)
                #board_corners, board_indices, _ = cv2.aruco.detectMarkers(image, board['board_dict'], parameters=detector_params)
                board_corners, board_indices, _ = cv2.aruco.detectMarkers(image, board['board_dict'])

                # Show detections
                if args.debug:
                    image_copy = image.copy()
                    if board_indices is not None:
                        cv2.aruco.drawDetectedMarkers(image_copy, board_corners, board_indices)
                    cv2.imshow('{}_board_corners'.format(board_name), image_copy)

                # Ensure board was found
                if board_indices is None or len(board_indices) < 4:
                    data[board_name] = None  # None rather than {} to signify board was not detected
                    continue

                board['src_points'] = []
                for marker_index, corner in sorted(zip(board_indices, board_corners)):
                    board['src_points'].append(corner.squeeze(0).mean(axis=0).tolist())
            else:
                # Warp the board
                M = cv2.getPerspectiveTransform(np.asarray(board['src_points'], dtype=np.float32), np.asarray(board['dst_points'], dtype=np.float32))
                warped_image = cv2.warpPerspective(image, M, (room_length_pixels, room_width_pixels))

                # Detect markers in warped image
                corners, indices, _ = cv2.aruco.detectMarkers(warped_image, board['marker_dict'], parameters=detector_params)

                # Show detections
                if args.debug:
                    image_copy = warped_image.copy()
                    if indices is not None:
                        cv2.aruco.drawDetectedMarkers(image_copy, corners, indices)
                    image_copy = cv2.resize(image_copy, (int(image_copy.shape[1] / 2), int(image_copy.shape[0] / 2)))
                    cv2.imshow(board_name, image_copy)

                if indices is None:
                    continue

                # Compute poses
                board_data = {}
                for marker_index, corner in zip(indices, corners):
                    marker_index = marker_index.item()
                    marker_corners = corner.squeeze(0)
                    marker_center = marker_corners.mean(axis=0)

                    # Compute heading estimates for each of the four marker corners
                    diffs = [c - marker_center for c in marker_corners]
                    angles = np.array([np.arctan2(-diff[1], diff[0]) for diff in diffs])
                    angles = angles + np.radians([-135, -45, 45, 135])
                    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi

                    # If multiple markers are detected on same cube, use the marker on top (which should have the lowest angle_std)
                    angle_std = angles.std()
                    if board_name == 'cubes' and marker_index in board_data and angle_std > board_data[marker_index]['angle_std']:
                        continue

                    # Compute position and heading
                    position = [
                        (marker_center[0] / marker_params['pixels_per_mm'] - room_length_mm / 2) / 1000,
                        (room_width_mm / 2 - (marker_center[1] / marker_params['pixels_per_mm'])) / 1000
                    ]
                    heading = angles.mean()
                    marker_data = {'position': position, 'heading': heading}
                    if board_name == 'cubes':
                        marker_data['angle_std'] = angle_std
                    board_data[marker_index] = marker_data
                data[board_name] = board_data

        return data

    while True:
        if cv2.waitKey(1) == 27:  # Esc key
            break

        if conn is None:
            print('Waiting for connection...')
            conn = listener.accept()
            print('Connected!')

        _, image = cap.read()
        if image is None:
            continue

        data = process_image(image)
        try:
            conn.send(data)
        except:
            conn = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-id', type=int, default=0)
    parser.add_argument('--camera-name', default='logitech-c930e')
    parser.add_argument('--room-length', type=float, default=1.0)
    parser.add_argument('--room-width', type=float, default=0.5)
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    main(parser.parse_args())
