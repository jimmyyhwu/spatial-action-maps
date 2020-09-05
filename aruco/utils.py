from pathlib import Path
import cv2
import usb.core

################################################################################
# Board and markers

def get_marker_parameters():
    params = {}
    params['dict_id'] = cv2.aruco.DICT_4X4_50
    params['marker_length'] = 0.018  # 18 mm
    params['marker_length_pixels'] = 6
    params['pixels_per_mm'] = 2  # 2 gives much better marker detections than 1
    params['sticker_length_mm'] = {'robots': 25, 'cubes': 28, 'corners': 24}
    return params

def get_charuco_board_parameters():
    params = get_marker_parameters()
    params['squares_x'] = 10
    params['squares_y'] = 7
    params['square_length'] = 0.024  # 24 mm
    square_length_pixels = (params['marker_length_pixels'] / params['marker_length']) * params['square_length']
    assert not square_length_pixels - int(square_length_pixels) > 1e-8
    params['square_length_pixels'] = int(square_length_pixels)
    return params

def get_paper_parameters(orientation='P'):
    width, height, padding = 8.5, 11, 0.5
    if orientation == 'L':
        width, height = height, width
    params = {}
    params['mm_per_in'] = 25.4
    params['width_mm'] = width * params['mm_per_in']
    params['height_mm'] = height * params['mm_per_in']
    params['padding_mm'] = padding * params['mm_per_in']
    params['ppi'] = 600
    return params

def get_marker_ids(marker_type):
    if marker_type == 'robots':
        return list(range(10))
    if marker_type == 'cubes':
        return list(range(10, 34))
    if marker_type == 'corners_robots':
        return list(range(42, 46))
    if marker_type == 'corners_cubes':
        return list(range(46, 50))
    if marker_type == 'corners':
        return get_marker_ids('corners_robots') + get_marker_ids('corners_cubes')
    raise Exception

################################################################################
# Camera

def get_video_cap(frame_width, frame_height, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Might not actually do anything on macOS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduces latency
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == frame_width, (cap.get(cv2.CAP_PROP_FRAME_WIDTH), frame_width)
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == frame_height, (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), frame_height)
    return cap

def get_usb_device_serial(camera_name):
    if camera_name == 'logitech-c930e':
        id_vendor, id_product = (0x046d, 0x0843)
    else:
        raise Exception
    dev = usb.core.find(idVendor=id_vendor, idProduct=id_product)
    return usb.util.get_string(dev, dev.iSerialNumber)

def get_camera_identifier(camera_name):
    try:
        return get_usb_device_serial(camera_name)
    except:
        return 'unknown-camera'

def get_camera_params_file_path(camera_name='logitech-c930e'):
    camera_params_dir = Path('camera-params')
    identifier = get_camera_identifier(camera_name)
    return str(camera_params_dir / '{}.yml'.format(identifier))

def get_camera_params(camera_params_file_path):
    assert Path(camera_params_file_path).exists()
    fs = cv2.FileStorage(camera_params_file_path, cv2.FILE_STORAGE_READ)
    image_width = fs.getNode('image_width').real()
    image_height = fs.getNode('image_height').real()
    camera_matrix = fs.getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode('distortion_coefficients').mat()
    fs.release()
    return image_width, image_height, camera_matrix, dist_coeffs
