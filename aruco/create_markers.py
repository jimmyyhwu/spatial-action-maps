import tempfile
from pathlib import Path
import cv2
from fpdf import FPDF
from PIL import Image
import utils

def create_markers(marker_type):
    marker_ids = utils.get_marker_ids(marker_type)
    if marker_type == 'robots':
        marker_ids = 5 * marker_ids + marker_ids[:4]
    elif marker_type == 'cubes':
        marker_ids = [marker_id for marker_id in marker_ids[:8] for _ in range(6)]
    elif marker_type == 'corners':
        marker_ids = 7 * marker_ids

    output_dir = 'printouts'
    pdf_name = 'markers-{}.pdf'.format(marker_type)
    orientation = 'P'
    sticker_padding_mm = 3

    marker_params = utils.get_marker_parameters()
    paper_params = utils.get_paper_parameters(orientation)

    marker_length_mm = 1000 * marker_params['marker_length']
    scale_factor = (marker_length_mm / paper_params['mm_per_in']) * paper_params['ppi'] / marker_params['marker_length_pixels']
    sticker_length_mm = marker_params['sticker_length_mm'][marker_type]
    stickers_per_row = int((paper_params['width_mm'] - 2 * paper_params['padding_mm']) / (sticker_length_mm + sticker_padding_mm))
    aruco_dict = cv2.aruco.Dictionary_get(marker_params['dict_id'])

    # Create PDF
    pdf = FPDF(orientation, 'mm', 'letter')
    pdf.add_page()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        for i, marker_id in enumerate(marker_ids):
            image_path = str(Path(tmp_dir_name) / '{}.png'.format(marker_id))
            Image.fromarray(cv2.aruco.drawMarker(aruco_dict, marker_id, int(scale_factor * marker_params['marker_length_pixels']))).save(image_path)
            center_x = (sticker_length_mm + sticker_padding_mm) * (i % stickers_per_row + 1)
            center_y = (sticker_length_mm + sticker_padding_mm) * (i // stickers_per_row + 1)
            pdf.rect(
                x=(center_x - sticker_length_mm / 2 - pdf.line_width / 2),
                y=(center_y - sticker_length_mm / 2 - pdf.line_width / 2),
                w=(sticker_length_mm + pdf.line_width),
                h=(sticker_length_mm + pdf.line_width))
            pdf.image(image_path, x=(center_x - marker_length_mm / 2), y=(center_y - marker_length_mm / 2), w=marker_length_mm, h=marker_length_mm)

    # Save PDF
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    pdf.output(output_dir / pdf_name)

if __name__ == '__main__':
    create_markers('robots')
    create_markers('cubes')
    create_markers('corners')
