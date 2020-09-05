import tempfile
from pathlib import Path
import cv2
from fpdf import FPDF
from PIL import Image
import utils

marker_ids = list(range(34, 38))
receptacle_width_mm = 150
padding_mm = 4
line_width = 5
dash_length = 10.15
output_dir = 'printouts'
pdf_name = 'trash-receptacle.pdf'
orientation = 'L'

marker_params = utils.get_marker_parameters()
paper_params = utils.get_paper_parameters(orientation)

marker_length_mm = 1000 * marker_params['marker_length']
scale_factor = (marker_length_mm / paper_params['mm_per_in']) * paper_params['ppi'] / marker_params['marker_length_pixels']
aruco_dict = cv2.aruco.Dictionary_get(marker_params['dict_id'])

# Create PDF
pdf = FPDF(orientation, 'mm', 'letter')
pdf.add_page()
pdf.set_line_width(line_width)
with tempfile.TemporaryDirectory() as tmp_dir_name:
    for marker_id, (corner_x, corner_y) in zip(marker_ids, [(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        image_path = str(Path(tmp_dir_name) / '{}.png'.format(marker_id))
        Image.fromarray(cv2.aruco.drawMarker(aruco_dict, marker_id, int(scale_factor * marker_params['marker_length_pixels']))).save(image_path)
        x = paper_params['width_mm'] / 2 + corner_x * receptacle_width_mm / 2 - (corner_x == 1) * marker_length_mm
        y = paper_params['height_mm'] / 2 + corner_y * receptacle_width_mm / 2 - (corner_y == 1) * marker_length_mm
        pdf.image(image_path, x=x, y=y, w=marker_length_mm, h=marker_length_mm)
    offset1 = receptacle_width_mm / 2 - marker_length_mm - padding_mm - line_width / 2
    offset2 = receptacle_width_mm / 2 - line_width / 2
    for (x1, x2, y1, y2) in [
            (-offset1, offset1, -offset2, -offset2),
            (offset2, offset2, -offset1, offset1),
            (offset1, -offset1, offset2, offset2),
            (-offset2, -offset2, offset1, -offset1),
        ]:
        pdf.dashed_line(
            paper_params['width_mm'] / 2 + x1, paper_params['height_mm'] / 2 + y1,
            paper_params['width_mm'] / 2 + x2, paper_params['height_mm'] / 2 + y2,
            dash_length=dash_length, space_length=dash_length + 2 * line_width
        )

# Save PDF
output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir(parents=True)
pdf.output(output_dir / pdf_name)
