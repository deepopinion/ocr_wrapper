from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import InterpolationMode, rotate

# We define a minimum number of pixels - we don't do an angle estimation with less and return the angle 0° instead
MIN_NB_DARK_PIXEL = 100
# We filter white pixel and keep only black. However, real live has shades of gray. One criterion for dark is being over
# the average value (pitch-dark is 1, and white is 0, due to inversion of color). Further, we add a MINIMAL_DARK_VALUE value,
# as second condition. This might be interesting for other types of images than pure text-pictures.
MINIMAL_DARK_VALUE = 0.1
RADIAN2DEGREE = 180 / torch.pi
ANGLE90 = torch.pi / 2
EXTRA_ANGLE_RADIAN = ANGLE90 / 3
EXTRA_ANGLE_DEGREE = EXTRA_ANGLE_RADIAN * RADIAN2DEGREE


class DetectTilt:
    """
    `DetectTilt` provides the function `find_angle`, which can be used to estimate the tilt angle in degree of an image of
    a text-document, which is passed to the function. During the instantiation of the class, several search parameter can be
    set, mainly to balance speed and precision against each other.

    Args:
        device: The device for torch tensors. Standard is "cpu".
        nb_pixel: Each document is downsized if needed such that its number of pixels given by height x width is roughly nb_pixel.
            standard is half a million (5e5).
        nb_pre_scan: Number of steps in the prescan. In the prescan, the range from -90° to 90° is tested (due to
            mirror-symmetry in the results, this covers 360°). The step-size is chosen such that in total nb_pre_scan angles
            are tested. Standard is 120, which corresponds to a step of 1.5° (180° / 120)
        grid_half_size: Standard is 20. After the prescan, the best area according to the prescan is further tested.
            The total grid is size is given by 2 * grid_half_size + 1. Be aware that the grid is iteratively refined later on,
            see `nb_fine_scan`.
        nb_fine_scan: The fine scan is preformed in `nb_fine_scan` rounds. After each round, step_size is halfed, as well
            as the search area. The standard value is 5.
    """

    def __init__(
        self,
        device: str = "cpu",
        nb_pixel: int = int(5e5),
        nb_pre_scan: int = 120,
        grid_half_size: int = 20,
        nb_fine_scan: int = 5,
    ):
        self.device = device
        self.nb_pixel = nb_pixel
        self.loop = range(nb_fine_scan)
        pre_scan_unit = torch.pi / nb_pre_scan
        self.pre_scan_angles = torch.arange(nb_pre_scan) * pre_scan_unit - ANGLE90
        int_grid = torch.linspace(
            -grid_half_size,
            grid_half_size,
            2 * grid_half_size + 1,
            dtype=torch.long,
            device=self.device,
        )
        scale = 2 * pre_scan_unit / grid_half_size
        self.zero_grid = scale * int_grid.to(torch.float)
        self.refine_grid = scale * int_grid[int_grid % 2 == 1].to(torch.float)

    def find_angle(self, image) -> float:
        """
        Args:
            image: image of the document

        Returns:
            Estimated tilt-angle in degree
        """

        def _calc_scatter_index():
            angles = test_angles + shift_angle
            scatter_index = (torch.outer(angles.cos(), work_y) + torch.outer(angles.sin(), work_x) + y_shift).to(
                torch.long
            )
            return scatter_index

        def _calc_square_sum():
            nb_grid = test_angles.shape[0]
            scatter_index = _calc_scatter_index()

            sum_tensor = torch.zeros([nb_grid, stretch * radius], dtype=torch.float, device=self.device)
            sum_tensor.scatter_add_(1, scatter_index, dark_tensor.expand(nb_grid, -1))
            sum_tensor = sum_tensor.cumsum(dim=1)
            sum_tensor = sum_tensor[:, stretch:] - sum_tensor[:, :-stretch]
            sum_tensor -= sum_tensor.mean(dim=1, keepdim=True)
            projection_range_factor = (
                (test_angles.sin() * width).square() + (test_angles.cos() * height).square()
            ).sqrt()
            square_sum = sum_tensor.square().sum(1) * projection_range_factor
            return square_sum

        width, height = image.size
        # Convert image to grayscale tensor and invert
        image_tensor = 1.0 - ToTensor()(image.convert("L"))
        reduce_factor = (self.nb_pixel / (width * height)) ** 0.5
        if reduce_factor >= 1:
            reduce_factor = 1
            small_image_tensor = image_tensor.squeeze(0)
        else:
            # The image_tensor has the shape [1, height, width]. To use interpolate, we need a 4D tensor of shape
            # [1, 1, height, width]. At the end, we remove the two dummy dims.
            small_image_tensor = (
                torch.nn.functional.interpolate(
                    image_tensor.unsqueeze(0),
                    scale_factor=reduce_factor,
                )
                .squeeze(0)
                .squeeze(0)
            )
        height, width = small_image_tensor.shape[0], small_image_tensor.shape[1]
        stretch = int(4 / reduce_factor + 0.5)
        radius = int((width**2 + height**2) ** 0.5)
        if radius % 2 == 0:
            radius += 2
        else:
            radius += 3
        y_shift = stretch * radius / 2
        shift_angle = EXTRA_ANGLE_RADIAN
        work_tensor = rotate(
            small_image_tensor.unsqueeze(0),
            EXTRA_ANGLE_DEGREE,
            expand=True,
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0)
        work_height, work_width = work_tensor.shape[0], work_tensor.shape[1]
        dark_threshold = torch.maximum(
            torch.tensor(MINIMAL_DARK_VALUE, device=self.device),
            small_image_tensor.mean(),
        )
        indices = torch.where(work_tensor > dark_threshold)
        work_y = stretch * (indices[0] - work_height / 2)
        work_x = stretch * (indices[1] - work_width / 2)
        if work_x.shape[0] < MIN_NB_DARK_PIXEL:
            return 0
        dark_tensor = work_tensor[indices].view(-1)
        test_angles = self.pre_scan_angles
        results = _calc_square_sum()
        best_angle = test_angles[results.argmax()]
        if best_angle.abs() > (EXTRA_ANGLE_RADIAN - best_angle).abs():
            work_tensor = small_image_tensor
            shift_angle = 0
            indices = torch.where(work_tensor > dark_threshold)
            work_y = stretch * (indices[0] - height / 2)
            work_x = stretch * (indices[1] - width / 2)
            dark_tensor = work_tensor[indices].view(-1)
        accumulated_results = torch.tensor([], device=self.device)
        accumulated_angles = torch.tensor([], device=self.device)
        test_angles = best_angle + self.zero_grid
        refine_angles = self.refine_grid.clone()
        for _ in self.loop:
            results = _calc_square_sum()
            accumulated_results = torch.cat([accumulated_results, results])
            accumulated_angles = torch.cat([accumulated_angles, test_angles])
            best_angle = accumulated_angles[accumulated_results.argmax()]
            # prepare next round
            refine_angles /= 2
            test_angles = best_angle + refine_angles
        # We return an angle in degree and with -90 <= angle < 90
        best_angle_degree = (best_angle * RADIAN2DEGREE + 90) % 180 - 90
        return best_angle_degree.item()


def _closest_90_degree_distance(angle: float) -> float:
    """
    Returns the smallest distance to the nearest multiple of 90 degrees.
    The distance is negative if the angle is below the nearest multiple of 90,
    and positive if it is above.
    """
    nearest_multiple_of_90 = round(angle / 90) * 90
    distance = angle - nearest_multiple_of_90
    return distance


def correct_tilt(image: Image.Image, tilt_threshold: float = 4) -> tuple[Image.Image, float]:
    """
    Corrects the tilt (small rotations) of an image of a document page

    Args:
        image: Image to correct the tilt of
        tilt_threshold: The maximum tilt angle to correct. If the angle is larger than this, the image is not rotated at all.

    Returns:
        The rotated image and the angle of rotation
    """
    detect_tilt = DetectTilt()
    angle = detect_tilt.find_angle(image)
    angle = _closest_90_degree_distance(angle)  # We round to the nearest multiple of 90 degrees
    # We only rotate if the angle is small enough to prevent bugs introduced by the algorithm
    angle = angle if abs(angle) < tilt_threshold else 0.0
    rotated_image = image.rotate(-angle, expand=True, fillcolor="white")
    return rotated_image, angle
