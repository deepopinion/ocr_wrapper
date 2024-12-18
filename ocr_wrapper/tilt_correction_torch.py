from __future__ import annotations

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed, but is required for tilt-correction. Either install the 'torch' package or set the environment variable OCR_WRAPPER_NO_TORCH to use numpy and scipy instead."
    )

try:
    from torchvision.transforms import ToTensor
    from torchvision.transforms.functional import InterpolationMode, rotate
except ImportError:
    raise ImportError(
        "The 'torchvision' package is not installed, but is required for tilt-correction. Please install the 'torchvision' package."
    )

from opentelemetry import trace
from PIL import Image

tracer = trace.get_tracer(__name__)


# ---------------- GENERAL IDEA ----------------------------------------------------------------------------------------------
# We like to find a potential tilt angle a document scan might have picked up.
# Usually, text-documents contain text in lines. To find the tilt angle, we look for the direction of the lines.
# The idea is to define a gain-function and calculate this gain-function for various tilt-angle.
# We start with the simple observation that text-lines contain a lot of dark pixels, while the spaces between the
# lines have no black pixel. Now, in the simplest form of the algorithm, we project all pixels of the 2D image onto
# the y-axis - i.e. we sum over the x-axis. If the lines are parallel to the x-axis, the projection contains sections with
# a large sum of black pixels and sections with ideally no black pixels. That's what we ar looking for: strong variations
# in the black-pixel density of the projection. A gain-function calculated as sum of the squared values of the projection
# values measures these fluctuations.
# The real algorithm uses some extra tricks as e.g. rotating the projection axis instead of the image, but this is explained
# in the code.


# We define a minimum number of pixels - we don't do an angle estimation with less and return the angle 0° instead
MIN_NB_DARK_PIXEL = 100
# We filter white pixel and keep only black pixels. However, real live has shades of gray. One criterion for dark is being over
# the average value (pitch-dark is 1, and white is 0, due to inversion of color). Further, we define a MINIMAL_DARK_VALUE,
# as second condition. This might be interesting for other types of images than pure text-pictures. However, currently,
# it's just an option and we use the value 0.0, which has no effect.
MINIMAL_DARK_VALUE = 0.0
RADIAN2DEGREE = 180 / torch.pi
ANGLE90 = torch.pi / 2
# For our tests, we work with a rotated document to shift the low-resolution area around 0°. We use a radian value, which
# avoids coming to close to rational fraction in degree (!)
EXTRA_ANGLE_RADIAN = 0.65
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
        nb_fine_scan: The fine scan is performed in `nb_fine_scan` rounds. After each round, step_size is halved, as well
            as the search area. The standard value for `nb_fine_scan` is 5.
    """

    def __init__(
        self,
        device: str = "cpu",
        nb_pixel: int = 500000,
        nb_pre_scan: int = 120,
        grid_half_size: int = 20,
        nb_fine_scan: int = 5,
    ):
        self.device = device
        self.nb_pixel = nb_pixel
        self.loop = range(nb_fine_scan)
        pre_scan_unit = torch.pi / nb_pre_scan
        # The pre-scan grid covers angles in the interval [-90°, 90°[.
        self.pre_scan_angles = torch.arange(nb_pre_scan) * pre_scan_unit - ANGLE90
        # We define our fine-scan grids. We start with an auxiliary int_grid, which has the right amount of values
        # left and right from the zero value. However, the step-size is one. The latter will changed for the grids we'll
        # really use.
        int_grid = torch.linspace(
            -grid_half_size,
            grid_half_size,
            2 * grid_half_size + 1,
            dtype=torch.long,
            device=self.device,
        )
        # If the gain-function was a smooth function with just one maximum and without local bumps, we could use a very strict
        # search strategy. Alas, real gain-function might have bumps. Therefore, for the first fine-scan, we search the area
        # given by two pre-scan-grid-points left and two pre-scan-grid-points right of the best-angle. Using this approach,
        # We first define the scale - i.e. the distance between two grid-points (angles) in the fine-scan.
        scale = 2 * pre_scan_unit / grid_half_size
        # Here, we define a grid around 0°. Later, this is easily adapted adding the best pre-scan angle, which shifts the
        # center of the grid to this value.
        self.zero_grid = scale * int_grid.to(torch.float)
        # For the fine-scan, local bumps can extend over longer distance. So, the following refinements can't be as drastic
        # as after the pre-scan. We'll just half the grid size and with that the grid steps. Sill, these iterative
        # halvings will happen several times and we don't do them here. However, we will recycle results and won't recalculate
        # results already obtained. As a consequence, after the first fine-scan, we can remove every second grid point for
        # the refined grid.
        self.refine_grid = scale * int_grid[int_grid % 2 == 1].to(torch.float)
        # We define a contrast-kernel used for convolution. We subtract a weighted average of the surroundings from the
        # central pixel. If all pixels have the same grayscale, the result is zero.
        # The definition we use is not mandatory - many alternatives are possible. This form simply worked well.
        self.contrast_kernel = -torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 1.0, 1.0, 1.0, 0.5],
                [0.5, 1.0, -16, 1.0, 0.5],
                [0.5, 1.0, 1.0, 1.0, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            device=self.device,
        )
        # Torch's conv2d function expects a batch- and a color-dimension at the beginning of the kernel.
        self.contrast_kernel = self.contrast_kernel.unsqueeze(0).unsqueeze(0)

    @tracer.start_as_current_span(name="DetectTilt.find_angle-torch")
    def find_angle(self, image: Image.Image) -> float:
        """
        Args:
            image: image of the document

        Returns:
            Estimated tilt-angle in degree
        """

        def _calc_scatter_index() -> torch.Tensor:
            """
            We use "scatter_add_" to sum the pixels and this function needs an index which tells it where each pixel
            goes, respectively to which position its value is added. This index is creates here. It depends on the
            (x, y) position of the original pixel.

            Returns:
                An 2D index of shape [angle, goto-position] for each pixel in the "dark_tensor" (defined in main function)
            """
            # Due to resolution problems around the 0° position, we rotated the image by a dummy angle. Here, we do
            # a correction with "shift_angle". That is test_angles corresponds to the tilt of original document and "angles"
            # to the rotated image, whose values are added later on.
            angles = test_angles + shift_angle
            # We don't rotate the image but "rotate" (i.e. adapt) the position to which the pixels are added. We do a
            # rotation around the center and half of the pixels coordinates translate into negative valued positions if
            # we just do the rotation. Since we need positive array-insices as result, we add "y_shift".
            scatter_index = (torch.outer(angles.cos(), work_y) + torch.outer(angles.sin(), work_x) + y_shift).to(
                torch.long
            )
            return scatter_index

        def _calc_square_sum() -> torch.Tensor:
            """
            This function calculates the gain-function used to evaluate the angles.

            Returns:
                gain-function-tensor with one value for each tested angle.
            """

            # We use scatter_add_ to add each pixel listed in dark_tensor to a certain position in the sum_tensor.
            # This position depends on the tested angle. This is equivalent to rotating the image and projecting the
            # 2D pixel tensor onto the y-axis - just without the need of rotation and with a stretch y-axis.
            nb_grid = test_angles.shape[0]
            # First, we get the angle-depended target-positions as "sctter_index"
            scatter_index = _calc_scatter_index()
            # sum_tensor has the shape [tested_angle, 1D projection]. Here, we stretch the projection axis to have a
            # higher resolution.
            sum_tensor = torch.zeros([nb_grid, stretch * diameter], dtype=torch.float, device=self.device)
            sum_tensor.scatter_add_(1, scatter_index, dark_tensor.expand(nb_grid, -1))
            # The stretching alone is not enough. We also have to smear out the results. If we have e.g. a stretch factor of
            # 10, we need to smear out each sum_tensor point over 10 positions. Without going into details: If we don't do it,
            # our stretching might genrate "holes", which cause strong variances. Since we search for high variances, we are not
            # allowed to create false fluctuations. Here, "smearing out" means each value is also added the 9 consecutive
            # index positions (for our example of stretch 10). This is done most efficiently by the difference of two
            # cumulative sums (cumsum). That is, If you sum all numbers from 1 to (n + 10) and subtract all numbers from
            # 1 to n, you get n + (n+1) + (n+2) + ... + (n+10). Thats the idea. For more details, fallow the code.
            sum_tensor = sum_tensor.cumsum(dim=1)
            sum_tensor = sum_tensor[:, stretch:] - sum_tensor[:, :-stretch]
            sum_tensor -= sum_tensor.mean(dim=1, keepdim=True)
            # The gain-function is the sum of squared values. To have a far comparison of angles, we have to make sure that
            # e.g. the projection over the short image side has no advantage compared to the projection over the long side.
            # To see the problem, assume a uniform pixel distribution where the number of pixels on a line corresponds just
            # to the length of the line. This value is squared so i.e. length**2, which might be width**2 for 0° and
            # height**2 for 90°. Assuming that the number of lines depends on the perpendicular axis, we get width**2 * height
            # for 0° and height**2 * width for 90° in total. Remember: This is for a uniform distribution, where we should find no
            # angle preference. To counter this effect, we introduce the projection_range_factor, which is defined for all
            # angles. Actually, for angles different from 0° and 90°, the correct way would be calculating an individual
            # factor for each line. However, we go with a single factor, which should suffice as approximation.
            projection_range_factor = (
                (test_angles.sin() * width).square() + (test_angles.cos() * height).square()
            ).sqrt()
            square_sum = sum_tensor.square().sum(1) * projection_range_factor
            return square_sum

        width, height = image.size
        # Convert image to grayscale tensor and invert value. After that, black is 1 and white is 0
        image_tensor = 1.0 - ToTensor()(image.convert("L"))
        # As approximation, we reduce all images to roughly the same number of pixels given by self.nb_pixel.
        # We calculate a reduction_factor. Using the square root of the fraction of the new number of pixels divided by
        # original number of pixels (width * heigt), we can use the reduction_factor on the height and width, later on.
        reduce_factor = (self.nb_pixel / (width * height)) ** 0.5
        # Notice that next to downsizing, we unsqueeze dim=0
        if reduce_factor >= 1:
            # small image - stays unchanged
            reduce_factor = 1
            small_image_tensor = image_tensor.unsqueeze(0)
        else:
            small_image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                scale_factor=reduce_factor,
            )
        # Originally, the algorithm was designed for images of white paper with black ink. To extend the algorithm beyond
        # these limitations, we use a "contrast_kernel" to transform colorful pictures with text into mostly black and white
        # objects. The contrast_kernel is a contrast filter which erases homogene pixel areas of similar gray scale
        # (i.e. turns them into 0, which corresponds to white, after the inversion we made at the beginning).
        # Only contrast survives (as letters).
        small_image_tensor = (
            torch.nn.functional.conv2d(
                small_image_tensor,
                self.contrast_kernel,
                padding="valid",
            )
            .squeeze(0)
            .squeeze(0)
        )
        # The contrast filter might have created negative values, which we remove now.
        # After that, we square the results. The reason is that the contrast filter only removes perfectly homogenous areas
        # of the some color - for all other pixel-areas we just reduce the pixel values. Squaring suppresses these small
        # values even further
        small_image_tensor = torch.maximum(torch.tensor(0, device=self.device), small_image_tensor).square()
        # The contrast filter reduces the image slightly, so we refresh width and height.
        height, width = small_image_tensor.shape[0], small_image_tensor.shape[1]
        # For the gain function, the image is projected onto a 1D tensor, which in a first step, can be thought of as a
        # rotated y-axis. However, we stretch this y-axis to increase the angle resolution (especially around 0°).
        # Using a stretch comes with the necessity of a post-processing step. We choose a big stretch, which is still small
        # enough to have a significant fast postprocessing.
        stretch = int(4 / reduce_factor + 0.5)
        # We simulate rotating the image by actually rotating the projection axis. The rotation is around the center of the
        # image. We have to consider the maximal length of this projection axis (without stretch, first). This is given by
        # the length of the diagonal (here called diameter), which we calculate with Pythagoras
        diameter = int((width**2 + height**2) ** 0.5)
        # We add some slack for round off mistakes and make sure the diameter is even -> integer radius = diameter / 2
        if diameter % 2 == 0:
            diameter += 2
        else:
            diameter += 3
        # We emulate rotation around the center, where half of the coordinates is negative. Since we need positive numbers,
        # we define a y_shift, which is the stretched radius.
        y_shift = stretch * diameter / 2
        # Ideally, most images have roughly the correct rotation which is around 0° tilt. Unfortunately, that's the area of
        # lowest resolution. We use the simple trick to rotate the image a bit, to place the low-resolution-angle in a
        # unlikely position. After the pre-scan, we might change this angle in case of the unlikely event that the tilt angle
        # matches our EXTRA_ANGLE_RADIAN.
        shift_angle = EXTRA_ANGLE_RADIAN
        work_tensor = rotate(
            small_image_tensor.unsqueeze(0),
            EXTRA_ANGLE_DEGREE,
            expand=True,
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0)
        # During rotation, we expanded the image, so we have a new image size. However, for some values (as the diameter),
        # we can still work with the old width and height (since the new pixels are just white and are removed in the
        # dark_tensor, calculated below).
        work_height, work_width = work_tensor.shape[0], work_tensor.shape[1]
        # For none of the following calculations, we need the 2D structure of the image - so we can use a 1D list of pixels.
        # The advantage is that we can drop all pixels, which are not sufficiently dark.
        # So, we define a threshold for "sufficiently dark". One part of the threshold is the average darkness, the other
        # part is the predefined value MINIMAL_DARK_VALUE. Alas, currently, this value is zero because due to the contrast
        # filter and the fact that we use all kinds of images, it's hard to determine a meaningful cutoff that works for all
        # image types.
        dark_threshold = torch.maximum(
            torch.tensor(MINIMAL_DARK_VALUE, device=self.device),
            small_image_tensor.mean(),
        )
        # Determine all pixel-indices of pixels above the dark_threshold (i.e. the ones which survive the filtering)
        indices = torch.where(work_tensor > dark_threshold)
        # We emulate rotating the image for various test_angles. For the rotation, we need the (x,y) positions of the pixels
        # which survived the filtering. However, we need to transform the results a bit. First note that we emulate rotating
        # around the center - so we subtract the center positions (work_height / 2 and work_width / 2). Second, we stretch
        # the projection. Mathematical, we pretend the image was larger and hence, we stretch the coordinates.
        work_y = stretch * (indices[0] - work_height / 2)
        work_x = stretch * (indices[1] - work_width / 2)
        if work_x.shape[0] < MIN_NB_DARK_PIXEL:
            # We don't have enough dark pixels. It makes no sense to estimate the tilt angle. Our best chance is to
            # leave the document unchanged and return 0° as tilt angle
            return 0
        # The following line is mere precaution - thr shape should already match the view.
        dark_tensor = work_tensor[indices].view(-1)

        # The purely image depended preparations are done.
        # Now, we prepare the scans. First a pre-scan is made followed by iteratively refined fine-scans.
        test_angles = self.pre_scan_angles  # angles from -90° to 90°
        # We calculate the gain-function
        pre_scan_results = _calc_square_sum()
        # Get the best angle (highest gain-function)
        best_angle = test_angles[pre_scan_results.argmax()]
        # Remember that we added a dummy angle to shift the 0° position into an unlikely region. However, if the true
        # tilt angle comes to close to the dummy angle, we remove the dummy angle. This is done by simply taking the
        # original document (which is no longer a problem, because the tilt angle is far from 0°)
        if (EXTRA_ANGLE_RADIAN - best_angle).abs() < EXTRA_ANGLE_RADIAN / 3:
            work_tensor = small_image_tensor
            shift_angle = 0
            # We need to repeat some of the calculations above
            indices = torch.where(work_tensor > dark_threshold)
            work_y = stretch * (indices[0] - height / 2)
            work_x = stretch * (indices[1] - width / 2)
            dark_tensor = work_tensor[indices].view(-1)
        # Next we do the fine-scan. We do several iteration loops. Each time, we divide the step-size by two. Further, we
        # recycle previous results (no need to calculate them again). We use "accumulated_results" and the corresponding
        # "accumulated_angles" to remember all.
        accumulated_results = torch.tensor([], device=self.device)
        accumulated_angles = torch.tensor([], device=self.device)
        # We predefined "self.zero_grid" which stretches left and right around 0°. By adding the so far best "best_angle",
        # we obtain the grid for the first fine_scan. After that, we do a similar trick with self_refined_grid. Due to
        # recycling, we need to calculate less angles after the first fine-scan. However, this reduction of the number of grid
        # angles works only once. This is best seen with an example. Assume we calculated the even angles 2, 4, 6, 8, ... .
        # When we half the grid steps, we obtain the angles 1, 2, 3, 4, ... . Evidently, we do not need to recalculate the
        # even angles, Leaving us with 1, 3, 5, ... . However, the next halving generates the angles 0.5, 1.5, 2.5, ... .
        # Here, we can't discard any grid angles. Same for the next halving 0.25, 0.75, 1.25, ... and all further halvings.
        test_angles = best_angle + self.zero_grid
        refine_angles = self.refine_grid.clone()
        for _ in self.loop:
            results = _calc_square_sum()
            accumulated_results = torch.cat([accumulated_results, results])
            accumulated_angles = torch.cat([accumulated_angles, test_angles])
            best_angle = accumulated_angles[accumulated_results.argmax()]
            # prepare next round - see explanation above
            refine_angles /= 2
            test_angles = best_angle + refine_angles
        # Done! We obtained a best_angle in radian, which we transform is degree such that the returned angles is
        # -90° and 90
        best_angle_degree = (best_angle * RADIAN2DEGREE + 90) % 180 - 90
        return best_angle_degree.item()
