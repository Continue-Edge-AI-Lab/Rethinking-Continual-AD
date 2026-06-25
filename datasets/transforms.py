"""
Code to perform our transformations for the MTD dataset given for the continual learning task.
"""

from datasets import *

# Set random seed for reproducibility
random.seed(42)

class color_transform():
    """
    Applies a color transform to the image.
    Args:
        img: Image to be transformed
        window: Tuple of window values to be used for the color transform
    Returns:
        Transformed image
    """
    def __init__(self, window, replay):
        self.window = window
        self.replay = replay
        return

    def __call__(self, img):
        # Apply color jitter
        value = random.uniform(0 if self.replay else self.window[0], self.window[1])
        pos = random.randint(0, 1)
        value = value + 1 if pos else 1 - value

        img = F.adjust_brightness(img, value)
        img = F.adjust_contrast(img, value)
        img = F.adjust_saturation(img, value)

        return img

class geometric_transform():
    """
    Applies a geometric transform to the image.
    Args:
        img: Image to be transformed
        degrees: +/- degree range to rotate the image
        translate: Tuple of translation values to be used for the transform
        scale: Scale factor
        shear: Shear factor
    Returns:
        Transformed image
    """
    def __init__(self, degrees, translate, scale, shear, replay):
        self.replay = replay

        if replay:
            self.degrees = degrees
            self.translate = translate
            self.scale = scale
            self.shear = shear
        else:
            self.degrees = random.uniform(degrees-2, degrees)
            self.translate = [random.uniform(translate-1, translate),
                              random.uniform(translate-1, translate)]
            self.scale = random.uniform(scale-0.01, scale)
            self.shear = random.uniform(shear-1, shear)

        return

    def __call__(self, img):
        if self.replay:
            # Will give us a rotation of +/- degrees
            degrees = random.uniform(-self.degrees, self.degrees)
            # Need two random numbers, one for vertical and one for horizontal
            translate = [random.uniform(0, self.translate),
                              random.uniform(0, self.translate)]
            scale = random.uniform(1 - self.scale, 1 + self.scale)
            shear = random.uniform(-self.shear, self.shear)
            return F.affine(img,
                            degrees,
                            translate,
                            scale,
                            shear)

        else:
            return F.affine(img,
                        self.degrees,
                        self.translate,
                        self.scale,
                        self.shear)
