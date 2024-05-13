import albumentations as A
import math


def transform_to_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1

    return num


def medical_augment(level=5):
    pixel_transforms = [
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, transform_to_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level)
    ]

    spatial_transforms = [
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                 crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 2 * level), 'y': (0, 0)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 0), 'y': (0, 2 * level)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level)
    ]

    transforms_1_2 = [
        A.Compose([
            A.OneOf(pixel_transforms, p=1),
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(spatial_transforms, p=1),
        ], p=1 / 3),
        A.Compose([
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(pixel_transforms, p=1),
            A.OneOf(spatial_transforms, p=1),
        ], p=1 / 3),
        A.Compose([
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(pixel_transforms, p=1),
        ], p=1 / 3)
    ]

    transforms_0_3 = [
        A.Compose([
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(spatial_transforms, p=1)
        ], p=1),

    ]

    transforms_0_2 = [
        A.Compose([
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(spatial_transforms, p=1)
        ], p=1),
    ]

    transforms_1_1 = [
        A.Compose([
            A.OneOf(pixel_transforms, p=1),
            A.OneOf(spatial_transforms, p=1)
        ], p=1 / 2),
        A.Compose([
            A.OneOf(spatial_transforms, p=1),
            A.OneOf(pixel_transforms, p=1)
        ], p=1 / 2)
    ]

    MedAugment = A.OneOf([
        A.OneOf(transforms_1_2, p=1),
        A.OneOf(transforms_0_3, p=1),
        A.OneOf(transforms_0_2, p=1),
        A.OneOf(transforms_1_1, p=1)
    ], p=1)

    return MedAugment
