from os.path import join

from torch.utils.data import DataLoader

from image_loader import ImageLoader
from image_loader import calculate_image


def get_path(root: str, name: str) -> (str, str):
    base_path = join(root, name)
    return join(base_path, 'images'), join(base_path, 'masks')


class ISICImageLoader(ImageLoader):
    def __init__(
            self,
            mode,
            data_dir=None,
            one_hot=True,
            image_size=224,
            aug=None,
            aug_empty=None,
            transform=None,
            img_transform=None,
            msk_transform=None,
            add_boundary_mask=False,
            add_boundary_dist=False,
            support_types: [str] = None,
            gt_format: str = "{}.png"
    ):
        assert mode == 'train' or mode == 'valid', "Mode must be one of ['train', 'valid']"
        name = 'train' if mode == 'train' else 'val'
        self.gt_format = gt_format
        super().__init__(
            mode=mode,
            data_dir=join(data_dir, name),
            one_hot=one_hot,
            image_size=image_size,
            aug=aug,
            aug_empty=aug_empty,
            transform=transform,
            img_transform=img_transform,
            msk_transform=msk_transform,
            add_boundary_mask=add_boundary_mask,
            add_boundary_dist=add_boundary_dist,
            support_types=support_types

        )

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        try:
            return self.gt_format.format(origin_image_name)
        except:
            return origin_image_name + "_segmentation.png"

