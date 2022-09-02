import numpy as np
import webdataset as wds
from ldm.data.base import Txt2ImgIterableBaseDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class ImageDataset(Dataset):

    def __init__(self, image_list_file,
                 size=256,
                 hflip=False,
                 scale=(0.25, 1.0)):
        super().__init__()

        # Open image list file assuming one path for each line
        with open(image_list_file, "r") as f:
            images = f.readlines()

        images = [l.strip() for l in images]
        print(f"num files: {len(images)}")
        print(f"top 5 files: {images[:5]}")
        self.images = images

        size = (size, size) if not isinstance(size, (list, tuple)) else size
        self.transform = transforms.Compose(
            ([transforms.RandomHorizontalFlip(p=0.5), ] if hflip else [])
            + [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=scale,
                    interpolation=3
                )
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        # XXX Following ldm.data.imagenet.ImageNetSR
        img = np.asarray(img)

        # Normalize
        # [0, 255] -> [-1, 1]
        img = (img/np.float32(255) - 0.5) * 2

        return {
            "image": img
        }


class WebdatasetImageCaptionDataset(Txt2ImgIterableBaseDataset):

    def __init__(
        self,

        urls,
        shuffle=1000,

        num_records=0,
        valid_ids=None,

        size=256,

        # Image augmentations
        hflip=True,
        random_crop=True,
        random_crop_scale=(0.5, 1.0)
    ):
        self.urls = urls
        self.shuffle = shuffle

        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

        self.size = size

        self.hflip = hflip
        self.random_crop = random_crop
        self.random_crop_scale = random_crop_scale

    def __len__(self):
        return self.num_records

    def transform(self, d):
        # Get data
        img = d["jpg"]
        caption = d["json"]["caption"]
        key = d["__key__"]

        size = self.size
        size = (size, size) if not isinstance(size, (list, tuple)) else size

        flip = [transforms.RandomHorizontalFlip(p=0.5), ] if self.hflip else []
        crop = [
            transforms.RandomResizedCrop(
                size=size,
                scale=self.random_crop_scale,
                interpolation=3
            )
        ] if self.random_crop else [transforms.Resize(size, interpolation=3), ]

        transform = transforms.Compose(
            flip + crop
        )

        # Apply transforms
        img = transform(img)
        # PIL to np array
        img = np.array(img)
        # Normalize [0, 255] -> [-1, 1]
        img = ((img / np.float32(255)) - 0.5) * 2

        return {
            "key": key,
            "image": img,
            "caption": caption
        }

    def __iter__(self):
        ds = wds.WebDataset(
            self.urls,
            nodesplitter=wds.split_by_node,
            handler=wds.handlers.warn_and_continue,
            verbose=True
        )
        if self.shuffle:
            ds = ds.shuffle(self.shuffle)
        ds = ds.decode("pil")
        ds = ds.map(self.transform)
        return iter(ds)
