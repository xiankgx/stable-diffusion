import glob
import os

import numpy as np
import pandas as pd
import webdataset as wds
from ldm.data.base import Txt2ImgIterableBaseDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class ImageCaptioningDataset(Dataset):

    def __init__(self, annotations_csv_file=None,
                 image_col="image",
                 text_col="caption",
                 image_root=None,
                 size=256,
                 hflip=False,
                 random_crop=True,
                 random_crop_scale=(0.8, 1.0),
                 repeats=1):
        super().__init__()

        if annotations_csv_file is not None:
            # Assuming image paths and captions providied in annotations csv file

            df = pd.read_csv(annotations_csv_file)

            if image_root is not None:
                df[image_col] = df[image_col].map(
                    lambda p: os.path.join(image_root, p)
                )

            images = df[image_col].tolist()
            captions = df[text_col].tolist()
        else:
            # No annotations csv file provided, assuming image filenames as captions

            assert image_root is not None, "Must provide at least one of 'annotations_csv_file' and 'image_root'."
            images = glob.glob(image_root + "/**/*.jpg", recursive=True) + \
                glob.glob(image_root + "/**/*.png", recursive=True)
            captions = list(map(lambda p: os.path.splitext(os.path.basename(p))[0],
                                images))

        # Repeat lists for some number of times
        if repeats > 1:
            images = images * repeats
            captions = captions * repeats

        print(f"num files: {len(images)}")
        # print(f"top 5 files: {images[:5]}")
        self.images = images
        self.captions = captions

        # Image transforms
        size = (size, size) if not isinstance(size, (list, tuple)) else size
        flip = [transforms.RandomHorizontalFlip(p=0.5), ] if hflip else []
        crop = [
            transforms.RandomResizedCrop(
                size=size,
                scale=random_crop_scale,
                ratio=(1, 1),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        ] if random_crop else [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
        ]
        self.transform = transforms.Compose(
            flip + crop
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        caption = self.captions[idx]

        img = self.transform(img)

        # Output image in np.ndarray format
        img = np.asarray(img)

        # Normalize: [0, 255] -> [-1, 1]
        img = (img/np.float32(255) - 0.5) * 2

        return {
            "image": img,
            "caption": caption
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
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        ] if self.random_crop else [transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), ]

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
