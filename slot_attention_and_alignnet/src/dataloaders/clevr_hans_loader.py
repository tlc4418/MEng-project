import json
import os
import jax.numpy as jnp
import jax
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import numpy as np
from tensorflow.data.experimental import AUTOTUNE
from pathlib import Path
from pycocotools import mask as maskUtils

tf.config.experimental.set_visible_devices([], "GPU")


class CLEVRHansLoader:
    def __init__(self, datapath, data_cfg, split, variant=3, get_masks=False):
        # Datapath should be
        assert split in ["train", "validation", "test"]
        self.split = split
        self.data_cfg = data_cfg
        self.variant = variant
        self.get_masks = get_masks  # in case masks are desired at test time
        self.load_ds(datapath, split, data_cfg["batch_size"], variant)

    def load_ds(self, datapath, split, batch_size, variant):
        builder = CLEVRHans(data_dir=datapath, variant=variant)
        builder.download_and_prepare()

        ds = builder.as_dataset(split=split)

        ds.cache()

        if split == "train":
            ds = ds.shuffle(1000 * batch_size)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)

        if split == "train":
            ds = ds.repeat()

        self.ds = tfds.as_numpy(ds)
        self.ds_iter = iter(self.ds)

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            try:
                batch = next(self.ds_iter)
            except StopIteration:  # For testing, reset iterator after one full pass
                self.ds_iter = iter(self.ds)
                raise StopIteration

            # Resize images to 128x128 as per past research, and normalize
            images = jnp.array(batch["image"])
            images = images[:, 16:304, 96:384]
            images = jax.image.resize(
                images, (images.shape[0], 128, 128, 3), "bilinear"
            ).astype(np.uint8)
            images = ((images / 255.0) - 0.5) * 2.0

            # Retrieve mask for computing ARI scores if requested
            masks = []
            if self.get_masks:
                rle_masks = batch["objects"]["mask"]
                counts = rle_masks["counts"]
                sizes = rle_masks["size"]
                masks = np.stack(list(map(mask_decode, counts, sizes)), axis=0)
                masks = jnp.array(masks, dtype=bool)
                masks = masks[:, :, 16:304, 96:384]
                masks = jax.image.resize(
                    masks, (masks.shape[0], masks.shape[1], 128, 128), "bilinear"
                ).astype(bool)

            return images, masks


def mask_decode(counts, sizes):
    return np.stack(
        list(
            map(lambda x, y: maskUtils.decode({"counts": x, "size": y}), counts, sizes)
        ),
        axis=0,
    )


# DATA BUILDER
_BASE_URL = "https://github.com/ml-research/CLEVR-Hans"
_DOWNLOAD_URL_3 = "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2611/CLEVR-Hans3.zip"
_DOWNLOAD_URL_7 = "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2618/CLEVR-Hans7.zip"


class CLEVRHans(tfds.core.GeneratorBasedBuilder):
    """CLEVRHans dataset."""

    MAX_OBJECTS = 10
    # For padding
    null_mask = maskUtils.encode(np.zeros((320, 480, 1), order="F", dtype=np.uint8))[0]
    NULL_OBJ = {
        "color": "gray",
        "material": "metal",
        "shape": "none",
        "size": "small",
        "rotation": 0.0,
        "pixel_coords": [0.0, 0.0, 0.0],
        "3d_coords": [0.0, 0.0, 0.0],
        "mask": null_mask,
    }

    def __init__(self, *args, **kwargs):
        if "variant" in kwargs:
            variant = str(kwargs["variant"])
            assert variant in ["3", "7"], ValueError("Variant must be 3 or 7")
            self.VERSION = tfds.core.Version(f"{variant}.0.0")
            del kwargs["variant"]
        else:  # default to 3
            self.VERSION = tfds.core.Version("3.0.0")

        return super().__init__(*args, **kwargs)

    def _info(self):
        features = {
            "image": tfds.features.Image(),
            "file_name": tfds.features.Text(),
            "class_id": tfds.features.ClassLabel(
                names=list(map(str, range(int(str(self.VERSION)[0]))))
            ),
            "num_obj": tfds.features.Tensor(shape=(), dtype=tf.uint8),
            "objects": tfds.features.Sequence(
                {
                    "color": tfds.features.ClassLabel(
                        names=[
                            "gray",
                            "blue",
                            "brown",
                            "yellow",
                            "red",
                            "green",
                            "purple",
                            "cyan",
                        ]
                    ),
                    "material": tfds.features.ClassLabel(names=["rubber", "metal"]),
                    "shape": tfds.features.ClassLabel(
                        names=["cube", "sphere", "cylinder", "none"]
                    ),
                    "size": tfds.features.ClassLabel(names=["small", "large"]),
                    "rotation": tfds.features.Tensor(shape=(), dtype=tf.float32),
                    "3d_coords": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                    "pixel_coords": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                    "mask": tfds.features.FeaturesDict(
                        {
                            "counts": tf.string,
                            "size": tfds.features.Tensor(shape=(2,), dtype=tf.float32),
                        }
                    ),
                }
            ),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(features),
            homepage=_BASE_URL,
        )

    def _split_generators(self, dl_manager):
        """Returns splits."""
        variant = str(self.VERSION)[0]
        download_url = _DOWNLOAD_URL_3 if variant == "3" else _DOWNLOAD_URL_7

        path = Path(dl_manager.download_and_extract(download_url))
        path = path / f"CLEVR-Hans{variant}"
        splits = []
        name_map = {
            "train": tfds.Split.TRAIN,
            "val": tfds.Split.VALIDATION,
            "test": tfds.Split.TEST,
        }

        for split_name in ["train", "val", "test"]:
            splits.append(
                tfds.core.SplitGenerator(
                    name=name_map[split_name],
                    gen_kwargs={
                        "images_dir_path": path / split_name / "images",
                        "scenes_description_file": path
                        / split_name
                        / f"CLEVR_HANS_scenes_{split_name}.json",
                    },
                )
            )

        return splits

    def _generate_examples(self, images_dir_path, scenes_description_file):

        with tf.io.gfile.GFile(scenes_description_file) as f:
            scenes_json = json.load(f)

        attrs = [
            "color",
            "material",
            "shape",
            "size",
            "rotation",
            "pixel_coords",
            "3d_coords",
            "mask",
        ]
        for scene in scenes_json["scenes"]:
            objects = scene["objects"]
            num_obj = len(objects)
            fname = scene["image_filename"]
            record = {
                "image": os.path.join(images_dir_path, fname),
                "file_name": fname,
                "class_id": str(scene["class_id"]),
                "num_obj": num_obj,
                "objects": [{attr: obj[attr] for attr in attrs} for obj in objects]
                + [self.NULL_OBJ]
                * (
                    self.MAX_OBJECTS - num_obj
                ),  # pylint: disable=g-complex-comprehension
            }
            yield fname, record
