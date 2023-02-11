import functools
from pathlib import Path
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


class DataLoader:
    def __init__(self, datacontroller, data_cfg, masks=False, split="train"):
        self.split = split
        self.data_cfg = data_cfg
        self.masks = masks
        self.load_ds(datacontroller)

    def load_ds(self, datacontroller):
        ds = datacontroller.get_ds(self.split)

        if self.split == "train":
            ds.shuffle(buffer_size=100 * self.data_cfg["batch_size"])

        if self.split == "train":
            ds = ds.repeat()

        self.ds = tfds.as_numpy(ds)
        self.ds_iter = iter(self.ds)

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            try:
                record = next(self.ds_iter)
                images = record["image"]
                images = (
                    (jnp.array(tf.squeeze(images)) / 255.0) - 0.5
                ) * 2.0  # Normalization
            except StopIteration:  # For testing, reset iterator after one full pass
                self.ds_iter = iter(self.ds)
                raise StopIteration

            # Return masks if the record has them
            if self.masks:
                return images, record["mask"]
            return images


class DataController(object):
    # Dictionary describing features present per record type
    JUST_IMAGE = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    IMAGE_AND_MASK = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
    }

    def __init__(
        self,
        save_path,
        load_mode=False,
        test_train_split=0.8,
        transformations=None,
        shuffle=False,
        batch_size=1,
        file_name="dataset",
        overwrite=False,
        uint=True,
        unbatch=False,
        gzip=False,
        latent_dim_size=64,
        masks=False,
    ):
        self.load_mode = load_mode
        self.batch_size = batch_size
        self.uint = uint  # Whether to use uint8 or float32
        self.latent_dim = latent_dim_size
        self.dataset_length = 0

        self._check_path(save_path, file_name)
        if load_mode:
            self._load_data(
                test_train_split, transformations, shuffle, unbatch, gzip, masks
            )
        else:
            self.writer = self._create_writer(overwrite)

    def _create_writer(self, overwrite):
        if Path(self.file_path).exists() and not overwrite:
            raise FileExistsError(
                "You are attempting to overwrite a pre-existing file. Please set the overwrite flag accordingly."
            )
        writer = tf.io.TFRecordWriter(self.file_path)
        print("Succesfully openened writer")
        return writer

    def _check_path(self, save_path, file_name):
        if save_path[-1] != "/":
            save_path += "/"

        # Check that we aren't overwriting a file, and create folders
        self.file_path = save_path + file_name + ".tfr"
        if self.load_mode:
            if not Path(self.file_path).exists():
                raise FileExistsError("No saved TF Record at specified location")
        else:
            Path(save_path).mkdir(parents=True, exist_ok=True)

    def clear_data(self):
        # Empty contents of the writer
        if hasattr(self, "writer"):
            self.writer.flush()
            print("Cleared data in open writer")
        else:
            print("Please load a file for writing before trying to clear.")

    def write_data(self, obs, extra=None, uint=False):
        if self.load_mode:
            raise Exception(
                "Attempting to add data to a loaded TFRecord is not currently supported."
            )

        if not isinstance(obs, list) and obs.ndim == 3:  # Not a batch of images
            obs = [obs]

        for image in obs:
            tf_example = self._create_tfr_image_entry(image)
            self.writer.write(tf_example.SerializeToString())

    def write_batched_data(self, obs, batch_size=1):
        if self.load_mode:
            raise Exception(
                "Attempting to add data to a loaded TFRecord is not currently supported."
            )

        if not isinstance(obs, list) and obs.ndim == 3:  # Not a batch of images
            obs = [obs]

        if len(obs) != batch_size:
            raise Exception(
                "Trying to write a batch of data that does not match batch size."
            )

        images = []
        for image in obs:
            images.append(tf.convert_to_tensor(image, dtype=tf.uint8))

        tf_example = self._create_tfr_image_entry(tf.stack(images))
        self.writer.write(tf_example.SerializeToString())

    def take(self, i):
        return self.data.shuffle(buffer_size=100 * self.batch_size).take(i)

    def finish(self):
        # Close the writer
        if not self.load_mode:
            self.writer.close()
        print("Succesfully closed DataConnector")

    def get_ds(self, split="train"):
        if split == "train":
            return self.train_ds
        elif split == "validation":
            return self.val_ds
        else:
            raise NotImplementedError

    def _load_data(self, test_train_split, transforms, shuffle, unbatch, gzip, masks):
        compression = (
            tf.io.TFRecordOptions.get_compression_type_string("GZIP") if gzip else None
        )
        raw_data = tf.data.TFRecordDataset(self.file_path, compression_type=compression)
        data = raw_data.map(
            functools.partial(self._parse_image_function, masks=masks),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )  # De-serialize data

        # False when sequences needed, e.g. for AlignNet
        if unbatch:
            data = data.unbatch()

        self.dataset_length = sum(1 for _ in data)
        print(f"Dataset length: {self.dataset_length}")

        if shuffle:
            data = data.shuffle(self.dataset_length)
        if transforms is not None:
            data = data.map(
                transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        train_size = int(test_train_split * self.dataset_length)
        print(f"Train size: {train_size}")
        train_ds = data.take(train_size)
        print(f"Validation size: {self.dataset_length - train_size}")
        val_ds = data.skip(train_size)

        self.train_ds = self._prefetch_batch(train_ds)
        self.val_ds = self._prefetch_batch(val_ds)

    def _prefetch_batch(self, data):
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data.batch(self.batch_size)

    def __iter__(self):
        if hasattr(self, "data"):
            if not self.shuffle:
                return iter(self.data)
            else:
                return iter(self.data.shuffle(buffer_size=100 * self.batch_size))
        else:
            raise Exception("Please load a TFRecord before trying to access data.")

    def _create_tfr_image_entry(self, images):
        feature = {"image": self._bytes_feature(tf.io.serialize_tensor(images))}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _parse_image_function(self, example_proto, masks):
        # Parse the input tf example using the corresponding dictionary
        example_type = self.IMAGE_AND_MASK if masks else self.JUST_IMAGE
        example = tf.io.parse_single_example(example_proto, example_type)
        example["image"] = tf.io.parse_tensor(
            example["image"], out_type=tf.uint8 if self.uint else tf.float32
        )
        if masks:
            example["mask"] = tf.io.parse_tensor(example["mask"], out_type=tf.bool)
        return example

    # The following functions can be used to convert a value to a type compatible with tf.Example.
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
