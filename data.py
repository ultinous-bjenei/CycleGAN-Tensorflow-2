import numpy as np
import tensorflow as tf
import tf2lib as tl


def make_dataset(img_paths, batch_size, size, training, drop_remainder=True, shuffle=True, repeat=1, greyscale=False):
    @tf.function
    def _map_fn(img):  # preprocessing
        img = tf.image.rgb_to_hsv(img)
        if greyscale:
            img = img[...,2]

        img = tf.dtypes.cast(img,dtype=tf.float32)
        img = tf.image.resize_with_crop_or_pad(img, target_height=size[0], target_width=size[1])
        img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
        img = img * 2 - 1

        return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)

def make_zip_dataset(img_paths, batch_size, size, training, shuffle=True, repeat=False):
    A_dataset = make_dataset(img_paths, batch_size, size, training, drop_remainder=True, shuffle=shuffle, repeat=repeat, greyscale=True)
    B_dataset = make_dataset(img_paths, batch_size, size, training, drop_remainder=True, shuffle=shuffle, repeat=repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = len(img_paths) // batch_size

    return A_B_dataset, len_dataset
