import tensorflow as tf

# BBOX Indices
BBOX_XMIN_IDX = 0
BBOX_XMAX_IDX = 1
BBOX_YMIN_IDX = 2
BBOX_YMAX_IDX = 3
BBOX_XCENTER_IDX = 4
BBOX_YCENTER_IDX = 5  # (This might be the same as CLASS_IDX)
BBOX_WIDTH_IDX = 6
BBOX_HEIGHT_IDX = 7
BBOX_CLASS_IDX = 8
BBOX_CANVAS_TOP_IDX = 9
BBOX_CANVAS_LEFT_IDX = 10


# number of digits to overlay on canvas
# num_of_digits = 3

# max digits to define the shape of prediction output
MAX_DIGITS = 5

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomTranslation(
        height_factor=0.2, width_factor=0.2, fill_value=0.0, fill_mode="constant", seed=42),

    tf.keras.layers.RandomZoom(
        height_factor=0.2, width_factor=0.2, fill_value=0.0, fill_mode="constant", seed=42),

    tf.keras.layers.RandomRotation(
        factor=0.1, fill_value=0.0, fill_mode="constant", seed=42),
])