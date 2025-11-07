import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from keras.datasets import mnist

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src import graph_compatible_data_generator, yolo_object_detection_model, training_utils
from src import training_utils as tu
from src import post_processing as pp


_, (x_test, y_test) = mnist.load_data()

X_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
# X_tensor = tf.reshape(X_tensor, shape=(-1, 28, 28, 1))
y_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

batch_size = 32
raw_dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

# create a generator for 2 digits
data_gen_2_digits = graph_compatible_data_generator.create_data_generator(2)
data_gen_3_digits = graph_compatible_data_generator.create_data_generator(3)
data_gen_4_digits = graph_compatible_data_generator.create_data_generator(4)
data_gen_5_digits = graph_compatible_data_generator.create_data_generator(5)

data_dir = Path("..", "data")
models_dir = Path("..", "models")

processed_test_dataset_2 = raw_dataset.map(
    data_gen_2_digits).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

# Get one batch of test data to reuse
for batch in processed_test_dataset_2.take(1):
    canvas_batch, true_labels_batch = batch
    break  # Stop after one batch


def get_width_height_from_bbox(bbox):
    width = bbox[3]
    height = bbox[4]
    return width, height


def get_center_from_bbox(bbox):
    x_center = bbox[1]
    y_center = bbox[2]
    return x_center, y_center


def get_class_from_bbox(bbox):
    # Find the predicted class
    pred_class = np.argmax(bbox[5:])
    pred_probability = bbox[5 + pred_class]
    return pred_class, pred_probability


def get_min_coor_from_bbox(bbox):
    x_center, y_center = get_center_from_bbox(bbox=bbox)
    width, height = get_width_height_from_bbox(bbox=bbox)

    # Calculate top-left (x_min, y_min) and clip to canvas bounds [0, 100]
    x_min = np.clip(np.floor(x_center - (width / 2)), 0, 100)
    y_min = np.clip(np.floor(y_center - (height / 2)), 0, 100)

    # x_min = np.clip(np.floor(x_center - (width / 2) + 0.5), 0, 100)
    # y_min = np.clip(np.floor(y_center - (height / 2) + 0.5), 0, 100)

    return x_min, y_min


def visualize_comparison(experiment_results, image_index=0):
    """
    Generates a side-by-side comparison for a single image 
    across multiple experiment results.

    Args:
        canvas_batch (tf.Tensor): The batch of canvas images (e.g., shape [32, 100, 100, 1]).
        true_values_batch (tf.Tensor): The batch of true labels (e.g., shape [32, 5, 15]).
        experiment_results (list): A list of tuples, where each tuple is:
                                   ("Experiment Name", post_processed_predictions_tensor)
        image_index (int): The index of the image *from the batch* to display.
    """

    # --- 1. Setup the Plot ---

    # Get the number of experiments to plot
    num_experiments = len(experiment_results)

    # Create a figure with 'num_experiments' subplots, arranged horizontally
    # 'figsize' is (width, height) in inches. We make it 8" wide per plot.
    fig, axes = plt.subplots(
        1, num_experiments, figsize=(8 * num_experiments, 8))

    # Matplotlib quirk: If you only have 1 plot, 'axes' is not a list.
    # We force it to be a list so our 'for' loop works either way.
    if num_experiments == 1:
        axes = [axes]

    # --- 2. Get the Base Data ---

    # Get the single canvas and true_value for the selected image
    # We call .numpy() to move the data from a TensorFlow tensor to a NumPy array for plotting
    canvas_to_show = canvas_batch[image_index].numpy()
    # Keep this as a tensor for now
    true_value = true_labels_batch[image_index]

    # --- 3. Loop Through Each Experiment ---

    # 'enumerate' gives us both the index (i) and the data (experiment_tuple)
    for i, (experiment_name, predictions_ragged) in enumerate(experiment_results):

        # Select the correct subplot (axis) for this experiment
        ax = axes[i]

        # Display the base canvas image
        ax.imshow(canvas_to_show, cmap='gray')

        # Set the title for this subplot
        ax.set_title(experiment_name, fontsize=16)

        # --- 4. Plot Predicted Boxes (in color) ---

        # Get the ragged predictions for the *single image* we're plotting
        prediction = predictions_ragged[image_index]
        prediction_boxes = prediction.shape[0]

        # Define colors for the (up to 5) predicted boxes
        colors = ['b', 'g', 'r', 'c', 'm']

        for j in range(prediction_boxes):
            # Convert the single box tensor to NumPy and de-normalize
            bbox = (prediction[j]).numpy() * 100

            # Unpack the 15-vector
            flag = bbox[0]

            x_center, y_center = get_center_from_bbox(bbox=bbox)
            x_min, y_min = get_min_coor_from_bbox(bbox=bbox)
            width, height = get_width_height_from_bbox(bbox=bbox)
            pred_class, pred_probs = get_class_from_bbox(bbox=bbox)
            # print(f"Pred Class : {pred_class},x_center : {x_center}, y_center : {y_center}")

            # Create the rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                # Use modulo to avoid index error
                edgecolor=colors[j % len(colors)],
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add text label (Class, Confidence)
            ax.text(
                x_min, y_min,  # Position text slightly above the box
                # Format to 2 decimal places
                f'Pred: {pred_class} ({flag:.2f})',
                color=colors[j % len(colors)],
                fontsize=12,
                fontweight='bold'
            )

        # --- 5. Plot True Boxes (in Yellow) ---

        true_value_boxes = true_value.shape[0]

        for j in range(true_value_boxes):
            bbox = (true_value[j]).numpy() * 100

            flag = bbox[0]
            # Only draw true boxes if they are "real" (flag > 0)
            if flag > 0:
                x_center, y_center = get_center_from_bbox(bbox=bbox)
                x_min, y_min = get_min_coor_from_bbox(bbox=bbox)
                width, height = get_width_height_from_bbox(bbox=bbox)
                true_class, true_probs = get_class_from_bbox(bbox=bbox)

                # print(f"True Class : {true_class},x_center : {x_center}, y_center : {y_center}")
                # Create a simple, dashed yellow rectangle
                rect = patches.Rectangle(
                    (x_min, y_min),
                    width,
                    height,
                    linewidth=2,
                    edgecolor='yellow',
                    facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)

                # Add text label (True Class)
                ax.text(
                    x_min, y_min + height,  # Position text slightly below the box
                    f'True: {true_class}',
                    color='yellow',
                    fontsize=12,
                    fontweight='bold'
                )

    # Finally, show the entire figure with all subplots
    plt.tight_layout()  # Adjusts plots to prevent title overlap
    plt.show()


def get_predictions(model_names):
    prediction_list = []
    # Get Predictions
    custom_objects = {
        "calculate_model_loss": tu.calculate_model_loss,
        # "objectness_metrics": tu.objectness_metrics,
        # "bounding_box_metrics": tu.bounding_box_metrics,
        # "classification_metrics": tu.classification_metrics,
        "YoloObjectDetectionModel": yolo_object_detection_model.YoloObjectDetectionModel}

    for experiment_name, model_name in model_names:
        # load model
        model = tf.keras.models.load_model(
            Path(models_dir, model_name), custom_objects=custom_objects)
        # predict bounding boxes
        prediction = model.predict(canvas_batch)
        # post process data
        # post_processed_data = pp.post_process(
        #     prediction,
        #     confidence_score_threshold=0.98, iou_threshold=0.2, max_boxes=5)
        # create tuple & update list
        prediction_list.append((experiment_name, prediction))
    return prediction_list


def post_process(prediction_list):
    post_processed_predictions = []
    for experiment_name, prediction in prediction_list:
        # post process data
        post_processed_data = pp.post_process(
            prediction,
            confidence_score_threshold=0.98, iou_threshold=0.2, max_boxes=5)
        # create tuple & update list
        post_processed_predictions.append(
            (experiment_name, post_processed_data))
    return post_processed_predictions
