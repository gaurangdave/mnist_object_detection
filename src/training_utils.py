import tensorflow as tf

# helper function to convert box values to corner coordinates


def convert_boxes_to_corners(box_center_format):
    # we'll use the following formulas
    # x_min = floor(x_center - (width/2))
    # x_max = floor(x_center + (width/2))
    # y_min = floor(y_center - (height/2))
    # y_max = floor(y_cetner + (height /2))
    # calculate true values.
    x_min = tf.floor(
        box_center_format[:, :, :, 0] - (box_center_format[:, :, :, 2])/2)
    x_max = tf.floor(
        box_center_format[:, :, :, 0] + (box_center_format[:, :, :, 2])/2)
    y_min = tf.floor(
        box_center_format[:, :, :, 1] - (box_center_format[:, :, :, 3])/2)
    y_max = tf.floor(
        box_center_format[:, :, :, 1] + (box_center_format[:, :, :, 3])/2)

    coordinates = tf.stack(values=[x_min, y_min, x_max, y_max], axis=3)
    return coordinates

# helper function to find the intersection box corners from given 2 boxes


def calculate_intersection_corners(box_1_corners, box_2_corners):
    x_min_for_intersection = tf.maximum(
        box_1_corners[:, :, :, 0], box_2_corners[:, :, :, 0])
    y_min_for_intersection = tf.maximum(
        box_1_corners[:, :, :, 1], box_2_corners[:, :, :, 1])
    x_max_for_intersection = tf.minimum(
        box_1_corners[:, :, :, 2], box_2_corners[:, :, :, 2])
    y_max_for_intersection = tf.minimum(
        box_1_corners[:, :, :, 3], box_2_corners[:, :, :, 3])
    intersection_box_corners = tf.stack(
        values=[x_min_for_intersection, y_min_for_intersection, x_max_for_intersection, y_max_for_intersection], axis=3)
    return intersection_box_corners

# helper function to calculate the area of intersection between two boxes


def calculate_intersection_area(intersection_box_corners):
    # find the width = x_max - x_min, if the boxes are not intersecting, this value could be negative or 0
    intersection_width = tf.maximum(
        0.0, intersection_box_corners[:, :, :, 2] - intersection_box_corners[:, :, :, 0])
    # find the height = y_max - y_min, if the boxes are not intersecting, this value could be negative or 0
    intersection_height = tf.maximum(
        0.0, intersection_box_corners[:, :, :, 3] - intersection_box_corners[:, :, :, 1])
    # intersection area = width * height
    intersection_area = intersection_width * intersection_height
    return intersection_area

# helper function to calcualte the area of union between two boxes


def calculate_union_area(box_1_dimensions, box_2_dimensions, intersection_area):
    box_1_area = box_1_dimensions[:, :, :, 0] * box_1_dimensions[:, :, :, 1]
    box_2_area = box_2_dimensions[:, :, :, 0] * box_2_dimensions[:, :, :, 1]
    union_area = box_1_area + box_2_area - intersection_area
    return union_area

# helper function to calculate the IOU ration between boxes


def calculate_iou(intersection_area, union_area):
    iou = intersection_area / (union_area + 1e-8)
    return iou

# helper function to calculate indices for the grid cells that contain the object


def calculate_grid_cell_indices(y_true, y_pred):
    x_grid_size = y_pred.shape[1]

    # Read the bounding box centers
    # Each instance in the bach will have 5 bounding box centers
    bounding_box_centers = y_true[:, :, 1:3]

    # TODO:  here we are assuming number of rows and columns in grid is same. Confirm the assumption.
    # The general formula is: grid_index = floor(pixel_coordinate * (grid_size / image_size))
    # convert each 5 bounding box centers to 5 possible grids for each instance
    grid_indices = tf.cast(
        tf.floor(bounding_box_centers * (x_grid_size / 100)), dtype=tf.int32)
    print(f"grid indices shape {grid_indices.shape}")

    return grid_indices

# Helper function to calculate anchor box indices.


def calculate_anchorbox_indices(y_true, y_pred, grid_cell_indices):
    x_grid_size = y_pred.shape[1]
    y_grid_size = y_pred.shape[2]

    anchor_boxes = tf.reshape(
        y_pred, shape=(-1, x_grid_size, y_grid_size, 3, 15))
    print(f"anchor_boxes.shape {anchor_boxes.shape}")

    selected_anchor_boxes = tf.gather_nd(
        anchor_boxes, batch_dims=1, indices=grid_cell_indices)
    print(f"selected_anchor_boxes.shape :{selected_anchor_boxes.shape}")

    # calcualte the IOU between anchor boxes and ground truth
    expanded_y_true = tf.expand_dims(y_true, axis=2)

    # calculate min and max values for ground truth and anchor boxes
    y_true_box_corners = convert_boxes_to_corners(
        expanded_y_true[:, :, :, 1:5])
    y_pred_box_corners = convert_boxes_to_corners(
        selected_anchor_boxes[:, :, :, 1:5])
    print(f"y_true_boxes.shape {y_true_box_corners.shape}")
    print(f"y_pred_boxes.shape {y_pred_box_corners.shape}")

    # calculate the intersection coordinates between ground truth and anchor boxes
    intersection_box_corners = calculate_intersection_corners(
        y_true_box_corners[:, :, :, 0:], y_pred_box_corners[:, :, :, 0:])
    print(f"intersection_box_corners.shape {intersection_box_corners.shape}")

    # calculate the IOU
    # calculate intersection area
    intersection_area = calculate_intersection_area(intersection_box_corners)
    print(f"intersection_area.shape {intersection_area.shape}")

    # calculate union area
    # we just need the width and length for union area
    union_area = calculate_union_area(
        expanded_y_true[:, :, :, 3:5], selected_anchor_boxes[:, :, :, 3:5], intersection_area)
    print(f"union_area.shape {union_area.shape}")

    # calculate IOU
    iou = calculate_iou(intersection_area, union_area)
    print(f"iou.shape {iou.shape}")

    # select the anchor box based on best iou score
    # select the index with highest iou
    highest_iou_index = tf.argmax(iou, axis=2, output_type=tf.int32)
    print(f"highest_iou_index.shape {highest_iou_index.shape}")

    highest_iou_index = tf.expand_dims(highest_iou_index, axis=2)
    # highest_iou_index = tf.reshape(highest_iou_index, shape=(iou.shape[0],iou.shape[1],-1))
    print(f"expanded highest_iou_index.shape {highest_iou_index.shape}")
    return highest_iou_index

# helper function to calculate best anchor boxes


def calculate_best_anchor_boxes(y_true, y_pred):
    x_grid_size = y_pred.shape[1]
    y_grid_size = y_pred.shape[2]

    print("----- True Values -----")
    print(f"y_true.shape {y_true.shape}")

    print("----- Pred Values -----")
    print(f"y_pred.shape {y_pred.shape}")

    # we have 6x6, each grid cell has 3 anchor box i.e 108 anchor boxes per insantance
    anchor_boxes = tf.reshape(
        y_pred, shape=(-1, x_grid_size, y_grid_size, 3, 15))
    print(f"anchor_boxes.shape {anchor_boxes.shape}")

    grid_cell_indices = calculate_grid_cell_indices(
        y_true=y_true, y_pred=y_pred)

    # out of 36 grid cells (per instance) select at most 5 grid cells that have ground truth bounding box
    # so out of 108 anchor boxes (per instance) we only need to check 15 anchor boxes
    selected_anchor_boxes = tf.gather_nd(
        anchor_boxes, batch_dims=1, indices=grid_cell_indices)
    print(f"selected_anchor_boxes.shape :{selected_anchor_boxes.shape}")

    highest_iou_index = calculate_anchorbox_indices(
        y_true=y_true, y_pred=y_pred, grid_cell_indices=grid_cell_indices)
    # select the anchor box based on the index
    best_anchor_boxes = tf.gather(
        selected_anchor_boxes, indices=highest_iou_index, batch_dims=2)
    print(f"best_anchor_boxes.shape {best_anchor_boxes.shape}")

    return best_anchor_boxes

# helper function to split and calculate loss


def calculate_loss(predicted_values, true_values):
    # slice the 3 properties that we are tyring to calculate loss against
    # predicted values

    y_pred_objectness = predicted_values[:, :, :, 0]
    print(f"y_pred_objectness.shape : {y_pred_objectness.shape}")

    y_pred_bounding_box = predicted_values[:, :, :, 1:5]
    print(f"y_pred_bounding_box.shape : {y_pred_bounding_box.shape}")

    y_pred_classification = predicted_values[:, :, :, 5:]
    print(f"y_pred_classification.shape : {y_pred_classification.shape}")

    # True Values
    y_true_objectness = true_values[:, :, :, 0]
    print(f"y_true_objectness.shape : {y_true_objectness.shape}")

    y_true_bounding_box = true_values[:, :, :, 1:5]
    print(f"y_true_bounding_box.shape : {y_true_bounding_box.shape}")

    y_true_classification = true_values[:, :, :, 5:]
    print(f"y_true_classification.shape : {y_true_classification.shape}")

    # Apply activation functions to predicted values
    y_pred_objectness = tf.keras.activations.sigmoid(y_pred_objectness)
    print(
        f"Post Activation y_pred_objectness.shape : {y_pred_objectness.shape}")

    y_pred_classification = tf.keras.activations.softmax(y_pred_classification)
    print(
        f"Post Activation y_pred_classification.shape : {y_pred_classification.shape}")

    # Calculate loss
    objectness_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False)(y_true_objectness, y_pred_objectness)
    bounding_box_loss = tf.keras.losses.MeanSquaredError()(
        y_true_bounding_box, y_pred_bounding_box)
    classification_loss = tf.keras.losses.CategoricalCrossentropy()(
        y_true_classification, y_pred_classification)

    return objectness_loss, bounding_box_loss, classification_loss

# helper function to calculate loss for object less cells


def calculate_objectless_loss(y_true, y_pred):
    # step 1: create placeholder y_true
    batch_size = y_pred.shape[0]
    y_true_objectless = tf.zeros(shape=y_true.shape, dtype=tf.float32)
    print(f"y_true_objectless.shape {y_true_objectless.shape}")

    # step 2: prepare mask for positive values
    # hard coding the grid size
    positive_mask = tf.constant(False, shape=(batch_size, 6, 6, 3))
    print(f"positive_mask.shape {positive_mask.shape}")

    grid_cell_indices = calculate_grid_cell_indices(
        y_true=y_true, y_pred=y_pred)
    print(f"grid_cell_indices {grid_cell_indices.shape}")
    # grid cell indices will have shape (m, 5, 2)
    # here 5 is max images and 2 is row and column index

    highest_iou_index = calculate_anchorbox_indices(
        y_true=y_true, y_pred=y_pred, grid_cell_indices=grid_cell_indices)
    print(f"highest_iou_index {highest_iou_index.dtype}")
    # highest iou index will have shpae (m,5,1)
    # here 5 is max images and 1 represents best anchor box in the cell.

    # we need to combine both the indices to create tensor of shape (m, row indices, column indices, box index)
    combine_update_index = tf.range(y_pred.shape[0])
    # expand dims
    combine_update_index = tf.reshape(
        combine_update_index, shape=(y_pred.shape[0], 1, 1))
    # combine_update_index = tf.expand_dims(combine_update_index, axis=2)
    combine_update_index = tf.tile(
        combine_update_index, [1, 5, 1])
    combine_update_index = tf.concat(
        [combine_update_index, grid_cell_indices, highest_iou_index], axis=2)
    combine_update_index = tf.reshape(combine_update_index, shape=(-1, 4))

    print(f"combine_update_index.shape : {combine_update_index.shape}")

    positive_mask = tf.scatter_nd(
        indices=combine_update_index, 
        shape=positive_mask.shape, 
        updates=tf.constant(True, shape=(combine_update_index.shape[0],)))
    print(f"positive_mask.shape : {positive_mask}")
    pass

# loss function for the model


def calculate_model_loss(y_true, y_pred):
    # Find best anchor box
    expanded_y_true = tf.expand_dims(y_true, axis=2)
    best_anchor_boxes = calculate_best_anchor_boxes(y_true, y_pred)

    # Loss Calculation
    objectness_loss, bounding_box_loss, classification_loss = calculate_loss(
        best_anchor_boxes, expanded_y_true)
    print(f"objectness_loss : {objectness_loss}")
    print(f"bounding_box_loss : {bounding_box_loss}")
    print(f"classification_loss : {classification_loss}")

    # objectless loss calculation
    print("\n\n----- Calculation Object Less Loss -----")
    calculate_objectless_loss(y_true=y_true, y_pred=y_pred)

    # scale the losses
    lambda_objectness = 1
    lambda_bounding_box = 0.001
    lambda_classification = 1

    total_loss = (objectness_loss * lambda_objectness) + (bounding_box_loss *
                                                          lambda_bounding_box) + (classification_loss * lambda_classification)

    print(f"\n\nTotal Loss : {total_loss}")

    return total_loss


def objectness_metrics(y_true, y_pred):
    expanded_y_true = tf.expand_dims(y_true, axis=2)
    best_anchor_boxes = calculate_best_anchor_boxes(y_true, y_pred)

    # Loss Calculation
    objectness_loss, bounding_box_loss, classification_loss = calculate_loss(
        best_anchor_boxes, expanded_y_true)
    return objectness_loss


def bounding_box_metrics(y_true, y_pred):
    expanded_y_true = tf.expand_dims(y_true, axis=2)
    best_anchor_boxes = calculate_best_anchor_boxes(y_true, y_pred)

    # Loss Calculation
    objectness_loss, bounding_box_loss, classification_loss = calculate_loss(
        best_anchor_boxes, expanded_y_true)
    return bounding_box_loss


def classification_metrics(y_true, y_pred):
    expanded_y_true = tf.expand_dims(y_true, axis=2)
    best_anchor_boxes = calculate_best_anchor_boxes(y_true, y_pred)

    # Loss Calculation
    objectness_loss, bounding_box_loss, classification_loss = calculate_loss(
        best_anchor_boxes, expanded_y_true)
    return classification_loss
