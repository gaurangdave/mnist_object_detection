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
    x_grid_size = tf.shape(y_pred)[1]


    # Read the bounding box centers
    # Each instance in the bach will have 5 bounding box centers
    # select boxes with objectness equal to 1
    # objectness_mask = y_true[:, :, 0] == 1.0
    # bounding_boxes_with_objects = tf.boolean_mask(y_true, mask=objectness_mask)
    # # tf.print(f"bounding_boxes_with_objects.shape : {bounding_boxes_with_objects.shape}")
    bounding_box_centers = y_true[:, :, 1:3]

    # TODO:  here we are assuming number of rows and columns in grid is same. Confirm the assumption.
    # The general formula is: grid_index = floor(pixel_coordinate * (grid_size / image_size))
    # convert each 5 bounding box centers to 5 possible grids for each instance
    
    normalized_grid_size = tf.cast((x_grid_size), dtype=tf.float32)
    
    grid_indices = tf.cast(
        tf.floor(bounding_box_centers * normalized_grid_size), dtype=tf.int32)

    # grid_indices = tf.reshape(grid_indices,shape=(batch_size,-1,2))
    # # tf.print(f"grid indices shape {grid_indices.shape}")

    return grid_indices

# Helper function to calculate anchor box indices.


def calculate_anchorbox_indices(y_true, y_pred, grid_cell_indices):
    y_pred_shape = tf.shape(y_pred)
    x_grid_size = y_pred_shape[1]
    y_grid_size = y_pred_shape[2]

    anchor_boxes = tf.reshape(
        y_pred, shape=(-1, x_grid_size, y_grid_size, 3, 15))
    
    # # tf.print(f"anchor_boxes.shape {tf.shape(anchor_boxes)}")

    selected_anchor_boxes = tf.gather_nd(
        anchor_boxes, batch_dims=1, indices=grid_cell_indices)
    # # tf.print(f"selected_anchor_boxes.shape :{tf.shape(selected_anchor_boxes)}")

    # calcualte the IOU between anchor boxes and ground truth
    expanded_y_true = tf.expand_dims(y_true, axis=2)

    # calculate min and max values for ground truth and anchor boxes
    y_true_box_corners = convert_boxes_to_corners(
        expanded_y_true[:, :, :, 1:5])
    y_pred_box_corners = convert_boxes_to_corners(
        selected_anchor_boxes[:, :, :, 1:5])
    # # tf.print(f"y_true_boxes.shape {y_true_box_corners.shape}")
    # # tf.print(f"y_pred_boxes.shape {y_pred_box_corners.shape}")

    # calculate the intersection coordinates between ground truth and anchor boxes
    intersection_box_corners = calculate_intersection_corners(
        y_true_box_corners[:, :, :, 0:], y_pred_box_corners[:, :, :, 0:])
    # # tf.print(f"intersection_box_corners.shape {intersection_box_corners.shape}")

    # calculate the IOU
    # calculate intersection area
    intersection_area = calculate_intersection_area(intersection_box_corners)
    # # tf.print(f"intersection_area.shape {intersection_area.shape}")

    # calculate union area
    # we just need the width and length for union area
    union_area = calculate_union_area(
        expanded_y_true[:, :, :, 3:5], selected_anchor_boxes[:, :, :, 3:5], intersection_area)
    # # tf.print(f"union_area.shape {union_area.shape}")

    # calculate IOU
    iou = calculate_iou(intersection_area, union_area)
    # # tf.print(f"iou.shape {iou.shape}")

    # select the anchor box based on best iou score
    # select the index with highest iou
    highest_iou_index = tf.argmax(iou, axis=2, output_type=tf.int32)
    # # tf.print(f"highest_iou_index.shape {highest_iou_index.shape}")

    highest_iou_index = tf.expand_dims(highest_iou_index, axis=2)
    # highest_iou_index = tf.reshape(highest_iou_index, shape=(iou.shape[0],iou.shape[1],-1))
    # # tf.print(f"expanded highest_iou_index.shape {highest_iou_index.shape}")
    return highest_iou_index

# helper function to calculate best anchor boxes


def calculate_best_anchor_boxes(y_true, y_pred):
    
    y_pred_shape = tf.shape(y_pred)
    x_grid_size = y_pred_shape[1]
    y_grid_size = y_pred_shape[2]

    # # tf.print("----- True Values -----")
    # # tf.print(f"y_true.shape {tf.shape(y_true)}")

    # # tf.print("----- Pred Values -----")
    # # tf.print(f"y_pred.shape {tf.shape(y_pred)}")

    # we have 6x6, each grid cell has 3 anchor box i.e 108 anchor boxes per insantance
    anchor_boxes = tf.reshape(
        y_pred, shape=(-1, x_grid_size, y_grid_size, 3, 15))
    # # tf.print(f"anchor_boxes.shape {tf.shape(anchor_boxes)}")

    grid_cell_indices = calculate_grid_cell_indices(
        y_true=y_true, y_pred=y_pred)

    # out of 36 grid cells (per instance) select at most 5 grid cells that have ground truth bounding box
    # so out of 108 anchor boxes (per instance) we only need to check 15 anchor boxes
    selected_anchor_boxes = tf.gather_nd(
        anchor_boxes, batch_dims=1, indices=grid_cell_indices)
    # # tf.print(f"selected_anchor_boxes.shape :{tf.shape(selected_anchor_boxes)}")

    highest_iou_index = calculate_anchorbox_indices(
        y_true=y_true, y_pred=y_pred, grid_cell_indices=grid_cell_indices)
    # select the anchor box based on the index
    best_anchor_boxes = tf.gather(
        selected_anchor_boxes, indices=highest_iou_index, batch_dims=2)
    # # tf.print(f"best_anchor_boxes.shape {tf.shape(best_anchor_boxes)}")

    return best_anchor_boxes

# helper function to split and calculate loss


def calculate_loss(predicted_values, true_values):

    objectness_mask = true_values[:, :, :, 0] == 1.0

    true_values_with_objects = tf.boolean_mask(
        true_values, mask=objectness_mask)
    predicted_values_with_objects = tf.boolean_mask(
        predicted_values, mask=objectness_mask)

    # # tf.print(f"true_values_with_objects.shape : {tf.shape(true_values_with_objects)}")
    # # tf.print(
    #     f"predicted_values_with_objects.shape : {tf.shape(predicted_values_with_objects)}")
    # slice the 3 properties that we are tyring to calculate loss against
    # predicted values

    y_pred_objectness = predicted_values_with_objects[:, 0]
    # # tf.print(f"y_pred_objectness.shape : {tf.shape(y_pred_objectness)}")

    y_pred_bounding_box = predicted_values_with_objects[:, 1:5]
    # # tf.print(f"y_pred_bounding_box.shape : {tf.shape(y_pred_bounding_box)}")

    y_pred_classification = predicted_values_with_objects[:, 5:]
    # # tf.print(f"y_pred_classification.shape : {tf.shape(y_pred_classification)}")

    # True Values
    y_true_objectness = true_values_with_objects[:, 0]
    # # tf.print(f"y_true_objectness.shape : {tf.shape(y_true_objectness)}")

    y_true_bounding_box = true_values_with_objects[:, 1:5]
    # # tf.print(f"y_true_bounding_box.shape : {tf.shape(y_true_bounding_box)}")

    y_true_classification = true_values_with_objects[:, 5:]
    # # tf.print(f"y_true_classification.shape : {tf.shape(y_true_classification)}")

    # Apply activation functions to predicted values
    y_pred_objectness = tf.keras.activations.sigmoid(y_pred_objectness)
    # # tf.print(
    #     f"Post Activation y_pred_objectness.shape : {tf.shape(y_pred_objectness)}")

    y_pred_classification = tf.keras.activations.softmax(y_pred_classification)
    # # tf.print(
    #     f"Post Activation y_pred_classification.shape : {tf.shape(y_pred_classification)}")

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
    """
        Objectless loss is a bit different from regular loss function and the reason for that is difference in shape between pred and true values. 
        True value shape will be (m,5,15) and pred shape would be (m,6,6,3,15) or (m,6,6,45)
        If we have only 2 digits per image total bbox with object in true would 30 and object less would be 45
        But in prediction object true would be 30 but objectless should be 1590, which is (6 * 6 * 3 * 15) - 30
        So we need to penalize these 1590 predictions if they have predited these anchor boxes with objects. 
        To to that we'll create a tensor with zeroes that matches shape of the prediction, and calculate the loss with respect to that. 
    """
    
    
    # step 1: create placeholder y_true
    y_pred_shape = tf.shape(y_pred)
    batch_size = y_pred_shape[0]
    grid_h = y_pred_shape[1]
    grid_w = y_pred_shape[2]
    NUM_ANCHORS = 3
    NUM_FEATURES = 15 # (1 score + 4 coords + 10 classes)

    bounding_box_with_object_mask = y_true[:, :, 0] == 1.0

    # step 2: prepare mask for positive values
    # hard coding the grid size
    # default mask with everything false
    positive_mask_shape = [batch_size, grid_h, grid_w, NUM_ANCHORS]
    positive_mask = tf.constant(False, shape=positive_mask_shape)
    # tf.print(f"positive_mask.shape {tf.shape(positive_mask)}")


    # indices to grid cells that have the objects
    grid_cell_indices = calculate_grid_cell_indices(
        y_true=y_true, y_pred=y_pred)

    # grid cell indices will have shape (m, 5, 2)
    # here 5 is max images and 2 is row and column index
    # returns indices of the anchor boxes with highest IOU in given grid cells
    highest_iou_index = calculate_anchorbox_indices(
        y_true=y_true, y_pred=y_pred, grid_cell_indices=grid_cell_indices)

    # highest iou index will have shpae (m,5,1)
    # here 5 is max images and 1 represents best anchor box in the cell.
    # we need to combine both the indices to create tensor of shape (m, row indices, column indices, box index)
    # create batch index
    combine_update_index = tf.range(batch_size)
    # expand dims
    combine_update_index = tf.reshape(
        combine_update_index, shape=(batch_size, 1, 1))
    
    # repeat axis 0 1 time, anxis 1 5 times and axis 2 1 time
    """
        Understanding tile output
        if batch size is 3 the combine_update_index tensor would be
        <tf.Tensor: shape=(3, 1, 1), dtype=int32, numpy=
        array([[[0]],
              [[1]],
              [[2]]], dtype=int32)>
        After tile operation - repeat axis 0 1 time, anxis 1 5 times and axis 2 1 time
        <tf.Tensor: shape=(3, 5, 1), dtype=int32, numpy=
        array([[[0],
                [0],
                [0],
                [0],
                [0]],

            [[1],
                [1],
                [1],
                [1],
                [1]],

            [[2],
                [2],
                [2],
                [2],
                [2]]], dtype=int32)>
        So all we did here was create a tensor with batch indices.
    """
    combine_update_index = tf.tile(
        combine_update_index, [1, 5, 1])
    # concat finally creates a tensor with indices pointing to all the best anchor boxes in cells with objects
    combine_update_index = tf.concat(
        [combine_update_index, grid_cell_indices, highest_iou_index], axis=2)

    # boolean mask filters out all the anchor boxes without object in it since bounding_box_with_object_mask is created using true values
    combine_update_index = tf.boolean_mask(
        combine_update_index, mask=bounding_box_with_object_mask)
    
    combine_update_index_shape = tf.shape(combine_update_index)
    # # tf.print(f"combine_update_index.shape : {combine_update_index_shape}")

    ## set the mask values to true where actual bounding boxes are present
    positive_mask = tf.scatter_nd(
        indices=combine_update_index,
        shape=positive_mask_shape,
        updates=tf.constant(True, shape=(combine_update_index_shape[0],)))

    # select predicted anchor boxes based on negative masked values
    negative_mask = ~positive_mask
    # tf.print(f"negative_mask.shape : {tf.shape(negative_mask)}")
    
    # select all the anchor boxes that are suppose to be object less
    y_pred_reshaped = tf.reshape(
        y_pred, shape=(batch_size, grid_h, grid_w, NUM_ANCHORS, NUM_FEATURES))
    objectless_anchorboxes = tf.boolean_mask(y_pred_reshaped, mask=negative_mask)
    # tf.print(f"masked_values.shape : {tf.shape(objectless_anchorboxes)}")
    
    objectless_anchorboxes_shape = tf.shape(objectless_anchorboxes)
    # create a fake true tensor that matches the shape with all 0
    # we'll use this tensor to calculate the object less loss. 
    y_true_objectless = tf.zeros(
        shape=objectless_anchorboxes_shape, dtype=tf.float32)
    # tf.print(f"y_true_objectless.shape {tf.shape(y_true_objectless)}")


    y_pred_objectness = objectless_anchorboxes[:, 0]
    # tf.print(f"y_pred_objectness.shape : {tf.shape(y_pred_objectness)}")

    # True Values
    y_true_objectness = y_true_objectless[:, 0]
    # tf.print(f"y_true_objectness.shape : {tf.shape(y_true_objectness)}")


    # Apply activation functions to predicted values
    y_pred_objectness = tf.keras.activations.sigmoid(y_pred_objectness)
    # tf.print(f"Post Activation y_pred_objectness.shape : {tf.shape(y_pred_objectness)}")

    # Calculate loss
    objectless_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False)(y_true_objectness, y_pred_objectness)
    
    return objectless_loss

# loss function for the model


def calculate_model_loss(y_true, y_pred):
    # Find best anchor box
    expanded_y_true = tf.expand_dims(y_true, axis=2)
    best_anchor_boxes = calculate_best_anchor_boxes(y_true, y_pred)

    # Loss Calculation
    objectness_loss, bounding_box_loss, classification_loss = calculate_loss(
        best_anchor_boxes, expanded_y_true)
    
    # # tf.print("\n\n----- Localization Loss -----")
    # # tf.print(f"objectness_loss : {objectness_loss}")
    # tf.print(f"bounding_box_loss : {bounding_box_loss}")
    # tf.print(f"classification_loss : {classification_loss}")

    # objectless loss calculation
    # tf.print("\n\n----- Calculation Object Less Loss -----")
    objectless_loss = calculate_objectless_loss(
        y_true=y_true, y_pred=y_pred)
    
    # # tf.print("\n\n----- Object Less Loss -----")
    # # tf.print(f"objectless_loss : {objectless_loss}")

    # scale the losses
    lambda_objectness = 1
    lambda_bounding_box = 0.001
    lambda_classification = 1
    lambda_objectless = 1

    total_loss = (objectness_loss * lambda_objectness) + (bounding_box_loss *
                                                          lambda_bounding_box) + (classification_loss * lambda_classification) + (objectless_loss * lambda_objectless)

    # tf.print(f"\n\nTotal Loss : {total_loss}")

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
