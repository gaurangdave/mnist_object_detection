import tensorflow as tf

def reshape_prediction(prediction_object):
    prediction_shape = tf.shape(prediction_object)
    batch_size = prediction_shape[0]
    grid_width = prediction_shape[1]
    grid_height = prediction_shape[2]
    reshaped_prediction = tf.reshape(prediction_object, shape=(batch_size,grid_width,grid_height,-1,15))
    return reshaped_prediction

@tf.function
def post_process(predictions, confidence_score_threshold=0.5, iou_threshold=0.5, max_boxes=5):

    @tf.function
    def _process_single_prediction_image(prediction_image):
        # tf.print("prediction_image shape : ", tf.shape(prediction_image))
        image_shape = tf.shape(prediction_image)
        grid_width = tf.cast(image_shape[0], dtype=tf.float32)
        grid_height = tf.cast(image_shape[1], dtype=tf.float32)

        # step 0: normalize image
        normalized_image = tf.sigmoid(prediction_image[..., :5])
        # print("normalized image shape : ", tf.shape(normalized_image))
        # step 1: calculate confidence score
        confidence_scores = normalized_image[..., 0]
        # tf.print("confidence_scores shape : ", tf.shape(confidence_scores))

        # step 2: create boolean mask based on confidence score
        confidence_score_mask = confidence_scores[...,
                                                  :] > confidence_score_threshold
        # tf.print("confidence_score_mask ", tf.shape(confidence_score_mask))

        # create coordinate grid to decode the grid coordinates
        gridx_coordinate_range = tf.range(grid_width, dtype=tf.float32)
        gridy_coordinate_range = tf.range(grid_height, dtype=tf.float32)

        grid_y, grid_x = tf.meshgrid(
            gridy_coordinate_range, gridx_coordinate_range, indexing="ij")
        image_grid = tf.stack(values=[grid_y, grid_x], axis=-1)
        image_grid = image_grid[:, :, tf.newaxis, :]

        # decode the coordinates
        # Get the offsets and apply sigmoid
        x_offset = normalized_image[..., 1]
        y_offset = normalized_image[..., 2]

        # Get the grid indices (notice the swapped 0 and 1!)
        grid_x_index = image_grid[..., 1]
        grid_y_index = image_grid[..., 0]

        # 3. Apply the correct formula
        decoded_x_norm = (grid_x_index + x_offset) / grid_width
        decoded_y_norm = (grid_y_index + y_offset) / grid_height

        width = normalized_image[..., 3]
        height = normalized_image[..., 4]

        decoded_box = tf.stack(
            [decoded_x_norm, decoded_y_norm, width, height], axis=-1)

        # decode class
        class_scores = tf.nn.softmax(prediction_image[..., 5:], axis=-1)

        # decoded prediction
        confidence_scores = confidence_scores[:, :, :, tf.newaxis]
        decoded_prediction = tf.concat(
            [confidence_scores, decoded_box, class_scores], axis=-1)

        # # step 3: filter boxes based on the mask
        filtered_boxes = tf.boolean_mask(
            decoded_prediction, confidence_score_mask)
        # tf.print("filtered_boxes : ", tf.shape(filtered_boxes))

        # step 4: filter scores based on the mask
        filtered_scores = filtered_boxes[:, 0]
        # tf.print("filtered_scores : ", tf.shape(filtered_scores))

        # step 5: read and decode the values for NMS algorithm
        # prediction object flag, x_center, y_center, width, height, one hot encoded class values (0 to 9)
        x_center = filtered_boxes[..., 1] * 100
        y_center = filtered_boxes[..., 2] * 100
        width = filtered_boxes[..., 3] * 100
        height = filtered_boxes[..., 4] * 100

        # calculate min and max values
        x_min = tf.floor(x_center - (width / 2))
        x_max = tf.floor(x_center + (width / 2))
        y_min = tf.floor(y_center - (height / 2))
        y_max = tf.floor(y_center + (height / 2))

        boxes = tf.stack([y_min, x_min, y_max, x_max], axis=1)
        # tf.print("boxes shape : ", tf.shape(boxes))

        # step 5: perform NMLS
        nms_indices = tf.image.non_max_suppression(
            boxes=boxes, scores=filtered_scores, max_output_size=max_boxes, iou_threshold=iou_threshold)
        # tf.print("nms_indices : ", nms_indices)

        # step 6: Final boxes
        final_boxes = tf.gather(filtered_boxes, nms_indices)
        # tf.print("final_boxes.shape : ", tf.shape(final_boxes))

        return final_boxes

    # step 1: reshape predictions
    reshaped_prediction = reshape_prediction(predictions)
    # tf.print("reshaped_prediction shape : ", tf.shape(reshaped_prediction))

    # step 2: loop through the predictions and apply NMS to each prediction
    spec_final_data = tf.RaggedTensorSpec(
        shape=(None, 15), dtype=tf.float32, ragged_rank=0)

    final_predictions = tf.map_fn(
        _process_single_prediction_image, reshaped_prediction, parallel_iterations=20, fn_output_signature=spec_final_data)

    return final_predictions