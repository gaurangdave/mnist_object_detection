import tensorflow as tf
import numpy as np

# ============= MODEL DEFINITION =============
def build_model_inline(batch_size=None):
    inputs = tf.keras.Input(shape=(100, 100, 1), batch_size=batch_size, name="input_layer")
    x = tf.keras.layers.Rescaling(scale=1./255, name="rescaling")(inputs)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    outputs = tf.keras.layers.Conv2D(filters=45, kernel_size=1, padding='same', activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ============= MAIN EXPORT SCRIPT =============
if __name__ == "__main__":
    from src import yolo_object_detection_model
    from src import training_utils as tu
    
    MODEL_PATH = "models/yolo_experiment_1_digits_5_20_0.13.keras" 
    TEMP_EXPORT = "models/temp_base_model"
    EXPORT_PATH = "models/vertex_ai_deployment"
    
    # Step 1: Load and extract weights
    print(f"Loading model from {MODEL_PATH}...")
    custom_objects = {
        "calculate_model_loss": tu.calculate_model_loss,
        "YoloObjectDetectionModel": yolo_object_detection_model.YoloObjectDetectionModel
    }
    
    dirty_wrapper = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects=custom_objects
    )
    
    print("Extracting weights...")
    weights = [np.array(w, copy=True) for w in dirty_wrapper.model.get_weights()]
    
    del dirty_wrapper
    
    # Step 2: Build fresh model
    print("Building fresh model...")
    clean_model = build_model_inline(batch_size=None)
    
    # Step 3: Set weights
    print("Setting weights...")
    clean_model.set_weights(weights)
    del weights
    
    # Step 4: Test
    print("Testing clean model...")
    dummy_input = tf.zeros([1, 100, 100, 1], dtype=tf.float32)
    test_output = clean_model(dummy_input, training=False)
    print(f"Model test successful! Output shape: {test_output.shape}")
    
    # Step 5: Save base model WITHOUT post-processing using Keras export
    print(f"Exporting base model to {TEMP_EXPORT}...")
    try:
        clean_model.export(TEMP_EXPORT)
        print("Base model exported successfully using Keras export.")
    except Exception as e:
        print(f"Keras export failed: {e}")
        print("Trying alternative save method...")
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 100, 100, 1], dtype=tf.float32)])
        def predict_raw(x):
            return clean_model(x, training=False)
        
        concrete_fn = predict_raw.get_concrete_function()
        
        tf.saved_model.save(
            clean_model,
            TEMP_EXPORT,
            signatures=concrete_fn
        )
        print("Base model exported with alternative method.")
    
    print("\n" + "="*60)
    print("Base model saved successfully!")
    print("="*60 + "\n")
    
    # Step 6: Now load it back and add post-processing
    print(f"Loading base model from {TEMP_EXPORT}...")
    loaded_model = tf.saved_model.load(TEMP_EXPORT)
    
    # Get the inference function
    infer = loaded_model.signatures["serving_default"]
    
    print("Adding post-processing layer...")
    
    # Post-processing inline
    def reshape_prediction(prediction_object):
        prediction_shape = tf.shape(prediction_object)
        batch_size = prediction_shape[0]
        grid_width = prediction_shape[1]
        grid_height = prediction_shape[2]
        reshaped_prediction = tf.reshape(prediction_object, shape=(batch_size, grid_width, grid_height, -1, 15))
        return reshaped_prediction

    @tf.function
    def post_process_inline(predictions, confidence_score_threshold=0.5, iou_threshold=0.5, max_boxes=5):
        @tf.function
        def _process_single_prediction_image(prediction_image):
            image_shape = tf.shape(prediction_image)
            grid_width = tf.cast(image_shape[0], dtype=tf.float32)
            grid_height = tf.cast(image_shape[1], dtype=tf.float32)

            normalized_image = tf.sigmoid(prediction_image[..., :5])
            confidence_scores = normalized_image[..., 0]
            confidence_score_mask = confidence_scores[..., :] > confidence_score_threshold

            gridx_coordinate_range = tf.range(grid_width, dtype=tf.float32)
            gridy_coordinate_range = tf.range(grid_height, dtype=tf.float32)
            grid_y, grid_x = tf.meshgrid(gridy_coordinate_range, gridx_coordinate_range, indexing="ij")
            image_grid = tf.stack(values=[grid_y, grid_x], axis=-1)
            image_grid = image_grid[:, :, tf.newaxis, :]

            x_offset = normalized_image[..., 1]
            y_offset = normalized_image[..., 2]
            grid_x_index = image_grid[..., 1]
            grid_y_index = image_grid[..., 0]

            decoded_x_norm = (grid_x_index + x_offset) / grid_width
            decoded_y_norm = (grid_y_index + y_offset) / grid_height
            width = normalized_image[..., 3]
            height = normalized_image[..., 4]
            decoded_box = tf.stack([decoded_x_norm, decoded_y_norm, width, height], axis=-1)

            class_scores = tf.nn.softmax(prediction_image[..., 5:], axis=-1)
            confidence_scores = confidence_scores[:, :, :, tf.newaxis]
            decoded_prediction = tf.concat([confidence_scores, decoded_box, class_scores], axis=-1)

            filtered_boxes = tf.boolean_mask(decoded_prediction, confidence_score_mask)
            filtered_scores = filtered_boxes[:, 0]

            x_center = filtered_boxes[..., 1] * 100
            y_center = filtered_boxes[..., 2] * 100
            width = filtered_boxes[..., 3] * 100
            height = filtered_boxes[..., 4] * 100

            x_min = tf.floor(x_center - (width / 2))
            x_max = tf.floor(x_center + (width / 2))
            y_min = tf.floor(y_center - (height / 2))
            y_max = tf.floor(y_center + (height / 2))

            boxes = tf.stack([y_min, x_min, y_max, x_max], axis=1)
            nms_indices = tf.image.non_max_suppression(
                boxes=boxes, scores=filtered_scores, max_output_size=max_boxes, iou_threshold=iou_threshold)
            final_boxes = tf.gather(filtered_boxes, nms_indices)

            return final_boxes

        reshaped_prediction = reshape_prediction(predictions)
        spec_final_data = tf.RaggedTensorSpec(shape=(None, 15), dtype=tf.float32, ragged_rank=0)
        final_predictions = tf.map_fn(
            _process_single_prediction_image, reshaped_prediction, 
            parallel_iterations=20, fn_output_signature=spec_final_data)

        return final_predictions
    
    # Create wrapper that tracks the loaded model
    class PostProcessWrapper(tf.Module):
        def __init__(self, loaded_model):
            super().__init__()
            # Track the entire loaded model, not just the inference function
            self.loaded_model = loaded_model
            self.inference_fn = loaded_model.signatures["serving_default"]
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 100, 100, 1], dtype=tf.float32)])
        def __call__(self, input_images):
            # Use the tracked inference function
            raw_output = self.inference_fn(input_images)
            
            # Extract the actual tensor (handle dict output)
            if isinstance(raw_output, dict):
                raw_predictions = list(raw_output.values())[0]
            else:
                raw_predictions = raw_output
            
            final_boxes_ragged = post_process_inline(
                raw_predictions, 
                confidence_score_threshold=0.5, 
                iou_threshold=0.5
            )
            final_boxes_padded = final_boxes_ragged.to_tensor(default_value=-1.0)
            return {"outputs": final_boxes_padded}
    
    wrapper = PostProcessWrapper(loaded_model)  # Pass the whole loaded_model
    
    # Test the wrapper
    print("Testing post-processing wrapper...")
    result = wrapper(dummy_input)
    print(f"Wrapper test successful! Output shape: {result['outputs'].shape}")
    
    # Step 7: Save the final model
    print(f"\nExporting final model to {EXPORT_PATH}...")
    tf.saved_model.save(
        wrapper,
        EXPORT_PATH,
        signatures={"serving_default": wrapper.__call__}
    )
    
    print("\n" + "="*60)
    print("SUCCESS! Model ready for Vertex AI.")
    print(f"Location: {EXPORT_PATH}")
    print("="*60)