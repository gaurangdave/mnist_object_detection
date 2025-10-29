# TensorFlow Data Pipeline Deep Dive: `graph_compatible_data_generator.py`

**Philosophy**: *Mastery Over Milestone*  
This document explains every function in the TensorFlow-based data pipeline, focusing on building deep, intuitive understanding of how vectorized GPU operations replace traditional loops.

---

## `get_mnist_data`

### What It Does (High-Level Summary)
This function loads the MNIST dataset from Keras and caches it as TensorFlow constants in global memory. It ensures the 60,000 training images and labels are loaded exactly once, then returns the cached tensors on all subsequent calls.

### Why It Exists (Its Purpose in the Pipeline)
Loading MNIST data from disk is expensive (multiple seconds). Without caching, every call to `sample_base_digits` would trigger a full reload, crushing performance. This function implements the singleton pattern, guaranteeing the data is loaded once and reused throughout the pipeline's lifetime.

### The "Data Story" (Inputs & Outputs)
**Input**: None (reads from Keras datasets internally)  
**Output**: A tuple of two TensorFlow constants:
- `ALL_MNIST_DATA_PIXELS_TF`: shape `(60000, 28, 28)`, dtype `float32` — the raw pixel data
- `ALL_MNIST_DATA_CLASSES_TF`: shape `(60000,)`, dtype `float32` — the digit labels (0-9)

### 💡 "Vectorized Thinking" & Key Logic
The key insight is using a **global cache** (`_mnist_data_cache`) combined with TensorFlow constants. Once loaded, the data lives on the GPU/TPU as immutable tensors. The `global` keyword in Python ensures all pipeline workers share the same cached instance, avoiding redundant disk I/O. The `tf.constant()` conversion is critical: it tells TensorFlow this data is fixed and can be optimized into the computation graph.

---

## `get_sample_indices`

### What It Does (High-Level Summary)
This function generates a tensor of random integer indices to sample from the MNIST dataset. It returns `size` random integers between 0 and `dataset_len - 1`.

### Why It Exists (Its Purpose in the Pipeline)
We need to randomly select digits from the 60,000-image MNIST pool. This function provides the random indices that `sample_base_digits` uses to gather specific images. The `@tf.function` decorator compiles this into a TensorFlow graph, making it GPU-compatible and traceable.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `dataset`: A 1D tensor of any length (typically the class labels tensor)
- `size`: Python int, the number of random indices to generate (e.g., 5)

**Output**: A 1D tensor of shape `(size,)`, dtype `int32`, containing random indices.

### 💡 "Vectorized Thinking" & Key Logic
Instead of using a Python `for` loop with `random.randint()`, this function uses **`tf.random.uniform`** to generate all `size` random numbers in a single GPU operation. The `minval=0, maxval=dataset_len` parameters define the valid range, and `dtype=tf.int32` ensures the output is usable as tensor indices. This is orders of magnitude faster than sequential random number generation.

---

## `sample_base_digits`

### What It Does (High-Level Summary)
This function randomly selects `num_of_digits` images and their corresponding class labels from the cached MNIST dataset, returning them as tensors ready for augmentation and bounding box calculation.

### Why It Exists (Its Purpose in the Pipeline)
Each training example needs multiple digits (e.g., 5 digits on a canvas). This function efficiently samples those digits from the master MNIST dataset without loading data from disk. It's the "feeder" function that provides raw digit data to the pipeline.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `num_of_digits`: Python int (e.g., 5)

**Output**: A tuple of two tensors:
- `sample_pixels`: shape `(num_of_digits, 28, 28, 1)`, dtype `float32` — the digit images
- `sample_values`: shape `(num_of_digits, 1)`, dtype `float32` — the class labels

### 💡 "Vectorized Thinking" & Key Logic
The magic here is **`tf.gather`** with `batch_dims=1`. Instead of a loop like:
```python
for i in range(num_of_digits):
    sample_pixels[i] = ALL_MNIST_DATA[sample_indices[i]]
```
`tf.gather` executes all `num_of_digits` lookups in a single, parallelized GPU operation. The `axis=0` parameter tells TensorFlow to index along the batch dimension, and `batch_dims=1` ensures each index grabs exactly one image. The subsequent `tf.reshape` adds the channel dimension `(28, 28, 1)` required by Keras Conv2D layers.

---

## `augment_digits`

### What It Does (High-Level Summary)
This function applies random geometric transformations (translation, zoom, rotation) to a batch of 28×28 digit images using Keras preprocessing layers, returning the augmented images as a tensor.

### Why It Exists (Its Purpose in the Pipeline)
Data augmentation increases model robustness by creating variations of the same digit. However, Keras augmentation layers **do not update bounding box coordinates**—they only transform pixels. This function performs the pixel transformation; downstream functions like `calculate_tight_bbox` must recalculate the bounding boxes from scratch.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `digits`: A tensor of shape `(m, 28, 28, 1)` where `m` is the batch size

**Output**: A tensor of the same shape `(m, 28, 28, 1)` with randomly transformed digits

### 💡 "Vectorized Thinking" & Key Logic
The key is using **Keras preprocessing layers** wrapped in a `tf.keras.Sequential` model. These layers are GPU-accelerated and operate on the entire batch simultaneously:
- `RandomTranslation` shifts all `m` images by random offsets in one kernel call
- `RandomZoom` scales all images in parallel
- `RandomRotation` rotates all images in a single matrix multiplication

The `fill_mode="constant"` with `fill_value=0.0` ensures that empty regions created by transformations are filled with black (background), preserving the assumption that non-zero pixels represent the digit.

---

## `calculate_tight_bbox`

### What It Does (High-Level Summary)
This function takes a batch of augmented digit images and calculates a tight, minimal bounding box for the "lit-up" (non-zero) pixels in each image, returning a tensor with bounding box coordinates and dimensions.

### Why It Exists (Its Purpose in the Pipeline)
After augmentation (rotation, zoom, translation), the pixel locations of each digit change, but Keras does **not** provide updated bounding box coordinates. This function reconstructs the bounding boxes by analyzing which pixels are non-zero, computing the minimum and maximum x/y coordinates, and packaging them with width, height, center, and class label.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `pixels`: shape `(m, 28, 28, 1)` — batch of augmented digit images
- `class_values`: shape `(m, 1)` — class labels
- `padding`: int (default 1) — extra pixels to add around the tight bbox

**Output**: A tensor of shape `(m, 9)` containing:
```
[x_min, x_max, y_min, y_max, x_center, y_center, width, height, class_id]
```

### 💡 "Vectorized Thinking" & Key Logic
This is one of the most sophisticated functions in the pipeline. Instead of looping through each of the `m` images:

1. **Projection Trick**: Use `tf.reduce_sum(pixels, axis=[2, 3])` to "collapse" the image into a 1D array. This creates `active_rows` of shape `(m, 28)` where each value is the sum of pixels in that row. Similarly, `active_cols` sums along columns. Now finding non-zero rows/cols is a simple inequality check.

2. **Coordinate Extraction**: Use `tf.where(active_rows != 0)` to get a 2D tensor of `[batch_index, coordinate]` pairs for all non-zero locations across all `m` images simultaneously.

3. **Segment Operations**: The genius move is **`tf.math.segment_min`** and **`tf.math.segment_max`**. These functions take:
   - `segment_ids`: the batch index (which image does this coordinate belong to?)
   - `data`: the actual coordinate value
   
   And they compute the min/max coordinate **per batch index** in one parallel operation. This replaces what would be a nested loop:
   ```python
   for i in range(m):
       y_min[i] = min(coordinates where batch_index == i)
   ```

4. **Vectorized Padding**: Instead of an `if` statement per image, use `tf.where(condition, padded_value, original_value)` to apply padding conditionally across all `m` images at once.

The final `tf.concat` stacks all 9 pieces of information (coordinates, dimensions, class) into a single `(m, 9)` tensor.

---

## `get_bbox_corners`

### What It Does (High-Level Summary)
This utility function extracts the four corner coordinates (x_min, x_max, y_min, y_max) from a bounding box information tensor, casting them to `int32` for use as array indices.

### Why It Exists (Its Purpose in the Pipeline)
Bounding box data is packed into a single tensor for efficiency. This function provides clean, readable access to specific components instead of using magic numbers like `bbox[..., 0]`. It improves code maintainability and reduces errors.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `bbox_info`: shape `(..., 9)` or `(..., 11)` or `(..., 13)` — tensor with bbox coordinates at indices 0-3

**Output**: A tuple of four tensors, each of shape `(...)`:
- `y_min, x_min, y_max, x_max`

### 💡 "Vectorized Thinking" & Key Logic
The `[..., INDEX]` syntax is TensorFlow's ellipsis notation—it means "preserve all dimensions except the last one, then extract element INDEX from that last dimension." This works on batches of any shape without hardcoding the batch size. The `tf.cast` ensures the coordinates are integers, which is required for tensor slicing operations like `tf.slice` downstream.

---

## `get_bbox_dimensions`

### What It Does (High-Level Summary)
This utility function extracts the width and height from a bounding box information tensor, casting them to `int32`.

### Why It Exists (Its Purpose in the Pipeline)
Similar to `get_bbox_corners`, this function provides clean access to bbox dimensions. Width and height are needed to create patches, calculate canvas placement, and normalize predictions.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `bbox_info`: shape `(..., 9)` or larger — tensor with width at index 6, height at index 7

**Output**: A tuple of two tensors, each of shape `(...)`:
- `height, width`

### 💡 "Vectorized Thinking" & Key Logic
The ellipsis notation `[..., INDEX]` allows this function to work on single bboxes `(9,)`, batches `(m, 9)`, or even higher-dimensional inputs without modification. This is a key principle in TensorFlow: write functions that are **shape-agnostic** in the batch dimensions.

---

## `get_bbox_center`

### What It Does (High-Level Summary)
This utility function extracts the center coordinates (x_center, y_center) from a bounding box information tensor, casting them to `int32`.

### Why It Exists (Its Purpose in the Pipeline)
The center coordinates are used for placement calculations and are part of the final prediction output (in normalized form). This function provides clean, typed access to these values.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `bbox_info`: shape `(..., 9)` or larger — tensor with centers at indices 4 and 5

**Output**: A tuple of two tensors, each of shape `(...)`:
- `y_center, x_center`

### 💡 "Vectorized Thinking" & Key Logic
Same ellipsis pattern as above—this function is batch-agnostic. The `tf.cast` to `int32` ensures these values can be used in integer arithmetic for grid calculations.

---

## `get_canvas_placement`

### What It Does (High-Level Summary)
This utility function extracts the final canvas placement coordinates (top, left) from the extended bounding box information tensor, which contains the position where the digit will be placed on the 100×100 canvas.

### Why It Exists (Its Purpose in the Pipeline)
After calculating random placement positions within grid cells, those positions are appended to the bbox tensor. This function retrieves them for use in canvas construction and prediction generation.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `bbox_info`: shape `(..., 11)` or larger — tensor with canvas coordinates at indices 9 and 10

**Output**: A tuple of two tensors, each of shape `(...)`:
- `canvas_top, canvas_left`

### 💡 "Vectorized Thinking" & Key Logic
This is the same pattern as the other `get_bbox_*` functions. The indices 9 and 10 are defined in the constants module (C.BBOX_CANVAS_TOP_IDX, C.BBOX_CANVAS_LEFT_IDX), making the code self-documenting. The ellipsis syntax ensures the function works on batches.

---

## `generate_grid`

### What It Does (High-Level Summary)
This function creates a 2D coordinate grid of shape `(grid_size, grid_size, 2)`, where each element contains its `[x, y]` coordinate. For example, `generate_grid(3)` creates a 3×3 grid of coordinates from `[0,0]` to `[2,2]`.

### Why It Exists (Its Purpose in the Pipeline)
To prevent digits from overlapping on the 100×100 canvas, we divide the canvas into a grid (e.g., 3×3 = 9 cells). This function generates all possible grid cell coordinates, which are then randomly sampled and scaled to canvas coordinates.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `grid_size`: int (e.g., 3)

**Output**: A tensor of shape `(grid_size, grid_size, 2)` containing coordinates:
```
[[[0,0], [1,0], [2,0]],
 [[0,1], [1,1], [2,1]],
 [[0,2], [1,2], [2,2]]]
```

### 💡 "Vectorized Thinking" & Key Logic
The key is **`tf.meshgrid`**, which creates two 2D arrays:
- `grid_X`: A matrix where each row is `[0, 1, 2, ..., grid_size-1]`
- `grid_Y`: A matrix where each column is `[0, 1, 2, ..., grid_size-1]`

Then `tf.stack([grid_X, grid_Y], axis=2)` combines them into coordinate pairs. This replaces a double nested loop:
```python
for i in range(grid_size):
    for j in range(grid_size):
        grid[i, j] = [i, j]
```
with a single, GPU-optimized operation.

---

## `get_canvas_grid_cells`

### What It Does (High-Level Summary)
This function generates `batch_size` random grid cell coordinates from a grid of size `grid_size`, scales them to the 100×100 canvas, and returns both the top-left corner and bottom-right corner (width/height limits) of each cell.

### Why It Exists (Its Purpose in the Pipeline)
We need to place `batch_size` digits on the canvas without overlapping. This function divides the canvas into a grid, randomly selects `batch_size` cells (without replacement via shuffle), and provides the coordinate boundaries for each cell.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `batch_size`: int (e.g., 5)
- `grid_size`: int (e.g., 3)

**Output**: A tensor of shape `(batch_size, 4)` where each row is:
```
[grid_cell_x, grid_cell_y, grid_cell_x_limit, grid_cell_y_limit]
```
All coordinates are scaled to the 100×100 canvas.

### 💡 "Vectorized Thinking" & Key Logic
1. **Grid Generation**: Call `generate_grid(grid_size)` to get all possible cells.
2. **Flatten and Shuffle**: Reshape the grid to `(grid_size², 2)` and use `tf.random.shuffle` to randomize the order. This ensures random selection without replacement.
3. **Slice**: Take the first `batch_size` rows using `shuffled_grid[:batch_size, :]`.
4. **Scale**: Multiply by `grid_cell_size = 100 / grid_size` to convert from grid coordinates to canvas coordinates.
5. **Calculate Limits**: Add `grid_cell_size` to get the bottom-right corner of each cell.

The genius is using **`tf.random.shuffle`** instead of a rejection-sampling loop to ensure unique cell assignments.

---

## `map_bbox_to_grid_cells`

### What It Does (High-Level Summary)
This function takes a bounding box and its assigned grid cell boundaries, then randomly places the bbox within that cell by generating a random `(top, left)` coordinate that ensures the bbox fits entirely inside the cell.

### Why It Exists (Its Purpose in the Pipeline)
We know which grid cell a digit should go in, but we don't want every digit in the exact same corner of its cell—that would create artificial patterns. This function adds randomness by placing the bbox at a random position within the cell, constrained by the bbox's width and height.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `bbox_grid_cells`: A tensor of shape `(13,)` containing:
  - Indices 0-8: standard bbox info (x_min, x_max, y_min, y_max, centers, width, height, class)
  - Indices 9-12: grid cell boundaries (x, y, x_limit, y_limit)

**Output**: A tensor of shape `(2,)` containing `[top, left]` — the random placement within the cell.

### 💡 "Vectorized Thinking" & Key Logic
Instead of a loop or rejection sampling, this function uses **`tf.random.uniform`** with carefully calculated bounds:
- `max_left = grid_cell_width_limit - bbox_width` (the rightmost position where the bbox still fits)
- `max_top = grid_cell_height_limit - bbox_height` (the bottommost position)

Then `tf.random.uniform(minval=grid_cell_x, maxval=max_left+1)` generates a random position in the valid range. The `+1` is required because `maxval` is exclusive in `tf.random.uniform`. This guarantees no out-of-bounds placement.

---

## `map_bbox_to_patch_indices`

### What It Does (High-Level Summary)
This function takes a single digit image and its bounding box information (including canvas placement), extracts the relevant pixel patch from the 28×28 image, and generates two sets of indices: one for the original image coordinates and one for the 100×100 canvas coordinates where the patch will be placed.

### Why It Exists (Its Purpose in the Pipeline)
To place a digit on the canvas, we need to:
1. Read the correct pixels from the 28×28 image (the region inside the bbox)
2. Map those pixels to the correct location on the 100×100 canvas

This function computes both mappings, preparing the data for `tf.scatter_nd` to perform the actual placement.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `elems`: A tuple of:
  - `single_image_data`: shape `(28, 28, 1)` — one digit image
  - `bbox_grid_cell_top_left`: shape `(11,)` — bbox info with canvas placement

**Output**: A tuple of:
- `patch_data`: shape `(height × width,)` — flattened pixel values (a ragged tensor)
- `canvas_indices`: shape `(height × width, 2)` — canvas coordinates for each pixel (a ragged tensor)

### 💡 "Vectorized Thinking" & Key Logic
This function is called via `tf.map_fn`, so it processes one digit at a time but prepares data for a batched operation:

1. **Meshgrid for Patch Coordinates**: Use `tf.meshgrid(tf.range(bbox_height), tf.range(bbox_width))` to create a grid of relative coordinates `(0,0), (0,1), ..., (height-1, width-1)`.

2. **Offset Addition**: Add the bbox's `(y_min, x_min)` to shift the patch grid to the correct location in the 28×28 image. This is done with broadcasting: `patch_indices = y_min_x_min_slice[newaxis, newaxis, :] + patch_grid`.

3. **Slice Image Data**: Use `tf.slice(single_image_data, begin=[bbox_y_min, bbox_x_min, 0], size=[bbox_height, bbox_width, 1])` to extract exactly the pixels inside the bbox. This is a zero-copy operation on GPUs.

4. **Canvas Coordinate Calculation**: Repeat the same offset trick, but add the canvas `(top, left)` instead of the image `(y_min, x_min)`. This maps patch coordinates to canvas coordinates.

5. **Flatten**: Reshape both `patch_data` and `canvas_indices` to 1D and 2D respectively, preparing them for aggregation.

The key insight is using **broadcasting** to add offsets to entire grids at once, avoiding per-pixel loops.

---

## `place_digit_on_canvas`

### What It Does (High-Level Summary)
This function takes a batch of `m` digit images and their bounding boxes, randomly places each digit in a different grid cell on a 100×100 canvas, and returns the final canvas image with all digits composited together.

### Why It Exists (Its Purpose in the Pipeline)
This is the core composition function. It combines all previous operations (grid generation, placement calculation, patch extraction) and uses `tf.scatter_nd` to "paste" all `m` digits onto the canvas in a single, massively parallel GPU operation.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `pixels`: shape `(m, 28, 28, 1)` — batch of augmented digit images
- `class_values_with_bbox`: shape `(m, 9)` — bbox info for each digit

**Output**: A tuple of:
- `canvas`: shape `(100, 100)` — the final composed image
- `bbox_grid_cell_top_left`: shape `(m, 11)` — extended bbox info with canvas placement

### 💡 "Vectorized Thinking" & Key Logic
This is the most sophisticated function in the entire pipeline. The genius is avoiding a `for` loop over the `m` digits:

1. **Grid Cell Assignment**: Call `get_canvas_grid_cells` to get `m` random, non-overlapping grid cells.

2. **Placement Calculation**: Use `tf.map_fn(map_bbox_to_grid_cells, ...)` to calculate random `(top, left)` positions within each cell. This is parallelized across the batch.

3. **Patch Extraction**: Call `tf.map_fn(map_bbox_to_patch_indices, ...)` with `fn_output_signature=(spec_patch_data, spec_canvas_indices)`. This is critical—the output is **ragged tensors** because different digits have different bbox sizes (e.g., a "1" might be 15×8 pixels, while a "0" might be 20×18).

4. **Flatten All Patches**: Use `.flat_values` to concatenate all patch data and indices across all `m` digits into two large, dense tensors:
   - `all_updates`: shape `(total_pixels,)` — pixel values from all patches
   - `all_indices`: shape `(total_pixels, 2)` — canvas coordinates for all pixels

5. **Single Scatter**: Call `tf.scatter_nd(indices=all_indices, updates=all_updates, shape=[100, 100])` **once** to place all pixels from all digits simultaneously. This is a single GPU kernel call that writes to the canvas in parallel.

The key insight: Instead of looping through `m` digits and calling `scatter_nd` `m` times, we **aggregate all patches and indices first**, then scatter once. This is orders of magnitude faster.

---

## `translate_bbox_to_prediction`

### What It Does (High-Level Summary)
This function converts bounding box information (in canvas coordinates) into a normalized prediction tensor suitable for training an object detection model. It calculates normalized centers, widths, heights, one-hot encodes the class labels, and assembles everything into a `(MAX_DIGITS, 15)` tensor.

### Why It Exists (Its Purpose in the Pipeline)
Object detection models like YOLO expect predictions in a specific format: normalized coordinates (0-1 range), one-hot encoded classes, and a "flag" indicating whether an object is present. This function performs that transformation, creating the ground-truth label tensor for training.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `bbox_grid_cell_top_left`: shape `(num_digits, 11)` — bbox info with canvas placement

**Output**: A tensor of shape `(MAX_DIGITS, 15)` where each row is:
```
[flag, x_center, y_center, width, height, one_hot_class_0, ..., one_hot_class_9]
```
Rows beyond `num_digits` are zero-filled (representing "no object").

### 💡 "Vectorized Thinking" & Key Logic
1. **Canvas Center Calculation**: Use the formula `canvas_x_center = (2 * bbox_canvas_left + bbox_width - 1) / 2` to find the center of the bbox on the canvas. This is vectorized across all `num_digits` digits.

2. **Normalization**: Divide by 100.0 to convert from pixel coordinates to the 0-1 range expected by neural networks.

3. **One-Hot Encoding**: Use `tf.one_hot(indices=bbox_class_val, depth=10)` to convert class integers (0-9) into 10-element vectors like `[0,0,1,0,0,0,0,0,0,0]` for class 2.

4. **Scatter into Fixed-Size Tensor**: The tricky part is placing `num_digits` predictions into a `(MAX_DIGITS, 15)` tensor. This uses a clever **double-indexing trick**:
   - Create `batch_indices = [0, 1, 2, ..., num_digits-1]` and broadcast it to shape `(num_digits, 15, 1)`.
   - Create `feature_indices = [0, 1, 2, ..., 14]` and broadcast it to shape `(num_digits, 15, 1)`.
   - Concatenate them to get `final_indices` of shape `(num_digits, 15, 2)`, where each element is `[batch_idx, feature_idx]`.
   - Call `tf.scatter_nd(indices=final_indices, updates=updates, shape=(MAX_DIGITS, 15))` to place each prediction row.

The key insight: By creating indices that map every element of the `updates` tensor to its position in the output tensor, we can fill the entire `(num_digits, 15)` block in one `scatter_nd` call instead of looping.

---

## `_generate_training_example_tf`

### What It Does (High-Level Summary)
This is the main orchestration function that ties the entire pipeline together. It takes a single digit image and label, samples additional digits, augments them, calculates bounding boxes, places them on a canvas, and generates the prediction tensor—producing one complete training example.

### Why It Exists (Its Purpose in the Pipeline)
This function implements the full data generation logic: from raw MNIST digit to augmented multi-digit canvas with ground-truth labels. It's called by the `tf.data.Dataset` pipeline to generate batches of training examples on-the-fly.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `x`: shape `(28, 28)` — a single digit image (from the base dataset)
- `y`: shape `()` — a single class label
- `num_of_digits`: int — total number of digits to place on the canvas (e.g., 5)

**Output**: A tuple of:
- `canvas`: shape `(100, 100)` — the composed image with multiple digits
- `prediction`: shape `(MAX_DIGITS, 15)` — ground-truth labels for object detection

### 💡 "Vectorized Thinking" & Key Logic
This function orchestrates the pipeline, so the key insight is in the **workflow**:

1. **Combine Base and Sampled Digits**: The input `x, y` is one digit from the dataset iteration. If `num_of_digits > 1`, call `sample_base_digits(num_of_digits - 1)` to get additional digits, then concatenate them into a batch using `tf.concat`.

2. **Augmentation and Cleanup**: Pass the batch through `augment_digits`, then apply `tf.nn.relu` to ensure no negative pixel values (Keras augmentation can produce slight negatives due to interpolation).

3. **Sequential Processing**: Call `calculate_tight_bbox`, `place_digit_on_canvas`, and `translate_bbox_to_prediction` in sequence. Each function adds more information to the tensor (e.g., bbox → bbox+canvas_coords → prediction).

4. **Graph Compatibility**: Because this is decorated with `@tf.function`, TensorFlow traces it once and compiles it into a static graph. The `num_of_digits` parameter must be a `tf.constant` (not a Python variable) to avoid retracing for every different value.

The key design principle: **Each function outputs the input tensor plus additional columns**, creating a "growing" tensor that accumulates information as it flows through the pipeline. This avoids creating intermediate dictionaries or complex data structures.

---

## `create_data_generator`

### What It Does (High-Level Summary)
This is a **factory function** that takes a Python integer `num_of_digits` and returns a `tf.function`-decorated function suitable for use with `tf.data.Dataset.map()`. It solves the problem of passing parameters to map functions in TensorFlow pipelines.

### Why It Exists (Its Purpose in the Pipeline)
`tf.data.Dataset.map()` expects a function with signature `(x, y) -> (new_x, new_y)`, but our `_generate_training_example_tf` function needs a third parameter `num_of_digits`. We can't use a lambda because lambdas aren't traceable by `@tf.function`. This factory creates a closure that "bakes in" the `num_of_digits` parameter.

### The "Data Story" (Inputs & Outputs)
**Input**:
- `num_of_digits`: Python int (e.g., 5)

**Output**: A function with signature:
```python
@tf.function
def generate_example_map_fn(x, y) -> (canvas, prediction)
```

### 💡 "Vectorized Thinking" & Key Logic
The key insight is the **closure pattern**:

1. **Convert to TensorFlow Constant**: `num_digits_tf = tf.constant(num_of_digits, dtype=tf.int32)`. This ensures the value is part of the TensorFlow graph, not a Python variable that would trigger retracing.

2. **Define Inner Function**: The `generate_example_map_fn` function "closes over" `num_digits_tf`, meaning it has access to this variable from the outer scope.

3. **Return the Function**: Return the inner function (not its result). The caller then does `dataset.map(create_data_generator(5))`, which is equivalent to `dataset.map(lambda x, y: _generate_training_example_tf(x, y, 5))`, but traceable.

This is a classic **functional programming** pattern that solves a fundamental limitation of TensorFlow's graph execution model. Without this pattern, you'd need to create separate functions for `generate_3_digits`, `generate_5_digits`, etc.

---

## Summary: The Complete Pipeline Flow

Here's how all these functions connect:

```
1. get_mnist_data() → Loads and caches 60,000 MNIST images
2. sample_base_digits() → Randomly selects digits using get_sample_indices()
3. augment_digits() → Applies geometric transformations
4. calculate_tight_bbox() → Finds non-zero pixels and computes bbox
5. generate_grid() + get_canvas_grid_cells() → Creates non-overlapping placement zones
6. map_bbox_to_grid_cells() → Randomly places bbox within its grid cell
7. map_bbox_to_patch_indices() → Extracts pixel patches and computes indices
8. place_digit_on_canvas() → Uses tf.scatter_nd to composite all digits
9. translate_bbox_to_prediction() → Converts bbox to normalized labels
10. _generate_training_example_tf() → Orchestrates steps 2-9
11. create_data_generator() → Wraps step 10 for tf.data.Dataset.map()
```

**The Core Philosophy**: Replace every `for` loop with a TensorFlow operation:
- `for i in range(m): sample[i] = data[indices[i]]` → `tf.gather`
- `for i in range(m): y_min[i] = min(coords[i])` → `tf.math.segment_min`
- `for i in range(m): canvas += digit[i]` → `tf.scatter_nd` (once, with all patches)
- `for i in range(grid_size): for j in range(grid_size): grid[i,j] = [i,j]` → `tf.meshgrid`

By thinking in parallel, vectorized operations instead of sequential loops, this pipeline achieves GPU-accelerated performance with minimal CPU overhead.
