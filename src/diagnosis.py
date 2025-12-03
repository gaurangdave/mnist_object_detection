import tensorflow as tf
from src import training_utils as tu
from src import yolo_object_detection_model

# Load the problematic model
MODEL_PATH = "models/yolo_experiment_1_digits_5_20_0.13.keras"

custom_objects = {
    "calculate_model_loss": tu.calculate_model_loss,
    "YoloObjectDetectionModel": yolo_object_detection_model.YoloObjectDetectionModel
}

dirty_wrapper = tf.keras.models.load_model(
    MODEL_PATH, 
    compile=False, 
    custom_objects=custom_objects
)

print("=== Wrapper Type ===")
print(f"Type: {type(dirty_wrapper)}")
print(f"Class: {dirty_wrapper.__class__}")

print("\n=== Inner Model ===")
print(f"Has 'model' attribute: {hasattr(dirty_wrapper, 'model')}")
if hasattr(dirty_wrapper, 'model'):
    print(f"Inner model type: {type(dirty_wrapper.model)}")
    print(f"Inner model class: {dirty_wrapper.model.__class__}")

print("\n=== Checking for _DictWrapper ===")
for attr_name in dir(dirty_wrapper):
    if not attr_name.startswith('_'):
        try:
            attr = getattr(dirty_wrapper, attr_name)
            if '_DictWrapper' in str(type(attr)):
                print(f"Found _DictWrapper in attribute: {attr_name}")
                print(f"  Type: {type(attr)}")
        except:
            pass

if hasattr(dirty_wrapper, 'model'):
    print("\n=== Checking inner model for _DictWrapper ===")
    for attr_name in dir(dirty_wrapper.model):
        if not attr_name.startswith('_'):
            try:
                attr = getattr(dirty_wrapper.model, attr_name)
                if '_DictWrapper' in str(type(attr)):
                    print(f"Found _DictWrapper in inner model attribute: {attr_name}")
                    print(f"  Type: {type(attr)}")
            except:
                pass

print("\n=== Wrapper __dict__ check ===")
try:
    print(f"Wrapper dict keys: {list(dirty_wrapper.__dict__.keys())[:10]}")
except Exception as e:
    print(f"Error accessing wrapper __dict__: {e}")

if hasattr(dirty_wrapper, 'model'):
    print("\n=== Inner model __dict__ check ===")
    try:
        print(f"Inner model dict keys: {list(dirty_wrapper.model.__dict__.keys())[:10]}")
    except Exception as e:
        print(f"Error accessing inner model __dict__: {e}")