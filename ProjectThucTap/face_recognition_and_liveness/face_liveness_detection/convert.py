import tensorflow as tf

# Path to the saved model directory
saved_model_dir = 'model/1'

# Convert the saved model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_file = 'Tflite/FaceRecog.tflite'
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)