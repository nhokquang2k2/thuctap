import tensorflow as tf

class LivenessNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        INPUT_SHAPE = (height, width, depth)
        chanDim = -1
        if tf.keras.backend.image_data_format() == 'channels_first':
            INPUT_SHAPE = (depth, height, width)
            chanDim = 1
        model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu' ,input_shape=INPUT_SHAPE),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(classes, activation='softmax')
            ])
        
        return model