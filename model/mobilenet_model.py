#import library tensorflow to build a model
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_mobilenetv2_model(num_classes: int,
                            input_shape: tuple=(224,224,3),
                            base_trainable: bool=False,
                            fine_tune_at: int=100,
                            dropout_rate: float=0.4,
                            l2_reg: float=1e-4,
                            learning_rate: float=1e-4) -> tf.keras.Model:
    """
    Build and compile a MobileNetV2 model with a custom head.

    Args:
        num_classes: number of output classes
        input_shape: image input shape
        base_trainable: whether to set base MobileNetV2 trainable initially
        fine_tune_at: if fine-tuning, the layer index to start training from (closer to output)
        dropout_rate: dropout rate in head
        l2_reg: L2 regularization for Dense layers
        learning_rate: optimizer LR
    Returns:
        compiled tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)
    # Data augmentation can be added outside or here:
    x = inputs

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'  # transfer learning
    )
    base_model.trainable = base_trainable

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Complex head: Dense -> BN -> Dropout -> Dense -> Dropout -> Output
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Optionally fine-tune: unfreeze from fine_tune_at
    if fine_tune_at is not None:
        base_model.trainable = True
        # Freeze earlier layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
