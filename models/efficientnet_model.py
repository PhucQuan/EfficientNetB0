from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def create_model(
        num_classes,
        image_size=224,
        dropout_rate=0.55,
        l2_value=0.001,
        weights="imagenet",
        train_backbone=False):

    inputs = keras.Input(
        shape=(image_size, image_size, 3),
        name="input_image"
    )

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_tensor=inputs
    )
    backbone.trainable = train_backbone

    x = backbone.output
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)
    x = layers.Dropout(
        dropout_rate,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2_value),
        name="predictions"
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="fruit8_efficientnetb0"
    )
