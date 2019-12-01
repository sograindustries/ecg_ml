import tensorflow.keras as keras
from keras import backend as K

STRIDE = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
CHANNEL_GROWTH = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
KERNEL_SIZE = 16

def build_network():
    input_layer = keras.layers.Input(shape=[2560, 1], dtype='float32', name='input')

    # First layer
    layer = keras.layers.Conv1D(filters=32, kernel_size=KERNEL_SIZE, strides=1, padding='same', use_bias=False)(input_layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.SpatialDropout1D(0.5)(layer)

    # Creates first resnet layer
    shortcut = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(layer)
    layer = keras.layers.SeparableConv1D(filters=32, kernel_size=KERNEL_SIZE, strides=1, padding='same', use_bias=False)(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.SpatialDropout1D(0.5)(layer)
    layer = keras.layers.SeparableConv1D(filters=32, kernel_size=KERNEL_SIZE, strides=1, padding='same', use_bias=False)(layer)
    layer = keras.layers.Add()([shortcut, layer])

    # Creates 15 Resnet blocks
    filters = 32
    for block in range(15):
        channel_mult = CHANNEL_GROWTH[block]
        filters *= channel_mult
        shortcut = keras.layers.MaxPooling1D(pool_size=2, strides=STRIDE[block], padding='same')(layer)
        if channel_mult > 1:
            shortcut = keras.layers.SeparableConv1D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False)(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Activation('relu')(layer)
        layer = keras.layers.SeparableConv1D(filters=filters, kernel_size=KERNEL_SIZE, strides=STRIDE[block], padding='same', use_bias=False)(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Activation('relu')(layer)
        layer = keras.layers.SpatialDropout1D(0.5)(layer)
        layer = keras.layers.SeparableConv1D(filters=filters, kernel_size=KERNEL_SIZE, strides=1, padding='same', use_bias=False)(layer)
        layer = keras.layers.Add()([shortcut, layer])

    # Final layer
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation('relu')(layer)
    layer = keras.layers.Flatten()(layer)
    layer = keras.layers.Dropout(0.8)(layer)
    layer = keras.layers.Dense(2)(layer)
#    layer = keras.layers.TimeDistributed(keras.layers.Dense(2))(layer)
    output = keras.layers.Activation('softmax', name='output')(layer)

    model = keras.Model(inputs=input_layer, outputs = output)
    optimizer = keras.optimizers.Adam(
        lr=0.0001,
        clipnorm=1)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
#    model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
    return model
