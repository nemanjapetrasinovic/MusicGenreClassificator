import tensorflow as tf
import numpy as np


def create_model(x_train, x_test, classes, trainable_encoder=False):
    input_shape = x_train.shape[1:]

    x = tf.keras.layers.Input(shape=input_shape, name='input')

    backbone = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet',
                                                input_tensor=x, pooling='avg', classes=classes)

    if not trainable_encoder:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    y = backbone.output

    y = tf.keras.layers.Dropout(rate=0.5)(y)
    y = tf.keras.layers.Dense(classes, activation='softmax', name='output')(y)

    model = tf.keras.models.Model(inputs=x, outputs=y)

    x_train_pp = tf.keras.applications.resnet_v2.preprocess_input(x_train)
    x_test_pp = tf.keras.applications.resnet_v2.preprocess_input(x_test)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
 
    model.summary()

    return x_train_pp, x_test_pp, model
