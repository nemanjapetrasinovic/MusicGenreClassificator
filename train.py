import matplotlib
from tensorflow.core.framework.dataset_options_pb2 import AUTO
from model import *
import model_resnet as modelres
import model_new as model_new_cnn
import tensorflow as tf
from data import *
from settings import *

########################################################################################################################
# LOADING DATA
########################################################################################################################

import matplotlib.pyplot as plt
plt.clf()

categories, train_data, test_data, x_train, y_train, x_test, y_test = load_data(DATA_PATH)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


########################################################################################################################
# BUILDING MODEL
########################################################################################################################

# Model CNN
# model = create_model(x_train, x_test, classes=len(categories), trainable_encoder=False)

# Model ResNet
x_train_pp, x_test_pp, model = modelres.create_model(x_train, x_test, classes=len(categories), trainable_encoder=False)

# Model CNN New
#model = model_new_cnn.create_model(x_train, x_test, classes=len(categories), trainable_encoder=False)

########################################################################################################################
# TRAINING MODEL
########################################################################################################################


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',
                                                  patience=30, restore_best_weights=True, verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', mode='max',
                                                 factor=0.1, patience=10, verbose=1)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(TMP_PATH, 'training.csv'))

# Model CNN
# hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1000,
#                 validation_data=(x_test, y_test), verbose=2,
#                 callbacks=[early_stopping, reduce_lr, csv_logger])

# Model Resnet
hist = model.fit(x_train_pp, y_train, batch_size=BATCH_SIZE, epochs=1000,
                 validation_data=(x_test_pp, y_test), verbose=2,
                 callbacks=[early_stopping, reduce_lr, csv_logger])


path = os.path.join(TMP_PATH, 'trained_model.h5')
model.save(path, include_optimizer=False)

plt.clf()
plt.plot(hist.history['loss'], label="loss")
plt.plot(hist.history['val_loss'], label="val_loss")
plt.legend()
plt.savefig(os.path.join(TMP_PATH, 'training_loss.png'))

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'], label="accuracy")
plt.plot(hist.history['val_sparse_categorical_accuracy'], label="val_accuracy")
plt.legend()
plt.savefig(os.path.join(TMP_PATH, 'training_accuracy.png'))


########################################################################################################################
# EVALUATE MODEL
########################################################################################################################


res = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(res)

y_out = model.predict(x_test, batch_size=BATCH_SIZE)
y_out = np.argmax(y_out, axis=1)
i = 0
plt.figure(figsize=(4, 4))
for img, out, exp in zip(x_test, y_out, y_test):
    if out != exp:
        # plt.clf()
        # plt.imshow(img)
        title = '{} misclassified as {}'.format(categories[exp], categories[out])
        print(title)
        # plt.title(title)
        i += 1
        # plt.savefig(os.path.join(TMP_PATH, '{} ({}).png'.format(i, title)))