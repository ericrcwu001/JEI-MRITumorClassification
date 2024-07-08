from model import feature_fusion
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


batch_size = 8  # Batch Size
epochs = 8  # Epoch number


def generators(directory):
    # Using the ImageDataGenerator to prepare the images
    generator_train = ImageDataGenerator(rescale=None,
                                         featurewise_center=False,
                                         samplewise_center=False,
                                         featurewise_std_normalization=False,
                                         samplewise_std_normalization=False,
                                         zca_whitening=False,
                                         rotation_range=0,
                                         zoom_range=0,
                                         width_shift_range=0,
                                         height_shift_range=0,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         validation_split=0.2)

    generator_test = ImageDataGenerator(rescale=None,
                                        featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        rotation_range=0,
                                        zoom_range=0,
                                        width_shift_range=0,
                                        height_shift_range=0,
                                        horizontal_flip=False,
                                        vertical_flip=False)

    # Creating the train and test data.
    train = generator_train.flow_from_directory(directory+'/Training', target_size=(256, 256),
                                                batch_size=batch_size, class_mode="categorical", color_mode='rgb',
                                                subset='training')

    val = generator_train.flow_from_directory(directory+'/Training', target_size=(256, 256),
                                              batch_size=batch_size, class_mode="categorical", color_mode='rgb',
                                              subset='validation')

    test = generator_test.flow_from_directory(directory+'/Testing', target_size=(256, 256),
                                              batch_size=batch_size, class_mode="categorical", color_mode='rgb', shuffle=False)

    return train, val, test


def save_model(model):
    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/model.keras")
    print("Saved model to disk")


def train(model, train_generator, val_generator):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=0.0001, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    # Stop training if loss doesn't keep decreasing.
    model1_es = EarlyStopping(monitor='loss', min_delta=1e-11, patience=6, verbose=1)
    model1_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

    history = model.fit(train_generator, steps_per_epoch=train.samples // batch_size, epochs=epochs,
                        validation_data=val_generator, validation_steps=test.samples // batch_size,
                        callbacks=[model1_es, model1_rlr])

    save_model(model)
    return history


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Deep Feature Fusion Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.show()

    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Deep Feature Fusion Model Categorical Accuracy')
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("categorical_accuracy.png")
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Deep Feature Fusion Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    plt.show()


if __name__ == '__main__':
    train, val, test = generators("processed")
    model = feature_fusion()
    history = train(model, train, val)
    plot_history(history)
