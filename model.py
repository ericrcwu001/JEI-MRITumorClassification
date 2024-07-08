import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.applications import VGG19, Xception, xception, vgg19

num_classes = 4  # Number of Classes


def feature_fusion():
    # Instantiating the Xception and VGG-19 Models
    xcep = Xception(weights='imagenet', include_top=False, pooling=max)
    vgg = VGG19(weights='imagenet', include_top=False, pooling=max)

    # Creating the Input List
    input = Input(shape=(256, 256, 3))

    # Xception and VGG-19 pre-processing
    xcepTemp = tf.cast(input, tf.float32)
    xcepTemp = xception.preprocess_input(xcepTemp)

    vggTemp = tf.cast(input, tf.float32)
    vggTemp = vgg19.preprocess_input(vggTemp)

    # Instantiating the Xception and VGG-19 Models
    xcep = Xception(weights='imagenet', include_top=False, pooling=max, input_tensor=xcepTemp,
                    input_shape=(224, 224, 3))
    vgg = VGG19(weights='imagenet', include_top=False, pooling=max, input_tensor=vggTemp, input_shape=(299, 299, 3))
    for i in range(len(vgg.layers)):
        vgg.layers[i]._name += "V"

    xcep_last = xcep.get_layer("block14_sepconv2_act")
    vgg_last = vgg.get_layer("block5_poolV")
    # Creating the Model
    xcepFlatten = Flatten(name="xcep_flatten")(xcep_last.output)
    vggFlatten = Flatten(name="vgg_flatten")(vgg_last.output)

    fusionFeatures = Concatenate(axis=1)([xcepFlatten, vggFlatten])
    fusion = Dense(1024, activation='relu', name='fusion_fc1')(fusionFeatures)
    fusion = BatchNormalization()(fusion)
    fusion = Dropout(0.5)(fusion)
    fusion = Dense(1024, activation='relu', name='fusion_fc2')(fusion)
    fusion = BatchNormalization()(fusion)
    fusion = Dropout(0.5)(fusion)
    fusion_pred = Dense(num_classes, activation='softmax', name='fusion_predictions')(fusion)

    model = Model(input, fusion_pred)

    return model
