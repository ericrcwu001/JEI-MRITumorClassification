from train import generators
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix


def load_model():
    json_file = open(r"models/model.json")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(r"models/model.keras")
    return model


def conf_matrix(y_pred, test, name):
    class_labels = ["Glioma", "Meningioma", "No\nTumor", "Pituitary"]
    true_classes = test.classes

    # Prints a confusion matrix of the feature fusion model
    cm = confusion_matrix(true_classes, y_pred)
    # print(cm)
    cmdf = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    df_percentages = cmdf.div(cmdf.sum(), axis=0)
    df_logs = np.log10(df_percentages.mul(100).add(1))

    sns.set(font_scale=1.2)
    ax = sns.heatmap(data=df_logs, annot=cmdf, cmap="Blues", fmt="d",
                     cbar_kws={'label': '\nlog10(prediction accuracy added by 1)'})

    ax.set_title('Confusion Matrix\n', size=16)
    ax.set_xlabel('\nPredicted Values', size=15)
    ax.set_ylabel('Actual Values\n', size=15)

    fig = ax.get_figure()
    fig.savefig("output/confusion_matrix_" + name + "2.png")
    plt.show()


def class_report(y_pred, test, name):
    class_labels = ["Glioma", "Meningioma", "No\nTumor", "Pituitary"]
    true_classes = test.classes

    # Prints a classification report of the feature fusion model
    report = classification_report(true_classes, y_pred, target_names=class_labels, digits=6, output_dict=True)
    df = pd.DataFrame(report)
    # print(df)
    df.to_csv("output/class_report_" + name + "2.csv")


if __name__ == '__main__':
    model = load_model()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=0.0001, clipvalue=0.5),
        loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])

    # train, val, test = generators("processed")
    # model.evaluate(test)

    # processed
    train, val, test = generators("processed")
    Y_pred = model.predict(test)
    # print(type(Y_pred))
    # print(Y_pred)

    y_pred = np.argmax(Y_pred, axis=1)
    # print(type(y_pred))
    # print(y_pred)

    conf_matrix(y_pred, test, "aug")
    class_report(y_pred, test, "aug")

    # regular
    train, val, test = generators("dataset")
    Y_pred = model.predict(test)
    y_pred = np.argmax(Y_pred, axis=1)
    conf_matrix(y_pred, test, "reg")
    class_report(y_pred, test, "reg")
