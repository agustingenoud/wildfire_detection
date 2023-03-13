import os

import pandas as pd
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import sklearn.metrics

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def load_csv(path):
    df = pd.read_csv(path)
    df_rows = df[["filename", "xmin", "ymin", "xmax", "ymax"]].values.tolist()

    return df_rows


def format_images(images_list, images_path):
    """Format images. accepts the image list and the images_path"""
    # initialize the list of data (images),
    # our target output predictions
    # (bounding box coordinates), along with
    # the filenames of the individual images
    data = []
    targets = []
    filenames = []

    # loop over the rows
    for row in images_list:
        # break the row into the filename and bounding box coordinates
        (filename, startX, startY, endX, endY) = row
        imagePath = os.path.sep.join([images_path, filename])
        image = cv2.imread(imagePath)
        (h, w) = image.shape[:2]

        # SCALE the bounding box coordinates relative to the spatial
        # dimensions of the input image
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        # LOAD the image and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        data.append(image)
        targets.append((startX, startY, endX, endY))
        filenames.append(filename)

    # convert the data and targets to NumPy arrays, scaling the input
    # pixel intensities from the range [0, 255] to [0, 1]
    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")

    return (data, targets, filenames)


def get_predictions(model, image_path_list):
    image_predictions_list = []
    for image_path in image_path_list:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        # make bounding box predictions on the input image
        preds = model.predict(image)[0]
        (startX, startY, endX, endY) = preds
        to_list = [image_path, startX, startY, endX, endY]
        image_predictions_list.append(to_list)

    return image_predictions_list


def draw_predicted_bboxes(predictions_list):
    """
    From a list of predictions loads the image and draw the
    respcetive bounding boxes. returns a list pf the images with
    the bboxes
    """
    image_list = []

    for prediction in predictions_list:
        # Assign variables
        (image_path, startX, startY, endX, endY) = prediction

        image = cv2.imread(image_path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        # scale the predicted bounding box coordinates based on the image
        # dimensions
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)
        # draw the predicted bounding box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(img)

    return image_list


def IOU_metric(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)

    if w_intersection <= 0 or h_intersection <= 0:
        return 0

    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I
    IOU = I / U

    return IOU


def compute_precision_recall(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []

    # loop over each threshold from 0.2 to 0.65
    for threshold in thresholds:
        # yPred is smoke if prediction score greater than threshold
        # else no_smoke if prediction score less than threshold
        yPred = ["smoke" if score >= threshold else "no_smoke" for score in pred_scores]

        # compute precision and recall for each threshold
        precision = sklearn.metrics.precision_score(
            y_true=y_true, y_pred=yPred, pos_label="smoke"
        )
        recall = sklearn.metrics.recall_score(
            y_true=y_true, y_pred=yPred, pos_label="smoke"
        )

        precisions.append(np.round(precision, 3))
        recalls.append(np.round(recall, 3))

    return precisions, recalls


def plot_pr_curve(precisions, recalls):
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()
