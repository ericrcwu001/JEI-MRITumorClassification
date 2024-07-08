import os
import shutil

import cv2
import imutils
import numpy as np

global_dir = 'dataset'
global_resize_dir = 'processed'
train_dir = global_dir + '/Training'
test_dir = global_dir + '/Testing'
train_resize_dir = global_resize_dir + '/Training'
test_resize_dir = global_resize_dir + '/Testing'
boundary_size = (256, 256)  # the size of the desired images
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']


def check_directories():
    if not os.path.exists(global_resize_dir):
        os.mkdir(global_resize_dir)
    if not os.path.exists(train_resize_dir):
        os.mkdir(train_resize_dir)
    if not os.path.exists(test_resize_dir):
        os.mkdir(test_resize_dir)

    for cate_ele in categories:
        if not os.path.exists(train_resize_dir + '/' + cate_ele):
            os.mkdir(train_resize_dir + '/' + cate_ele)
        if not os.path.exists(test_resize_dir + '/' + cate_ele):
            os.mkdir(test_resize_dir + '/' + cate_ele)


# https://github.com/masoudnick/Brain-Tumor-MRI-Classification/blob/main/Preprocessing.py
def crop_img(img, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = add_pixels_value
    new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
              extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
    return new_img


def validate_size(target_path):
    for i in categories:
        resized_path = target_path + '/' + i
        if not os.path.exists(resized_path):
            continue
        for img in os.listdir(resized_path):

            img_array = cv2.imread(resized_path + '/' + img)

            # if img_array.shape[0] != boundary_size[0] or img_array.shape[1] != boundary_size[1]:
            #     print(img_array.shape)


def norm(target_path):
    for i in categories:
        resized_path = target_path + '/' + i
        for img in os.listdir(resized_path):
            img_array = cv2.imread(resized_path + '/' + img)
            norm = np.zeros((256,256))
            final = cv2.normalize(img_array, norm, 0, 255, cv2.NORM_MINMAX)
            os.remove(resized_path + "/" + img)
            cv2.imwrite(resized_path + "/" + img, final)


def sp_noise(image, prob):
    output = image.copy()
    if image.shape[2] == 1:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


def sp_noise_all(target_path):
    for i in categories:
        resized_path = target_path + '/' + i
        for img in os.listdir(resized_path):
            img_array = cv2.imread(resized_path + '/' + img)
            temp = sp_noise(img_array, 0.003)
            os.remove(resized_path + "/" + img)
            cv2.imwrite(resized_path + "/" + img, temp)


def grayscale(target_path):
    for i in categories:
        resized_path = target_path + '/' + i
        for img in os.listdir(resized_path):
            img_array = cv2.imread(resized_path + '/' + img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            os.remove(resized_path + "/" + img)
            cv2.imwrite(resized_path + "/" + img, gray)


def resize_and_flip_img(source_path, target_path):
    directions = ['0', '90', '180', '270']
    for i in categories:
        count = 0
        path = source_path + '/' + i
        if not os.path.exists(path):
            continue
        resized_path = target_path + '/' + i
        for img in os.listdir(path):
            img_array = cv2.imread(path + '/' + img)

            if img_array.shape[0] < boundary_size[0] or img_array.shape[1] < boundary_size[1]:
                continue

            count += 1
            if i != "meningioma" and count % 2 == 0:
                continue
            else:
                img_array = crop_img(img_array)

                img_array = cv2.resize(img_array, boundary_size, interpolation=cv2.INTER_CUBIC)

                for i in directions:
                    cv2.imwrite(resized_path + '/' + img[:-4] + '_' + str(i) + '_degree.jpg', img_array)
                    # Use Flip code 0 to flip vertically
                    img_array = cv2.flip(img_array, 0)
                    cv2.imwrite(resized_path + '/' + img[:-4] + '_' + str(i) + '_degree_fliped.jpg', img_array)
                    img_array = cv2.flip(img_array, 0)
                    img_array = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)

                    count += 2

    # print(count)


if __name__ == '__main__':
    shutil.rmtree(global_resize_dir)
    check_directories()
    resize_and_flip_img(train_dir, train_resize_dir)
    resize_and_flip_img(test_dir, test_resize_dir)
    validate_size(train_resize_dir)
    validate_size(test_resize_dir)
    grayscale(train_resize_dir)
    grayscale(test_resize_dir)
    sp_noise_all(train_resize_dir)
    sp_noise_all(test_resize_dir)
