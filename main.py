import os
import glob
import time
import cv2
import numpy as np
import pandas as pd


def mask_bbox(bbox, black_mask, color):
    # fill in the color according to the range of the bbox
    cv2.rectangle(
        black_mask,
        (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']),
        color, -1
    )


def mask_all_bbox(bbox_list, height, width):
    # create a black background mask with the same dimensions as the original image
    black_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # fill in black according to the range of all bbox
    bbox_list[['x1', 'y1', 'x2', 'y2']].apply(lambda row: mask_bbox(
        row, black_mask, (255, 255, 255)), axis=1)

    return black_mask


def data_clean(img, bbox_list_input, classes, classes_flag, basename, output_path):
    height, width = img.shape[:2]

    # build bbox_list for grouping by labels and convert bbox to OpenCV format
    bbox_list = pd.DataFrame(
        bbox_list_input, columns=['label', 'x', 'y', 'w', 'h'])
    bbox_list['x1'] = bbox_list['x'] - bbox_list['w']/2   # x1
    bbox_list['y1'] = bbox_list['y'] - bbox_list['h']/2   # y1
    bbox_list['x2'] = bbox_list['x'] + bbox_list['w']/2   # x2
    bbox_list['y2'] = bbox_list['y'] + bbox_list['h']/2   # y2

    # rescale range to [0, 255] and make sure the type are integer
    bbox_list[['x1', 'y1', 'x2', 'y2']] = \
        bbox_list[['x1', 'y1', 'x2', 'y2']] * [width, height, width, height]
    bbox_list[['label', 'x1', 'y1', 'x2', 'y2']] = \
        bbox_list[['label', 'x1', 'y1', 'x2', 'y2']].astype('int32')

    # mask all bbox first
    # invert black and white to easy to handle downstream
    black_mask = mask_all_bbox(bbox_list, height, width)

    bbox_group = bbox_list.groupby(by=['label'])

    for group in bbox_group:
        inverted_mask = black_mask.copy()
        label = classes[group[0]]
        classes_flag[group[0][0]] = True  # mark the class as found

        # fill in black according to the range of all bbox
        group[1][['x1', 'y1', 'x2', 'y2']].apply(lambda row: mask_bbox(
            row, inverted_mask, (0, 0, 0)), axis=1)

        # invert black and white to keep white areas
        mask = cv2.bitwise_not(inverted_mask)

        # use mask to cover areas of other labels, but not the area of ​​this label.
        masked_image = cv2.bitwise_and(img, mask)

        # check all output path
        result_path = os.path.join(output_path, label)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(os.path.join(result_path, 'images')):
            os.makedirs(os.path.join(result_path, 'images'))
        if not os.path.exists(os.path.join(result_path, 'labels')):
            os.makedirs(os.path.join(result_path, 'labels'))

        # save masked image and label
        cv2.imwrite(os.path.join(result_path, 'images',
                    basename + '.jpg'), masked_image)
        np.savetxt(
            os.path.join(result_path, 'labels', basename + '.txt'),
            group[1][['label', 'x', 'y', 'w', 'h']].values,
            fmt=['%d', '%f', '%f', '%f', '%f']
        )

        # cv2.imshow('Masked Image', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('My Image', masked_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def make_classes_dir(classes, output_path):
    for label in classes:
        result_path = os.path.join(output_path, label)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(os.path.join(result_path, 'images')):
            os.makedirs(os.path.join(result_path, 'images'))
        if not os.path.exists(os.path.join(result_path, 'labels')):
            os.makedirs(os.path.join(result_path, 'labels'))


if __name__ == '__main__':

    start_time = time.time()

    images_path = './coco128LW/train/images/'
    labels_path = './coco128LW/train/labels/'
    classes_path = './coco128LW/train/classes.txt'
    output_path = './result/'

    # record whether the classes appear in the dataset
    classes_flag = [False]*80

    # load classes and make directories
    classes = np.loadtxt(os.path.join(classes_path), delimiter=',', dtype=str)
    make_classes_dir(classes[:80], output_path)

    image_list = glob.glob(os.path.join(images_path, "*.jpg"))
    for image_path in image_list:
        base_name = os.path.basename(image_path).split(".jpg")[0]
        img = cv2.imread(image_path)
        try:
            bbox_list = np.loadtxt(os.path.join(labels_path, base_name + ".txt")).reshape(-1, 5)
        except:
            print(image_path, "has no label... skip")
            continue

        data_clean(img, bbox_list, classes,
                   classes_flag, base_name, output_path)
    print("Done!")

    # get the classes those do not appear in the dataset
    not_found_classes = ""
    not_found_classes_count = 0
    for i in range(80):
        if not classes_flag[i]:
            not_found_classes_count += 1
            not_found_classes += classes[i] + ", "
    print(
        f"\nThere are {not_found_classes_count} classes those do not appear in the dataset:", not_found_classes)

    # Timing
    end_time = time.time()
    print(f"\nTotal time spent: {end_time - start_time:.4f} seconds.")
