import sys
import os

import cv2
import json

import retinex

data_path = 'img'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)

for img_name in img_list:
    if img_name == '.gitkeep':
        continue

    img = cv2.imread(os.path.join(data_path, img_name))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    shape = img.shape
    cv2.imshow('Image', img)

    roi = img[121:186, 497:722]

    print('msrcr processing......')
    img_msrcr_roi = retinex.MSRCR(
        roi,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )
    img_msrcr = img

    img_msrcr[121:186, 497:722] = img_msrcr_roi
    cv2.imshow('MSRCR retinex', img_msrcr)
    cv2.imwrite("MSRCR_retinex.tif", img_msrcr);

    print('amsrcr processing......')
    img_amsrcr_roi = retinex.automatedMSRCR(
        roi,
        config['sigma_list']
    )
    img_amsrcr = img
    img_amsrcr[121:186, 497:722] = img_amsrcr_roi
    cv2.imshow('autoMSRCR retinex', img_amsrcr)
    cv2.imwrite('AutomatedMSRCR_retinex.tif', img_amsrcr)

    print('msrcp processing......')
    img_msrcp_roi = retinex.MSRCP(
        roi,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )
    img_msrcp = img
   # img_msrcp_roi = cv2.GaussianBlur(img_msrcp_roi, (5, 5), 0)
    img_msrcp[121:186, 497:722] = img_msrcp_roi



    cv2.imshow('MSRCP', img_msrcp)
    cv2.imwrite('MSRCP.tif', img_msrcp)
    cv2.waitKey()
