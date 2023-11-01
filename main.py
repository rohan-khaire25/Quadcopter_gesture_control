#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configargparse

import cv2 as cv

from utils import CvFpsCalc

from gestures import *

import threading


def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)
    parser.add("--buffer_len",
               help='Length of gesture buffer',
               type=int)

    args = parser.parse_args()

    return args


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def main():
    # init global vars
    global gesture_buffer
    global gesture_id

    # Argument parsing
    args = get_args()

    # Camera preparation
    # Use the webcam
    # cap = tello.get_frame_read()
    cap = cv.VideoCapture(0)
   
    # Gesture recognition initialization
    gesture_detector = GestureRecognition(False, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    while True:
        fps = cv_fps_calc.get()

        # Camera capture
        result, image = cap.read()
        
        debug_image, gesture_id = gesture_detector.recognize(image)
        gesture_buffer.add_gesture(gesture_id)

        debug_image = gesture_detector.draw_info(debug_image, fps)

        # Battery status and image rendering
        #cv.putText(debug_image, "Battery: {}".format(battery_status), (5, 720 - 5),
        #           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Gesture Recognition', debug_image)
        cv.waitKey(1)


if __name__ == '__main__':
    main()
