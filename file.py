import cv2
import imutils
import numpy as np
import ChangeClothes as cc
import random
import mediapipe as mp


def virtual():
    cap = cv2.VideoCapture(0)
    images = cc.loadImages()
    thres = [130, 40, 75, 130]
    size = 180
    curClothId = 1
    th = thres[0]

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while True:
        (ret, cam) = cap.read()
        cam = cv2.flip(cam, 1, 0)
        t_shirt = images[curClothId]
        resized = imutils.resize(cam, width=800)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Process the image and find the pose
        results = pose.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

        # Get the body landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Get coordinates for the shoulders and hips
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Calculate the center position for the t-shirt
            center_x = int((left_shoulder.x + right_shoulder.x) / 2 * cam.shape[1])
            center_y = int((left_hip.y + left_shoulder.y) / 2 * cam.shape[0])

            # Calculate size based on distance between shoulders
            shoulder_width = int(abs(left_shoulder.x - right_shoulder.x) * cam.shape[1])
            size = int(shoulder_width * 1.5)  # Scale the size appropriately

            # Limit size to a maximum and minimum
            size = max(100, min(size, 350))

            # Resize the t-shirt
            t_shirt = imutils.resize(t_shirt, width=size)

            f_height = cam.shape[0]
            f_width = cam.shape[1]
            t_height = t_shirt.shape[0]
            t_width = t_shirt.shape[1]

            # Calculate position to place the t-shirt
            height = int(center_y - t_height / 2)
            width = int(center_x - t_width / 2)

            # Ensure the region of interest (roi) is within the frame boundaries
            if height < 0:
                height = 0
            if width < 0:
                width = 0
            if height + t_height > f_height:
                height = f_height - t_height
            if width + t_width > f_width:
                width = f_width - t_width

            # Create mask for the t-shirt
            t_shirt_gray = cv2.cvtColor(t_shirt, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(t_shirt_gray, th, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)

            # Define the roi based on the calculated height and width
            roi = cam[height:height + t_height, width:width + t_width]

            # Perform bitwise operations
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img_fg = cv2.bitwise_and(t_shirt, t_shirt, mask=mask)

            # Overlay the t-shirt on the camera feed
            t_shirt_combined = cv2.add(img_bg, img_fg)
            cam[height:height + t_height, width:width + t_width] = t_shirt_combined

        # Display instructions
        font = cv2.FONT_HERSHEY_PLAIN
        x = 10
        y = 20
        cv2.putText(cam, "press 'n ' for next item, 'p' for previous, 'c' for snapshot", (x, y), font, .8,
                    (255, 255, 255), 1)

        cv2.namedWindow("Virtual Dressing Room", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Virtual Dressing Room", int(cam.shape[1] * 1.4), int(cam.shape[0] * 1.4))
        cv2.imshow('Virtual Dressing Room', cam)

        key = cv2.waitKey(10)
        if key & 0xFF == ord('n'):
            if curClothId < len(images) - 1:
                curClothId += 1
                th = thres[curClothId]
        if key & 0xFF == ord('c'):
            rand = random.randint(1, 999999)
            cv2.imwrite('output/' + str(rand) + '.png', cam)
        if key & 0xFF == ord('p'):
            if curClothId > 0:
                curClothId -= 1
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


virtual()
