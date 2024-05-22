import cv2
import numpy as np
import math
from KalmanFilter import Tracker
from helper import get_iob, get_center, apply_model
import time

cam = cv2.VideoCapture('plane.mp4')
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
frame_size = (frame_width,frame_height)
#output = cv2.VideoWriter('out_plane.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

history_length = 150
bg_subtractor.setHistory(history_length)

erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))

objects = []

num_history_frames_populated = 0

boxes_prev_frame = []
tracking_objects = {}
track_id = 0
trackers = {}
predictions = {}
id_to_delete = []
count = 0
start = time.time()
while True:
    grabbed, frame = cam.read()
    if not grabbed:
        break
    fg_mask = bg_subtractor.apply(frame)
    fg_mask_c = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    if num_history_frames_populated < history_length:
        num_history_frames_populated += 1
        continue
    cv2.imshow("mask", fg_mask_c & frame)
    count += 1
    _, thresh = cv2.threshold(fg_mask, 90, 255, cv2.THRESH_BINARY)

    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes_cur_frame = []
    for c in contours:
        if cv2.contourArea(c) > 2500:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
            boxes_cur_frame.append((int(x), int(y), int(w), int(h)))

    boxes_cur_frame_copy = boxes_cur_frame.copy()
    md_applyed = False
    if count <= 2:
        for box in boxes_cur_frame:
            for box2 in boxes_prev_frame:
                IOB = get_iob(box, box2, False)
                if IOB > 0.7:
                    if md_applyed == False:
                        md_boxes = apply_model(frame)
                        mp_applyed = True
                    for box3 in md_boxes:
                        IOB = get_iob(box, box3[:4], True)
                        if IOB > 0.7:
                            tracking_objects[track_id] = box
                            trackers[track_id] = Tracker(track_id, get_center(box, False),  box3[4])
                            track_id += 1
                            break
        md_applyed = False

    else:
        tracking_objects_copy = tracking_objects.copy()
        for object_id, box2 in tracking_objects_copy.items():
            object_exists = False
            for box in boxes_cur_frame_copy:
                IOB = get_iob(box2, box, False)
                if IOB > 0.7:
                    tracking_objects[object_id] = box
                    object_exists = True
                    trackers[object_id].predict()
                    trackers[object_id].update(get_center(box, False))
                    y, S = trackers[object_id].kf.y, trackers[object_id].kf.S
                    eps = y.T @ np.linalg.inv(S) @ y
                    eps_max = 4.
                    Q_scale_factor = 1000.
                    if eps > eps_max:
                        trackers[object_id].kf.Q *= Q_scale_factor
                        trackers[object_id].scaled = True
                    elif trackers[object_id].scaled and eps < eps_max:
                        trackers[object_id].kf.Q /= Q_scale_factor
                        trackers[object_id].scaled = False
                    if box in boxes_cur_frame:
                        boxes_cur_frame.remove(box)
                    continue

            if not object_exists:
                tracking_objects.pop(object_id)
                trackers[object_id].object_detected = False

    if len(trackers) != 0:
        for id in trackers:
            if trackers[id].object_detected == False:
                trackers[id].predict()
                for box in boxes_cur_frame:
                    x_c, y_c = get_center(box, False)
                    dist = math.hypot(int(trackers[id].kf.x[0]) - x_c, int(trackers[id].kf.x[2]) - y_c)
                    if dist < 200:
                        tracking_objects[id] = box
                        object_exists = True
                        trackers[id].object_detected = True
                        trackers[id].without_obj = 0
                        trackers[id].update(get_center(box, False))
                        y, S = trackers[id].kf.y, trackers[id].kf.S
                        eps = y.T @ np.linalg.inv(S) @ y
                        eps_max = 4.
                        Q_scale_factor = 1000.
                        if eps > eps_max:
                            trackers[id].kf.Q *= Q_scale_factor
                            trackers[id].scaled = True
                        elif trackers[object_id].scaled and eps < eps_max:
                            trackers[id].kf.Q /= Q_scale_factor
                            trackers[id].scaled = False
                        if box in boxes_cur_frame:
                            boxes_cur_frame.remove(box)
                        continue
                cv2.circle(frame, (int(trackers[id].kf.x[0]), int(trackers[id].kf.x[2])), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{trackers[id].obj}", (int(trackers[id].kf.x[0]), int(trackers[id].kf.x[2]) - 7), 0, 1, (0, 0, 255), 2)
                if trackers[id].object_detected == False:
                    trackers[id].without_obj += 1
                if trackers[id].without_obj == 100:
                    id_to_delete.append(id)
    for id in id_to_delete:
        trackers.pop(id)
    id_to_delete = []
    
    if count > 2:       
        for box in boxes_cur_frame:
            if md_applyed == False:
                md_boxes = apply_model(frame)
                md_applyed = True
            for box3 in md_boxes:
                IOB = get_iob(box, box3[:4], True)
                if IOB > 0.7:
                    tracking_objects[track_id] = box
                    trackers[track_id] = Tracker(track_id, get_center(box, False), box3[4])
                    track_id += 1
                    break
        md_applyed = False

    for object_id, box in tracking_objects.items():
        if trackers[object_id].obj == 1:
            cv2.putText(frame, 'plane', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        if trackers[object_id].obj == 2:
            cv2.putText(frame, 'helicopter', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.circle(frame, (int(trackers[object_id].kf.x[0]), int(trackers[object_id].kf.x[2])), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{trackers[object_id].obj}", (int(trackers[object_id].kf.x[0]), int(trackers[object_id].kf.x[2]) - 7), 0, 1, (0, 0, 255), 2)

   
    boxes_prev_frame = boxes_cur_frame_copy
    #output.write(frame)
    cv2.imshow('Objects tracked', frame)

    k = cv2.waitKey(10)

    if k == 27:
        break

end = time.time()
exec_time = end - start
print(count/exec_time)