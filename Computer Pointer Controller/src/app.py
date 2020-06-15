'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
''' 
import cv2
import os
import numpy as np
import logging as log
import time
import math
import argparse
import matplotlib.pyplot as plt
import logging
import sys 

from input_feeder import InputFeeder
from mouse_controller import MouseController 
from face_detection import Facedetectionmodel
from facial_landmarks_detection import Faciallandmarksmodel
from head_pose_estimation import Headposeestimationmodel
from gaze_estimantion import Gazeestimationmodel

def get_args():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-fdm", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection model xml file with a trained model")
    parser.add_argument("-flm", "--facial_landmarks_model", required=True, type=str,
                        help="Path to a facial landmarks detection model xml file with a trained model")
    parser.add_argument("-hpm", "--head_pose_model", required=True, type=str,
                        help="Path to a head pose estimation model xml file with a trained model")
    parser.add_argument("-gm", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model xml file with a trained model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
    parser.add_argument("-extension", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="(CPU)-targeted custom layers. and locates in opnevino app")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="CPU, GPU, FPGA or VPU (NCS2 or MYRIAD) is acceptable")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.4,
                        help="Probability threshold for detections in the frame"
                             "(0.5 by default)")
    parser.add_argument("-flags", "--visualization", required=False, nargs='+', default=[],
                        help="flags (fdm, flm, hpm, gm)"
                        "You can see with these flags different outputs")


    return parser

def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                   [0, 1, 0],
                   [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    # R = np.dot(Rz, Ry, Rx)
    # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    # R = np.dot(Rz, np.dot(Ry, Rx))
    R = Rz @ Ry @ Rx
    # print(R)
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame

def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix


def mouse_move_coord(x, y, z):
    costheta = math.cos(z * math.pi / 180)
    sintheta = math.sin(z * math.pi / 180)

    # calculating angles of vectors
    mouse_x = x * costheta + y * sintheta
    mouse_y = y * costheta + x * sintheta
     
    return (mouse_x, mouse_y)

def main():
    args = get_args().parse_args()
    flags = args.visualization
    input_file_path = args.input

    if input_file_path == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path): 
            logging.error("check your input file, it is not valid")
            exit(1)
        input_feeder = InputFeeder("video", input_file_path)
    
    # assigning parameters for constructor function
    # Face Detection Model
    face_detection = Facedetectionmodel(model_name=args.face_detection_model, device=args.device, threshold=args.prob_threshold, 
        extensions=args.cpu_extension)
    # Facial Landmarks Model
    facial_landmarks = Faciallandmarksmodel(model_name=args.facial_landmarks_model, device=args.device,
        extensions=args.cpu_extension)
    # Head Pose Estimation Model
    head_pose_estimation = Headposeestimationmodel(model_name=args.head_pose_model, device=args.device,
        extensions=args.cpu_extension)
    # Gaze Estimation Model
    gaze_estimantion = Gazeestimationmodel(model_name=args.gaze_estimation_model, device=args.device,
        extensions=args.cpu_extension)
    # face detection model load time
    face_detection_time = time.time()
    face_detection.load_model()
    face_detection_end = time.time() - face_detection_time
    logging.error("Face Detection model load time: {:.3f} ms".format(face_detection_end * 1000))
    # facial landmarks model load time
    facial_landmarks_time = time.time()
    facial_landmarks.load_model()
    facial_landmarks_end = time.time() - facial_landmarks_time
    logging.error("Facial Landmarks model load time: {:.3f} ms".format(facial_landmarks_end * 1000))
    # head pose estimation model load time
    head_pose_estimation_time = time.time()
    head_pose_estimation.load_model()
    head_pose_estimation_end = time.time() - head_pose_estimation_time
    logging.error("Head Pose Estimation model load time: {:.3f} ms".format(head_pose_estimation_end * 1000))
    # gaze estimation model load time
    gaze_estimantion_time = time.time()
    gaze_estimantion.load_model()
    gaze_estimantion_end = time.time() - gaze_estimantion_time
    logging.error("Gaze Estimation model load time: {:.3f} ms".format(gaze_estimantion_end * 1000))
    # determining precsion and speed for mouse controller 
    mouse_controller = MouseController('low', 'fast')

    logging.error("All models have been successfully loaded")
    # load inputs image, video and cam 
    input_feeder.load_data()
    counter=0 
    inference_time_start = time.time()

    for flag, frame in input_feeder.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        counter+=1

        # Face detection
        face_coordinates, fd_outputs = face_detection.predict(frame) 
        xmin, ymin, xmax, ymax = face_coordinates[0], face_coordinates[1], face_coordinates[2], face_coordinates[3]
        # croped image
        crop = frame[ymin:ymax, xmin:xmax]
        # facial landmark detection
        eyes_coordinates = facial_landmarks.predict(crop) 
        # croped eyes and adding 15 for callculating coordinates
        left_eye = crop[eyes_coordinates[0][0]-15:eyes_coordinates[0][0]+15, 
            eyes_coordinates[0][1]-15:eyes_coordinates[0][1]+15]
        right_eye = crop[eyes_coordinates[1][0]-15:eyes_coordinates[1][0]+15, 
            eyes_coordinates[1][1]-15:eyes_coordinates[1][1]+15]
        # head pose estimation
        angles = head_pose_estimation.predict(crop)
        # gaze estimation 
        gaze_vector = gaze_estimantion.predict(left_eye, right_eye, angles) 
        # find gaze angle
        mouse_move_coords = mouse_move_coord(gaze_vector[0], gaze_vector[1], gaze_vector[2])
 
        # dwaring bounding box ,axes and putting message and visualization
        if len(flags) != 0:
            temp_frame = frame.copy()
            if 'fdm' in flags: 
                cv2.rectangle(frame, (xmin, ymin), 
                    (xmax, ymax), (0, 255, 255), 1)  
            if 'flm' in flags: 
                cv2.rectangle(crop, (eyes_coordinates[0][0]-15, eyes_coordinates[0][1]-15),
                    (eyes_coordinates[0][0]+15, eyes_coordinates[0][1]+15), (0, 255, 255), 2)
                
                cv2.rectangle(crop, (eyes_coordinates[1][0]-15, eyes_coordinates[1][1]-15),
                    (eyes_coordinates[1][0]+15, eyes_coordinates[1][1]+15), (0, 255, 255), 2)
            if 'hpm' in flags:
                hdp = "Head angles: {:.2f}, {:.2f}, {:.2f}".format(angles[0], angles[1], angles[2])  
                ges = "x = {:.1f}, y = {:.1f}, z = {:.1f}".format(gaze_vector[0], gaze_vector[1], gaze_vector[2])
                cv2.putText(frame, hdp, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, ges, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

            
                x, y, z = mouse_move_coords[0], mouse_move_coords[1], gaze_vector[2] 
                
                # showing head movement but it does not work properly
                if x > 0 and y > 0:
                    cv2.putText(frame, "up right", (15, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                elif x > 0 and y < 0:
                    cv2.putText(frame, "up left", (15, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                elif x < 0 and y > 0 and z < 0:
                    cv2.putText(frame, "down right", (15, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                elif x < 0 and y < 0 and z < 0:
                    cv2.putText(frame, "down left", (15, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2) 
                elif x == 0 and y == 0:
                    continue
            if 'gm' in flags: 
                # vector
                yaw = gaze_vector[0]
                pitch = gaze_vector[1]
                roll = gaze_vector[2]

                focal_length = 950.0
                scale = 50
        
                center_of_face = (xmin + crop.shape[1] / 2, ymin + crop.shape[0] / 2, 0)
                draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length)
        
        frame = cv2.resize(frame, (740, 640))                    
        cv2.imshow('frame', frame) 


        # mouse move with gaze vector (x, y)
        mouse_controller.move(mouse_move_coords[0], mouse_move_coords[1])

    total_time = time.time() - inference_time_start
    total_inference_time=total_time
    fps=counter/total_inference_time
    logging.error("Inference time: {:.3f}".format(total_inference_time))
    logging.error("FPS: {}".format(fps))

    # writing all results to txt
    total_model_load_time = face_detection_end + facial_landmarks_end + head_pose_estimation_end + gaze_estimantion_end
    # getting directory name
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, 'stats.txt'), 'w') as f:
            f.write("Inference Time: {:.3f} ms".format(total_inference_time) + '\n')
            f.write("FPS: {:.1f}".format(fps) + '\n')
            f.write("All model loading times" + '\n')
            f.write("Face Detection model load time: {:.3f} ms".format(face_detection_end * 1000) + '\n')
            f.write("Facial Landmarks model load time: {:.3f} ms".format(facial_landmarks_end * 1000) + '\n')
            f.write("Head Pose Estimation model load time: {:.3f} ms".format(head_pose_estimation_end * 1000) + '\n')
            f.write("Gaze Estimation model load time: {:.3f} ms".format(gaze_estimantion_end * 1000) + '\n')
            f.write("Total: {:.3f} ms".format(total_model_load_time * 1000) + '\n')
    
    # drawing bar graph for results
    """ model_list = ['face detection', 'facial landmark', 'head pose', 'gaze', 'Total']
    times = [face_detection_end, facial_landmarks_end, head_pose_estimation_end, gaze_estimantion_end, 
        total_model_load_time]

    plt.bar(model_list, times)
    plt.xlabel('Models')
    plt.ylabel(' Loading Times')
    plt.show() """

    # Release the out writer, capture, and destroy any OpenCV windows
    input_feeder.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
