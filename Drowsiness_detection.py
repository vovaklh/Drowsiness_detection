from scipy.spatial import distance as dist
from imutils.video import FileVideoStream, VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import cv2
import time
import argparse
import dlib
import imutils
import playsound


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute ratio
    ear = (A + B) / (2 * C)

    return ear


def play_sound(path):
    playsound.playsound(path)


# construct parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", default="shape_predictor_68_face_landmarks.dat")
ap.add_argument("-s", "--siren", type=str, default='Resources/siren.mp3')
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam")
ap.add_argument("-t", "--thresh", type=float, default=0.3, help="eye aspect ratio threshold")
ap.add_argument("-f", "--frames", type=int, default=40, help="number of frames to indicate sleeping")
args = vars(ap.parse_args())

# create counter of blinks and flag to turn on siren
counter = 0
siren = False

# load face_detector and shape_predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['predictor'])

# constants to grab eye regions from face
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start webcam thread
vs = VideoStream(args['webcam']).start()
time.sleep(1)

# loop over frames
while True:
    # read frames, resize it and convert to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces on frame
    rects = detector(gray, 0)

    # loop over detected faces
    for rect in rects:
        # predict key-points on face and convert it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract left and right eyes and compute eye aspect ratio
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # average ear from both eyes
        ear = (left_ear + right_ear) / 2

        # compute hull contours and visualize it
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # check eye aspect ratio
        if ear < args['thresh']:
            counter += 1

            # check if eye are closed for a long time
            if counter >= args['frames']:
                # activate alarm
                if not siren:
                    siren = True
                    t = Thread(target=play_sound, args=(args['siren'],) )
                    t.daemon = True
                    t.start()

                # put the text
                cv2.putText(frame, "Drowsiness was detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            counter = 0
            siren = False
        # draw the ear
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# clean
cv2.destroyAllWindows()
vs.stop()