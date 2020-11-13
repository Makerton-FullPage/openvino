#!/usr/bin/env python3

from argparse import ArgumentParser

from responsive_voice.voices import UKEnglishMale

from inference import FaceDetection, MaskDetection
from pyvino_utils import InputFeeder
import cv2



def main():
    
    # Initialise the video stream
    input_feed = InputFeeder(input_feed='cam')
    # Initialise the speech output
    green=(0,255,0)
    face_detection = FaceDetection(
        model_name='./models/face-detection-adas-0001',
        device='CPU',
        threshold=0.60,
        input_feed=input_feed,
    )
    mask_detection = MaskDetection(
        model_name='./models/face_mask',
        device='CPU',
        threshold=0.30,
    )

    
    mask_detected_prob = -1
    try:
        # TODO: Convert to contextmanager
        cam=cv2.VideoCapture(0)
        while True:
            # count += 1
            _,frame=cam.read()
            fd_results = face_detection.predict(
                frame, show_bbox=False, mask_detected=mask_detected_prob
            )

            face_bboxes = fd_results["process_output"]["bbox_coord"]

            if face_bboxes:
                for face_bbox in face_bboxes:
                    
                    (x, y, w, h) = face_bbox
                    face = frame[y:h, x:w]
                    (face_height, face_width) = face.shape[:2]
                    if face_height < 20 or face_width < 20:
                        print('small')
                        continue
                    
                    face=cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
                    md_results = mask_detection.predict(
                        face, frame=frame
                    )
                    mask_detected_prob = md_results["process_output"][
                        "flattened_predictions"
                    ]
                    color,text= ((0,255,0),'wearing mask') if mask_detected_prob[0]>0.3 else ((0,0,255),"Don't wearing mask")
                    
                    frame=cv2.rectangle(frame,(x,y),(w,h),color,3)
                    cv2.putText(frame, text, (x,y-10), cv2.FONT_ITALIC , 1, color, 2)
                    print(mask_detected_prob[0])
                    



            cv2.imshow('cam',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
        input_feed.close()
    
        


if __name__ == "__main__":
    # Grab command line args
    main()
