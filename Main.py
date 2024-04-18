from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU') #convert GPU model to CPU
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes #load YOLOV4 package to filter boxes which contains vehicles
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker #deep sort tracker model to predict or track vehciles from YOLO processed video frame
from tools import generate_detections as gdet
from tqdm import tqdm
from collections import deque


pts = [deque(maxlen=30) for _ in range(9999)]

main = tkinter.Tk()
main.title("Road Traffic Analysis using YOLO-V4 & Deep Sort")
main.geometry("1300x1200")

global filename
global model, encoder, tracker, config
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
global accuracy, precision

def loadModel():
    global model, encoder, tracker, config
        
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
        
    pathlabel.config(text="YOLOv4 DeepSort Model Loaded")
    text.delete('1.0', END)
    text.insert(END,"YOLOv4 DeepSort Model Loaded\n\n");

def vehicleDetection():
    global model, encoder, tracker, config
    global accuracy, precision
    accuracy = 0
    precision = 0
    filename = filedialog.askopenfilename(initialdir="Videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    text.update_idletasks()
    saved_model_loaded = tf.saved_model.load('yolo/yolov4-416', tags=[tag_constants.SERVING])
    model = saved_model_loaded.signatures['serving_default']
    cap = cv2.VideoCapture(filename)
    start_time = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (416, 416))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            batch_data = tf.constant(image_data)
            pred_bbox = model(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                                                                                             scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                                                                                             max_output_size_per_class=50, max_total_size=50,
                                                                                             iou_threshold=0.45, score_threshold=0.50)
            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]
            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)
            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]
            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)
            # by default allow all classes in .names file
            #allowed_classes = list(class_names.values())        
            # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['car','truck']
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            cv2.putText(frame, "tracked: {}".format(count), (5, 70), 0, 5e-3 * 200, (0, 255, 0), 2)
            
        
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # Call the tracker
            print(scores)
            if accuracy == 0:
                accuracy = scores[0]
                text.insert(END,"YoloV4 DeepSort Accuracy  : "+str(scores[0])+"\n\n")
                text.insert(END,"YoloV4 DeepSort Precision : "+str(scores[1])+"\n\n")
                text.update_idletasks()
            tracker.predict()
            tracker.update(detections)
            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                # Tracking with historical trajectory 
                center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
                pts[track.track_id].append(center)
                thickness = 5
                # center point
                cv2.circle(frame,  (center), 1, color, thickness)
            	# draw motion path
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
            fps = 1.0 / (time.time() - start_time)
            #print("FPS: %.2f" % fps)
            frame = cv2.resize(frame,(800,800))
            cv2.putText(frame, "FPS: %f" %(fps), (5,150), 0, 5e-3 * 200, (0,255,0), 2)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Output Video", result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()                


def close():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Road Traffic Analysis using YOLO-V4 & Deep Sort',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Generate & Load YOLOv4-DeepSort Model", command=loadModel)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

markovButton = Button(main, text="Run Traffic Analysis", command=vehicleDetection)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=50,y=250)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
