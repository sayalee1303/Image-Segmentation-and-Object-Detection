# fastapi packages
import uvicorn
from fastapi import FastAPI, File, Form
from starlette.responses import Response
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import gc
import detectron2
from detectron2.detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.detectron2 import model_zoo
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.utils.visualizer import Visualizer
from detectron2.detectron2.data import MetadataCatalog
import logging
import numpy as np
import cv2
from datetime import date, datetime
import pathlib
import os
import json
import io
from PIL import Image
import base64

def log():
  # Log files
  # creating path for directory inside log
  parent_path = str(pathlib.Path(__file__).absolute().parent)
  full_path = os.path.join(parent_path,'logs',str(date.today()))
  os.makedirs(full_path,exist_ok=True)

  # getting current time for each log file
  # current date and time
  now = datetime.now()
  t = str(now.strftime("%H:%M:%S"))
  t = t.replace(':','-')
  filename = 'object detection' + r'-' + str(t) + '.log'
  file_path =  os.path.join( full_path, filename)      	    

  # create log file
  logging.basicConfig(
       filename=file_path,
       level=logging.INFO, 
       format= '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
       datefmt='%H:%M:%S'
   )
    
def make_json_compatible(result):
    if "pred_boxes" in result.keys():
        result["pred_boxes"] = result["pred_boxes"].tensor.tolist()
    if "pred_classes" in result.keys():        
        result["pred_classes"] = result["pred_classes"].tolist()
    if "pred_keypoint_heatmaps" in result.keys():
        result["pred_keypoint_heatmaps"] = result["pred_keypoint_heatmaps"].tolist()
    if "pred_keypoints" in result.keys():    
        result["pred_keypoints"] = result["pred_keypoints"].tolist()
    if "scores" in result.keys():    
        result["scores"] = result["scores"].tolist()
    return result


log()
app = FastAPI()
labels = ['???', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '???', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '???', 'backpack', 'umbrella', '???', '???', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', '???', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', '???', 'dining table', '???', '???', 'toilet', '???', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', '???', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
try:
    # Inference with a keypoint detection model
    logging.info('Inference with a keypoint detection model')
    cfg2 = get_cfg()
    cfg2.MODEL.DEVICE = "cpu"
    cfg2.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor2 = DefaultPredictor(cfg2)
except Exception as e:
    print('In keypoint detection model : ',e)
    logging.info('In keypoint detection model : '+str(e))
    
try:    
    # Inference with a panoptic segmentation model
    cfg3 = get_cfg()
    cfg3.MODEL.DEVICE = "cpu"
    cfg3.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg3.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor3 = DefaultPredictor(cfg3)
except Exception as e:
    print('In panoptic segmentation model : ',e)
    logging.info('In panoptic segmentation model : '+str(e))



@app.post("/object_keypoint_detection/{option}")
def object_keypoint_detection(option:str, file: bytes = File(...)):
    try:
        print("option : ",option)
        logging.info('In object_keypoint_detection')
        image = np.array(Image.open(io.BytesIO(file)))
        logging.info("Inference on input image")
        outputs2 = predictor2(image)
        result = outputs2["instances"].to("cpu")
        result_dict = result.get_fields()
        result_json = make_json_compatible(result_dict)
        num_boxes = len(result_json["pred_boxes"])
        label_category = [labels[int(result_json["pred_classes"][r])+1] for r in range(0, int(num_boxes))]
        if option == "image": 
            
            logging.info("visualising output")
            v2 = Visualizer(image[:,:,::-1], MetadataCatalog.get(cfg2.DATASETS.TRAIN[0]), scale=1.2)
            out2 = v2.draw_instance_predictions(result)
            output_image = out2.get_image()
            logging.info("converting output image to bytes")
            is_success, buffer = cv2.imencode(".jpg", output_image)
            bytes_io = io.BytesIO(buffer)
            output_image_bytes = bytes_io.getvalue()
            logging.info("converting byte image to base64 data")
            output_image_bytes_bs64 = base64.b64encode(output_image_bytes).decode('ascii')
            logging.info("creating response json")
            output_json = {"result" : "success","status" : "success", "image" : str(output_image_bytes_bs64), "text" : label_category }
            output_json_str = json.dumps(output_json)
            logging.info("convert response json to bytes")
            output_json_bytes = output_json_str.encode()
            return Response(output_json_bytes, media_type="application/json")
            
        elif option == "points":
            result_json["label_category"] = label_category
            output_json = {"result" : "success","status" : "success", "data" : result_json}
            output_json_str = json.dumps(output_json)
            logging.info("convert response json to bytes")
            output_json_bytes = output_json_str.encode()
            return Response(output_json_bytes, media_type="application/json")            

    except Exception as e:
        print('In object_keypoint_detection : ',e)
        logging.info('In object_keypoint_detection : '+str(e))         
        output_json = {"result" : "failed","status" : "failed", "error" : str(e), "text":"" }
        output_json_str = json.dumps(output_json)
        # convert to bytes
        output_json_bytes = output_json_str.encode()
        return Response(output_json_bytes, media_type="application/json")


@app.post("/object_segment_detection/{option}")
def object_segment_detection(option:str, file: bytes = File(...)):
    try:
        logging.info('In object_segment_detection')
        image = np.array(Image.open(io.BytesIO(file)))
        logging.info("Inference on input image")
        panoptic_seg, segments_info = predictor3(image)["panoptic_seg"]
        panoptic_seg = panoptic_seg.to("cpu")
        
        if option == "image": 
            logging.info("visualising output")
            v3 = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg3.DATASETS.TRAIN[0]), scale=1.2)
            out3 = v3.draw_panoptic_seg_predictions(panoptic_seg, segments_info)
            #output_image = out3.get_image()[:, :, ::-1]
            output_image = out3.get_image()
            logging.info("converting output image to bytes")
            is_success, buffer = cv2.imencode(".jpg", output_image)
            bytes_io = io.BytesIO(buffer)
            output_image_bytes = bytes_io.getvalue()
            logging.info("converting byte image to base64 data")
            output_image_bytes_bs64 = base64.b64encode(output_image_bytes).decode('ascii')
            logging.info("creating response json")
            output_json = {"result" : "success","status" : "success", "image" : str(output_image_bytes_bs64), "text":"" }
            output_json_str = json.dumps(output_json)
            logging.info("convert response json to bytes")
            output_json_bytes = output_json_str.encode()
            return Response(output_json_bytes, media_type="application/json")
        elif option == "points":
            panoptic_seg = panoptic_seg.tolist()
            result_json = {"segmentation" : panoptic_seg, "segments_info" : segments_info}
            output_json = {"result" : "success", "status" : "success", "data" : result_json}
            output_json_str = json.dumps(output_json)
            logging.info("convert response json to bytes")
            output_json_bytes = output_json_str.encode()
            return Response(output_json_bytes, media_type="application/json")  
        
    except Exception as e:
        print('In object_segment_detection : ',e)
        logging.info('In object_segment_detection : '+str(e))         
        output_json = {"result" : "failed", "status" : "failed", "error" : str(e), "text":""   }
        output_json_str = json.dumps(output_json)
        # convert to bytes
        output_json_bytes = output_json_str.encode()
        return Response(output_json_bytes, media_type="application/json")


if __name__ == '__main__':
    uvicorn.run(app,host = '0.0.0.0',port=20000)
