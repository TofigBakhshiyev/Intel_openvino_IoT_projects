

# People Counter App at the Edge

In this application, app counts the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. Furthermore, app send values to Mosca server namely, total counted people, current count and the duration of person. This project is a part of Intel Edge AI for IoT Developers Nanodegree program by Udacity. [project video](https://drive.google.com/file/d/1lvTWPZAgfjXR24qpM-lWrWGGNQF5Pxsg/view)

## Explaining Custom Layers  

Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom. These custom layers need to be specifically registered following the steps corresponding to each framework after which the Model Optimizer will take it into consideration while producing the Intermediate Representation. Intel openVINO already has extensions for custom layers. 

## Comparing Model Performance 
In this table, you can see differences in inference time, model size and accuracy and this example we see how openvino increases efficiency

| Pre-Trained Model (Framework)                             | Inference Time ms					| 	Size of Model MB|
| -----------------------------------         |:---------------------------------:| :-------:|
| ssd_mobilenet_v1_coco (TF) 		          | 89| 57 |
| ssd_mobilenet_v1_coco (IR Conversion)       | 44 									| 28  |
| faster_rcnn_inception_v2_coco (TF)          | 342 								| 202    |	 
| faster_rcnn_inception_v2_coco (IR Conversion)| 155                                | 98	|	 
| ssd_mobilenet_v2_coco (TF)		  		  | 104                               	| 135 	|	 
| ssd_mobilenet_v2_coco (IR Conversion)		  | 68									| 67	|	 

##### Comparing Edge and Cloud 
Edge Computing is suitable for small systems which they use low energy and resources, but Cloud Computing is good for high computations, data storage and big projects

## Assess Model Use Cases

Some of the potential use cases of the people counter app are checking election boxes, restriction areas, warehouses and passport control in airports, as a result, app could check people in the frame and may not allow two people in these areas.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows person may not be detected in dark place because of weak lighting and if model accuracy is low, app may not detect person properly in the frame. 

## Model Research

 All models taken from TensorFlow Object Detection Model Zoo and Open model zoo, respectively [TF detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), [OpenVINO public models](https://github.com/opencv/open_model_zoo/blob/master/models/public/index.md)

In investigating potential people counter models, I tried each of the following three models:

### ssd_mobilenet_v1_coco
  Download model: 
  ```
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
  ```
  Extracting the tar.gz file: `tar -xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz`
  Change path to `cd ssd_mobilenet_v1_coco_2018_01_28`
  Convert the model to an Intermediate Representation with the following arguments: 
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  The model was insufficient for the app because does not detect person properly in th frame 
  
### faster_rcnn_inception_v2_coco
 Download model: 
 ```
 wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
 ```
Extracting the tar.gz file: `tar -xzvf ssd_inception_v2_coco_2018_01_28.tar.gz`
  Change path to `cd ssd_inception_v2_coco_2018_01_28`
  Convert the model to an Intermediate Representation with the following arguments: 
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  The model was insufficient for the app because of slow fps

### ssd_mobilenet_v2_coco 
 Download model: 
 ```
 wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
 ```
  Extracting the tar.gz file: `tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
  Change path to `cd ssd_mobilenet_v2_coco_2018_03_29`
  Convert the model to an Intermediate Representation with the following arguments: 
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  This model is better than others and gives expected results with 0.2 probability threshold 

## Run the application

From the main directory:

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js

```

You should see the following message, if successful:

```
Mosca server started.

```

### Step 2 - Start the GUI

Open new terminal and run below commands.

```
cd webservice/ui
npm run dev

```

You should see the following message in the terminal.

```
webpack: Compiled successfully

```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.

```
sudo ffserver -f ./ffmpeg/server.conf

```

### Step 4 - Run the code

Open a new terminal to run the code.

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```
ou should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

#### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at:

```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
```
#### Run app
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/mobilenetssdv2/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm 
```
To see the output on a web based interface, open the link  [http://0.0.0.0:3004](http://0.0.0.0:3004/)  in a browser.
