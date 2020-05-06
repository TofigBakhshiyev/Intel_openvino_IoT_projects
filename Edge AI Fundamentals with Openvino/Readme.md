# OpenVINO Object detection and recognition 

The Intel OpenVINO toolkit is a free toolkit facilitating the optimization of a Deep Learning model from a framework and deployment using an inference engine onto Intel hardware. You can download from [here](https://software.intel.com/en-us/openvino-toolkit/choose-download) and you can also check [documentaion](https://docs.openvinotoolkit.org/latest/index.html)

# Pre-Trained Models
You can use openvino pre-trained models from these links or you can use other models
- [Pretrained Models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) 

# Object Detection
-m is model path
Run code: `python vehicle_and_pedesterian/app.py -m "model_path.xml"`
    

# Object Recogntion 
-t is model type
-c is cpu extension
Run code: `python handle_pose_text_car/app.py -i "image_path" -t "CAR_META" -m "model_path.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"`
 
## Example outputs

Ouput videos and images [link](https://drive.google.com/drive/folders/147xS07k21KB6dgL6tJgTzVBBr2-uDH7X?usp=sharing):