source ~/envs/udacity/bin/activate
source /opt/intel/openvino/bin/setupvars.sh
python src/app.py -fdm models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 -flm models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hpm models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -gm models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i bin/demo.mp4 -extension "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so" -d CPU
