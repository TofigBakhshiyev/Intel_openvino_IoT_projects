cd C:\Program Files (x86)\IntelSWTools\openvino\bin\ 
setupvars.bat
python src/app.py -fdm models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 -flm models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hpm models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -gm models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i bin/demo.mp4 -extension "C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll" -d CPU
