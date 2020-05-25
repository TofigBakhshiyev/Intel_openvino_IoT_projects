
# People Counter App at the Edge

In this application, Edge AI system directs people to less-congested queues in the rush hour and this app was tested in three different scneraios, namely manufacturing, retail and transportation. This project is a part of Intel Edge AI for IoT Developers Nanodegree program by Udacity. 
##### Project videos
[Manufacturing](https://drive.google.com/file/d/13lTu6l9dZ2IKq4dST-1DygDmI0t_kIBB/view?usp=sharing) <br/>
[Retail](https://drive.google.com/file/d/1_4knzl-YP7zH0YMof4WpOGoGdzdEFCQW/view?usp=sharing) <br/>
[Transportation](https://drive.google.com/file/d/1IcfxAbDiFEWf_2YiEOl_w3xL0xzFzOm_/view?usp=sharing) <br/>
## Requirements

### Hardware 
* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* VPU - Intel® Neural Compute Stick 2 (NCS2)  
* FPGA

### Software

-   Intel® Distribution of OpenVINO™ toolkit 2020.1 release 
-   Intel DevCloud

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)

 ## Downloading model from Intel OpenVINO model zoo
 Change directory (opt exist in Linux: opt contains add-on software, larger programs may install here rather than in /usr)
 ```
 cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
 ```
 Install model
 ```
 sudo ./downloader.py --name person-detection-retail-0013 -o 'your direcotry'
 ```

###  Run the Edge AI 

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
Every scenario has its queue file (manufacturing.npy, retail.npy and transportation.npy)
```
python  python3 person_detect.py  --model "your model path/<MODEL PRECISION>/person-detection-retail-0013.xml" \
                          --device "CPU" \
                          --video "video path" \
                          --queue_param "queue file path" \
                          --output_path "your output path"\
                          --max_people "4" \
```
#### Testing app in different Intel Architecture
##### Intel DevCloud is best for this case because there are different devices which we can test our app
Before run the notebooks, you should change all path in all notebooks 
##### Run notebooks
- [Create_Python_Script.ipynb](./Create_Python_Script.ipynb)
- [Create_Job_Submission_Script.ipynb](./Create_Job_Submission_Script.ipynb)
#### Choose scenario
- [Manufacturing_Scenario.ipynb](./Manufacturing_Scenario.ipynb)
- [Retail_Scenario.ipynb](./Retail_Scenario.ipynb)
- [Transportation_Scenario.ipynb](./Transportation_Scenario.ipynb)
#### For test results and scenarios, you can check pdf file
[Results in different devices](./choose-the-right-hardware-proposal-template)

