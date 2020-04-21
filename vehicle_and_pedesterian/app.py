import argparse
import cv2
from inference import Network

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    t_desc = "Different confidence thresholds used to draw bounding boxes"
    c_desc = "choosing the color of the bounding boxes (Yellow and Green)"
   
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default = 0.5)
    optional.add_argument("-t", help=t_desc, default = "YELLOW")
    args = parser.parse_args() 
   
    return args

def draw_boxes(frame, output, args, width, height): 
    for box in output[0][0]:
        confidence = box[2]
        if confidence > args.c:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            ## (RGB (0, 255, 255)) is yellow
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (40, 143, 59), 1) 

    return frame
        
def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255), "YELLOW": (237, 230, 14)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['YELLOW']

def infer_on_video(args):
    #args.t = convert_color(args.t)
    #args.c = float(args.c)

    ### TODO: Initialize the Inference Engine
    ie = Network()
    ### TODO: Load the network model into the IE
    ie.load_model(args.m, args.d, CPU_EXTENSION)
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    net_input_shape = ie.get_input_shape()
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        pos_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        pos_frame = pos_frame.transpose((2,0,1))
        pos_frame = pos_frame.reshape(1, *pos_frame.shape)
        ### TODO: Perform inference on the frame
        ie.async_inference(pos_frame)
        ### TODO: Get the output of inference
        
        ### TODO: Update the frame to include detected bounding boxes 
    
        if ie.wait() == 0:
            output = ie.extract_output()
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, output, args, width, height)
            # Write out the frame
            out.write(frame)
            # Break if escape key pressed
            if key_pressed == 27:
                break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args() 
    infer_on_video(args)


if __name__ == "__main__":
    main()
