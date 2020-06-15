'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import os
import numpy as np 
import pprint
from openvino.inference_engine import IENetwork, IECore

class Faciallandmarksmodel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device 
        self.device_extensions = extensions
        self.net = None
        self.core = None
        self.model = None
        
        # check model
        self.check_model(self.model_structure, self.model_weights)

        # getting inputs and outputs of model 
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self): 
        self.core = IECore()

        supported_layers = self.core.query_network(self.model, self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        # checking unsupported layers, if there is any unsupported layers, code will add extension
        if len(unsupported_layers) != 0:  
            if self.device_extensions and "CPU" in self.device:
                self.core.add_extension(self.device_extensions, self.device)
        # loading model
        self.net = self.core.load_network(network = self.model, device_name = self.device, num_requests = 0)

    def predict(self, image):
        # Pre-process the input image
        frame_image = self.preprocess_input(image) 
        input_dict = { self.input_name: frame_image }
        # Start asynchronous inference for specified request
        self.net.requests[0].async_infer(input_dict) 
        # Wait for the result
        status = self.net.requests[0].wait(-1)
        if status == 0:
            # print layer performance
            """ pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(self.net.requests[0].get_perf_counts()) """
            # Get the results of the inference request    
            outputs = self.net.requests[0].outputs[self.output_name] 
            # Pre-process the output image and get eyes coordinates 
            eyes_coordinates = self.preprocess_output(outputs[0], image) 

        return eyes_coordinates

    def check_model(self, model_structure, model_weights): 
        try:
            self.model=IENetwork(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA) 
        image = image.transpose((2,0,1)) 
        image = image.reshape(1, *image.shape)  
        return image

    def preprocess_output(self, outputs, image):
        left_eye = []
        right_eye = []
        # get width and height
        width, height = image.shape[1], image.shape[0]
        # extracting coordinates of eyes from all face coordinates
        left_x = int(outputs[0] * width)        
        left_y = int(outputs[1] * height)
        right_x = int(outputs[2] * width)
        right_y = int(outputs[3] * height)

        left_eye.append([left_x, left_y])
        right_eye.append([right_x, right_y])
        eyes_coordinates = left_eye + right_eye

        return eyes_coordinates 