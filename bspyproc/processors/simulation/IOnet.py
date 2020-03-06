# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:02:39 2020

Defines a new child-class of DNPU, which adds an extra scaling input or output layer (I/O layer).

@author: Jochem
"""
from bspyproc.processors.simulation.dopanet import DNPU
import torch.nn as nn
import torch


class InputScaleNet(DNPU):

    def __init__(self, configs):
        # Load DNPU parent class
        super().__init__(configs)
        # Create parameters. Initial value will be overwritten by reset.
        if configs['IOinfo']['mode'] == 'single_scaler':
            self.offset = nn.Parameter(torch.zeros(1))
            self.scaling = nn.Parameter(torch.zeros(1))
            self.forward = self.forward_single_scaler
        elif configs['IOinfo']['mode'] == 'multi_scaler':
            self.offset = nn.Parameter( torch.zeros(1,self.input_no) )
            self.scaling = nn.Parameter( torch.zeros(1,self.input_no) )
            #self.scale_matrix = torch.mm(torch.ones(2**self.input_no), self.scaling )
            self.forward = self.forward_multi_scaler
        self.reset()             #randomly initialize all variables.
        # Variables used by methods:
        self.input_low = self.min_voltage[self.input_indices]
        self.input_high = self.max_voltage[self.input_indices]
        self.output_high = configs['IOinfo']['output_high']
        self.output_low = configs['IOinfo']['output_low']

        # Register parameters
        # IS this automatically done for nn.Parameters?
        #self.register_parameter('ioscaler', self.scaling)
        #self.register_parameter('ioscaler', self.offset)

        # Parameters must also be added to optimizer!
        # Goes correctly if this class is initiated before the optimizer, so should be fine?
        # Optimizer loading is defined in GDData (gd.py) by init_optimizer, perhaps also the place where variable learning rate can be defined.

    def reset(self):
        super().reset()
        # Max and min values for random initialization are hardcoded for now...
        self.offset.data[:].uniform_(0, 0.6)
        self.scaling.data[:].uniform_(0.1, 1.5)

    def forward_single_scaler(self, x):
        # Add a single layer which can add an offset and a bias to the inputs given.
        self.input = x*self.scaling + self.offset   #self.input is used by the regularizer, and by readout (since it gets trianed)
        self.output = super().forward(self.input)     #self.output is used by regularizer
        return self.output

    def forward_multi_scaler(self,x):
        # Add a single layer which can add an offset and a scaling
        # For multiscale, every electrode can get its own offset and scaling individually.
        # Differs slightly from forward_single scaler because it needs a matrix multiplication.
        self.input = x*torch.mm(torch.ones(len(x),1), self.scaling) + self.offset   #self.input is used by the regularizer, and by readout (since it gets trianed)
        self.output = super().forward(self.input)     #self.output is used by regularizer
        return self.output

    def get_input_scaling(self):
        return self.offset.tolist(), self.scaling.tolist()
        # def get_control_voltages(self):
        #     return next(self.parameters()).detach() * self.scaling.detach() + self.offset.detach()

    def regularizer(self):
        # regularize output
        loss = torch.sum(torch.relu(self.output_low - self.output) + torch.relu(self.output - self.output_high))
        # regularize input
        loss += torch.sum(torch.relu(self.input_low - self.input) + torch.relu(self.input - self.input_high))
        # Regularize controls
        loss += super().regularizer()
        return loss
        # from super: return torch.sum(torch.relu(self.control_low - self.bias) + torch.relu(self.bias - self.control_high))

    def get_output(self, input_matrix=None):
        # If no input matrix is supplied, return the most recent output.
        if input_matrix == None:
            return self.output
        else:
            super().get_output(input_matrix)

# FOr adding a complete first layer with inputs trainable a piece, the nn.Sequential might be called with nn.Linear

# Question: calling self.parameters() does not display the added 'layer'?
# But when doing
#for params in excel_results[0]['processor'].parameters():
#    print(params.shape)
# the two extra layers get displayed, but only after a layer defining the control voltages...?The inputs x should be the inputs right, not control voltages being scaled?

class InputSuperScaler(DNPU):

    def __init__(self, configs):
        # Load DNPU parent class
        super().__init__(configs)

        self.scale_layer = nn.Linear()