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

        # Variables used by methods:
        self.input_low = self.min_voltage[self.input_indices]
        self.input_high = self.max_voltage[self.input_indices]

        # Specific IOinfo key loading
        if 'output_high' in configs['IOinfo'].keys():
            self.output_high = configs['IOinfo']['output_high']
        else:
            # Default value.
            self.output_high = 76
        if 'output_low' in configs['IOinfo'].keys():
            self.output_low = configs['IOinfo']['output_low']
        else:
            self.output_low = -320
        if 'offset_low' in configs['IOinfo'].keys():
            self.offset_low = configs['IOinfo']['offset_low']
        else:
            self.offset_low = 0
        if 'offset_high' in configs['IOinfo'].keys():
            self.offset_high = configs['IOinfo']['offset_high']
        else:
            self.offset_high = 0.6
        if 'scaling_low' in configs['IOinfo'].keys():
            self.scaling_low = configs['IOinfo']['scaling_low']
        else:
            self.scaling_low = 0.1
        if 'scaling_high' in configs['IOinfo'].keys():
            self.scaling_high = configs['IOinfo']['scaling_high']
        else:
            self.scaling_high = 1.5
        # Create parameters. Initial value will be overwritten by reset.
        if configs['IOinfo']['mode'] == 'single_scaler':
            self.offset = nn.Parameter(torch.zeros(1))
            self.scaling = nn.Parameter(torch.zeros(1))
            self.forward = self.forward_single_scaler
            self.reset()
        elif configs['IOinfo']['mode'] == 'multi_scaler':
            self.offset = nn.Parameter( torch.zeros(1,self.input_no) )
            self.scaling = nn.Parameter( torch.zeros(1,self.input_no) )
            #self.scale_matrix = torch.mm(torch.ones(2**self.input_no), self.scaling )
            self.forward = self.forward_multi_scaler
            self.reset()
        elif configs['IOinfo']['mode'] == 'None':
            self.offset = torch.tensor([[0.0]]) # not trainable
            self.scaling = torch.tensor([[1.0]])
            self.forward = self.forward_single_scaler
            super().reset()
        else:
            raise ValueError('Unknown IOnet mode supplied.')


    def reset(self):
        super().reset()
        # Max and min values for random initialization are hardcoded for now...
        self.offset.data[:].uniform_(self.offset_low, self.offset_high)
        self.scaling.data[:].uniform_(self.scaling_low, self.scaling_high)

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

    def forward_without_scaling(self, x):
        self.input = x
        self.output = super().forward(x)
        return self.output

    def get_input_scaling(self):
        return self.offset.tolist(), self.scaling.tolist()

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