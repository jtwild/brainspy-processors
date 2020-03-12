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
        self.alpha = configs['IOinfo']['alpha']

        # Specific IOinfo key loading
        if 'output_high' in configs['IOinfo'].keys():
            self.output_high = configs['IOinfo']['output_high']
        else:
            # Default value.
            self.output_high = self.info['data_info']['clipping_value'][1]
        if 'output_low' in configs['IOinfo'].keys():
            self.output_low = configs['IOinfo']['output_low']
        else:
            # Default value
            self.output_low = self.info['data_info']['clipping_value'][0]
        if 'offset_low' in configs['IOinfo'].keys():
            self.input_offset_low = configs['IOinfo']['offset_low']
        else:
            # Default, automatic, values
            # Default value for offset is zero. Scaling done by scaling only, (so actually, offset always initialized the same), such that it does not get initialized outside the valid range.
            self.input_offset_low = self.offset  # loaded by the super()
        if 'offset_high' in configs['IOinfo'].keys():
            self.input_offset_high = configs['IOinfo']['offset_high']
        else:
            # Default value the same as the offset_low
            self.input_offset_high = self.offset[self.input_indices]
        if 'scaling_low' in configs['IOinfo'].keys():
            self.scaling_low = configs['IOinfo']['scaling_low']
        else:
            self.scaling_low = [0]
        if 'scaling_high' in configs['IOinfo'].keys():
            self.scaling_high = configs['IOinfo']['scaling_high']
        else:
            self.scaling_high = self.amplitude[self.input_indices]

        # Create parameters. Initial value will be overwritten by reset.
        if configs['IOinfo']['mode'] == 'single_scaler':
            # Reduce the interval of sclaing and offset, since we have to choose one vlaue for all electrodes.
            self.scaling_low = min(self.scaling_low)
            self.scaling_high = max(self.scaling_high)
            self.input_offset_low = max(self.input_offset_low)
            self.input_offset_high = min(self.input_offset_high)
            # Register new parameters
            self.input_offset = nn.Parameter(torch.zeros(1))
            self.scaling = nn.Parameter(torch.zeros(1))
            self.forward = self.forward_single_scaler
            self.reset = self.reset_single_scaler
        elif configs['IOinfo']['mode'] == 'multi_scaler':
            self.input_offset = nn.Parameter( torch.zeros(1,self.input_no) )
            self.scaling = nn.Parameter( torch.zeros(1,self.input_no) )
            self.forward = self.forward_multi_scaler
            self.reset = self.reset_multi_scaler
        elif configs['IOinfo']['mode'] == 'None':
            self.input_offset = torch.tensor([[0.0]]) # not trainable
            self.scaling = torch.tensor([[1.0]])
            self.forward = self.forward_none_scaler
            self.reset= self.reset_none_scaler
        else:
            raise ValueError('Unknown IOnet mode supplied.')
        # Reinitialize all parameters accordingly.
        self.reset()


    def reset_multi_scaler(self):
        super().reset()
        print('Deliberately left a bug here in IOnet , reset multi scaler. Only the first index gets initialized randomly upon a new attemtps. The others remian the same and are thus continued in their training.')
        for k in range(len(self.input_offset)):
        #for k in range(len(self.input_indices)):
            self.input_offset.data[:, k].uniform_(self.input_offset_low[k], self.input_offset_high[k])
            self.scaling.data[:, k].uniform_(self.scaling_low[k], self.scaling_high[k])
        #We should be clearing gradients here?

    def reset_single_scaler(self):
        super().reset()
        self.input_offset.data[:].uniform_(self.input_offset_low, self.input_offset_high)
        self.scaling.data[:].uniform_(self.scaling_low, self.scaling_high)

    def reset_none_scaler(self):
        super().reset()

    def forward_single_scaler(self, x):
        # Add a single layer which can add an offset and a bias to the inputs given.
        self.input = x*self.scaling + self.input_offset   #self.input is used by the regularizer, and by readout (since it gets trianed)
        self.output = super().forward(self.input)     #self.output is used by regularizer
        return self.output

    def forward_multi_scaler(self,x):
        # Add a single layer which can add an offset and a scaling
        # For multiscale, every electrode can get its own offset and scaling individually.
        # Differs slightly from forward_single scaler because it needs a matrix multiplication.
        self.input = x*torch.mm(torch.ones(len(x),1), self.scaling) + self.input_offset   #self.input is used by the regularizer, and by readout (since it gets trianed)
        self.output = super().forward(self.input)     #self.output is used by regularizer
        return self.output

    def forward_none_scaler(self, x):
        self.input = x
        self.output = super().forward(x)
        return self.output

    def get_input_scaling(self):
        return self.input_offset.tolist(), self.scaling.tolist()

    def regularizer(self):
        # regularize output
        loss = torch.sum(torch.relu(self.output_low - self.output) + torch.relu(self.output - self.output_high))
        # regularize input
        loss += torch.sum(torch.relu(self.input_low - self.input) + torch.relu(self.input - self.input_high)) * self.alpha
        # Regularize controls
        loss += super().regularizer() * self.alpha
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