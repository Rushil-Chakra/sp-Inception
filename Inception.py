from typing import Optional, Union, Tuple, List, Sequence, Iterable

import torch
import os
import numpy as np
from torch.nn.modules.utils import _pair

from bindsnet.network import Network
from bindsnet.network.nodes import Input, DiehlAndCookNodes, IFNodes
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.learning import PostPre, NoOp, WeightDependentPostPre

from ConcatConnection import ConcatConnection

class sp_Inception(Network):
	def __init__(
		self,
		n_input: int,
		n_neurons: int,
		n_classes: int,
		kernel_size: Union[Sequence[int], Sequence[Tuple[int, int]]],
		stride: Union[Sequence[int], Sequence[Tuple[int, int]]],
		n_filters: Sequence[int],
		n_fc: int = 1,
		inh: float = 17.5,
		dt: float = 1.0,
		nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
		reduction: Optional[callable] = None,
		wmin: float = 0.0,
		wmax: float = 1.0,
		norm: float = 78.4,
		tc_decay: float = 100.0,
		theta_plus: float = 0.05,
		tc_theta_decay: float = 1e7,
		input_shape: Optional[Iterable[int]] = None,	

	) -> None:  
	
	#param definitions
	
		super().__init__(dt=dt)
	
		self.n_input = n_input
		self.n_neurons = n_neurons
		self.n_fc = n_fc
		self.kernel_size = kernel_size
		self.stride = stride
		self.n_filters = n_filters
		self.inh = inh
		self.dt = dt
		self.nu = nu
		self.reduction = reduction
		self.wmin = wmin
		self.wmax = wmax
		self.norm = norm
		self.theta_plus = theta_plus
		self.tc_theta_decay = tc_theta_decay
		self.input_shape = input_shape
	
		input_layer = Input(
			n=self.n_input, shape=self.input_shape, traces=True, tc_traces=20.0
		)
		self.add_layer(input_layer, name='Input')
		
		total_neuron = 0
		
		for i in range(n_fc):
			total_neuron += n_neurons

			fc_output = DiehlAndCookNodes(
				n=n_neurons,
				traces=True,
				tc_trace=20.0,
				thresh=-52.0,
				rest=-65.0,
				reset=-65.0,
				refrac=5,
				tc_decay=tc_decay,
				theta_plus=theta_plus,
				tc_theta_decay=tc_theta_decay,
			)
			fc_name = 'fc_output' + str(i)
			self.add_layer(fc_output, name=fc_name)

			w = 0.3 * torch.rand(self.n_input, self.n_neurons)
			fc_input_output_conn = Connection(
				source=input_layer,
				target=fc_output,
				w=w,
				nu=nu,
				update_rule=WeightDependentPostPre,
				reduction=reduction,
				wmin=wmin,
				wmax=wmax,
				norm=norm,
			)

			self.add_connection(fc_input_output_conn, source='Input', target=fc_name)

			w = -self.inh * (
				torch.ones(self.n_neurons, self.n_neurons)
				- torch.diag(torch.ones(self.n_neurons))
			)

			fc_output_comp_conn = Connection(
				source=self.layers[fc_name],
				target=self.layers[fc_name],
				w=w,
				wmin=-self.inh,
				wmax=0
			)

			self.add_connection(fc_output_comp_conn, source=fc_name, target=fc_name)
			
		num_lc_layers = len(n_filters)
		
		for i in range(num_lc_layers):
			conv_size = [0, 0]
			kernel_size[i] = _pair(kernel_size[i])
			stride[i] = _pair(stride[i])

			if kernel_size[i] == input_shape[1:]:
				conv_size = [1, 1]
			else:
				conv_size = (
					int((input_shape[1] - kernel_size[i][0]) / stride[i][0]) + 1,
					int((input_shape[2] - kernel_size[i][1]) / stride[i][1]) + 1,
				)

			total_neuron += self.n_filters[i] * conv_size[0] * conv_size[1]

			lc_output = DiehlAndCookNodes(
				n=self.n_filters[i] * conv_size[0] * conv_size[1],
				traces=True,
				tc_trace=20.0,
				thresh=-52.0,
				rest=-65.0,
				reset=-65.0,
				refrac=5,
				tc_decay=tc_decay,
				thetaplus=theta_plus,
				tc_theta_decay=tc_theta_decay,
			)

			lc_name = 'lc_output' + str(i)
			self.add_layer(lc_output, name=lc_name)

			lc_input_output_conn = LocalConnection(
				source=input_layer,
				target=self.layers[lc_name],
				kernel_size=kernel_size[i],
				stride=stride[i],
				n_filters=n_filters[i],
				nu=nu,
				reduction=reduction,
				update_rule=WeightDependentPostPre,
				wmin=wmin,
				wmax=wmax,
				norm=0.2,
				input_shape=input_shape[1:],
			)	   

			self.add_connection(lc_input_output_conn, source="Input", target=lc_name)
			
			#makes weights so that competition is in each receptive field
			w = torch.zeros(n_filters[i], *conv_size, n_filters[i], *conv_size)
			for fltr1 in range(n_filters[i]):
				for fltr2 in range(n_filters[i]):
					if fltr1 != fltr2:
						for j in range(conv_size[0]):
							for k in range(conv_size[1]):
								w[fltr1, j, k, fltr2, j, k] = -inh

			w = w.view(
				n_filters[i] * conv_size[0] * conv_size[1],
				n_filters[i] * conv_size[0] * conv_size[1],
			)
			
			lc_output_comp_conn = Connection(source=self.layers[lc_name], target=self.layers[lc_name], w=w)
		
			self.add_connection(lc_output_comp_conn, source=lc_name, target=lc_name)

		'''
		concat_layers = {i:self.layers[i] for i in self.layers if i!='Input'}

		vfa_layer = IFNodes(n=n_classes, learning=False, thresh=-np.inf)
		self.add_layer(vfa_layer, name='vfa_layer')

		concat_conn = ConcatConnection(
			source=concat_layers,
			target=vfa_layer,
		)

		self.add_connection(concat_conn, source='concat_layers', target='vfa_layer')
		'''