from typing import Optional, Union, Tuple, List, Sequence, Iterable

import torch
import os
import numpy as np
from torch.nn.modules.utils import _pair
from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.network import Network
from bindsnet.network.nodes import Input, DiehlAndCookNodes, IFNodes
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.learning import PostPre, NoOp
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder

from ConcatConnection import ConcatConnection

class sp_Inception(Network):
	def __init__(
		self,
		n_input: int,
		n_neurons: int,
		n_classes: int,
		n_fc: int = 1,
		kernel_size: Union[Sequence[int], Sequence[Tuple[int, int]]],
		stride: Union[Sequence[int], Sequence[Tuple[int, int]]],
		n_filters: Sequence[int],
		exc: float = 22.5,
		inh: float = 17.5,
		dt: float = 1.0,
		nu: Optional[Union[float, Sequence[float]]] = [0.01, 0.0001],
		reduction: Optional[callable] = None,
		wmin: float = 0.0,
		wmax: float = 1.0,
		norm: float = 78.4,
		tc_decay: float = 100.0,
		theta_plus: float = 0.05,
		tc_theta_decay: float = 1e7,
		input_shape: Optional[Iterable[int]] = None,	
		R: Optional[float] = 60.0,

	) -> None:  
	
	#param definitions
	
		super().__init__(dt=dt)
	
		self.n_input = n_input
		self.n_neurons = n_neurons
		self.n_fc = n_fc
		self.kernel_size = kernel_size
		self.stride = stride
		self.n_filters = n_filters
		self.exc = exc
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
				update_rule=PostPre,
				reduction=reduction,
				wmin=wmin,
				wmax=wmax,
				norm=norm,
			)

			self.add_connection(fc_input_output_conn, source=Input, target=self.layers[fc_name])

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

			self.add_connection(fc_output_comp_conn, source=self.layers[fc_name], target=self.layers[fc_name])

		num_lc_layers = len(n_filters)
		conv_sizes = [0] * num_lc_layers


		for i in range(num_lc_layers):
			kernel_size[i] = _pair(kernel_size[i])
			stride[i] = _pair(stride[i])

			if kernel_size[i] == input_shape:
				conv_sizes[i] = [1, 1]
			else:
				conv_sizes[i] = (
					int((input_shape[0] - kernel_size[i][0]) / stride[i][0]) + 1,
					int((input_shape[1] - kernel_size[i][1]) / stride[i][1]) + 1,
				)

			total_neuron += self.n_filters[i] * conv_sizes[i][0] * conv_sizes[i][1]

			lc_output = DiehlAndCookNodes(
				n=self.n_filters[i] * conv_sizes[i][0] * conv_sizes[i][1],
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

			w = 0.3 * torch.rand(self.n_input, self.n_filters[i] * conv_sizes[i][0] * conv_sizes[i][1])
			lc_input_output_conn = LocalConnection(
				source=input_layer,
				target=self.layers[name],
				kernel_size=kernel_size[i],
				stride=stride[i],
				n_filters=n_filters[i],
				nu=nu,
				reduction=reduction,
				update_rule=PostPre,
				wmin=wmin,
				wmax=wmax,
				norm=norm,
				input_shape=input_shape,
			)	   

			self.add_connection(lc_input_output_conn, source=Input, target=self.layers[lc_name])
			
			#makes weights so that competition is in each receptive field
			w = torch.zeros(n_filters, *conv_size, n_filters, *conv_size)
			for fltr1 in range(n_filters):
				for fltr2 in range(n_filters):
					if fltr1 != fltr2:
						for i in range(conv_size[0]):
							for j in range(conv_size[1]):
								w[fltr1, i, j, fltr2, i, j] = -inh

			w = w.view(
				n_filters * conv_size[0] * conv_size[1],
				n_filters * conv_size[0] * conv_size[1],
			)
			
			lc_output_comp_conn = Connection(source=self.layers[lc_name], target=self.layers[lc_name], w=w)
		
			self.add_connection(lc_output_comp_conn, source=self.layers[lc_name], target=self.layers[lc_name])

		vfa_layer = IFNodes(n=n_classes, learning=False)
		self.add_layer(vfa_layer, name='vfa_layer', thresh=-np.inf)

		concat_layers = dict(set(self.layers) - {'Input'})

		concat_conn = ConcatConection(
			source=concat_layers,
			target=vfa_layer,
		)

		self.add_connection(concat_conn, source='concat_layers', target='vfa_layer')
