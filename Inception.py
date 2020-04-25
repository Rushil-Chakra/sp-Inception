from typing import Optional, Union, Tuple, List, Sequence, Iterable

import torch
import os
import numpy as np
from torch.nn import AdaptiveMaxPool2d
from torch.nn.modules.utils import _pair
from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes, DiehlAndCookNodes
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.learning import PostPre, NoOp
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder

class sp_Inception(Network):
	def __init__(
		self,
		n_input: int,
		n_neurons: int,
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
		self.kernel_size = kernel_size
		self.stride = stride
		self.n_filters = n_filters
		self.exc = exc
		self.inh = inh
		self.dt = dt
		self.theta_plus = theta_plus
		self.tc_theta_decay = tc_theta_decay
		self.input_shape = input_shape
	
		input_layer = Input(
			n=self.n_input, shape=self.input_shape, traces=True, tc_traces=20.0
		)
		self.add_layer(input_layer, name='Input')
		
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

		self.add_layer(fc_output, name='fc_output')

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

		self.add_connection(fc_input_output_conn, source=Input, target=fc_output)

		w = -self.inh * (
			torch.ones(self.n_neurons, self.n_neurons)
			- torch.diag(torch.ones(self.n_neurons))
		)

		fc_output_comp_conn = Connection(
			source=fc_output,
			target=fc_output,
			w=w,
			wmin=-self.inh,
			wmax=0
		)

		self.add_connection(fc_output_comp_conn, source=fc_output, target=fc_output)

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

			name = 'lc_output' + str(i)
			self.add_layer(lc_output, name=name)

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

			self.add_connection(lc_input_output_conn, source=Input, target=self.layers[name])
			
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
			
			lc_output_comp_conn = Connection(source=self.layers[name], target=self.layers[name], w=w)
		
			self.add_connection(lc_output_comp_conn, source=self.layers[name], target=self.layers[name])

		#start count at -n_neurons so it doesn't count input neurons in count
		concat_n = -n_input
		for layer in self.layers:
			concat_n += layer.n

		concat_layer = AdaptiveLIFNodes(
			n=concat_n,
			traces=False,
			tc_trace=20.0,
			thresh=-52.0,
			rest=-65.0,
			reset=-65.0,
			refrac=5,
			tc_decay=tc_decay,
			thetaplus=theta_plus,
			tc_theta_decay=tc_theta_decay,
		)

		self.add_layer(concat_layer, name='concat_layer')

		#TODO figure out how to concatenate weighs, currently creating a connection from everything to concat layer
		#then making a recurrent connection to itself as a concatenation of all layers (nned to make everything into a vector and concat along one axis)
		concat_w = torch.tensor(())
		for connection in self.connections:
			if "Input_to" in connection.name:
				w = torch.flatten(connection.w)
				concat_w = torch.cat((concat_w, w.view(concat_w.size)), -1)
				source_layer = connection.target
				concat_connection = Connection(source=source_layer, target=concat_layer, w=w)

seed = 0
n_neurons = 112
n_epochs = 1
n_test = 10000
n_workers = -1
theta_plus = 0.05
time = 100
dt = 1
intensity = 128
progress_interval = 16
update_steps = 10
batch_size = 16
train = True
plot = False
gpu = True

if not train:
	update_steps = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

update_interval = update_steps * batch_size


# Sets up Gpu use
if gpu:
	torch.cuda.manual_seed_all(seed)
else:
	torch.manual_seed(seed)
	
# Determines number of workers to use
if n_workers == -1:
	n_workers = gpu * 4 * torch.cuda.device_count()

network = sp_Inception(
	n_input=784,
	n_neurons=n_neurons,
	kernel_size=[24, 16],
	stride=[4, 6],
	n_filters=[112, 112],
	dt=dt,
	theta_plus=theta_plus,
	input_shape=(28, 28),
)

if gpu:
	#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	network.to("cuda")


# Load MNIST data.
dataset = MNIST(
	PoissonEncoder(time=time, dt=dt),
	None,
	root=os.path.join("data", "MNIST"),
	download=True,
	transform=transforms.Compose(
	[transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
	),
)

