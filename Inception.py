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
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
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
		dt: float = 1.0,
		nu: Optional[Union[float, Sequence[float]]] = [0.01, 0.0001],
		reduction: Optional[callable] = None,
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
		self.dt = dt
		self.theta_plus = theta_plus
		self.tc_theta_decay = tc_theta_decay
		self.input_shape = input_shape
	
		input_layer = Input(
			n=self.n_input, shape=self.input_shape, traces=True, tc_traces=20.0
		)
		self.add_layer(input_layer, name='Input')
		
		fc_layer = DiehlAndCookNodes(
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

		self.add_layer(fc_layer, name='fc_layer')

		w = 0.3 * torch.rand(self.n_input, self.n_neurons)
		fc_connection = Connection(
			source=input_layer,
			target=fc_layer,
			w=w,
			nu=nu,
			update_rule=PostPre,
		)

		self.add_connection(fc_connection, source='Input', target='fc_layer')

		
		fc_output = LIFNodes(
			n=self.n_neurons,
			traces=False,
			rest=-65.0,
			reset=-65.0,
			thresh=-52.0,
			tc_decay=tc_delay,
			refrac=5,
			tc_trace=20.0,
		)

		self.add_layer(fc_output, name='fc_output')

		w = torch.diag(torch.ones(self.n_neurons))
		#add next connection to competitive layer, do same for lc layers

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

			lc_layer = DiehlAndCookNodes(
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

			name = 'lc_layer' + str(i)
			self.add_layer(lc_layer, name=name)

			w = 0.3 * torch.rand(self.n_input, self.n_filters[i] * conv_sizes[i][0] * conv_sizes[i][1])
			lc_connection = LocalConnection(
				source=input_layer,
				target=lc_layer,
				kernel_size=kernel_size[i],
				stride=stride[i],
				n_filters=n_filters[i],
				nu=nu,
				update_rule=PostPre,
			)	   

			target=name
			self.add_connection(lc_connection, source='Input', target=target)

			#concatenate and reshape layer, figure out how to do this
			n_concat=0
			for layer in self.layers:
				n_concat += layer.n

			concat_layer = DiehlAndCookNodes(
				n=n_concat,
				shape=input_shape
				traces=True,
			)
			



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

