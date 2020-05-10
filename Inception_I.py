import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Parameter

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
	plot_input,
	plot_spikes,
	plot_weights,
	plot_assignments,
	plot_performance,
	plot_voltages,
)

from vfa_voting import vfa_assignment, vfa_prediction
from Inception import sp_Inception

seed = 0
n_neurons = 112
n_classes = 10
n_epochs = 1
n_test = 10000
n_workers = -1
exc = 22.5
inh = 120
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

n_total = 1568

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
	n_classes= n_classes,
	kernel_size=[24, 16],
	stride=[4, 6],
	n_filters=[n_neurons, n_neurons],
	dt=dt,
	theta_plus=theta_plus,
	input_shape=(1, 28, 28),
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

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_total)
voltage_record = torch.zeros(update_interval, time, n_classes)

# Neuron assignments and spike proportions.
proportions = torch.zeros(n_total, n_classes)
rates = torch.zeros(n_total, n_classes)

# Sequence of accuracy estimates.
accuracy = []

vfa_voltage_monitor = Monitor(network.layers["vfa_layer"], ["v"], time=time)
network.add_monitor(vfa_voltage_monitor, name="vfa_voltage")

spikes = {}
for layer in set(network.layers) - {'Input', 'vfa_layer'}:
	spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
	network.add_monitor(spikes[layer], name="%s_spikes" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

print("\nBegin training.\n")
start = t()

for epoch in range(n_epochs):
	labels = []

	if epoch % progress_interval == 0:
		print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
		start = t()

	# Create a dataloader to iterate and batch data
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=n_workers,
		pin_memory=gpu,
	)

	for step, batch in enumerate(dataloader):
		# Get next input sample.
		inputs = {"Input": batch["encoded_image"]}
		if gpu:
			inputs = {k: v.cuda() for k, v in inputs.items()}

		if step % update_steps == 0 and step > 0:
			# Convert the array of labels into a tensor
			label_tensor = torch.tensor(labels)		

			predictions = vfa_prediction(voltages=voltage_record)

			accuracy.append(
			100
			* torch.sum(label_tensor.long() == predictions).item()
			/len(label_tensor)
			)

			print(
				"Accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
				% (
					accuracy[-1],
					np.mean(accuracy),
					np.max(accuracy),
				)
			)

		#assign weights for concat connection
			weights = vfa_assignment(
				spikes=spike_record,
				labels=label_tensor,
				n_labels=n_classes,
				rates=rates
			)
			network.connections[('concat_layers', 'vfa_layer')].w = Parameter(weights, requires_grad=False)

			labels = []
	
		labels.extend(batch["label"].tolist())

		# Run the network on the input.
		network.run(inputs=inputs, time=time, input_time_dim=1)

		s = torch.cat(tuple(monitor.get('s').permute((1, 0, 2)) for monitor in list(spikes.values())), dim=2)
		spike_record[
			(step * batch_size)
			% update_interval : (step * batch_size % update_interval)
			+ s.size(0)
		] = s

		v = vfa_voltage_monitor.get('v').permute((1, 0, 2))
		voltage_record[
			(step * batch_size)
			% update_interval: (step * batch_size % update_interval)
			+ v.size(0)
		] = v

		network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")