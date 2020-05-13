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
inh = 120
theta_plus = 0.05
time = 100
dt = 1
intensity = 128
progress_interval = 10
update_steps = 256
batch_size = 16
train = True
plot = True
gpu = True

n_total = 1568

if not train:
	update_steps = n_test

n_sqrt = int(np.ceil(np.sqrt(448)))
start_intensity = intensity

update_interval = update_steps * batch_size


# Sets up Gpu use
if gpu:
	torch.cuda.manual_seed_all(seed)
else:
	torch.manual_seed(seed)
	
# Determines number of workers to use
if n_workers == -1:
	n_workers = gpu * 8 * torch.cuda.device_count()

network = sp_Inception(
	n_input=784,
	n_neurons=n_neurons,
	n_classes=n_classes,
	inh=inh,
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
assignments = -torch.ones(n_total)

# Sequence of accuracy estimates.
accuracy = {'accuracy':[]}

spikes = {}
for layer in set(network.layers) - {'Input'}:
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

	for step, batch in enumerate(tqdm(dataloader)):
		# Get next input sample.
		inputs = {"Input": batch["encoded_image"]}
		if gpu:
			inputs = {k: v.cuda() for k, v in inputs.items()}

		if step % update_steps == 0 and step > 0:
			# Convert the array of labels into a tensor
			label_tensor = torch.tensor(labels)	
			
			predictions = vfa_prediction(
				spikes=spike_record,
				proportions=proportions
			)

			accuracy['accuracy'].append(
			100
			* torch.sum(label_tensor.long() == predictions).item()
			/len(label_tensor)
			)

			print(
				"Accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
				% (
					accuracy['accuracy'][-1],
					np.mean(accuracy['accuracy']),
					np.max(accuracy['accuracy']),
				)
			)

			proportions, rates = vfa_assignment(
				spikes=spike_record,
				labels=label_tensor,
				n_labels=n_classes,
				rates=rates
			)

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

		if plot:
			#image = batch["image"][:, 0].view(28, 28)
			#inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
			input_exc_weights = network.connections[("Input", "lc_output0")].w
			square_weights = get_square_weights(
				input_exc_weights.view(784, 448), n_sqrt, 28
			)
			
			weights_im = plot_weights(square_weights, im=weights_im)
			perf_ax = plot_performance(accuracy, x_scale=update_steps * batch_size, ax=perf_ax)
			#voltage_ims, voltage_axes = plot_voltages(
			#	voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
			#S)

			plt.pause(1e-8)

		network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")