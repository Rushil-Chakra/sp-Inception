import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import (
	all_activity,
	proportion_weighting,
	assign_labels,
	vfa,
)
from bindsnet.models import LocallyConnectedNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import (
	plot_input,
	plot_spikes,
	plot_weights,
	plot_performance,
	plot_assignments,
	plot_voltages,
	plot_locally_connected_weights
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_steps", type=int, default=256)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_workers = args.n_workers
update_steps = args.update_steps
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu

update_interval = update_steps * batch_size

# Sets up Gpu use
if gpu and torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
else:
	torch.manual_seed(seed)

# Determines number of workers to use
if n_workers == -1:
	n_workers = gpu * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons*4)))
start_intensity = intensity

network = LocallyConnectedNetwork(
	n_inpt=784,
	kernel_size=24,
	stride=4,
	n_filters=n_neurons,
	inh=inh,
	dt=dt,
	nu=(1e-4, 1e-2),
	theta_plus=theta_plus,
	input_shape=(1, 28, 28),
)

if gpu:
	network.to("cuda")

# Load MNIST data.
dataset = MNIST(
	PoissonEncoder(time=time, dt=dt),
	None,
	root=os.path.join(ROOT_DIR, "data", "MNIST"),
	download=True,
	transform=transforms.Compose(
		[transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
	),
)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons * 4)
proportions = torch.zeros(n_neurons * 4, n_classes)
rates = torch.zeros(n_neurons * 4, n_classes)
vfa_proportions = torch.zeros(n_neurons * 4, n_classes)
vfa_rates = torch.zeros(n_neurons * 4, n_classes)

# Sequence of accuracy estimates.
accuracy = {"vfa": [], "all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
	spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
	network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
	voltages[layer] = Monitor(
		network.layers[layer], state_vars=["v"], time=time
	)
	network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros(update_interval, time, n_neurons * 4)

# Train the network.
print("\nBegin training.\n")
start = t()

for epoch in range(n_epochs):
	labels = []

	if epoch % progress_interval == 0:
		print(
			"Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start)
		)
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
		inputs = {"X": batch["encoded_image"]}
		if gpu:
			inputs = {k: v.cuda() for k, v in inputs.items()}
		
		if step % update_steps == 0 and step > 0:
			# Convert the array of labels into a tensor
			label_tensor = torch.tensor(labels)
			
			# Get network predictions.
			vfa_pred, vfa_rates = vfa(
				spikes=spike_record,
				labels=label_tensor,
				n_labels=n_classes,
				rates=vfa_rates
			)
			all_activity_pred = all_activity(
				spikes=spike_record,
				assignments=assignments,
				n_labels=n_classes,
			)
			proportion_pred = proportion_weighting(
				spikes=spike_record,
				assignments=assignments,
				proportions=proportions,
				n_labels=n_classes,
			)

			# Compute network accuracy according to available classification strategies.
			accuracy["vfa"].append(
				100
				* torch.sum(label_tensor.long() == vfa_pred).item()
				/ len(label_tensor)
			)
			accuracy["all"].append(
				100
				* torch.sum(label_tensor.long() == all_activity_pred).item()
				/ len(label_tensor)
			)
			accuracy["proportion"].append(
				100
				* torch.sum(label_tensor.long() == proportion_pred).item()
				/ len(label_tensor)
			)
			print(
				"\nvfa accuracy: %.2f (last), %.2f (average), %.2f (best)"
				% (
					accuracy["vfa"][-1],
					np.mean(accuracy["vfa"]),
					np.max(accuracy["vfa"]),
				)
			)
			print(
				"All activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
				% (
					accuracy["all"][-1],
					np.mean(accuracy["all"]),
					np.max(accuracy["all"]),
				)
			)
			print(
				"Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
				% (
					accuracy["proportion"][-1],
					np.mean(accuracy["proportion"]),
					np.max(accuracy["proportion"]),
				)
			)

			# Assign labels to excitatory layer neurons.
			
			assignments, proportions, rates = assign_labels(
				spikes=spike_record,
				labels=label_tensor,
				n_labels=n_classes,
				rates=rates,
			)

			labels = []

		labels.extend(batch["label"].tolist())
		# Run the network on the input.
		network.run(inputs=inputs, time=time, input_time_dim=1)

		# Add to spikes recording.
		s = spikes["Y"].get("s").permute((1, 0, 2))
		spike_record[
			(step * batch_size)
			% update_interval : (step * batch_size % update_interval)
			+ s.size(0)
		] = s

		# Get voltage recording.
		exc_voltages = exc_voltage_monitor.get("v")

		# Optionally plot various simulation information.
		if plot:
			#image = batch["image"][:, 0].view(28, 28)
			#inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
			input_exc_weights = network.connections[("X", "Y")].w
			square_weights = get_square_weights(
				input_exc_weights.view(784, n_neurons*4), n_sqrt, 28
			)
			#square_assignments = get_square_assignments(assignments, n_sqrt)
			#spikes_ = {
			#	layer: spikes[layer].get("s")[:, 0].contiguous()
			#	for layer in spikes
			#}
			#voltages = {"Y": exc_voltages}
			#inpt_axes, inpt_ims = plot_input(
			#	image, inpt, label=labels[step * batch_size % update_interval], axes=inpt_axes, ims=inpt_ims
			#)
			#spike_ims, spike_axes = plot_spikes(
			#	spikes_, ims=spike_ims, axes=spike_axes
			#)
			weights_im = plot_weights(square_weights, im=weights_im)
			#assigns_im = plot_assignments(square_assignments, im=assigns_im)
			perf_ax = plot_performance(accuracy, x_scale=update_steps * batch_size, ax=perf_ax)
			#voltage_ims, voltage_axes = plot_voltages(
			#	voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
			#)

			plt.pause(1e-8)

		network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")