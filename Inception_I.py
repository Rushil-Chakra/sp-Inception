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
	#vfa,
)
from bindsnet.models import DiehlAndCook2015v2
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

from Inception import sp_Inception
from vfa import vfa

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=112)
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

n_classes = 10
n_total = 1568

if not train:
	update_steps = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons*9)))
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

vfa_proportions = torch.zeros(n_total, n_classes)
vfa_rates = torch.zeros(n_total, n_classes)

# Sequence of accuracy estimates.
accuracy = {'vfa':[], 'all':[], 'proportion':[]}

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

			accuracy['vfa'].append(
			100
			* torch.sum(label_tensor.long() == vfa_predictions).item()
			/len(label_tensor)
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
				"\nVFA Accuracy: %.2f (last), %.2f (average), %.2f (best)"
				% (
					accuracy['vfa'][-1],
					np.mean(accuracy['vfa']),
					np.max(accuracy['vfa']),
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
			
			assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

			labels = []

		labels.extend(batch["label"].tolist())

		# Run the network on the input.
		lab = batch['label'][0].item()
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
			#input_fc0_weights = network.connections[("Input", "lc_output1")].w
			#square_weights = get_square_weights(
			#	input_fc0_weights.view(n_total, n_classes), n_sqrt, 28
			#)
			#input_lc0_weights = network.connection[("Input", "lc_output1")].w

			#weights_im = plot_weights(vfa_proportions, im=weights_im)
			perf_ax = plot_performance(accuracy, x_scale=update_steps * batch_size, ax=perf_ax)
			#voltage_ims, voltage_axes = plot_voltages(
			#	voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
			#S)

			plt.pause(1e-8)

		network.reset_state_variables()  # Reset state variables.

if plot:
	plt.savefig('Inception_batch_size-32.png')
print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")