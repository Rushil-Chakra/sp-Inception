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


from Inception import sp_Inception

seed = 0
n_neurons = 448
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

total_n = 13440

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
	kernel_size=[24, 16, 10],
	stride=[4, 6, 6],
	n_filters=[n_neuron, n_neuron],
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

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, total_n)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(total_n)
proportions = torch.zeros(total_n, n_classes)
rates = torch.zeros(total_n, n_classes)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

concat_voltage_monitor = Monitor(network.layers["concat_layer"], ["v"], time=time)
network.add_monitor(concat_voltage_monitor, name="concat_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
	spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
	network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"Input"}:
	voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
	network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Train the network.
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

			# Get network predictions.
			all_activity_pred = all_activity(
				spikes=spike_record, assignments=assignments, n_labels=n_classes
			)
			proportion_pred = proportion_weighting(
				spikes=spike_record,
				assignments=assignments,
				proportions=proportions,
				n_labels=n_classes,
			)
			
			# Compute network accuracy according to available classification strategies.
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
				"\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
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
        s = spikes["concat_layer"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        # Get voltage recording.
        concat_voltages = concat_voltage_monitor.get("v")

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"][:, 0].view(28, 28)
            inpt = inputs["Input"][:, 0].view(time, 784).sum(0).view(28, 28)
            concat_concat_weights = network.connections[("concat_layer", "concat_layer")].w
            spikes_ = {
                layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
            } 
            voltages = {"concat": concat_voltages}

            # inpt_axes, inpt_ims = plot_input(
            #     image, inpt, label=labels[step], axes=inpt_axes, ims=inpt_ims
            # )
            # spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            perf_ax = plot_performance(accuracy, ax=perf_ax)
            # voltage_ims, voltage_axes = plot_voltages(
            #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            # )

            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")