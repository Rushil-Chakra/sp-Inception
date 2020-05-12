import torch
from typing import Optional, Tuple, Dict

def vfa_assignment(
	spikes: torch.Tensor,
	labels: torch.Tensor,
	n_labels: int,
	rates: Optional[torch.Tensor] = None,
	alpha: float = 1.0,
) -> torch.Tensor:

	n_neurons = spikes.size(2)

	if rates is None:
		rates = torch.zeros(n_neurons, n_labels)

	# Sum over time dimension (spike ordering doesn't matter).
	spikes = spikes.sum(1)

	for i in range(n_labels):
		# Count the number of samples with this label.
		n_labeled = torch.sum(labels == i).float()

		if n_labeled > 0:
			# Get indices of samples with this label.
			indices = torch.nonzero(labels == i).view(-1)
			# Compute average firing rates for this label.
			rates[:, i] = alpha * rates[:, i] + (
				torch.sum(spikes[indices], 0) / n_labeled
			)
	# Compute proportions of spike activity per class.
	proportions = rates / rates.sum(1, keepdim=True)
	proportions[proportions != proportions] = 0  # Set NaNs to 0
	return proportions, rates

def vfa_prediction(
	spikes: torch.Tensor,
	#labels: torch.Tensor,
	proportions: torch.Tensor,
	#n_label: int,
) -> torch.Tensor:

	spikes = spikes.sum(1)

	voltages = spikes @ proportions
	predictions = torch.sort(voltages, dim=1, descending=True)[1][:, 0]

	return predictions