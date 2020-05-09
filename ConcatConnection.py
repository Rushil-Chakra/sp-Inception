from typing import Union, Tuple, Optional, Sequence, Dict

import numpy as np
import torch
from torch.nn import Module, Parameter

from bindsnet.network.nodes import Nodes

class ConcatConnection(AbstractConnection):
	def __init__(
		self,
		source: Dict[Nodes],
		target: Nodes,
		nu: Optional[Union[float, Sequence[float]]] = None,
		reduction: Optional[callable] = None,
		weight_decay: float = 0.0,
		**kwargs
	) -> None:

		super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

		w = kwargs.get("w", None)
		source_n = np.sum(source.values().n)
		
		if w is None:
			if self.wmin == -np.inf or self.wmax == np.inf:
				w = torch.clamp(
					torch.zeros(source_n, target.n), self.wmin, self.wmax
				)
			else:
				w = self.wmin + torch.zeros(source_n, target.n) * (
					self.wmax - self.wmin
				)
		else:
			if self.wmin != -np.inf or self.wmax != np.inf:
				w = torch.clamp(w, self.wmin, self.wmax)

		self.w = Parameter(w, requires_grad=False)
		self.b = Parameter(
			kwargs.get("b", torch.zeros(target.n)), requires_grad=False
		)

	def compute(self, s: Dict[torch.Tensor]) -> torch.Tensor:
		# language=rst
		"""
		Compute pre-activations given spikes using connection weights.
		:param s: Incoming spikes.
		:return: Incoming spikes multiplied by synaptic weights (with or without
				 decaying spike activation).
		"""
		# Compute multiplication of spike activations by weights and add bias.
		s = torch.cat(tuple(s.values()), dim=0)
		post = s.float().view(s.size(0), -1) @ self.w + self.b
		return post.view(s.size(0), *self.target.shape)

	def update(self, **kwargs) -> None:
		# language=rst
		"""
		Compute connection's update rule.
		"""
		super().update(**kwargs)

	def normalize(self) -> None:
		# language=rst
		"""
		Normalize weights so each target neuron has sum of connection weights equal to
		``self.norm``.
		"""
		if self.norm is not None:
			w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
			w_abs_sum[w_abs_sum == 0] = 1.0
			self.w *= self.norm / w_abs_sum

	def reset_state_variables(self) -> None:
		# language=rst
		"""
		Contains resetting logic for the connection.
		"""
		super().reset_state_variables()