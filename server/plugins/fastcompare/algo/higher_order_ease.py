from abc import ABC
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from plugins.fastcompare.algo.algorithm_base import (
		AlgorithmBase,
		Parameter,
		ParameterType,
)


class HigherOrderEASE(AlgorithmBase, ABC):
		"""Implementation of HOEASE (Higher-Order EASE) recommender
		based on 'Don’t Go Deeper, Go Higher' (RecSys 2021).
		Extends EASE by adding higher-order (pair) interactions.

		Reference: https://dl.acm.org/doi/10.1145/3460231.3474273
		"""

		def __init__(self, loader, positive_threshold, l2, min_support=50, **kwargs):
			self._ratings_df = loader.ratings_df
			self._loader = loader
			self._all_items = self._ratings_df.item.unique()

			self._rating_matrix = (
				self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
				.fillna(0)
				.values
			)

			self._threshold = positive_threshold
			self._l2 = l2
			self._min_support = min_support

			self._items_count = np.shape(self._rating_matrix)[1]

			self._weights = None   # pairwise weights (like EASE)
			self._C = None         # higher-order weights
			self._pair_index = {}  # map pair -> column index

		# Train the model
		def fit(self):
			# Step 1: Binary user–item matrix
			X = np.where(self._rating_matrix >= self._threshold, 1, 0).astype(np.float32)

			# Step 2: Pairwise part (same as EASE)
			G = X.T @ X
			G += self._l2 * np.eye(self._items_count)
			P = np.linalg.inv(G)
			diag_P = np.diag(P)
			B = -P / diag_P[None, :]
			np.fill_diagonal(B, 0)
			self._weights = B

			# Step 3: Count frequent item-pairs
			pair_counts = defaultdict(int)
			for u in range(X.shape[0]):
				items = np.where(X[u] == 1)[0]
				for i, j in combinations(items, 2):
					pair = (min(i, j), max(i, j))
					pair_counts[pair] += 1
			# Keep only frequent pairs
			self._pair_index = {
				p: idx
				for idx, (p, c) in enumerate(pair_counts.items())
				if c >= self._min_support
			}

			# Step 4: Build ZᵀZ implicitly
			num_pairs = len(self._pair_index)
			ZtZ = np.zeros((num_pairs, num_pairs), dtype=np.float32)

			for u in range(X.shape[0]):
				items = np.where(X[u] == 1)[0]
				active_pairs = []
				for i, j in combinations(items, 2):
					pair = (min(i, j), max(i, j))
					if pair in self._pair_index:
						active_pairs.append(self._pair_index[pair])
				# Fill ZᵀZ
				for a in active_pairs:
					ZtZ[a, a] += 1
				for a, b in combinations(active_pairs, 2):
					ZtZ[a, b] += 1
					ZtZ[b, a] += 1

			# Step 5: Solve for higher-order weights C
			H = ZtZ + self._l2 * np.eye(num_pairs)
			Q = np.linalg.inv(H)
			diag_Q = np.diag(Q)
			C = -Q / diag_Q[None, :]
			np.fill_diagonal(C, 0)
			self._C = C

		# Predict for a new user
		def predict(self, selected_items, filter_out_items, k):
			# Pairwise EASE part
			user_vector = np.zeros((self._items_count,), dtype=np.float32)
			for i in selected_items:
				user_vector[i] = 1.0
			preds = user_vector @ self._weights

			# Higher-order part
			active_pairs = []
			for i, j in combinations(selected_items, 2):
				pair = (min(i, j), max(i, j))
				if pair in self._pair_index:
					active_pairs.append(self._pair_index[pair])
			if active_pairs:
				z_user = np.zeros((len(self._pair_index),), dtype=np.float32)
				z_user[active_pairs] = 1.0
				preds += z_user @ self._C

			# Filter candidates
			candidates = np.setdiff1d(self._all_items, selected_items)
			candidates = np.setdiff1d(candidates, filter_out_items)

			candidates_by_prob = sorted(((preds[c], c) for c in candidates), reverse=True)
			return [x for _, x in candidates_by_prob][:k]

		@classmethod
		def name(cls):
			return "HOEASE"

		@classmethod
		def parameters(cls):
			return [
				Parameter(
					"l2",
					ParameterType.FLOAT,
					0.1,
					help="L2-norm regularization",
					help_key="hoease_l2_help",
				),
				Parameter(
					"positive_threshold",
					ParameterType.FLOAT,
					2.5,
					help="Threshold for conversion of n-ary rating into binary.",
				),
				Parameter(
					"min_support",
					ParameterType.INT,
					50,
					help="Minimum frequency for item-pair to be included as a higher-order feature.",
				),
			]
