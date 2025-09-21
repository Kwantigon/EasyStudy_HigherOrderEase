from abc import ABC
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from plugins.fastcompare.algo.algorithm_base import (
	AlgorithmBase,
	Parameter,
	ParameterType,
)


class HOEASE_NoADMM(AlgorithmBase, ABC):
	"""
	Higher-Order EASE without ADMM constraint.
	- Keeps memory-efficient construction of XtZ and ZtZ.
	- Learns B (pairwise) same as EASE.
	- Learns C (higher-order) by ridge regression (no projection).
	"""

	def __init__(
		self,
		loader,
		positive_threshold,
		l2,
		lambda_c=None,
		min_support=500,
		max_pairs=10,
		**kwargs,
	):
		self._ratings_df = loader.ratings_df
		self._loader = loader
		self._all_items = np.asarray(sorted(self._ratings_df.item.unique()))

		self._rating_matrix = (
			self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
			.fillna(0)
			.values
		)

		self._threshold = positive_threshold
		self._l2 = l2
		self._lambda_c = l2 if lambda_c is None else lambda_c
		self._min_support = min_support
		self._max_pairs = max_pairs
		self._items_count = np.shape(self._rating_matrix)[1]

		# Learned params
		self._B = None
		self._C = None
		self._pair_index = {}
		self._index_pair = []

	def fit(self):
		# Step 1: binarize X
		X = np.where(self._rating_matrix >= self._threshold, 1.0, 0.0).astype(np.float32)
		num_users, num_items = X.shape
		assert num_items == self._items_count

		# === Learn B (EASE closed form) ===
		G = X.T @ X
		G += self._l2 * np.eye(self._items_count, dtype=np.float32)
		P = np.linalg.inv(G)
		diag_P = np.diag(P)
		diag_P_safe = np.where(diag_P == 0.0, 1e-12, diag_P)
		B = np.eye(self._items_count, dtype=np.float32) - P.dot(np.diag(1.0 / diag_P_safe))
		np.fill_diagonal(B, 0.0)
		self._B = B

		# === Build frequent pairs ===
		pair_counts = defaultdict(int)
		for u in range(num_users):
			items = np.nonzero(X[u])[0]
			for i, j in combinations(items, 2):
				pair = (min(i, j), max(i, j))
				pair_counts[pair] += 1

		frequent_pairs = [(p, c) for p, c in pair_counts.items() if c >= self._min_support]
		frequent_pairs.sort(key=lambda x: -x[1])
		if self._max_pairs is not None:
			frequent_pairs = frequent_pairs[: self._max_pairs]

		self._pair_index = {p: idx for idx, (p, _) in enumerate(frequent_pairs)}
		self._index_pair = [p for p, _ in frequent_pairs]
		m = len(self._index_pair)

		if m == 0:
			self._C = np.zeros((0, self._items_count), dtype=np.float32)
			return

		# === Build XtZ and ZtZ implicitly ===
		XtZ_rows, XtZ_cols, XtZ_data = [], [], []
		ZtZ_rows, ZtZ_cols, ZtZ_data = [], [], []
		pindex = self._pair_index

		for u in range(num_users):
			items = np.nonzero(X[u])[0]
			active_pairs = []
			for i, j in combinations(items, 2):
				r = pindex.get((min(i, j), max(i, j)))
				if r is not None:
					active_pairs.append(r)

			if not active_pairs:
				continue

			for k in items:
				for r in active_pairs:
					XtZ_rows.append(k)
					XtZ_cols.append(r)
					XtZ_data.append(1.0)

			for a in active_pairs:
				ZtZ_rows.append(a)
				ZtZ_cols.append(a)
				ZtZ_data.append(1.0)
			for a, b in combinations(active_pairs, 2):
				ZtZ_rows += [a, b]
				ZtZ_cols += [b, a]
				ZtZ_data += [1.0, 1.0]

		XtZ = sp.coo_matrix((XtZ_data, (XtZ_rows, XtZ_cols)), shape=(self._items_count, m)).tocsr()
		ZtZ = sp.coo_matrix((ZtZ_data, (ZtZ_rows, ZtZ_cols)), shape=(m, m)).tocsr()

		# === Learn C (ridge regression, no constraint) ===
		H = ZtZ + self._lambda_c * sp.eye(m, format="csr")
		C = np.zeros((m, self._items_count), dtype=np.float32)
		ZtX = XtZ.T
		rhs = ZtX - ZtX.dot(self._B)

		# Solve H * C[:,j] = rhs[:,j] for each item j
		solve_H = splinalg.factorized(H.tocsc())
		for j in range(self._items_count):
			# rhs[:, j] could be numpy matrix/2D; make sure it’s 1-D float64
			b_col = np.asarray(rhs[:, j]).astype(np.float64).ravel()
			x_col = solve_H(b_col)
			C[:, j] = np.asarray(x_col).ravel().astype(np.float32)

		self._C = C

	def predict(self, selected_items, filter_out_items, k):
		user_vector = np.zeros((self._items_count,), dtype=np.float32)
		for i in selected_items:
			user_vector[i] = 1.0

		preds = user_vector.dot(self._B)

		if self._C is not None and self._C.shape[0] > 0 and len(selected_items) >= 2:
			active_pairs = []
			for i, j in combinations(selected_items, 2):
				idx = self._pair_index.get((min(i, j), max(i, j)))
				if idx is not None:
					active_pairs.append(idx)
			if active_pairs:
				z_user = np.zeros((len(self._index_pair),), dtype=np.float32)
				z_user[active_pairs] = 1.0
				preds += z_user.dot(self._C)

		candidates = np.setdiff1d(self._all_items, np.asarray(selected_items))
		if len(filter_out_items) > 0:
			candidates = np.setdiff1d(candidates, np.asarray(filter_out_items))

		candidates_by_prob = sorted(((preds[int(c)], int(c)) for c in candidates), reverse=True)
		return [x for _, x in candidates_by_prob][:k]

	@classmethod
	def name(cls):
		return "HOEASE_NoADMM"

	@classmethod
	def parameters(cls):
		return [
			Parameter("l2", ParameterType.FLOAT, 0.1, help="lambda_B regularization"),
			Parameter("lambda_c", ParameterType.FLOAT, 0.1, help="lambda_C regularization"),
			Parameter("positive_threshold", ParameterType.FLOAT, 2.5, help="Threshold for positive"),
			Parameter("min_support", ParameterType.INT, 100, help="Minimum support for a pair"),
			Parameter("max_pairs", ParameterType.INT, None, help="Optional cap on number of pairs"),
		]
