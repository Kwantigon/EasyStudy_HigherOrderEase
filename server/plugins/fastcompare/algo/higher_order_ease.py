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


class HigherOrderEASE(AlgorithmBase, ABC):
	"""
	Implementation of the higher-order EASE algorithm according to the
	paper: https://dl.acm.org/doi/pdf/10.1145/3460231.3474273
	"""

	def __init__(
		self,
		loader,
		positive_threshold,
		l2,
		lambda_c=None,
		min_support=500,
		max_pairs=10,
		rho=1.0,
		admm_iters=20,
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
		self._l2 = float(l2)
		self._lambda_c = float(l2 if lambda_c is None else lambda_c)
		self._min_support = int(min_support)
		self._max_pairs = int(max_pairs)
		self._items_count = np.shape(self._rating_matrix)[1]
		
		self._rho = float(rho)
		self._admm_iters = int(admm_iters)

		# Learned params
		self._B = None	# B is the same as EASE.
		self._C = None	# C is the new higher-order EASE matrix.
		self._pair_index = {}
		self._index_pair = []

	def fit(self):
		# Binarize the user-items matrix X.
		X = np.where(self._rating_matrix >= self._threshold, 1.0, 0.0).astype(np.float64)
		num_users, num_items = X.shape
		assert num_items == self._items_count

		# Learn B just like in EASE.
		G = X.T @ X
		G += self._l2 * np.eye(num_items, dtype=np.float64)
		P = np.linalg.inv(G)
		diag_P = np.diag(P).copy()
		diag_P_safe = np.where(diag_P == 0.0, 1e-12, diag_P) # To avoid division by zero.
		B = np.eye(num_items, dtype=np.float64) - P.dot(np.diag(1.0 / diag_P_safe))
		np.fill_diagonal(B, 0.0)
		self._B = B

		# Count item pair occurences.
		pair_counts = defaultdict(int)	# keys are pairs of items (i,j), and values are how many users have that pair.
		for u in range(num_users):
			# For each user, look at the items they interacted with and record the pairs.
			items = np.nonzero(X[u])[0]
			for i, j in combinations(items, 2):
				pair = (min(i, j), max(i, j))
				pair_counts[pair] += 1

		# Prune pairs to save memory.
		# Only keep pairs that have occured at least `min_support` times.
		frequent_pairs = [(p, c) for p, c in pair_counts.items() if c >= self._min_support]
		# Sort pairs by frequency.
		frequent_pairs.sort(key=lambda x: -x[1])
		# Only keep `max_pairs` number of most frequent pairs.
		if self._max_pairs is not None:
			frequent_pairs = frequent_pairs[: self._max_pairs]

		# Indexing structure for pairs.
		# For example, (item 1, item 3) -> row 42.
		self._pair_index = {p: idx for idx, (p, _) in enumerate(frequent_pairs)}
		# Row 42 -> (item 1, item 3).
		self._index_pair = [p for p, _ in frequent_pairs]
		m = len(self._index_pair)

		# If there are no pairs left, fall back to plain EASE
		if m == 0:
			self._C = np.zeros((0, num_items), dtype=np.float32)
			return

		# Build XtZ and ZtZ using sparse matrices.
		# XtZ[k, r] = the number of users who had item k together with pair r.
		XtZ_rows, XtZ_cols, XtZ_data = [], [], []
		# ZtZ[a, b] = the number of users who had both pair a and pair b.
		ZtZ_rows, ZtZ_cols, ZtZ_data = [], [], []

		for u in range(num_users):
			items = np.nonzero(X[u])[0]
			active_pairs = []
			for i, j in combinations(items, 2):
				r = self._pair_index.get((min(i, j), max(i, j)))
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

		# These matrices will be used later in ADMM.
		XtZ = sp.coo_matrix((XtZ_data, (XtZ_rows, XtZ_cols)), shape=(num_items, m), dtype=np.float64).tocsr()
		ZtZ = sp.coo_matrix((ZtZ_data, (ZtZ_rows, ZtZ_cols)), shape=(m, m), dtype=np.float64).tocsr()

		ZtX = XtZ.T.tocsr() # Transpose for ADMM loops.
		# For row r (pair (i,j)), we must keep C[r, i] = C[r, j] = 0.
		pair_left = np.array([ij[0] for ij in self._index_pair], dtype=np.int64)
		pair_right = np.array([ij[1] for ij in self._index_pair], dtype=np.int64)
		row_idx = np.arange(m, dtype=np.int64)

		# ADMM state
		C = np.zeros((m, num_items), dtype=np.float64)	# coefficients for higher-order features
		D = np.zeros_like(C)
		Gamma = np.zeros_like(C)

		# Single factorization reused over all ADMM iterations
		# H = ZtZ + (λc ​+ ρ)Im ​∈ Rm×m
		H = (ZtZ + (self._lambda_c + self._rho) * sp.eye(m, format="csr", dtype=np.float64)).tocsc()
		solve_H = splinalg.factorized(H)  # solves H x = b for many b's cheaply

		# ADMM loop enforcing C ⊙ M = 0 and updating B with C.
		I_n = np.eye(num_items, dtype=np.float64)
		for it in range(self._admm_iters):
			# C-step: (ZtZ + (λc + ρ)I) C = ZtX (I - B) + ρ(D - Gamma)
			# compute RHS once as a dense (m x n)
			RHS = ZtX - ZtX.dot(self._B) + self._rho * (D - Gamma)
			# Solve column-wise using the cached factorization.
			for j in range(num_items):
				b_col = np.asarray(RHS[:, j]).ravel()
				C[:, j] = solve_H(b_col)

			# D = (1 - M) ⊙ C  (zero out the two columns per pair)
			D[:, :] = C
			D[row_idx, pair_left] = 0.0
			D[row_idx, pair_right] = 0.0

			# Dual update
			Gamma += (C - D)

			# To enforce zero diagonal, add a diagonal correction: B=I−P(A−diag(η)).
			XtZC = XtZ.dot(C)
			PC = P.dot(XtZC)
			diag_eta = (np.diag(PC) - 1.0) / diag_P_safe
			A = XtZC.copy()
			A[np.arange(num_items), np.arange(num_items)] -= diag_eta  # A = XtZC - diag(η)
			self._B = I_n - P.dot(A)
			
			# Floating-point round-off can bleed tiny values onto the diagonal.
			# Enforce the EASE constraint by setting 0.
			np.fill_diagonal(self._B, 0.0)

		# Store learned params
		self._C = C.astype(np.float32)
		self._B = self._B.astype(np.float32)

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
		return "Higher-order EASE"

	@classmethod
	def parameters(cls):
		return [
			Parameter("l2", ParameterType.FLOAT, 0.1, help="lambda_B regularization"),
			Parameter("lambda_c", ParameterType.FLOAT, 0.1, help="lambda_C regularization"),
			Parameter("positive_threshold", ParameterType.FLOAT, 2.5, help="Threshold for positive"),
			Parameter("min_support", ParameterType.INT, 500, help="Minimum support for a pair"),
			Parameter("max_pairs", ParameterType.INT, 10, help="Optional cap on number of pairs"),
			Parameter("rho", ParameterType.FLOAT, 1.0, help="ADMM penalty parameter"),
			Parameter("admm_iters", ParameterType.INT, 20, help="Number of ADMM iterations"),
		]
	