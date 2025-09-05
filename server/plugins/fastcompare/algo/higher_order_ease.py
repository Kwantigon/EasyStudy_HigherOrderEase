import numpy as np
from abc import ABC
from itertools import combinations
from typing import Iterable, List, Tuple, Dict, Optional

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class HigherOrderEASE(AlgorithmBase, ABC):
    """
    EASE with higher‑order (pair) interactions, following
      Steck & Liang (RecSys 2021): "Negative Interactions for Improved Collaborative Filtering:
      Don't go Deeper, go Higher" (DOI: 10.1145/3460231.3474273).

    Objective (implicit, binary X):
        min_{B,D} || X - X B - Z D ||_F^2 + \lambda (||B||_F^2 + ||D||_F^2)
        s.t. diag(B) = 0 and D_{(i,k),j} = 0 if j in {i,k}

    Where
        X \in R^{n_users x m_items} is binary interactions (thresholded),
        Z \in R^{n_users x r_pairs} contains pairwise features z_{u,(i,k)} = X_{u,i} * X_{u,k} (i<k),
        B \in R^{m x m} are item->item weights (as in EASE),
        D \in R^{r x m} are pair->item weights (higher‑order).

    Notes
    -----
    * This implementation keeps a *sparse* representation for Z by default
      and selects only frequent pairs using a min_support and optional cap (max_pairs).
    * Closed‑form is solved per‑target j with masked ridge regression over allowed features,
      which mirrors the constrained solution in the paper while remaining practical.
    * For very large catalogs, tune min_pair_support and/or max_pairs.
    """

    def __init__(
        self,
        loader,
        positive_threshold: float = 2.5,
        l2: float = 0.1,
        min_pair_support: int = 5,
        max_pairs: Optional[int] = None,
        use_sparse: bool = True,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self._loader = loader
        self._ratings_df = loader.ratings_df
        self._all_items = self._ratings_df.item.unique()
        self._threshold = positive_threshold
        self._l2 = float(l2)
        self._min_pair_support = int(min_pair_support)
        self._max_pairs = max_pairs
        self._use_sparse = use_sparse and (sp is not None)
        self._rng = np.random.default_rng(random_state)

        # Rating/user‑item matrix
        self._rating_matrix = (
            self._loader.ratings_df.pivot(index="user", columns="item", values="rating")
            .fillna(0)
            .values
        )
        self._n_users, self._m_items = self._rating_matrix.shape

        # Learned weights
        self._B = None  # (m x m)
        self._D = None  # (r x m)

        # Pair bookkeeping
        self._pair_index: Dict[Tuple[int, int], int] = {}
        self._pairs: List[Tuple[int, int]] = []

    # ---------- utilities ----------
    @staticmethod
    def _make_binary(X: np.ndarray, thr: float) -> np.ndarray:
        return (X >= thr).astype(np.float32)

    def _select_pairs(self, X_bin: np.ndarray) -> List[Tuple[int, int]]:
        # Co‑occurrence (Gram) counts
        C = X_bin.T @ X_bin  # (m x m)
        C = np.triu(C, k=1)
        ii, kk = np.where(C >= self._min_pair_support)
        pairs = list(zip(ii.tolist(), kk.tolist()))
        if self._max_pairs is not None and len(pairs) > self._max_pairs:
            self._rng.shuffle(pairs)
            pairs = pairs[: self._max_pairs]
        # sort pairs for deterministic order
        pairs.sort()
        return pairs

    def _build_Z_sparse(self, X_bin: np.ndarray, pairs: List[Tuple[int, int]]):
        # Build Z as CSR without materializing dense
        rows = []
        cols = []
        data = []
        for col_id, (i, k) in enumerate(pairs):
            prod = X_bin[:, i] * X_bin[:, k]
            nz = np.nonzero(prod)[0]
            rows.extend(nz.tolist())
            cols.extend([col_id] * nz.size)
            data.extend([1.0] * nz.size)
        Z = sp.csr_matrix((data, (rows, cols)), shape=(self._n_users, len(pairs)), dtype=np.float32)
        return Z

    def _build_Z_dense(self, X_bin: np.ndarray, pairs: List[Tuple[int, int]]):
        Z = np.empty((self._n_users, len(pairs)), dtype=np.float32)
        for col_id, (i, k) in enumerate(pairs):
            Z[:, col_id] = X_bin[:, i] * X_bin[:, k]
        return Z

    # ---------- training ----------
    def fit(self):
        X = self._make_binary(self._rating_matrix, self._threshold)

        # 1) Standard EASE closed‑form for B
        G = X.T @ X
        G = G + self._l2 * np.eye(self._m_items, dtype=np.float32)
        P = np.linalg.inv(G)
        diagP = np.diag(P)
        B = -P / diagP[:, None]
        np.fill_diagonal(B, 0.0)
        self._B = B.astype(np.float32)

        # 2) Higher‑order (pair) features
        self._pairs = self._select_pairs(X)
        self._pair_index = {p: idx for idx, p in enumerate(self._pairs)}
        r = len(self._pairs)
        if r == 0:
            # nothing to learn
            self._D = np.zeros((0, self._m_items), dtype=np.float32)
            return

        Z = self._build_Z_sparse(X, self._pairs) if self._use_sparse else self._build_Z_dense(X, self._pairs)

        # 3) Solve ridge per target with constraints:
        #    forbid using x_j and pair features (i,k) where j in {i,k}.
        #    Allowed feature matrix F_S = [X_{-j} | Z_{pairs_without_j}].
        D = np.zeros((r, self._m_items), dtype=np.float32)

        # Precompute normal blocks
        XtX = G  # already computed
        if self._use_sparse:
            XtZ = X.T @ Z  # (m x r) dense
            ZtZ = (Z.T @ Z).toarray()
        else:
            XtZ = X.T @ Z  # (m x r)
            ZtZ = Z.T @ Z  # (r x r)

        I_m = np.eye(self._m_items, dtype=np.float32)

        for j in range(self._m_items):
            # Mask out item j (diag(B)=0 is already enforced above)
            allow_item = np.ones(self._m_items, dtype=bool)
            allow_item[j] = False

            # Mask out pairs containing j
            allow_pair = np.ones(r, dtype=bool)
            for idx, (i, k) in enumerate(self._pairs):
                if i == j or k == j:
                    allow_pair[idx] = False

            idx_items = np.where(allow_item)[0]
            idx_pairs = np.where(allow_pair)[0]

            # Build normal equations for allowed subset
            A11 = XtX[np.ix_(idx_items, idx_items)]
            A12 = XtZ[np.ix_(idx_items, idx_pairs)]
            A21 = A12.T
            A22 = ZtZ[np.ix_(idx_pairs, idx_pairs)]

            # Regularize
            A11 = A11 + self._l2 * np.eye(A11.shape[0], dtype=np.float32)
            A22 = A22 + self._l2 * np.eye(A22.shape[0], dtype=np.float32)

            # RHS: X^T y and Z^T y where y = X[:, j]
            y = X[:, j:j+1]
            b1 = (X[:, idx_items].T @ y).astype(np.float32)
            if self._use_sparse:
                b2 = (Z[:, idx_pairs].T @ y).toarray().astype(np.float32)
            else:
                b2 = (Z[:, idx_pairs].T @ y).astype(np.float32)

            # Solve block system [A11 A12; A21 A22] [c; d] = [b1; b2]
            # Use Schur complement for numerical stability when r is large
            # Solve A11 c + A12 d = b1
            #       A21 c + A22 d = b2
            # => (A22 - A21 A11^{-1} A12) d = b2 - A21 A11^{-1} b1
            c = None
            try:
                A11_inv = np.linalg.inv(A11)
                S = A22 - A21 @ (A11_inv @ A12)
                rhs = b2 - A21 @ (A11_inv @ b1)
                d = np.linalg.solve(S, rhs)
                # back‑substitute for c
                c = A11_inv @ (b1 - A12 @ d)
            except np.linalg.LinAlgError:
                # Fallback to direct solve on the full block
                top = np.concatenate([A11, A12], axis=1)
                bottom = np.concatenate([A21, A22], axis=1)
                A = np.concatenate([top, bottom], axis=0)
                rhs = np.concatenate([b1, b2], axis=0)
                sol = np.linalg.solve(A, rhs)
                c = sol[: A11.shape[0]]
                d = sol[A11.shape[0] :]

            # Store only the pair weights for item j
            D[idx_pairs, j:j+1] = d.astype(np.float32)

        self._D = D

    # ---------- inference ----------
    def _user_features(self, selected_items: Iterable[int]):
        x = np.zeros((self._m_items,), dtype=np.float32)
        for i in selected_items:
            if 0 <= i < self._m_items:
                x[i] = 1.0
        if self._D is None or self._D.shape[0] == 0:
            return x, None
        # Build z only for existing pairs
        if len(selected_items) < 2:
            z = None
        else:
            sel = sorted(set(int(i) for i in selected_items if 0 <= int(i) < self._m_items))
            present = set(combinations(sel, 2))
            r = self._D.shape[0]
            if self._use_sparse and sp is not None:
                # sparse z
                cols = []
                data = []
                for (i, k) in present:
                    idx = self._pair_index.get((i, k))
                    if idx is not None:
                        cols.append(idx)
                        data.append(1.0)
                if cols:
                    z = sp.csr_matrix((data, ([0] * len(cols), cols)), shape=(1, r), dtype=np.float32)
                else:
                    z = None
            else:
                z = np.zeros((r,), dtype=np.float32)
                for (i, k) in present:
                    idx = self._pair_index.get((i, k))
                    if idx is not None:
                        z[idx] = 1.0
        return x, z

    def predict(self, selected_items: Iterable[int], filter_out_items: Iterable[int], k: int) -> List[int]:
        seen = set(int(i) for i in selected_items)
        banned = set(int(i) for i in filter_out_items)

        # Candidate pool
        candidates = [i for i in self._all_items if i not in seen and i not in banned]
        if not selected_items:
            if len(candidates) <= k:
                return candidates
            idx = self._rng.choice(len(candidates), size=k, replace=False)
            return [candidates[i] for i in idx]

        x, z = self._user_features(selected_items)

        # Scores: x B + z D
        scores = x @ self._B
        if z is not None and self._D is not None and self._D.shape[0] > 0:
            if self._use_sparse and sp is not None and sp.issparse(z):
                scores = scores + (z @ self._D).A.ravel()
            else:
                scores = scores + z @ self._D

        # rank candidates
        cand_idx = np.array(candidates, dtype=int)
        cand_scores = scores[cand_idx]
        order = np.argsort(-cand_scores)
        top = cand_idx[order][:k]
        return top.tolist()

    # ---------- metadata ----------
    @classmethod
    def name(cls):
        return "HigherOrderEASE"

    @classmethod
    def parameters(cls):
        return [
            Parameter("l2", ParameterType.FLOAT, 0.1, help="L2 regularization for both B and D."),
            Parameter("positive_threshold", ParameterType.FLOAT, 2.5, help="Threshold to binarize implicit ratings."),
            Parameter("min_pair_support", ParameterType.INT, 5, help="Minimum user co‑occurrence for an item pair (i,k) to be included."),
            Parameter("max_pairs", ParameterType.INT, 200000, help="Optional cap on number of pair features; set None for no cap."),
            Parameter("use_sparse", ParameterType.BOOL, True, help="Use scipy.sparse for Z and inference."),
        ]
