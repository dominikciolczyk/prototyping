from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import random
import numpy as np
import torch

@dataclass
class SeqEntry:
    X: torch.Tensor          # (1, L, F)
    y: torch.Tensor          # (1, H, T)
    err: float               # błąd przy wstawieniu (np. AsymSmoothL1)
    t: int                   # znacznik czasu (krok online)
    regime_id: Optional[int] = None

class BaseReplayBuffer:
    """Wspólny interfejs dla wszystkich buforów."""
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))

    def push(self, X: torch.Tensor, y: torch.Tensor, step_t: int, err: float = 0.0,
             regime_id: Optional[int] = None) -> None:
        raise NotImplementedError

    def sample(self, k: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[np.ndarray]]:
        """Zwraca listy [X_i], [y_i] oraz ewentualne wagi IS (np.ndarray shape (m,)) lub None."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class NoReplayBuffer(BaseReplayBuffer):
    """Brak replay – zawsze zwraca pustą próbkę."""
    def __init__(self):
        super().__init__(capacity=0)

    def push(self, *args, **kwargs) -> None:
        return

    def sample(self, k: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[np.ndarray]]:
        return [], [], None

    def __len__(self) -> int:
        return 0


class SlidingWindowBuffer(BaseReplayBuffer):
    """Prosty bufor FIFO trzymający ostatnie N sekwencji; sampling losowy z okna."""
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._buf: deque[SeqEntry] = deque(maxlen=self.capacity)

    def push(self, X: torch.Tensor, y: torch.Tensor, step_t: int, err: float = 0.0,
             regime_id: Optional[int] = None) -> None:
        self._buf.append(SeqEntry(X.detach().cpu(), y.detach().cpu(), float(err), int(step_t), regime_id))

    def sample(self, k: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[np.ndarray]]:
        if len(self._buf) == 0 or k <= 0:
            return [], [], None
        k = min(k, len(self._buf))
        idxs = random.sample(range(len(self._buf)), k)
        Xs = [self._buf[i].X for i in idxs]
        ys = [self._buf[i].y for i in idxs]
        return Xs, ys, None

    def __len__(self) -> int:
        return len(self._buf)


class CyclicReplayBuffer(BaseReplayBuffer):
    """Duży bufor z uniform sampling (experience replay bez priorytetów)."""
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._buf: List[SeqEntry] = []
        self._next = 0

    def push(self, X: torch.Tensor, y: torch.Tensor, step_t: int, err: float = 0.0,
             regime_id: Optional[int] = None) -> None:
        entry = SeqEntry(X.detach().cpu(), y.detach().cpu(), float(err), int(step_t), regime_id)
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
        else:
            self._buf[self._next] = entry
            self._next = (self._next + 1) % self.capacity

    def sample(self, k: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[np.ndarray]]:
        if len(self._buf) == 0 or k <= 0:
            return [], [], None
        k = min(k, len(self._buf))
        idxs = random.sample(range(len(self._buf)), k)
        Xs = [self._buf[i].X for i in idxs]
        ys = [self._buf[i].y for i in idxs]
        return Xs, ys, None

    def __len__(self) -> int:
        return len(self._buf)

class RandomReplayBuffer(BaseReplayBuffer):
    """Replay buffer with random eviction and uniform sampling."""
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._buf: List[SeqEntry] = []

    def push(self, X: torch.Tensor, y: torch.Tensor, step_t: int, err: float = 0.0,
             regime_id: Optional[int] = None) -> None:
        entry = SeqEntry(X.detach().cpu(), y.detach().cpu(), float(err), int(step_t), regime_id)
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
        else:
            idx = random.randrange(self.capacity)  # random slot to overwrite
            self._buf[idx] = entry

    def sample(self, k: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[np.ndarray]]:
        if len(self._buf) == 0 or k <= 0:
            return [], [], None
        k = min(k, len(self._buf))
        idxs = random.sample(range(len(self._buf)), k)
        Xs = [self._buf[i].X for i in idxs]
        ys = [self._buf[i].y for i in idxs]
        return Xs, ys, None

    def __len__(self) -> int:
        return len(self._buf)


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """
    Priorytetyzowany replay (PER) bez sum-tree (prosty, O(n) sampling).
    Priorytet: p_i = (|err| + eps)^alpha * exp(-lambda * age_i)
    Wagi IS: w_i ∝ (N * P(i))^{-beta} / max(...)
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 eps: float = 1e-3, half_life: int = 1000):
        super().__init__(capacity)
        self._buf: List[SeqEntry] = []
        self._next = 0
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.half_life = max(1, int(half_life))  # w krokach
        # lambda = ln(2)/half_life
        self._lam = math.log(2.0) / self.half_life

    def push(self, X: torch.Tensor, y: torch.Tensor, step_t: int, err: float = 0.0,
             regime_id: Optional[int] = None) -> None:
        entry = SeqEntry(X.detach().cpu(), y.detach().cpu(), float(err), int(step_t), regime_id)
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
        else:
            self._buf[self._next] = entry
            self._next = (self._next + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._buf)

    def _priorities(self, now_t: int) -> np.ndarray:
        if len(self._buf) == 0:
            return np.zeros((0,), dtype=np.float64)
        ages = np.array([max(0, now_t - e.t) for e in self._buf], dtype=np.float64)
        errs = np.array([abs(e.err) for e in self._buf], dtype=np.float64)
        pri = (errs + self.eps) ** self.alpha * np.exp(-self._lam * ages)
        s = pri.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback: równe szanse
            pri = np.ones_like(pri) / len(pri)
        else:
            pri = pri / s
        return pri

    def sample(self, k: int, now_t: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[np.ndarray]]:
        if len(self._buf) == 0 or k <= 0:
            return [], [], None
        if now_t is None:
            now_t = max(e.t for e in self._buf)
        probs = self._priorities(now_t)
        k = min(k, len(self._buf))
        idxs = np.random.choice(len(self._buf), size=k, replace=False, p=probs)
        Xs = [self._buf[i].X for i in idxs]
        ys = [self._buf[i].y for i in idxs]
        # Importance Sampling weights
        N = len(self._buf)
        P = probs[idxs]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = (N * P) ** (-self.beta)
        w = w / (w.max() + 1e-8)
        return Xs, ys, w.astype(np.float32)

    # (opcjonalnie) metoda do aktualizacji błędów po uczeniu – nie wymagana:
    def update_errors(self, idxs: List[int], new_errs: List[float]) -> None:
        for i, e in zip(idxs, new_errs):
            if 0 <= i < len(self._buf):
                self._buf[i].err = float(e)
