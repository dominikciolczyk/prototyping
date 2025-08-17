import os, pickle, math

def norm_cfg(cfg: dict):
    return tuple((k, round(float(v), 8)) for k, v in sorted(cfg.items()))

class FitnessCache:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.data = pickle.load(f)
            except Exception:
                self.data = {}

    def validate_cfg(self, cfg):
        for k, v in cfg.items():
            fv = float(v)
            assert math.isfinite(fv), f"Nieakceptowalna wartość dla {k}: {v}"

    def get(self, cfg: dict):
        self.validate_cfg(cfg)
        return self.data.get(norm_cfg(cfg))

    def set(self, cfg: dict, score: float):
        self.validate_cfg(cfg)
        self.data[norm_cfg(cfg)] = float(score)

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)
