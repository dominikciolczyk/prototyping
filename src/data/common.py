from .preprocess_cpu_only import preprocess as cpu_only
from .preprocess_full_features import preprocess as full_features
def run_preprocessing(name, config):
    return {"cpu_only":  cpu_only,
            "full":      full_features}[name](config)
