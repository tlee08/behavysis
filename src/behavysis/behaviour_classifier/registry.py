"""Model registry: name → adapter factory."""

from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .adapter import BaseAdapter, SklearnAdapter, TorchAdapter
from .torch.architectures import CNN1, CNN2, DNN1, DNN2, DNN3

ModelFactory = Callable[[], BaseAdapter]

MODEL_REGISTRY: dict[str, ModelFactory] = {
    "rf": lambda: SklearnAdapter(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "logreg": lambda: SklearnAdapter(
        LogisticRegression(max_iter=1000, random_state=42),
    ),
    "dnn1": lambda: TorchAdapter(lambda nf: DNN1(nf, window_frames=0)),
    "dnn2": lambda: TorchAdapter(lambda nf: DNN2(nf, window_frames=0)),
    "dnn3": lambda: TorchAdapter(lambda nf: DNN3(nf, window_frames=0)),
    "cnn1": lambda: TorchAdapter(lambda nf: CNN1(nf, window_frames=2)),
    "cnn2": lambda: TorchAdapter(lambda nf: CNN2(nf, window_frames=2)),
}
