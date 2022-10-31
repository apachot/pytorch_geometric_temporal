import json
import urllib
import numpy as np
from ..signal import StaticGraphTemporalSignal


class ProductspaceDatasetLoader(object):
    """A dataset of Harvard Product Space exports between 1995
    and 2020. The underlying graph is static - vertices are product classes in the HS nomenclature and
    edges are proximities. Vertex features are lagged yearly (we included 2 lags). The target is the yearly amount of
    exports the upcoming year. Our dataset consist of 25 snapshots (years).
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        f = open ('./dataset/productspace.json', "r")
        self._dataset = json.loads(f.read())
        f.close()
        
    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["X"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 1) -> StaticGraphTemporalSignal:
        """Returning the Product Space demand data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The PedalMe dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
