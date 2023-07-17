from scipy.stats import pearsonr, spearmanr, kendalltau, entropy
from scipy.signal import coherence
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
import numpy as np


def compute_coherence_score(curve1, curve2, fs=1/(15*60), interest_range=None):
    """

    :param curve1: Pandas Series that represent the first time series
    :param curve2: Pandas Series that represent the second time series
    :param fs: Sampling frequency
    :param interest_range: tuple that describe the range of interest
    :return: summarize the magnitude score for each frequency of interest obtained with a cross spectral density
    """
    f, Cxy = coherence(curve1, curve2, fs=fs)

    if interest_range is not None:
        freq_range = (f >= interest_range[0]) & (f <= interest_range[1])
    else:
        freq_range = np.ones(f.shape, dtype='bool')
    return np.trapz(np.abs(Cxy[freq_range]), f[freq_range])


pearson = lambda curve1, curve2: pearsonr(curve1, curve2)[0]
p_value = lambda curve1, curve2: pearsonr(curve1, curve2)[1]
spearman = lambda curve1, curve2: spearmanr(curve1, curve2)[0]
kendall = lambda curve1, curve2: kendalltau(curve1, curve2)[0]
cross_correlation = lambda curve1, curve2: np.correlate(curve1, curve2)[0]
mutual_info = lambda curve1, curve2: mutual_info_score(curve1, curve2)
Kullback_Leibler = lambda curve1, curve2: entropy(curve1, curve2)
cosine_distance = lambda curve1, curve2: cosine_distances(curve1.values.reshape(1, -1),
                                                          curve2.values.reshape(1, -1))[0, 0]
