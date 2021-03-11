import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist


trackers = []
trackers.extend(trackerlist('kys', 'default', [0], 'KYS'))

dataset = get_dataset('got10k_val')
print_results(trackers, dataset, 'GOT10k', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
plot_results(trackers, dataset, 'GOT10k', merge_results=True, plot_types=('success', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)


dataset = get_dataset('got10k_test')
print_results(trackers, dataset, 'GOT10k', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
plot_results(trackers, dataset, 'GOT10k', merge_results=True, plot_types=('success', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

