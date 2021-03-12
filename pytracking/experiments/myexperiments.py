from pytracking.evaluation import Tracker, get_dataset, trackerlist


def got_retrained_bl():
    # Evaluate GOT on a baseline KYS I trained
    trackers = trackerlist('kys', 'default', range(1))
    dataset = get_dataset('got10k_test', 'got10k_val')  # , 'got10k_ltrval')
    return trackers, dataset


def got_circuit_bl():
    # Evaluate GOT on a circuit KYS I trained
    trackers = trackerlist('kysc', 'default', range(1))
    dataset = get_dataset('got10k_test', 'got10k_val')  # , 'got10k_ltrval')
    return trackers, dataset

