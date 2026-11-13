"""Your decoder for the code check.

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README).

Everything that happens inside decode() counts towards your runtime, including building matrices from the circuit
or loading trained weights. Training doesn't count, but it must be reproducible from the code you hand in.
"""
import numpy as np
import stim

def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    # open train.csv, which has columns
    # gen_{basis}_{shot}, {basis}, D0, D1, ..., D69, L0, L1, L2, L3
    # for 200001 rows. 1 header, 100000 X shots, 100000 Z shots. The first column is a unique id, the second column is the basis, the next 70 columns are the detector events, and the last 4 columns are the observable flips.
    train_data = np.loadtxt("submission_template/train.csv", delimiter=",", skiprows=1, usecols=range(2, 76), dtype=np.uint8)

    # Return the labels (L0, L1, L2, L3) from the row of train.csv most similar to each detector event.
    result = np.zeros((len(detector_events), 4), dtype=np.uint8)

    # return np.zeros((len(detector_events), 4), dtype=np.uint8)

    for i, row in enumerate(detector_events):
        # Python ranges are half-open: [0, 1000) and [100000, 101000).
        most_similar_index = 0
        best_similarity = -1
        for start, stop in ((0, 1000), (100000, 101000)):
            candidates = train_data[start:stop, :70]
            similarities = np.sum(candidates == row, axis=1)
            candidate_offset = int(np.argmax(similarities))
            candidate_similarity = int(similarities[candidate_offset])
            if candidate_similarity > best_similarity:
                best_similarity = candidate_similarity
                most_similar_index = start + candidate_offset

        result[i] = train_data[most_similar_index, 70:74]

    return result