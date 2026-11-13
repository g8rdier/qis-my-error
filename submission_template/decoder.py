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
    # open train.csv, skipping the non-numeric id and basis columns
    train_data = np.loadtxt("submission_template/train.csv", delimiter=",", skiprows=1, usecols=range(2, 76), dtype=np.uint8)

    #return the L0,L1,L2,L3 in the row of train.csv that are most similar to the detector_events, for each row of detector_events. You can use any similarity metric you want, but the simplest is to count how many columns are equal.
    result = np.zeros((len(detector_events), 4), dtype=np.uint8)


    for i, row in enumerate(detector_events):
        # find the row in train_data that is most similar to row
        similarities = np.sum(train_data[:, :70] == row, axis=1)
        most_similar_row = np.argmax(similarities)
        result[i] = train_data[most_similar_row, 70:74]
        #print result so far
        if i % 100 == 0:
            print(f"decoded {i} shots")


    return result
    return np.zeros((len(detector_events), 4), dtype=np.uint8)