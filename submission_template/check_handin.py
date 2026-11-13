# check_handin.py: run inside your hand-in folder, in the fresh environment
import time
import numpy as np
import stim
from decoder import decode

total = 0.0
result = np.zeros(2)
for i, basis in enumerate("XZ"):
    circuit = stim.Circuit.from_file(f"submission_template/{basis}_circuit.stim")
    events, flips = circuit.compile_detector_sampler(seed=1).sample(100_000, separate_observables=True)
    start = time.perf_counter()
    predictions = np.asarray(decode(circuit, events))
    total += time.perf_counter() - start
    assert predictions.shape == flips.shape and np.isin(predictions, (0, 1)).all()
    print(f"{basis}: LER {np.mean(predictions != flips):.5f}")
    result[i] = np.mean(predictions != flips)
print(f"X: {result[0]:.5f}, Z: {result[1]:.5f}")
print(f"decode time: {total:.1f} s")
