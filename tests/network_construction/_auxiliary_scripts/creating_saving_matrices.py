import numpy as np

m1_2_int = np.array(
    [[0, 2],
    [2, 0]]
)
m1_2_prob = np.array(
    [[0, 1],
    [1, 0]]
)
np.save("/home/yehor/Desktop/AllostericPathwayAnalyzer/tests/small_test/output/tmc/1-2/interactions.npy", m1_2_int)
np.save("/home/yehor/Desktop/AllostericPathwayAnalyzer/tests/small_test/output/tmc/1-2/probabilities.npy", m1_2_prob)

m3_4_int = np.array(
    [[0, 10],
    [10, 0]]
)
m3_4_prob = np.array(
    [[0, 1],
    [1, 0]]
)
np.save("/home/yehor/Desktop/AllostericPathwayAnalyzer/tests/small_test/output/tmc/3-4/interactions.npy", m3_4_int)
np.save("/home/yehor/Desktop/AllostericPathwayAnalyzer/tests/small_test/output/tmc/3-4/probabilities.npy", m3_4_prob)
