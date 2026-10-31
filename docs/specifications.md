# Array storage specifications

## `ArrayStorage.block_iter(from_block_named, *, step=1)`

- `step` is a positive integer specifying the number of rows per chunk.
- Iterating with zero or a negative step raises `ValueError("step must be positive")`, including for empty and scalar arrays. As with other generator errors, validation occurs when iteration begins.
- Positive steps visit every row in order, once. The final chunk may have fewer than `step` rows; trailing dimensions and dtype are preserved.
- Chunks are detached NumPy copies: modifying a yielded chunk does not change the stored block.
- Empty arrays yield no chunks. Scalar arrays yield one scalar NumPy array for any positive step.
- Missing blocks still raise `KeyError`; named members that are not Zarr arrays still raise `TypeError`.
