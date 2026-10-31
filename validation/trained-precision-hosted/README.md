# Hosted historical native regression replay

The workflow fetches the existing public CPU v6 release, checks every numbered
part and the complete archive, and restores only the eight checkpoints and their
selected input rows authenticated by the development inventory. No executable
code is extracted or executed from that release. Historical failures are unchanged.

It builds native policy 15 on hosted macOS, pins NumPy 2.5.2 / PyTorch 2.14.1,
forces NEON, then replays all 89 historical anchors with B16 logits, loss and every
gradient. Six bounded width-1024 shards and one width-512 shard retain exact raw
arrays in the lossless member store. The independent collection auditor checks
raw numerical decisions, dual FP64 oracles, exact repeats, complete ordered
coverage and actual native dispatch. Failures remain retained. Artifacts expire
after seven days; the frozen public inputs, source and recipe permit regeneration.

These checks run on the hosted M1 virtual CPU. They neither qualify M4/SME nor
supply performance measurements. No GPU, fresh development trajectory, reserved
stream, formal fit or framework runtime repair is executed here. The initial
replay requires 24 GiB free after preparation and preserves an 8 GiB reserve.
The largest evidence allocation stays hosted; it is not automatically downloaded.
