# Repo containing code examples for pytorch distributed training

1. Multi GPU training with torch parallel (Has reduced performance due to GIL Global interpreter lock due to the way Cpython handles multi threading in a single process). Easiest to modify from single to multi gpu (one liner)
2. Multi GPU Training with torch data distributed parallel. Better performance but requires more code changes
