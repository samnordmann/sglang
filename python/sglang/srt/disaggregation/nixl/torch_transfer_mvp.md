# Core transfer experiment

Set `SGLANG_DISAGGREGATION_NIXL_USE_TORCH_TRANSFER=1` on test workers. Requires
the PyTorch `_transfer` prototype and NIXL `torch_transfer` provider. Default: off.

The NIXL agent is wrapped after creation, retaining its configured backends and
strict thread synchronization. Registration, prepared indexed WRITE, fallback
transfers and notifications route through Core. Bootstrap, allocation ownership,
CUDA ordering, staging and abort protocols are unchanged.

This is a **success-path review prototype**, not a production mode. The manager
must keep raw-pointer allocations stable. Active cancellation and failure recovery
are outside the MVP. GPU serving/staging and throughput remain unvalidated.
