# splits/

This folder contains:
- `train.txt`, `val.txt`, `test.txt`: one sample key (image path) per line
- `clients_alphaX_seedY/`: client partitions for FL
  - `client_0.txt` ... `client_{N-1}.txt`: subset of train keys per client

For `data.dataset_id: synthetic`, these files are auto-generated if missing at runtime.

