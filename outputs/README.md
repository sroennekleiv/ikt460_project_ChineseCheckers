This folder contains generated files from training and evaluation runs.

## Layout

- `models/afterstate/`
  - active Afterstate checkpoints such as `best.pth`, `trained.pth`, and `final.pth`
- `models/alphazero/`
  - active AlphaZero checkpoints such as `best.pth`, `external_best.pth`, `trained.pth`, and `final.pth`
- `models/archive/`
  - older snapshots, backups, and local checkpoints you do not want mixed into the active folders
- `plots/`
  - current report-facing plots for Afterstate and AlphaZero
- `plots/archive/`
  - older or diagnostic plots kept locally for reference
- `logs/`
  - saved console logs from important runs

Only Afterstate and AlphaZero outputs should live here now.
