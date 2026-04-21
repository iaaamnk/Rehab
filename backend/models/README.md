Trained model weights are not committed to this repository.

To generate them, run:
  python -m backend.models.train

This requires the Kinect dataset in data_new/ (see data_new/README.md).

Files produced:
  - lstm_model.pth    (BiLSTM weights)
  - norm_params.npz   (per-feature z-score normalisation parameters)
