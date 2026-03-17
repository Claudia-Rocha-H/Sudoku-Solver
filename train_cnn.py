from __future__ import annotations

import argparse

from sudoku.vision.cnn_trainer import TrainConfig, train_and_save_digit_cnn


def _status(text, color="#2f4858"):
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Train and save the digit CNN model for Sudoku")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
    )

    result = train_and_save_digit_cnn(config=config, status_callback=_status)

    print("Training completed")
    print(f"Model: {result['model_path']}")
    print(f"Epochs run: {result['epochs_ran']}")
    print(f"Best val_accuracy: {result['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
