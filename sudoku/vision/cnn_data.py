from __future__ import annotations


def load_mnist_digits_1_to_9(tf, np, validation_split=0.1, seed=42):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    mask = y_all > 0
    x_all = x_all[mask]
    y_all = y_all[mask] - 1

    x_all = x_all.astype("float32") / 255.0
    x_all = x_all[..., None]

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x_all))
    rng.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    split_index = int(len(x_all) * (1.0 - validation_split))
    x_train = x_all[:split_index]
    y_train = y_all[:split_index]
    x_val = x_all[split_index:]
    y_val = y_all[split_index:]

    return (x_train, y_train), (x_val, y_val)


def build_augmentation_pipeline(tf):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomTranslation(height_factor=0.12, width_factor=0.12),
            tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.GaussianNoise(0.08),
        ]
    )


def build_tf_datasets(tf, x_train, y_train, x_val, y_val, batch_size=128):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(min(len(x_train), 50000), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size)

    def random_invert(images, labels):
        probs = tf.random.uniform((tf.shape(images)[0], 1, 1, 1))
        inverted = 1.0 - images
        images = tf.where(probs < 0.18, inverted, images)
        return images, labels

    train_ds = train_ds.map(random_invert, num_parallel_calls=tf.data.AUTOTUNE)

    augmenter = build_augmentation_pipeline(tf)

    def augment(images, labels):
        return augmenter(images, training=True), labels

    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
