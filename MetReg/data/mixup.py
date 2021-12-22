import tensorflow as tf


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    """Define the mixup technique function
        To perform the mixup routine, we create new virtual datasets using the training data from
    the same dataset, and apply a lambda value within the [0, 1] range sampled from a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
    — such that, for example, `new_x = lambda * x1 + (1 - lambda) * x2` (where
    `x1` and `x2` are images) and the same equation is applied to the labels as well.
    """
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    print(images_one.shape)
    batch_size = tf.shape(images_one)[0]
    print(batch_size)
    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    
    x_l = tf.reshape(l, (batch_size, 1, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1, 1, 1, 1))
    print(x_l.shape)
    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = tf.cast(images_one, tf.float64) * tf.cast(x_l, tf.float64) + tf.cast(images_two, tf.float64) * tf.cast((1-x_l), tf.float64)
    labels = tf.cast(labels_one, tf.float64) * tf.cast(y_l, tf.float64) + tf.cast(labels_two, tf.float64) * tf.cast((1-y_l), tf.float64)
    print(images.shape)
    print(labels.shape)
    return (images, labels)


def augment(x_train, y_train, x_valid, y_valid, BATCH_SIZE):
    print(BATCH_SIZE)
    train_ds_one = (tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(5000).batch(BATCH_SIZE))
    train_ds_two = (tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(5000).batch(BATCH_SIZE))

    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
    # First create the new dataset using our `mix_up` utility
    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2),
        num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_valid, y_valid)).batch(BATCH_SIZE)

    return train_ds_one, val_ds
