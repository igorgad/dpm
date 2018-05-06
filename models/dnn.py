
import tensorflow as tf

lr = 0.001


def model_func(features, labels, mode, params):

    features['stfts'] = tf.contrib.signal.stft(features['samples'], frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'])

    top = tf.feature_column.input_layer(features, params["feature_columns"])
    for units in params.get('hidden_units'):
        top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)
    output_layer = tf.layers.dense(inputs=top, units=params['label_size'])

    complex_output = tf.reshape(tf.cast(output_layer, tf.complex64), tf.shape(features['stfts']))
    predictions = tf.contrib.signal.inverse_stft(complex_output,
                                                 frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'],
                                                 window_fn=tf.contrib.signal.inverse_stft_window_fn(params['frame_step']))

    features['predicted_samples'] = predictions

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=features)

    average_loss = tf.losses.mean_squared_error(labels, predictions)

    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    eval_metrics = {"rmse": rmse}
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)
