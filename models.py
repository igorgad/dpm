
import tensorflow as tf
import numpy as np
import ITL as ITL


############################################################################################################################
############################################################################################################################
def dnn(features, labels, mode, params):
    stft_shape = [(params['audio_samples_length'] - params['frame_length']) // params['frame_step'] + 1, params['fft_length'] // 2 + 1]
    stft_size = int(np.prod(stft_shape))

    hidden_units = [stft_size, stft_size, stft_size]
    output_size = stft_size

    features['input_stfts'] = tf.contrib.signal.stft(features['input_samples'], frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'])
    features['vst_stfts'] = tf.contrib.signal.stft(features['vst_samples'], frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'])

    features['input_stfts/mag'] = tf.image.per_image_standardization(tf.abs(features['input_stfts']))
    features['input_stfts/phase'] = tf.angle(features['input_stfts'])
    features['vst_stfts/mag'] = tf.image.per_image_standardization(tf.abs(features['vst_stfts']))
    features['vst_stfts/phase'] = tf.angle(features['vst_stfts'])

    params['feature_columns'].append(tf.feature_column.numeric_column(key='input_stfts/mag', shape=stft_shape))
    top = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in hidden_units:
        top = tf.layers.dense(inputs=top, units=units, use_bias=False)
        top = tf.layers.batch_normalization(top, center=True, scale=False)
        top = tf.nn.leaky_relu(top)
    output_layer = tf.layers.dense(inputs=top, units=output_size)

    features['predicted_stfts/mag'] = tf.reshape(output_layer, tf.shape(features['vst_stfts/mag']))
    features['predicted_stfts'] = tf.complex(features['predicted_stfts/mag'] + tf.reduce_min(features['predicted_stfts/mag']), features['vst_stfts/phase'])
    predictions = tf.contrib.signal.inverse_stft(features['predicted_stfts'],
                                                 frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'])

    features['predicted_samples'] = predictions

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=features)

    average_loss = tf.losses.sigmoid_cross_entropy(features['vst_stfts/mag'], features['predicted_stfts/mag'])
    # sigma = tf.Variable(10.0, trainable=True)
    # average_loss = ITL.correntropy_loss(features['vst_stfts/mag'], features['predicted_stfts/mag'], sigma=sigma)
    rmse = tf.metrics.root_mean_squared_error(features['vst_stfts/mag'], features['predicted_stfts/mag'])
    total_loss = tf.to_float(params['batch_size']) * average_loss

    tf.summary.histogram('outdnn', output_layer)
    tf.summary.image('predicted_stfts/mag', tf.expand_dims(features['predicted_stfts/mag'], axis=3))
    tf.summary.image('vst_stfts/mag', tf.expand_dims(features['vst_stfts/mag'], axis=3))
    tf.summary.scalar('rmse', rmse[1])
    tf.summary.scalar('sigma', sigma)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metrics = {"rmse": rmse}
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)


############################################################################################################################
############################################################################################################################
def mb_conv2d(features, labels, mode, params):
    stft_shape = [(params['audio_samples_length'] - params['frame_length']) // params['frame_step'] + 1, params['fft_length'] // 2 + 1]
    stft_size = int(np.prod(stft_shape))

    cnn_filter_sizes = [[9, 9, 1, 8],
                        [5, 5, 8, 16],
                        [5, 5, 16, 8],
                        [9, 9, 8, 1]]

    cnn_total_filter_size = int(np.sum([np.prod(i) for i in cnn_filter_sizes]))

    dnn_hidden_units = [cnn_total_filter_size, cnn_total_filter_size]
    dnn_output_size = cnn_total_filter_size

    features['input_stfts'] = tf.contrib.signal.stft(features['input_samples'], frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'])
    features['vst_stfts'] = tf.contrib.signal.stft(features['vst_samples'], frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'])

    features['input_stfts/mag'] = tf.image.per_image_standardization(tf.abs(features['input_stfts']))
    features['input_stfts/phase'] = tf.angle(features['input_stfts'])
    features['vst_stfts/mag'] = tf.image.per_image_standardization(tf.abs(features['vst_stfts']))
    features['vst_stfts/phase'] = tf.angle(features['vst_stfts'])

    # DNN NETWORK (CALCULATE CNN FILTERS FROM VST PARAMS)
    topdnn = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in dnn_hidden_units:
        topdnn = tf.layers.dense(inputs=topdnn, units=units, activation=tf.nn.leaky_relu)
    outdnn = tf.layers.dense(inputs=topdnn, units=dnn_output_size)
    conv_filters = tf.split(outdnn, [np.prod(k) for k in cnn_filter_sizes], axis=1)

    # CNN NETWORK (APPLY FILTERS ON STFT INPUT)
    topcnn = tf.expand_dims(features['input_stfts/mag'], axis=-1)
    topcnn = tf.expand_dims(topcnn, axis=0)
    for i, ff in enumerate(conv_filters):
        with tf.name_scope('cnn' + str(i)):
            filter_shape = [tf.shape(ff)[0]] + [(tf.convert_to_tensor(k)) for k in cnn_filter_sizes[i]]
            ff = tf.reshape(ff, filter_shape)
            topcnn = tf.nn.conv3d(input=topcnn, filter=ff, strides=[1, 1, 1, 1, 1], padding='SAME')
            topcnn = tf.layers.batch_normalization(topcnn, center=True, scale=False)
            topcnn = tf.nn.leaky_relu(topcnn)

    outcnn = tf.squeeze(tf.squeeze(topcnn, axis=0), axis=-1)
    outcnn.set_shape([params['batch_size'], stft_shape[0], stft_shape[1]])
    outcnn = tf.layers.dense(inputs=tf.layers.flatten(outcnn), units=stft_size)

    features['predicted_stfts/mag'] = tf.reshape(outcnn, tf.shape(features['vst_stfts/mag']))
    features['predicted_stfts'] = tf.complex(features['predicted_stfts/mag'] + tf.reduce_min(features['predicted_stfts/mag']), features['vst_stfts/phase'])
    predictions = tf.contrib.signal.inverse_stft(features['predicted_stfts'],
                                                 frame_length=params['frame_length'], frame_step=params['frame_step'], fft_length=params['fft_length'])

    features['predicted_samples'] = predictions

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=features)

    average_loss = tf.losses.sigmoid_cross_entropy(features['vst_stfts/mag'], features['predicted_stfts/mag'])
    # sigma = tf.Variable(10.0, trainable=True)
    # average_loss = ITL.correntropy_loss(features['vst_stfts/mag'], features['predicted_stfts/mag'], sigma=sigma)
    rmse = tf.metrics.root_mean_squared_error(features['vst_stfts/mag'], features['predicted_stfts/mag'])

    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss

    tf.summary.histogram('outdnn', outdnn)
    tf.summary.histogram('outcnn', outcnn)
    tf.summary.image('predicted_stfts/mag', tf.expand_dims(features['predicted_stfts/mag'], axis=3))
    tf.summary.image('vst_stfts/mag', tf.expand_dims(features['vst_stfts/mag'], axis=3))
    tf.summary.scalar('rmse', rmse[1])
    tf.summary.scalar('sigma', sigma)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metrics = {"rmse": rmse}
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)

