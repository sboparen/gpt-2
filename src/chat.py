#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=None,
    length=None,
    temperature=1,
    top_k=0,
):
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    length = min(length, 80)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        chatlog = ''
        userprompt = input('Your name: ') + ': '
        itsprompt = 'Deepfake: '
        while True:
            chatlog += userprompt + input(userprompt) + '\n'
            chatlog += itsprompt
            context_tokens = enc.encode(chatlog)
            fulltext = ''
            while True:
                out, = sess.run(output, feed_dict={
                    context: [context_tokens]})[:, len(context_tokens):]
                text = enc.decode(out)
                if '\n' in text:
                    text = text.split('\n')[0] + '\n'
                    fulltext += text
                    break
                else:
                    fulltext += text
            chatlog += fulltext
            print(itsprompt + fulltext, end='')

if __name__ == '__main__':
    fire.Fire(interact_model)

