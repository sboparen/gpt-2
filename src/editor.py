from PyQt5.QtWidgets import *
import sys

class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setMinimumSize(800, 600)
        self.setWindowTitle("GPT-2 Text Editor")
        self.layout = QVBoxLayout()
        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(self.layout)
        self.edit = QPlainTextEdit(self.centralWidget())
        self.layout.addWidget(self.edit)
        self.response = QPlainTextEdit(self.centralWidget())
        self.layout.addWidget(self.response)
        self.buttons = QWidget()
        self.layout.addWidget(self.buttons)
        self.buttons.setLayout(QHBoxLayout())
        button = QPushButton('Append')
        button.clicked.connect(self.append)
        self.buttons.layout().addWidget(button)
        button = QPushButton('Quick')
        button.clicked.connect(self.doit_quick)
        self.buttons.layout().addWidget(button)
        button = QPushButton('Long')
        button.clicked.connect(self.doit)
        self.buttons.layout().addWidget(button)
        self.gpt2 = GPT2()

    def append(self):
        self.edit.setPlainText(
            self.edit.toPlainText() + self.response.toPlainText())
        self.response.setPlainText('')

    def doit_quick(self):
        text = self.edit.toPlainText()
        self.response.setPlainText(self.gpt2.conditional(text, length=80))

    def doit(self):
        # TODO make asynchronous
        # TODO trigger automatically if idle for a few seconds
        # TODO save all generated texts to a log
        text = self.edit.toPlainText()
        self.response.setPlainText(self.gpt2.conditional(text))

def main():
    app = QApplication(sys.argv)
    root = MainWindow()
    root.show()
    sys.exit(app.exec_())

import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder

class GPT2:

    def __init__(self):
        self.cache = {}

    def setup(self, model_name='117M', seed=None, temperature=1, top_k=40,
              length=None):
        if length not in self.cache:
            enc = encoder.get_encoder(model_name)
            hparams = model.default_hparams()
            with open(os.path.join('models', model_name, 'hparams.json')) as f:
                hparams.override_from_dict(json.load(f))
            if length is None:
                length = hparams.n_ctx // 2
            sess = tf.Session(graph=tf.Graph()).__enter__()
            context = tf.placeholder(tf.int32, [1, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=1,
                temperature=temperature, top_k=top_k
            )
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
            saver.restore(sess, ckpt)
            self.cache[length] = context, enc, output, sess
        return self.cache[length]

    def conditional(self, raw_text, **kwargs):
        context, enc, output, sess = self.setup(**kwargs)
        context_tokens = enc.encode(raw_text[-1000:])
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(1)]
        })[:, len(context_tokens):]
        return enc.decode(out[0])

if __name__ == '__main__':
    main()
