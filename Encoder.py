import tensorflow as tf
import tensorflow_hub as hub


class UniversalSentenceEncoder:

    def encode_text(self, text):
        embed_fn = self.embed_useT('E:/universal_model')
        encoded_text = embed_fn(text)
        return encoded_text

    def embed_useT(self, module):
        with tf.Graph().as_default():
            sentences = tf.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})
