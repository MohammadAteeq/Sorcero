import nltk.data


class Tokenizer:
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def tokenize_to_sentences(self, document):
        sentences = self.tokenizer.tokenize(document)
        return sentences
