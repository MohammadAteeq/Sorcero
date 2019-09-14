from DataReader import FileReader
from Encoder import UniversalSentenceEncoder
from MathComputation import SimilarityComputation
from Optimizer import SummarizationOptimizer
from TextProcessor import Tokenizer

fileReader = FileReader()
document = fileReader.read_raw_file("data")
tokenizer = Tokenizer()
sentences = tokenizer.tokenize_to_sentences(document)
universalSentenceEncoder = UniversalSentenceEncoder()
encoding_sentences = universalSentenceEncoder.encode_text(sentences)
documents = [document]
encoded_document = universalSentenceEncoder.encode_text(documents)
similarityComputation = SimilarityComputation()

sentences_similarity = similarityComputation.inner_product(encoding_sentences, encoding_sentences)

coverage_matrix = []
for sentence_encoded in encoding_sentences:
    cosine_value = similarityComputation.compute_cosine(sentence_encoded, encoded_document[0])
    coverage_matrix.append(cosine_value)

print(coverage_matrix)
print(sentences_similarity)
for s in sentences:
    print(s + "\n")

summarizationOptimizer = SummarizationOptimizer()
best_result = summarizationOptimizer.start_optimization(len(sentences), coverage_matrix, sentences_similarity, 0.4)

print("Summary")
for result in best_result:
    print(sentences[result] + "\n")
