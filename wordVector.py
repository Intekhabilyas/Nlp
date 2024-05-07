""" Generate code in python for the word vectors using the bigram probabilities and calculate the cosine similarity between two given words. Use the given Transcript.

<s>we are from Jamia </s>
<s>we are Jamia student from engineering</s>
<s>students from Jamia are good </s>

<s> Jamia engineering students </s>

<s> Jamia students are best </s>"""
from collections import Counter
import math

def preprocess_transcript(transcript):
    # Split the transcript into sentences and tokenize each sentence
    sentences = transcript.split('\n')
    tokens = [sentence.split() for sentence in sentences]
    return tokens

def count_words_and_bigrams(tokens):
    word_counts = Counter()
    bigram_counts = Counter()

    for sentence in tokens:
        word_counts.update(sentence)
        bigram_counts.update(zip(sentence, sentence[1:]))

    return word_counts, bigram_counts

def calculate_word_probabilities(word_counts):
    total_words = sum(word_counts.values())
    word_probabilities = {word: count / total_words for word, count in word_counts.items()}
    return word_probabilities

def calculate_bigram_probabilities(word_counts, bigram_counts):
    bigram_probabilities = {}
    
    for bigram, count in bigram_counts.items():
        word1, word2 = bigram
        bigram_probabilities[bigram] = count / word_counts[word1]

    return bigram_probabilities

def calculate_word_vectors(word_probabilities, dimension=100):
    word_vectors = {}

    for word, probability in word_probabilities.items():
        vector = [probability] * dimension
        word_vectors[word] = vector

    return word_vectors

def cosine_similarity(vector1, vector2):
    dot_product = sum(x * y for x, y in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x ** 2 for x in vector1))
    magnitude2 = math.sqrt(sum(y ** 2 for y in vector2))
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity

def main():
    transcript = """
    <s>we are from Jamia </s>
    <s>we are Jamia student from engineering</s>
    <s>students from Jamia are good </s>
    <s> Jamia engineering students </s>
    <s> Jamia students are best </s>
    """

    tokens = preprocess_transcript(transcript)
    word_counts, bigram_counts = count_words_and_bigrams(tokens)
    word_probabilities = calculate_word_probabilities(word_counts)
    bigram_probabilities = calculate_bigram_probabilities(word_counts, bigram_counts)
    word_vectors = calculate_word_vectors(word_probabilities)

    word1 = "Jamia"
    word2 = "students"

    if word1 in word_vectors and word2 in word_vectors:
        similarity = cosine_similarity(word_vectors[word1], word_vectors[word2])
        print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")
    else:
        print("One or both of the words are not present in the transcript.")

if __name__ == "__main__":
    main()
