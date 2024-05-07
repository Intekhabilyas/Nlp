"""In python Implement the Byte pair Encoding Algorithm of tokenization and generate the tokens for test corpus.

Training Corpus: low, lowest, newer, wider, new

Test Corpus: newer, lower"""
from collections import defaultdict

def get_vocab(corpus):
    vocab = defaultdict(int)
    for word in corpus:
        vocab[' '.join(word)] += 1
    return vocab

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, in_vocab):
    out_vocab = {}
    bigram = ' '.join(pair)
    for word in in_vocab:
        out_word = word.replace(bigram, ''.join(pair))
        out_vocab[out_word] = in_vocab[word]
    return out_vocab

def tokenize(corpus, vocab):
    tokenized_corpus = []
    for word in corpus:
        while True:
            new_word = word
            found_bigram = False
            for pair in vocab:
                if ' '.join(pair) in new_word:
                    new_word = new_word.replace(' '.join(pair), ''.join(pair))
                    found_bigram = True
                    break
            if not found_bigram:
                break
            word = new_word
        tokenized_corpus.append(word.split())
    return tokenized_corpus

def byte_pair_encoding(training_corpus, test_corpus, num_merges):
    vocab = get_vocab(training_corpus)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        most_common_pair = max(pairs, key=pairs.get)
        if pairs[most_common_pair] == 1:
            break
        vocab = merge_vocab(most_common_pair, vocab)
    
    tokenized_test_corpus = tokenize(test_corpus, vocab)
    return tokenized_test_corpus

def main():
    training_corpus = ['low', 'lowest', 'newer', 'wider', 'new']
    test_corpus = ['newer', 'lower']
    num_merges = 10

    tokenized_test_corpus = byte_pair_encoding(training_corpus, test_corpus, num_merges)
    print("Tokenized Test Corpus:")
    for token in tokenized_test_corpus:
        print(token)

if __name__ == "__main__":
    main()
