from vocab2vec import *
from basicRNN import *

if __name__ == "__main__":
    
    VOCAB_SIZE = 40
    text = vocab2vec(VOCAB_SIZE, vocab_length = 10**6)
    tfrnn = TFBasicRNN(vocab_size=VOCAB_SIZE, hidden_size=100, seq_length=10)
    tfrnn.train(text)
