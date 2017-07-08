import numpy as np

def vocab2vec(vocab_size, vocab_length=10**7):
    f = open("../Combined_String.txt", "r")
    s = f.read()
    f.close()
    D = 'abcdefghijklmnopqrstuvwxyz .,\'1234567890";'
    res = []
    for i in range(vocab_length):
        c = s[i].lower()
        v = np.zeros((vocab_size))
        try:
            idx = D.index(c)
            v[idx] = 1
            res.append(v)
        except (ValueError, IndexError) as e:
            pass


    ret = np.array(res) # A list of shape (vocab_length,) one-hot encoded characters
    print ("shape is: {}".format(ret.shape))
    return ret
