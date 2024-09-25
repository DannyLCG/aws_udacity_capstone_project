import numpy as np
'''Add documentation.'''


# Define one-hot encoder
class OneHotEncoder:
  def __init__(self, max_len = None, stop_signal = True):
        self.max_len = max_len
        self.stop_signal = stop_signal

  def encode(self, sequences):
      vocab = {
          'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
          'Q': 5, 'E': 6, 'G': 7, 'H':8, 'I': 9,
          'L': 10, 'K': 11, 'M': 12, 'F' : 13, 'P': 14,
          'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
      }

      if self.stop_signal:
          vec_lenght = 20
      else:
          vec_lenght = 21
      encoded_sequences = []
      max_len = 0
      for sequence in sequences:
          sequence = sequence.upper()
          encoded_sequence = []
          for aa in sequence:
              vec = [0 for _ in range(vec_lenght)]
              pos = vocab[aa]
              vec[pos] = 1
              encoded_sequence.append(vec)
          encoded_sequences.append(encoded_sequence)
          max_len = max(max_len, len(sequence))

      if self.max_len is not None:
          max_len = self.max_len
      max_len += 1
      
      if self.stop_signal:
          for sequence in encoded_sequences:
              while len(sequence) < max_len:
                  vec = [0 for _ in range(vec_lenght)]
                  vec[-1] = 1
                  sequence.append(vec)
      return np.array(encoded_sequences)
  