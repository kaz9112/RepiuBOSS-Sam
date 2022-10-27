import re
import string
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

# A Function to use in the dataframe
kamus = pd.read_csv('kamus.txt', sep="	", header=None,names=['slang', 'fix'])
slang_list = kamus['slang'].tolist()
fix_list = kamus['fix'].tolist()

def TextProcess(text):

  # 1. Change all text to Lowercase
  text = text.lower()
  
  # 2. Removing Mentions
  text = re.sub("@[A-Za-z0-9_]+", " ", text)
  
  # 3. Removing Hashtags
  text = re.sub("#[A-Za-z0-9_]+", " ", text)
  
  # 4. Removing \n
  text = re.sub(r"\\n", " ",text)
  
  # 5. Removing Whitespaces
  text = text.strip()

  # 6. Removing Links
  text = re.sub(r"http\S+", " ", text)
  text = re.sub(r"www.\S+", " ", text)

  # 7. Removing non text characters such as Emojis, Mathematical symbols
  text = re.sub("[^A-Za-z\s']", " ", text)

  # 8. Removing RT
  text = re.sub("rt", " ",text)

  # 9. Removing Punctuations
  text = text.translate(str.maketrans('', '', string.punctuation))

  # 11. Tokenization
  token = word_tokenize(text)

  for x in range(len(token)):
    for i in range(len(slang_list)):
      if token[x] == slang_list[i]:
        token[x] = fix_list[i]
      else:
        pass

  from nltk.util import ngrams
  _2gram = [' '.join(e) for e in ngrams(token, 2)]
  _3gram = [' '.join(e) for e in ngrams(token, 3)]
  text = token + _2gram + _3gram

  
  #text2 = ' '.join(token)

  text = np.array(text)

  
  return text

def Label(num):
  if num == 0:
    topic = 'Baterai cepat habis'
  elif num == 1:
    topic = 'hp tidak berfungsi, tidak sesuai, tidak nyala'
  elif num == 2:
    topic = 'barang tidak sesuai deskripsi, hp mati'
  elif num == 3:
    topic = 'positif'
  elif num == 4:
    topic = 'barang tidak sesuai pesanan'
  elif num == 5:
    topic = 'barang rusak'
  elif num == 6:
    topic = 'barang tidak sesuai, suara tidak berfungsi'
  elif num == 7:
    topic = 'warna tidak sesuai, atau barang tidak sesuai gambar'
  elif num == 8:
    topic = 'barang tidak sesuai deskripsi, pengiriman lama'
  elif num == 9:
    topic = 'barang kosong, cancel, retur'
  else:
    pass
  return topic




