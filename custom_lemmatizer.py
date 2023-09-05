from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class custom_lemmatizer:

    tag_dict = {"ADJ": wordnet.ADJ,
                "NOUN": wordnet.NOUN}

    lemmatizer = WordNetLemmatizer()

    def lemmatize(self, word_pos_tuple):
        word = word_pos_tuple[0]
        pos_tag = word_pos_tuple[1]
        if pos_tag in self.tag_dict:
            return self.lemmatizer.lemmatize(word, self.tag_dict[pos_tag]).lower()
        else:
            return word.lower()

### EXAMPLE USE
### Given 'tagged' is a list of tuples, consisting of word-tag pairs such as ('civilization','NOUN'), obtained from nltk.pos_tag(..., tagset='universal'):

# import custom_lemmatizer
# ...
# cm = custom_lemmatizer.custom_lemmatizer()
# for t in tagged:
#     print(cm.lemmatize(t))

