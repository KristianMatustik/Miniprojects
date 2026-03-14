# small project, trying alg to detect stop words in token stream, cutting the stream at the first stop word, yielding all tokens before it (without stop word tokens)

import random
from typing import Generator


# Trie class for efficient storage and search of stop words (for <10 words there might be more efficient methods, but this should scale better)
class Trie:
    class TrieNode:
        def __init__(self):
            self.char = None
            self.children = {}              # dictionary of children nodes (char -> TrieNode)
            self.word_end = False           # if node with word_end=True is reached the word from used char sequence is in the trie

        def advance(self, char: str):   # moves to next character in word (next TrieNode), None if not found
            char = char.lower()             # comment if case sensitive search is desired
            return self.children.get(char, None)

    def __init__(self):
        self.root = Trie.TrieNode()

    def insert(self, word: str):    # inserts word into trie, sets last node (char of word) as word_end=True
        node = self.root
        for char in word:
            char = char.lower()     # comment if case sensitive search is desired
            if char not in node.children:
                node.children[char] = Trie.TrieNode()
                node.children[char].char = char
            node = node.children[char]
        node.word_end = True

# Linked list class with 3 data points, has 2 different uses described below in function cut_stream_stop_words() 
class List:
    class ListNode:
        def __init__(self, token = None, index = 0, trie_node = None):
            self.token = token
            self.index = index
            self.trie_node = trie_node
            self.next = None
            self.prev = None

    def __init__(self):
        self.head = None
        self.tail = None
    
    def push_back(self, token=None, index = 0, trie_node = None):
        if self.head is None:
            self.head = List.ListNode(token, index, trie_node)
            self.tail = self.head
            return
        node = List.ListNode(token, index, trie_node)
        self.tail.next = node
        node.prev = self.tail
        self.tail = node

    def pop_front(self):    
        if self.head is None:
            return
        node = self.head
        self.head = node.next
        if self.head is not None:
            self.head.prev = None
        else:
            self.tail = None

    def remove(self, node): # removes specified node from the list (which can be found through iteration to match different properties)
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        if self.head == node:
            self.head = node.next
        if self.tail == node:
            self.tail = node.prev

def string_to_stream(text: str) -> Generator[str, None, None]: # Generator function to split text into tokens, used for testing, simulating LLM token stream
    i=0
    while i<len(text):
        word = ''
        j=i+random.randint(1, 5)
        while i<len(text) and i<j:
            word += text[i]
            i += 1
        yield word


# Main function
# If stop words shouldnt be part of other words (eg. "end" is stop word, but "send" is not), they NEED TO BE passed with space at the beginning (" end")
    # Code can be easily modified to not require space if its allways desired behavior, but this implementation is more general, allows both cases

def cut_stream_stop_words(token_stream: Generator[str, None, None], stop_words: list[str]) -> Generator[str, None, None]:
    if not hasattr(cut_stream_stop_words, "trie"):  # Trie should be static to avoid creation on each call, words added only in first one...
        cut_stream_stop_words.trie = Trie()         # ...later can just add new stop words (speeds up the function)
    trie = cut_stream_stop_words.trie

    trie = Trie()                                   # Nonstatic if new Trie required for each call (used bellow in testing), comment/uncomment this as needed

    for word in stop_words:   # Trie initialization with stop words
        trie.insert(word)       # Here can " " be added manually if stop words should never be part of other words, also needs small modifications below (index of stop word start)
        
    tokens = List()     # list of tokens currently searched for stop words, they are yielded if no stop word starts in them
    words = List()      # list of TrieNodes to check for stop words, also contains for each one token and index of the first char (possible start of stop word)
    stop_word = None    # node from words list that is stop word

    for token in token_stream:
        tokens.push_back(token)          # add token as being searched for stop words
        for i, char in enumerate(token):
            word = words.head                           # iterate over tracked words, remove if not in trie, stop if stop word found
            while word is not None:
                if not word.trie_node.word_end:         # 1) ... condition can be removed (always advance)
                    word.trie_node = word.trie_node.advance(char)
                if word.trie_node is None:
                    words.remove(word)
                elif word.trie_node.word_end:
                    stop_word = word
                    if words.head == stop_word:         # 1) ... condition can be removed (always break)
                        break
                word = word.next
            if stop_word is not None and words.head == stop_word: # 1) ... 2nd condition can be removed
                break   

            val = trie.root.advance(char)  # add new word starting at current char
            if val is not None:
                if char == ' ':                         # this can be modified, simplified slightly if stop words should never be part of other words
                    words.push_back(token, i+1, val)
                else:
                    words.push_back(token, i, val)  

        while tokens.head.token!=token and (words.head is None or tokens.head.token!=words.head.token):
            yield tokens.head.token
            tokens.pop_front()   
        if stop_word is not None and words.head == stop_word: # 1) ...
            break
    
    # points 1) required if stop words can be nested (stop word "stream end now" would yield "stream " if "end" was also stop word)...
    # ...If stop words will not be like that it can be simplified and stop search on first stop_word found (changes as described in comments above)

    # remaining tokens if no stop word found
    if stop_word is None:
        while tokens.head is not None:
            yield tokens.head.token
            tokens.pop_front()
        return

    # remaining tokens if stop word found
    while tokens.head is not None and tokens.head.token != stop_word.token:
        yield tokens.head.token
        tokens.pop_front()
    if stop_word.index > 0:
        yield stop_word.token[:stop_word.index]



# Testing some edge cases I considered
token_stream = string_to_stream('this is. a test send stream end now please') # gibberish for my testing
token_stream_list = list(token_stream) # List instead of Gen. so I can have exact same stream for multiple tests (same token split), 1st test is with Gen.

# test with generator
stop_words = ["stop", "end"]
result = list(cut_stream_stop_words(string_to_stream('this is. a test send stream end now please'), stop_words))
print(result)

# case insensitivity, stop word within other word
stop_words = ["stop", "EnD"]
result = list(cut_stream_stop_words(token_stream_list, stop_words))
print(result)

# only new words
stop_words = [" stop", " end"]
result = list(cut_stream_stop_words(token_stream_list, stop_words))
print(result)

# stop words in other stop words
stop_words = [" end", " stream end now"]
result =  list(cut_stream_stop_words(token_stream_list, stop_words))
print(result)

# no stop word found
stop_words = [" stop", " end it"]
result =  list(cut_stream_stop_words(token_stream_list, stop_words))
print(result)


# some other tests
# char
stop_words = ["e"]
result =  list(cut_stream_stop_words(token_stream_list, stop_words))
print(result)

# start of word
stop_words = [" e"]
result =  list(cut_stream_stop_words(token_stream_list, stop_words))
print(result)

# no stop words
stop_words = []
result =  list(cut_stream_stop_words(token_stream_list, stop_words))
print(result)

# more extensive testing would be nice, but I think it should work and cover considered edge cases
# comments, naming, and code style could be improved, didnt have much time to polish it
# one "edge case" this doesnt consider is end of words, eg. " end" would stop in all: "end.", "end ", " ending", ...
# can be fixed by adding space/dot/... to the end of each token, adding multiple variants of each stop word to the trie