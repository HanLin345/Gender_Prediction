import re

def removeDuplicateWords(word):
	new_word = re.compile(r"(.)\1{1,}", re.DOTALL)
	return new_word.sub(r"\1\1",word)
