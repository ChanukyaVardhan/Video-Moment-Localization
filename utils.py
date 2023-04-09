import string

def get_tokens(query):
	return str(query).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()
