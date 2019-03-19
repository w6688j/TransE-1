def _tokenize(text):
    if not isinstance(text, str):
        return []
    else:
        return [x.lower() for x in text.split()]
