def get_delimiters(n, size, return_chunks=False):
    delim  = [(0, n)]
    chunks = [n]
    if return_chunks:
        return delim, chunks
    return delim
