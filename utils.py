def to_np(tensor):
    return tensor.detach().cpu().numpy()


def adjacent_pairs(seq: list) -> list:
    """Makes a list of adjacent pairs from the elements of a sequence.

    Examples:
    >>> adjacent_pairs([1, 2, 3])
    [(1, 2), (2, 3)]
    """
    seq = iter(seq)
    try:
        x, y = next(seq), next(seq)
    except StopIteration:
        return []
    while True:
        yield x, y
        try:
            x, y = y, next(seq)
        except StopIteration:
            break
