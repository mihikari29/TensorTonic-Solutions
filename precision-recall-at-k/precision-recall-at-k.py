def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    TP = 0
    for i in range(k):
        if recommended[i] in relevant:
            TP += 1
    precision = TP / k
    recall = TP / len(relevant)
    return [precision, recall]