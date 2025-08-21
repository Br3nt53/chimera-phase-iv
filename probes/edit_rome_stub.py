def localized_edit(weight_matrix, gradient, lr=0.1, l2=1e-3):
    return weight_matrix - lr * gradient / (l2 + 1e-9)
