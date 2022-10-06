import numpy as np

def get_block(matrix, row_start, row_end, col_start, col_end):
    if row_end > matrix.shape[0]:
        row_end = matrix.shape[0]
    if col_end > matrix.shape[1]:
        col_end = matrix.shape[1]

    return matrix[row_start:row_end, col_start:col_end], np.arange(row_start, row_end, 1), np.arange(col_start, col_end, 1)


def create_mcar_mask(n_rows, n_columns, percentage_of_missingness=0.1, seed=0):
    np.random.seed(seed)
    mask = np.ones((n_rows, n_columns))

    # now for each row I have to turn off some values
    # I think that the easiest thing (for now) is to consider
    # as 1 the pixel that arre observed and 0 the one missing
    # I think it easier in dealing with masking when computing the loss

    for i in range(mask.shape[0]):
        missing_values = np.random.choice(np.arange(0, n_columns, 1), size=int(percentage_of_missingness * n_columns),
                                          replace=False)
        mask[i, missing_values] = 0

    return mask


def create_mar_mask(data):

    masks = np.zeros((data.shape[0], data.shape[1]))
    for i, example in enumerate(data):
        h = (1. / (784. / 2.)) * np.sum(example[int(784 / 2):]) + 0.3
        pi = np.random.binomial(2, h)

        _mask = np.ones(example.shape[0])

        if pi == 0:
            # we have to remove the second quarter
            _mask[196:(2 * 196)] = 0

        elif pi == 1:
            # the whole top half is missing
            _mask[0:(2 * 196)] = 0

        elif pi == 2:
            # the first quarter is missing
            _mask[0:196] = 0

        else:
            print('There is a problem mate, the pub is close')
            break

        masks[i, :] = _mask

    return masks

def get_imputation(data, observed_mask, missing_mask, imputation_value):

    return data * observed_mask + missing_mask * imputation_value