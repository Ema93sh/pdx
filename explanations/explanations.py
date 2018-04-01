import numpy as np


class Explanation(object):
    """Explanation class related to generating explanations"""

    def __init__(self):
        super(Explanation, self).__init__()
        # TODO

    def get_pdx(self, q_values, selected_action, target_actions):
        """predicted delta explanations for the selected action in comparision with the target action.

        :param list q_values: decomposed q-values
        :param int selected_action:
        :param list target_actions:

        >>> Explanation().get_pdx(np.array([[1, 1, 1, 1], [4, 3, 2, 1]]), 0, [1, 2])
        ([[0, 0], [1, 2]], [[0.0, 0.0], [1.0, 1.0]])

        >>> Explanation().get_pdx(np.array([[1, 0, 2, 1], [3, 1, 2, 0]]), 0, [1, 2])
        ([[1, -1], [2, 1]], [[0.33, 0.0], [0.67, 1.0]])

        """
        pdx = [[(q_values[r][selected_action] - q_values[r][target]) for target in target_actions]
               for r in range(len(q_values))]

        _pdx = np.array(pdx.copy())
        _pdx[_pdx < 0] = 0
        _atomic_pdx = np.array(_pdx).sum(axis=0)
        contribution = [[round(_pdx[r][target] / (_atomic_pdx[target] + 0.001), 2)
                         for target in range(len(target_actions))]
                        for r in range(len(q_values))]
        return pdx, contribution


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)
