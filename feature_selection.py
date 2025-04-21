import pandas as pd
import numpy as np
from functools import reduce
def mutual_information_value(x, y, verbose=False):
    x.name = 'feature'
    y.name = 'target'
    if type(y) != pd.core.series.Series:
        y = pd.Series(y, name='target')
    if type(x) != pd.core.series.Series:
        x = pd.Series(x, name='feature')
    output = pd.crosstab(x, y, margins=True, margins_name='total')
    if verbose:
        print(output)
    output /= len(x)
    if verbose:
        print(output)
    def compute_mi(join, marginals):
        return join * np.log2(join / reduce(lambda x, y: x * y, marginals))
    total_mi = []
    rows = len(output.index) - 1
    cols = len(output.columns) - 1
    for r in range(rows):
        for c in range(cols):
            value = output.iloc[r, c]
            marg_a, marg_b = output.iloc[r, cols], output.iloc[rows, c]
            if verbose:
                print(value, [marg_a, marg_b], [r, c])
            if value != 0:
                mi_value = compute_mi(value, [marg_a, marg_b])
            else:
                mi_value = 0
            # Handle infinity values (e.g., log(0)) by setting them to zero
            if mi_value == np.inf:
                mi_value = 0
            total_mi.append(mi_value)
    if verbose:
        print(total_mi)
    return np.sum(total_mi)

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    df = pd.DataFrame({
        'feature1': [
            'orange', 'apple', 'banana', 'orange', 'banana', 'banana', 'banana',
            'banana', 'apple', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange'
        ],
        'target': [0, 1, 0, 1, 0,0,0,0,1,1,1,1,1,1,1],
    })

    # Calculate mutual information between feature1 and target
    mi_value = mutual_information_value(df['feature1'], df['target'], verbose=True)
    print(f"Mutual Information between feature1 and target: {mi_value}")


    