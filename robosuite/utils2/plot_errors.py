import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['A','B'])
for i in range(5): #add 5 rows of data
    df = df.concat({'A': i}, ignore_index=True)
    df = df.concat({'B': i+2}, ignore_index=True)

print()
