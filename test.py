# for i in range(10)[-1::]:
#     print(i)


# a = [0,1,2,3,4,5,6,7,8,9]
# print(a[1:-1])

import numpy as np
import pandas as pd


# df = pd.read_csv('/home/mauricio/Documents/Unesp/CFD/classyGen/curve.csv')
data = np.genfromtxt('/home/mauricio/Documents/Unesp/CFD/classyGen/curve.csv', delimiter=',', usecols=2)[1::]
print(len(data))
# print(df)

