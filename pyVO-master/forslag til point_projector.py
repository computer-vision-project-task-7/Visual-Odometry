
import numpy as np



points = np.array( [[1,  2, 3, 4, 5, 6, 7, 8, 9],
                    [10,11,12,13,14,15,16,17,18] ])


z = np.array([22,66,333,654,234,654,546,234,676])

# number of points
N = len(points[0])
# new row of zeros
nullere = np.zeros(N)
# adding new row of zeros to points
points = np.vstack( (points, nullere) )
# making list of points array to speed up iteration
points_list = list( zip(points[0], points[1]) )
# matrix to return
new = np.zeros( (3, N) )

i = 0
for x,y in points_list:

    new[0][i] = x
    new[1][i] = y
    new[2][i] = z[i]
    i += 1

print(new)
