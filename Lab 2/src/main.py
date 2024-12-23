import kernel
import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generating Test Data
numpy.random.seed(100)  # To get the same random data every time you run the program

# Generate two sets of random points in 2D, one for each class.
classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
     numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))

classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

# Concatenate the two sets into a single matrix of inputs.
inputs = numpy.concatenate((classA, classB))

# Generate a corresponding vector of target values.
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]),
     -numpy.ones(classB.shape[0])))

N = inputs.shape[0]  # Number of rows (samples)

# Randomly permute the order of the inputs and targets to ensure that the order of the data does not affect the
# learning process.
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# Creating the P-matrix
# This matrix is used to store the dot product of each pair of data points. The kernel function is used to transform the
# data into a higher-dimensional space where it is easier to find a hyperplane that separates the data points
# into different classes.
dimension = (N, N)
P = numpy.zeros(dimension)

for i in range(len(classA)):
    for j in range(len(classB + 1), N):
        P[i, j - 20] = targets[i] * targets[j] * kernel.polynomialKernel(inputs[i], inputs[j], 2)

P_array = numpy.array(P)


# Implement the function objective
def objective(alpha):
    sumEquation = -numpy.sum(alpha)
    for i in range(N):
        for j in range(N):
            sumEquation += alpha[i] * alpha[j] * P_array[i, j]

    sumEquation *= 0.5
    return sumEquation


def zerofun(alpha):
    return numpy.dot(alpha, targets)


# minimize function.
XC = {'type': 'eq', 'fun': zerofun}
C = 5
B = [(0, 0.15) for b in range(N)]
ret = minimize(objective, numpy.zeros(N), bounds=B, constraints=XC)

alphaVec = ret['x']
print(alphaVec)
if ret['success']:  # Om du ändrar None till typ 1 eller 10 så får du success, vet inte varför.
    print("Success")
else:
    print("Failure")

threshold = 1e-5
nonZero = []  # Represents the support vectors
dataPoints = []  # Represents the corresponding data points
targetValues = []  # Represents the corresponding target values
for i in range(len(alphaVec)):
    if alphaVec[i] > threshold:
        nonZero.append(alphaVec[i])
        dataPoints.append(inputs[i])
        targetValues.append(targets[i])


# Function to calculate the bias term
def calculateB():
    sumB = -targetValues[0]
    for n in range(len(nonZero)):
        sumB += nonZero[n] * targetValues[n] * kernel.polynomialKernel(dataPoints[n], dataPoints[0], 2)
    return sumB


def indicator(x, y):
    sumIndicator = -calculateB()
    for n in range(len(nonZero)):
        sumIndicator += nonZero[n] * targetValues[n] * kernel.polynomialKernel(dataPoints[n], [x, y], 2)
    return sumIndicator


# 6.1 Plotting the Decision Boundary

xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)
grid = numpy.array([[indicator(x, y)
                     for x in xgrid]
                    for y in ygrid])

plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))
# Plotting

plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')

plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

plt.axis('equal')  # Force same scale on both axes

plt.savefig('plots/svmplot.png')  # Save a copy in a file
plt.show()  # Show the plot on the screen
