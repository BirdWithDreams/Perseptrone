import perceptron as perc
import numpy as np

data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0, 1]])

ans = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

# MyPerceptron = perc.Perceptron(0.005, 30000, perc.ErrorFunctions.arctan, 'test')
# MyPerceptron.addLayer(9, MyPerceptron.actFunctions.ReLU)
# MyPerceptron.addLayer(6, MyPerceptron.actFunctions.ReLU)
# MyPerceptron.addLayer(2, MyPerceptron.actFunctions.ReLU)
MyPerceptron = perc.Perceptron.load('test_perceptron.json')
MyPerceptron.set_data(data, ans)
MyPerceptron.init()

# MyPerceptron.start()
MyPerceptron.test_loop()
# MyPerceptron.save()
