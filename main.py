import perceptron as perc
import numpy as np

data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0, 1]])

ans = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

MyPerceptron = perc.Perceptron(0.01, 10000, perc.ErrorFunctions.Arctan, 'test')
MyPerceptron.addLayer(9, MyPerceptron.actFunctions.ReLU)
MyPerceptron.addLayer(6, MyPerceptron.actFunctions.ReLU)
MyPerceptron.addLayer(2, MyPerceptron.actFunctions.ReLU)

MyPerceptron.set_data(data, ans)
MyPerceptron.init()
MyPerceptron.start()
MyPerceptron.test_loop()
MyPerceptron.save()

# p = perc.Perceptron.load("test_perceptron.json")
# p.set_data(data, ans)
# p.test_loop()
# p.a = 0.01
