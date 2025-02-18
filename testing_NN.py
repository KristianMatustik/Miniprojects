import pygame as pg
from drawingBoard import DrawingBoard
from NeuralNetworks import NeuralNetwork
import pandas as pd
import numpy as np

# XOR
# NN = NeuralNetwork()

# # NN = NeuralNetwork.load("files/NN_XOR.pkl")

# data = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
# data_targets = np.array(data)[:, 0]
# data_inputs = np.array(data)[:, 1:]
# data_inputs.flatten()

# NN.add_Layer_FullyConnected(2,3, NeuralNetwork.ReLU)
# NN.add_Layer_FullyConnected(3,1, NeuralNetwork.sigmoid)
# NN.train(data_inputs, data_targets, 2, 100, 0.2, 0, NeuralNetwork.cost_binaryCrossEntropy)

# for i in range(4):
#     print(data_inputs[i], NN.forward(data_inputs[i]), data_targets[i])

# # NN.save("files/NN_XOR.pkl")



# MNIST
data = pd.read_csv("files/MNIST_train.csv")
data = np.array(data)
data_targets = data[:, 0].astype(int)
data_inputs = data[:, 1:].astype(np.float32) / 255.0

# # Fully Connected
# NN = NeuralNetwork()
# NN.add_Layer_FullyConnected(784, 20, NeuralNetwork.ReLU)
# NN.add_Layer_FullyConnected(20, 10, NeuralNetwork.softmax)

# NN = NeuralNetwork.load("files/NN_MNIST.pkl") #pretrained, can improve further

# NN.train(data_inputs, data_targets, 16, 20, 0.01, 0.95, NeuralNetwork.cost_crossEntropy)
# NN.save("files/NN_MNIST.pkl")

# # CNN
# data_inputs = data_inputs.reshape(-1, 28, 28)
# NN = NeuralNetwork()
# NN.add_Layer_Convolutional((1,28,28),5,3,1,0, NeuralNetwork.ReLU)
# NN.add_Layer_Pooling(-1, 4, 4, "max")
# NN.add_Layer_Convolutional(-1, 5, 3, 1, -1, NeuralNetwork.ReLU)
# NN.add_Layer_FullyConnected(-1, 20, NeuralNetwork.ReLU)
# NN.add_Layer_FullyConnected(-1, 10, NeuralNetwork.softmax)

# # NN = NeuralNetwork.load("files/NN_MNIST_CNN.pkl") #pretrained, can improve further

# NN.train(data_inputs, data_targets, 16, 1, 0.05, 0.9, NeuralNetwork.cost_crossEntropy, True)
# NN.save("files/NN_MNIST_CNN2.pkl")



# Testing
# NN = NeuralNetwork.load("files/NN_MNIST.pkl")
NN = NeuralNetwork.load("files/NN_MNIST_CNN.pkl")

pg.init()
screen = pg.display.set_mode((800, 560))
board = DrawingBoard((0, 0), 28, 28, 560, 560, 1)
clock = pg.time.Clock()
font = pg.font.SysFont(None, 32)

def draw_predictions(screen, predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    for i, idx in enumerate(sorted_indices):
        text = f"{idx}: {predictions[idx] * 100:.2f}%"
        img = font.render(text, True, (255, 255, 255))
        screen.blit(img, (580, 20 + i * 30))

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            quit()
        if event.type == pg.MOUSEMOTION or event.type == pg.MOUSEBUTTONDOWN:
            if pg.mouse.get_pressed()[0]:
                board.paint(pg.mouse.get_pos())
            elif pg.mouse.get_pressed()[2]:
                board.erase(pg.mouse.get_pos())
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                board.clear()
            if event.key == pg.K_RIGHT:
                board.grid = data_inputs[np.random.randint(0, data_inputs.shape[0])].reshape(28, 28)
                
    prediction = NN.forward(board.grid)
    board.draw(screen)
    pg.draw.line(screen, (255, 255, 255), (570, 0), (570, 560), 2)
    draw_predictions(screen, prediction)
    pg.display.update()
    clock.tick(120)