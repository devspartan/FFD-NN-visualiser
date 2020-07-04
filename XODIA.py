import pygame
import numpy as np
import random
import cv2
import pandas as pd
from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
from keras import backend as K


base_model = load_model('../IdentifyMnistDigits/kerasModel.h5')

def getFeatures(img):
    layers = ['activation_1', 'activation_2', 'activation_3']
    features1 = []
    features2 = []
    features3 = []
    for i in range(3):
        model = Model(input=base_model.input, output=base_model.get_layer(layers[i]).output)
        if i == 0:
            features1 = model.predict(img)
        elif i == 1:
            features2 = model.predict(img)
        elif i == 2:
            features3 = model.predict(img)

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # get the features from this block
    features1 = np.resize(features1, [100, 1])
    features2 = np.resize(features2, [100, 1])
    features3 = np.resize(features3, [10, 1])

    return features1, features2, features3
    # get_1st_layer_output = K.function([model.layers[0].input],
    #                                   [model.layers[1].output])
    # layer_output = get_1st_layer_output([X])

    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name, layer.output_shape)


class Rectangle:
    def __init__(self, x, y, w, h, name, index, color=(2, 20, 200), value=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.value = value
        self.index = index
        self.layerName = name

    def drawRect(self):
        pygame.draw.rect(display, self.color, [self.x, self.y, self.w, self.h])

def getArray():
    tempArr = []
    k = 1
    for i in range(28):
        for j in range(28):
            tempArr.append(Rectangle(j*15 + 715, i*15 + 575, 13, 13, 'InputLayer', k))
            k += 1
    inputArr = np.array(tempArr)
    inputArr = inputArr.reshape(28, 28)

    tempArr = []
    for i in range(100):
        tempArr.append(Rectangle(i*18 + 25, 400, 14, 14, 'HiddenLayer1', i + 1))
    hiddenArr1 = np.array(tempArr)

    tempArr = []
    for i in range(100):
        tempArr.append(Rectangle(i*18 + 25, 200, 14, 14, 'HiddenLayer2', i + 1))
    hiddenArr2 = np.array(tempArr)

    tempArr = []
    for i in range(10):
        tempArr.append(Rectangle(i*25 + 800, 50, 20, 20, 'OutputLayer', i + 1))
    outputArr = np.array(tempArr)

    return inputArr, hiddenArr1, hiddenArr2, outputArr

def inputRect(list):
    for i in range(28):
        for j in range(28):
            list[i][j].drawRect()

def hiddenRect(list):
    for i in range(100):
        list[i].drawRect()

def outputRect(list):
    for i in range(10):
        list[i].drawRect()

def hoverInput(list):
    mousePos = pygame.mouse.get_pos()
    for i in range(28):
        for j in range(28):
            if j*15 + 715 < mousePos[0] < j*15 + 728 and i*15 + 575 < mousePos[1] < i*15 + 588:
                pygame.draw.rect(display, (123, 123, 123), [list[i][j].x - 2, list[i][j].y - 2, list[i][j].w + 4, list[i][j].h + 4])

                if list[i][j].index > 588:
                    clickOutput3(list[i][j])
                else:
                    clickOuput(list[i][j])

def hoverHidden(list, key):
    yCordinate = 0
    if key == 1:
        yCordinate = 400
    elif key == 2:
        yCordinate = 200

    mousePos = pygame.mouse.get_pos()
    for i in range(100):
        if i*18 + 25 < mousePos[0] < i*18 + 43 and yCordinate < mousePos[1] < yCordinate + 14:
            pygame.draw.rect(display, (123, 123, 123), [list[i].x - 2, list[i].y - 2, list[i].w + 4, list[i].h + 4])
            if key == 2:
                drawLinesOutHidd(hiddenArr2, i, hiddenArr1)
            else:
                drawLinesHiddInp(hiddenArr1, i, inputArr)

            if i >= 90:
                clickOutput2(list[i])
            else:
                clickOuput(list[i])

def hoverOutput(list):
    mousePos = pygame.mouse.get_pos()
    mouseClick = pygame.mouse.get_pressed()
    for i in range(10):
        if 800 + 25*i < mousePos[0] < 820 + 25*i and 50 < mousePos[1] < 70:
            pygame.draw.rect(display, (123, 123, 123), [list[i].x - 2, list[i].y - 2, list[i].w + 4, list[i].h + 4])
            drawLinesOutHidd(outputArr, i, hiddenArr2)
            clickOuput(list[i])

def drawLinesHiddInp(list1, pos, list2):
    for i in range(28):
        for j in range(28):
            pygame.draw.line(display, (222, 56, 122), (list1[pos].x + 8, list1[pos].y + 15),
                             (list2[i][j].x + 5, list2[i][j].y))

def drawLinesOutHidd(list1, pos, list2):
    for i in range(list2.size):
        pygame.draw.line(display, (222, 56, 122), (list1[pos].x+10, list1[pos].y+20), (list2[i].x+7, list2[i].y))

def clickOuput(node):
    pygame.draw.rect(display, (240, 200, 240), [node.x + node.w + 5, node.y - 40, 150, 140])
    font1 = pygame.font.Font('./fonts/comici.ttf', 20)
    layerName = font1.render('' + str(node.layerName), True, (0, 0, 0))
    thNode = font1.render('Node: ' + str(node.index), True, (0, 0, 0))
    output = font1.render('Output:', True, (0, 0, 0))
    value = font1.render('' + str(node.value), True, (23, 45, 67))

    display.blit(layerName, (node.x + node.w + 10, node.y - 35))
    display.blit(thNode, (node.x + node.w + 10, node.y - 5))
    display.blit(output, (node.x + node.w + 10, node.y + 25))
    display.blit(value, (node.x + node.w + 10, node.y+55))

def clickOutput2(node):
    pygame.draw.rect(display, (240, 200, 240), [node.x - 160, node.y - 40, 150, 140])
    font1 = pygame.font.Font('./fonts/comici.ttf', 20)
    layerName = font1.render('' + str(node.layerName), True, (0, 0, 0))
    thNode = font1.render('Node: ' + str(node.index), True, (0, 0, 0))
    output = font1.render('Output:', True, (0, 0, 0))
    value = font1.render('' + str(node.value), True, (23, 45, 67))

    display.blit(layerName, (node.x - 155, node.y - 35))
    display.blit(thNode, (node.x - 155, node.y - 5))
    display.blit(output, (node.x - 155, node.y + 25))
    display.blit(value, (node.x - 155, node.y + 55))

def clickOutput3(node):
    pygame.draw.rect(display, (240, 200, 240), [node.x + node.w + 5, node.y - 110, 150, 140])
    font1 = pygame.font.Font('./fonts/comici.ttf', 20)
    layerName = font1.render('' + str(node.layerName), True, (0, 0, 0))
    thNode = font1.render('Node: ' + str(node.index), True, (0, 0, 0))
    output = font1.render('Output:', True, (0, 0, 0))
    value = font1.render('' + str(node.value), True, (23, 45, 67))

    display.blit(layerName, (node.x + node.w + 10, node.y - 105))
    display.blit(thNode, (node.x + node.w + 10, node.y - 75))
    display.blit(output, (node.x + node.w + 10, node.y - 50))
    display.blit(value, (node.x + node.w + 10, node.y - 20))

def displayNumbers():
    font1 = pygame.font.Font('./fonts/comici.ttf', 24)
    for i in range(10):
        num = font1.render('' + str(i), True, (200, 12, 12))
        display.blit(num, (25*i + 803, 20))

def resetButton():
    font1 = pygame.font.Font('./fonts/comici.ttf', 24)
    pygame.draw.rect(display, (200, 215, 222), [12, 15, 80, 30])
    res = font1.render('Reset', True, (10, 10, 10))
    display.blit(res, (17, 12))

    mouseClick = pygame.mouse.get_pressed()
    mousePos = pygame.mouse.get_pos()
    if 13 < mousePos[0] < 93 and 15 < mousePos[1] < 45:
        pygame.draw.rect(display, (240, 240, 240), [12, 15, 80, 30])
        res = font1.render('Reset', True, (10, 10, 10))
        display.blit(res, (17, 12))
        if mouseClick[0] is 1:
            createModel()

def createModel():
    randomNum = random.randint(0, 9998)
    a = xTrain[randomNum, :]
    tempa = a
    a = a / 255
    a = np.reshape(a, [1, 784])

    features1, features2, features3 = getFeatures(a)

    features2 = features2/features2.max()
    features1 = features1/features1.max()

    img = np.reshape(tempa, [28, 28])

    for i in range(28):
        for j in range(28):
            inputArr[i][j].color = (2, img[i][j], 200)
            inputArr[i][j].value = img[i][j]

    for i in range(100):
        hiddenArr1[i].color = (2, int(features1[i]*220 + 30), 200)
        hiddenArr2[i].color = (2, int(features2[i]*220 + 30), 200)
        hiddenArr1[i].value = features1[i]
        hiddenArr2[i].value = features2[i]

    for i in range(10):
        outputArr[i].color = (2, int(features3[i]*250), 200)
        outputArr[i].value = features3[i].round(2)

xTrain = np.array(pd.read_csv('mnist_test.csv').iloc[:, 1:])
inputArr, hiddenArr1, hiddenArr2, outputArr = getArray()
createModel()
pygame.init()
display = pygame.display.set_mode((1850, 1000))
clock = pygame.time.Clock()

while True:
    display.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    resetButton()
    displayNumbers()
    outputRect(outputArr)
    hiddenRect(hiddenArr1)
    hiddenRect(hiddenArr2)
    inputRect(inputArr)

    hoverOutput(outputArr)
    hoverHidden(hiddenArr1, 1)
    hoverHidden(hiddenArr2, 2)
    hoverInput(inputArr)

    pygame.display.update()
    clock.tick(8)
