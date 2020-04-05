import numpy as np
import os, glob, random, cv2,cv2.ml
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time



def pca(data, k):
    data = np.float32(np.mat(data))
    rows, cols = data.shape  # 取大小
    data_mean = np.mean(data, 0)  # 求均值
    Z = data - np.tile(data_mean, (rows, 1))
    D, V = np.linalg.eig(Z * Z.T)  # 特征值与特征向量
    eigSortIndex = np.argsort(-D)
    #V1 = V[:, :k]  # 取前k个特征向量
    V1 = V[:, eigSortIndex[0:k]]
    V1 = Z.T * V1
    for i in range(k):  # 特征向量归一化
        V1[:, i] /= np.linalg.norm(V1[:, i])
    return np.array(Z * V1), data_mean, V1


def loadImageSet(folder=r'E:\test\face_recognition2', sampleCount = 8):  # 加载图像集，随机选择sampleCount张图片用于训练
    trainData = [];
    testData = [];
    yTrain = [];
    yTest = [];
    for k in range(10):
        yTrain1 = [k]
        yTest1 = [k]
        folder2 = os.path.join(folder, 's%d' % (k + 1))
        size = (128, 128)
        data=[]
        haar = cv2.CascadeClassifier(r'E:\clear f\haarcascade_frontalface_alt2.xml')
        for d in glob.glob(os.path.join(folder2, '*.jpg')):
            img=cv2.imread(d,0)
            # img2=cv2.resize(img,size)
            faceRects = haar.detectMultiScale(img, 1.2, 5)
            if len(faceRects) > 0:  # 大于0则检测到人脸
                for faceRect in faceRects:  # 单独框出每一张人脸
                    x, y, w, h = faceRect
                    # cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), color, 3)
                    crop = img[y:(h + y), x:(w + x)]
                    res=cv2.resize(crop, size)
                    # cv2.imshow('img',res)
                    # cv2.waitKey(0)
            # data.append(cv2.resize(crop, size))
            data.append(res)
            # cv2.imshow('img',y)
            # cv2.waitKey(0)
        #sample=[2,1,8,5,7]
        #samplecount = 5
        sample = random.sample(range(10), sampleCount)
        trainData.extend([data[i].ravel() for i in range(10) if i in sample])
        testData.extend([data[i].ravel() for i in range(10) if i not in sample])
        yTest.extend([k] * (10 - sampleCount))
        yTrain.extend([k] * sampleCount)
    return np.array(trainData), np.array(yTrain), np.array(testData), np.array(yTest)


def main():
    time_begin=time.time()
    xTrain_, yTrain, xTest_ , yTest = loadImageSet()
    num_train, num_test = xTrain_.shape[0], xTest_.shape[0]

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(xTrain_, yTrain)
    XPredict = (lda.score(xTest_, yTest))
    print(u'LDA-WITHOUT-PCA 法识别率: %.2f%%' % (XPredict.mean() * 100))

    xTrain, data_mean, V = pca(xTrain_, 20)
    xTest = np.array((xTest_ - np.tile(data_mean, (num_test, 1))) * V)  # 得到测试脸在特征向量下的数据

    yPredict = [yTrain[np.sum((xTrain - np.tile(d, (num_train, 1))) ** 2, 1).argmin()] for d in xTest]
    print(u'Euclidean_Distance法识别率: %.2f%%' % ((yPredict == yTest).mean() * 100))

    clf = SVC(C=0.1, cache_size=200, class_weight='balanced', coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    clf.fit(xTrain, yTrain)
    predict = clf.predict(xTest)
    print(u'SVM-WITH-PCA识别率: %.2f%%' % ((predict == np.array(yTest)).mean() * 100))
    #CPredict = (clf.score(xTest, yTest))
    #print(u'SVM-WITHOUT-PCA 法识别率: %.2f%%' % (CPredict.mean() * 100))

    #lda = LinearDiscriminantAnalysis(n_components=2)
    model = lda.fit(xTrain, yTrain)
    PCA_Predict = (model.score(xTest, yTest))
    print(u'LDA-WITH-PCA法识别率: %.2f%%' % (PCA_Predict.mean() * 100))
    time_over = time.time()
    print(time_over-time_begin,'s')

    # ldaa = LinearDiscriminantAnalysis(n_components=20)
    # x = ldaa.fit_transform(xTrain_, yTrain)
    # SVM = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
    #           decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
    #           max_iter=-1, probability=False, random_state=None, shrinking=True,
    #           tol=0.001, verbose=False)
    # SS=SVM.fit(x, yTrain)
    # Spredict = SS.predict(xTest)
    # print(u'SVM-WITH-PCA-LDA识别率: %.2f%%' % ((Spredict == np.array(yTest)).mean() * 100))
if __name__ == '__main__':
    main()