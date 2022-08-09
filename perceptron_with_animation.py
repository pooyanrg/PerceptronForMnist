import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.animation as animation

MnistTrainX = sio.loadmat ('../../../datasets/mnist/MnistTrainX')['MnistTrainX']
MnistTrainY = sio.loadmat ('../../../datasets/mnist/MnistTrainY')['MnistTrainY']
X = MnistTrainX
T = MnistTrainY
(N,d) = X.shape

K = 4
C = 10
maxEpoch = 2
#W = np.random.rand(K*C, d)
WPos = np.random.rand(K*C, d)
WNeg = np.random.rand(K*C, d)
WNegRecent = np.zeros([K*C, d])
b = np.zeros (K*C)

fig = plt.figure()
#ax = fig.add_subplot(111)
#mpl.pyplot.ion()
metadata = dict(title='Perceptron', artist='Ghiasi')
pillowWriter = animation.writers['pillow']
moviewriter = pillowWriter(fps=5, metadata=metadata)
moviewriter.setup(fig=fig, outfile='perceptron.gif', dpi=600)
plt.show(block=False)

for i in range (maxEpoch):
    for j in range (N):
        x = X[j,:]
        t = int(T[j])
        z = np.dot(WPos-WNeg, x) + b
        l = np.argmax (z)
        y = int (l / K)
        if (y != t):
            #W[l,:] -= x
            WNeg[l,:] += x
            WNegRecent[l,:] += x
            m = t*K + np.argmax(z[t*K:(t+1)*K])
            #W[m,:] += x
            WPos[m,:] += x
        if (j % 1000==0):
            print (j)
            wholeImage = np.zeros ([3*K * 29, C * 29])
            for u in range (C):
                for v in range (K):
                    img = np.reshape(WPos[u*K+v,:], [28,28])
                    img = img / (np.abs(np.max(img))+0.00000001)
                    wholeImage[v * 29:(v+1)* 29 -1, u * 29:(u+1) * 29-1] = img
                    img = np.reshape(WNeg[u*K+v,:], [28,28])
                    img = img / (np.abs(np.max(img))+0.00000001)
                    wholeImage[(v+K) * 29:(v+K+1)* 29 -1, u * 29:(u+1) * 29-1] = img            
                    img = np.reshape(WNegRecent[u*K+v,:], [28,28])
                    img = img / (np.abs(np.max(img))+0.00000001)
                    wholeImage[(v+2*K) * 29:(v+2*K+1)* 29 -1, u * 29:(u+1) * 29-1] = img            
                    
            plt.imshow (wholeImage, cmap='gray')
            moviewriter.grab_frame()

            fig.canvas.draw()
            fig.canvas.flush_events()

        if (j % 1000==1):
            WNegRecent = WNegRecent * 0
moviewriter.finish()
