# import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym # sympy to compute the partial derivatives

# from IPython import display
# display.set_matplotlib_formats('svg')

x=np.random.randn(20)
y=x**2 -np.random.randn(20)**2 /2

W2,W1, W0 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100),np.linspace(-2, 2, 100), indexing = 'ij')

def compute_loss(w2_val,w1_val, w0_val):
    Y_ = w2_val* x**2 + w1_val * x + w0_val
    loss = np.sum((Y_ - y)**2) / (2)
    return loss

loss_values = np.vectorize(compute_loss)(W2,W1, W0)

# loss_values.shape
# plt.imshow(loss_values)


sw2,sw1,sw0 = sym.symbols('sw2,sw1,sw0')

sZ = np.sum((sw2* x**2 + sw1 * x + sw0 - y)**2) * 0.5

df_w2 = sym.lambdify((sw2,sw1,sw0), sym.diff(sZ,sw2),'sympy')
df_w1 = sym.lambdify((sw2,sw1,sw0),sym.diff(sZ,sw1),'sympy')
df_w0 = sym.lambdify((sw2,sw1,sw0),sym.diff(sZ,sw0),'sympy')





training_epochs = 1000
learning_rate = 0.01
localmin = np.random.rand(3)*4-2 # also try specifying coordinates

startpnt = localmin[:] # make a copy, not re-assign

trajectory = np.zeros((training_epochs,3))
for i in range(training_epochs):
  grad = np.array([ df_w2(localmin[0],localmin[1],localmin[2]),
                    df_w1(localmin[0],localmin[1],localmin[2]), 
                    df_w0(localmin[0],localmin[1],localmin[2]) 
                  ])
  localmin = localmin - learning_rate*grad  # add _ or [:] to change a variable in-place
  trajectory[i,:] = localmin



print(localmin)
# plt.subplot(1,2,1)
# plt.imshow(np.flip(loss_values,0),extent=[-2,2,-2,2],vmin=-10,vmax=10)
# plt.plot(startpnt[0],startpnt[1],'bs')
# plt.plot(localmin[0],localmin[1],'ro')
# plt.plot(trajectory[:,0],trajectory[:,1],'r')
# plt.legend(['rnd start','local min'])
# plt.colorbar()
# plt.subplot(1,2,2)

plt.plot(x,y,'bs')

p = np.linspace(np.min(x) , np.max(x) , 20 )
plt.plot(p, localmin[0]*p**2 + localmin[1]* p + localmin[2] , 'r')

# plt.plot(trajectory[:,0],'g')
plt.show()
