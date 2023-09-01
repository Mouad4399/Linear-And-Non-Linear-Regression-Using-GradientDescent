# import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym # sympy to compute the partial derivatives

# from IPython import display
# display.set_matplotlib_formats('svg')

x=np.random.randn(20)
y=x-np.random.randn(20)/2

W1, W0 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

def compute_loss(w1_val, w0_val):
    Y_ = w1_val * x + w0_val
    loss = np.sum((Y_ - y)**2) / (2)
    return loss

loss_values = np.vectorize(compute_loss)(W1, W0)

# plt.imshow(np.flip(loss_values,0),extent=[-2,2,-2,2])



sw1,sw0 = sym.symbols('sw1,sw0')

sZ = np.sum((sw1 * x + sw0 - y)**2) * 0.5

df_w1 = sym.lambdify((sw1,sw0),sym.diff(sZ,sw1),'sympy')
df_w0 = sym.lambdify((sw1,sw0),sym.diff(sZ,sw0),'sympy')




training_epochs = 1000
learning_rate = 0.01
localmin = np.random.rand(2)*4-2 # also try specifying coordinates

startpnt = localmin[:] # make a copy, not re-assign

trajectory = np.zeros((training_epochs,2))
for i in range(training_epochs):
  grad = np.array([ df_w1(localmin[0],localmin[1]), 
                    df_w0(localmin[0],localmin[1]) 
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
plt.plot([[np.min(x) ],[ np.max(x)]], [ [np.min(x)*localmin[0] + localmin[1]] ,[np.max(x)*localmin[0] + localmin[1] ]] , 'r')

# plt.plot(trajectory[:,0],'g')
plt.show()



from sklearn.linear_model import LinearRegression
x=x.reshape((-1,1))
model = LinearRegression()
model.fit(x, y)
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")