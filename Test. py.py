import numpy as np
import matlotlib pyplot as plt
m=3.0
g=9.8
r=2.0
I=12.0
h=0.0025
I=2.0
cycle=10.0
t=np.arrange(0,cycle,h)
n=(int)((cycle)/h)
initial_angle=90.0
y=np.zeroes(n)
v=np.zeroes(n)
def accel(theta)
return-(m*g*r/I)*np.sin(theta)
y[0]=np.radians(initial_angle)
v[0]=np.radians(0.0)
for i in range (0,n-I)
k1y=h*v[i]
k1v=h*accel(y[i])
k2y=h*(v[i]+0.5*k1v)
k2v=h*accel(y[i]+0.5*k1v)
k3y=h*(v[i])0.5*k2v)
k3v=h*accel(y[i]+0.5*k2y)
k4y=h*(v[i]+k3v)
k4v=h*accel (y[i]+k3y)
y[i+1]=y[i]+(k1y+2*k2y+2*k3y+k4y)/6.0
v[i+1]=v[i]+(k1v+2*k2v+2*k3v+k4v)/6.0
plt.plot(t,y)
plt.title('pendulum motion:')
plt.xlabel('time(s)')
plt.ylabel('angle (rad)')
plt.grid(true)
plt.show()
