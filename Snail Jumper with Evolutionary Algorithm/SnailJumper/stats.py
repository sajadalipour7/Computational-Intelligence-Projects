from audioop import avg
import matplotlib.pyplot as plt

max_gen=[]
min_gen=[]
avg_gen=[]
f = open("stats.txt", "r")
x=f.readline()
while x!=None and x!='':
    max_gen.append(int(x))
    x=f.readline()
    min_gen.append(int(x))
    x=f.readline()
    avg_gen.append(float(x))
    x=f.readline()
f.close()

plt.plot(max_gen,label='Max')
plt.plot(min_gen,label='Min')
plt.plot(avg_gen,label='Average')
plt.legend()
plt.show()