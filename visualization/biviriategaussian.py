
import math
def comupte_bivariate_gaussian(ux,uy,sigmax,sigmay, x,y):
    res = ((x-ux) ** 2 )/(2*sigmax*sigmax) + ((y-uy) ** 2 )/(2*sigmay*sigmay)
    res = math.exp(-res)
    res = res/(2*math.pi * sigmax * sigmay)
    return res

sigmax=0.3
sigmay = 0.3
x = 2
y = 2

ux =2
uy = 1

res = comupte_bivariate_gaussian(ux,uy,sigmax,sigmay, x,y)
print(res)