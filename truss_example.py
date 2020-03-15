'''
==================================================================
Eample file for Space truss analysis and export
Req. python,numpy
2020.03
==================================================================
'''

'''
====================================================================
Import Part
====================================================================
'''
from truss_GEN import *

'''
====================================================================
Parameter Part
====================================================================
'''

xmin = -10 # minimum x coordinate (horizontal) value (m.)
xmax = 10 # maximum x coordinate (horizontal) value (m.)
zmin = -10 # minimum z coordinate (horizontal) value (m.)
zmax = 10 # maximum z coordinate (horizontal) value (m.)
avg = 1 # member's averge length (m.)
diff = 1 # space between upper cord and lower cord  (m.)
area = 0.05 # member's sectional area  (sqm.)
lx = 0 # load in x direction (N)
ly = -1000 # load in y direction (N)
lz = 0 # load in z direction (N)
Young = 10000000000 # member's Young modulus (N/sqm)


'''
====================================================================
Form Parameter Part
y (heigth) of each node will be represented by
y = (c1*(x**2) + c2*(x*z) + c3*(z**2) + c4*x + c5*z + c6) * c7
or please feel free to hack at
frame_GEN
class gen_model's def _Y_Val_Range
====================================================================
'''
c1= 1
c2= 0
c3= 1
c4= 0
c5= 0
c6= 0
c7= -0.1

'''
====================================================================
Generate model
====================================================================
'''
model_X = gen_model(xmin,xmax,zmin,zmax,avg,diff,area,lx,ly,lz,Young,c1,c2,c3,c4,c5,c6,c7)

'''
====================================================================
Print out global deformations of each node
====================================================================
'''
for i in range(len(model_X.model.nodes)):
    print(model_X.model.nodes[i].global_d)

'''
====================================================================
Print out Strain energy of the model
====================================================================
'''
print(model_X.model.U_full[0][0])

'''
====================================================================
Export as .obj file
====================================================================
'''
name = 'Truss.obj'
model_X.model.gen_obj(name)
