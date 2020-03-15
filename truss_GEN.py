'''
==================================================================
Space truss generate file
Req. python,numpy
2020.03
==================================================================
'''

'''
====================================================================
Import Part
====================================================================
'''
from FEM_truss import *
import numpy as np
import os
import random

import shutil
import csv
import ast

from os import listdir
from os.path import isfile, join


'''
====================================================================
Class Part
====================================================================
'''

class gen_model:
    def __init__(self,
        X_Val_Range_min,X_Val_Range_max,
        Z_Val_Range_min,Z_Val_Range_max,
        Avg_length,Diff_Space,xarea,
        loadx,loady,loadz,
        Young,
        c1,c2,c3,c4,c5,c6,c7):

        # Truss Dimension
        self.XRmin = X_Val_Range_min
        self.XRmax = X_Val_Range_max
        self.ZRmin = Z_Val_Range_min
        self.ZRmax = Z_Val_Range_max
        self.avgl = Avg_length
        self.diff = Diff_Space
        self.area = xarea

        # Truss load
        self.loadx = loadx
        self.loady = loady
        self.loadz = loadz

        # Truss Properties
        self.Young = Young

        # Generate lists and numbers
        self.n_u_x = []
        self.n_u_z = []
        self.n_l_coord = []
        self.n_u_coord = []
        self.n_l_name_div = []
        self.n_u_name_div = []

        # Parameters
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.c7 = c7

        self.model = None

        # Generate
        self.gennode()
        self.generate()

    # Function of X
    def _X_Val_Range(self,p):
        #Change the function of x in () after "return"
        return (  p  )

    # Function of Z
    def _Z_Val_Range(self,p):
        #Change the function of z in () after "return"
        return (  p  )

    # Y as a function of X and Z
    def _Y_Val_Range(self,x,z):

        return ((self.c1*(x**2) + self.c2*(x*z) + self.c3*(z**2) + self.c4*x + self.c5*z + self.c6 ) * self.c7)


    def gennode(self):
        self.n_u_x = []
        self.n_u_z = []
        self.n_l_coord = []
        self.n_u_coord = []
        self.n_l_name_div = []
        self.n_u_name_div = []

        for i in range(self.XRmin,self.XRmax+1):
            self.n_u_x.append(self._X_Val_Range(i))

        for i in range(self.ZRmin,self.ZRmax+1):
            self.n_u_z.append(self._Z_Val_Range(i))

        #Generate Node lower cord coordinates
        for i in range(len(self.n_u_x)-1):
            for j in range(len(self.n_u_z)-1):
                self.n_l_coord.append(
                    [self.avgl*(self.n_u_x[i]+self.n_u_x[i+1])/2,
                    (self.avgl*((self._Y_Val_Range(self.n_u_x[i],self.n_u_z[j])+self._Y_Val_Range(self.n_u_x[i+1],self.n_u_z[j+1]))/2))-self.diff,self.avgl*(self.n_u_z[j]+self.n_u_z[j+1])/2])
        #Generate Node upper cord coordinates
        for i in range(len(self.n_u_x)):
            for j in range(len(self.n_u_z)):
                self.n_u_coord.append([self.avgl*self.n_u_x[i],self.avgl*self._Y_Val_Range(self.n_u_x[i],self.n_u_z[j]),self.avgl*self.n_u_z[j]])

    def generate(self):
        l1 = Load()
        l1.set_name(1)
        l1.set_type(1)
        l1.set_size(self.loadx,self.loady,self.loadz,self.loadx,self.loady,self.loadz)
        '''
        ==================================
        Generate Node
        ==================================
        '''
        n = 'n'
        #Generate Node lower cord
        n_l_name=[]
        counter = 1
        for i in range(len(self.n_l_coord)):
            n_l_name.append(n+str(counter))
            counter+=1
        counter = 1
        for i in range(len(self.n_l_coord)):
            n_l_name[i] = Node()
            n_l_name[i].set_name(counter)
            n_l_name[i].set_coord(self.n_l_coord[i][0],self.n_l_coord[i][1],self.n_l_coord[i][2])
            n_l_name[i].set_res(0,0,0)
            counter+=1

        #Generate Node upper cord
        n_u_name=[]
        counter = 1
        for i in range(len(self.n_u_coord)):
            n_u_name.append(n+str(counter))
            counter+=1
        counter = 1
        for i in range(len(self.n_u_coord)):
            n_u_name[i] = Node()
            n_u_name[i].set_name(counter)
            n_u_name[i].set_coord(self.n_u_coord[i][0],self.n_u_coord[i][1],self.n_u_coord[i][2])
            n_u_name[i].set_res(0,0,0)
            counter+=1

        #Divide n_l_name and n_u_name into zrow

        for i in range(len(self.n_u_z)-1):
            self.n_l_name_div.append([])
        for i in range(len(n_l_name)):
            for j in range(len(self.n_u_z)-1):
                if n_l_name[i].coord[2] == (self.n_u_z[j]+self.n_u_z[j+1])/2:
                    self.n_l_name_div[j].append(n_l_name[i])

        for i in range(len(self.n_u_z)):
            self.n_u_name_div.append([])
        for i in range(len(n_u_name)):
            for j in range(len(self.n_u_z)):
                if n_u_name[i].coord[2] == self.n_u_z[j]:
                    self.n_u_name_div[j].append(n_u_name[i])

        #Set load to upper nodes
        for i in range(len(n_u_name)):
            n_u_name[i].set_load(l1)

        #Set node name
        node_pool =[]
        for i in range(len(n_l_name)):
            node_pool.append(n_l_name[i])
        for i in range(len(n_u_name)):
            node_pool.append(n_u_name[i])
        counter = 1
        for i in range(len(node_pool)):
            node_pool[i].set_name(counter)
            counter +=1

        '''
        ==================================
        Generate Member
        ==================================
        '''

        e = 'e'
        #E lower cord
        E_type1_name =[]
        counter = 1
        for num in range(len(self.n_l_name_div)):
            for i in range(len(self.n_l_name_div[num])-1):
                E_type1_name.append(e+str(counter))
                E_type1_name[-1] = Element()
                E_type1_name[-1].set_name(str(counter))
                E_type1_name[-1].set_nodes(self.n_l_name_div[num][i],self.n_l_name_div[num][i+1])
                E_type1_name[-1].set_em(self.Young)
                E_type1_name[-1].set_area(self.area)
                counter+=1

        #E lower connect lower
        E_type2_name =[]
        counter = len(E_type1_name)+1
        for num in range(len(self.n_l_name_div)-1):
            for i in range(len(self.n_l_name_div[num])):
                E_type2_name.append(e+str(counter))
                E_type2_name[-1] = Element()
                E_type2_name[-1].set_name(str(counter))
                E_type2_name[-1].set_nodes(self.n_l_name_div[num][i],self.n_l_name_div[num+1][i])
                E_type2_name[-1].set_em(self.Young)
                E_type2_name[-1].set_area(self.area)
                counter+=1

        #E lower Diagonal
        E_type3_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+1
        for num in range(len(self.n_l_name_div)-1):
            for i in range(len(self.n_l_name_div[num])-1):
                E_type3_name.append(e+str(counter))
                E_type3_name[-1] = Element()
                E_type3_name[-1].set_name(str(counter))
                E_type3_name[-1].set_nodes(self.n_l_name_div[num][i],self.n_l_name_div[num+1][i+1])
                E_type3_name[-1].set_em(self.Young)
                E_type3_name[-1].set_area(self.area)
                counter+=1

        #E lower connect upper 1-1
        E_type4_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+1
        for num in range(len(self.n_l_name_div)):
            for i in range(len(self.n_l_name_div[num])):
                E_type4_name.append(e+str(counter))
                E_type4_name[-1] = Element()
                E_type4_name[-1].set_name(str(counter))
                E_type4_name[-1].set_nodes(self.n_l_name_div[num][i],self.n_u_name_div[num][i])
                E_type4_name[-1].set_em(self.Young)
                E_type4_name[-1].set_area(self.area)
                counter+=1

        #E lower connect upper 1-2
        E_type5_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+len(E_type4_name)+1
        for num in range(len(self.n_l_name_div)):
            for i in range(len(self.n_l_name_div[num])):
                E_type5_name.append(e+str(counter))
                E_type5_name[-1] = Element()
                E_type5_name[-1].set_name(str(counter))
                E_type5_name[-1].set_nodes(self.n_l_name_div[num][i],self.n_u_name_div[num][i+1],)
                E_type5_name[-1].set_em(self.Young)
                E_type5_name[-1].set_area(self.area)
                counter+=1

        #E lower connect upper 2-1
        E_type6_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+len(E_type4_name)+len(E_type5_name)+1
        for num in range(len(self.n_l_name_div)):
            for i in range(len(self.n_l_name_div[num])):
                E_type6_name.append(e+str(counter))
                E_type6_name[-1] = Element()
                E_type6_name[-1].set_name(str(counter))
                E_type6_name[-1].set_nodes(self.n_l_name_div[num][i],self.n_u_name_div[num+1][i])
                E_type6_name[-1].set_em(self.Young)
                E_type6_name[-1].set_area(self.area)
                counter+=1

        #E lower connect upper 2-1
        E_type7_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+len(E_type4_name)+len(E_type5_name)+len(E_type6_name)+1
        for num in range(len(self.n_l_name_div)):
            for i in range(len(self.n_l_name_div[num])):
                E_type7_name.append(e+str(counter))
                E_type7_name[-1] = Element()
                E_type7_name[-1].set_name(str(counter))
                E_type7_name[-1].set_nodes(self.n_l_name_div[num][i],self.n_u_name_div[num+1][i+1])
                E_type7_name[-1].set_em(self.Young)
                E_type7_name[-1].set_area(self.area)
                counter+=1

        #E upper cord
        E_type8_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+len(E_type4_name)+len(E_type5_name)+len(E_type6_name)+len(E_type7_name)+1
        for num in range(len(self.n_u_name_div)):
            for i in range(len(self.n_u_name_div[num])-1):
                E_type8_name.append(e+str(counter))
                E_type8_name[-1] = Element()
                E_type8_name[-1].set_name(str(counter))
                E_type8_name[-1].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num][i+1])
                E_type8_name[-1].set_em(self.Young)
                E_type8_name[-1].set_area(self.area)
                counter+=1

        #E upper connect upper
        E_type9_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+len(E_type4_name)+len(E_type5_name)+len(E_type6_name)+len(E_type7_name)+len(E_type8_name)+1
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[num])):
                E_type9_name.append(e+str(counter))
                E_type9_name[-1] = Element()
                E_type9_name[-1].set_name(str(counter))
                E_type9_name[-1].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num+1][i])
                E_type9_name[-1].set_em(self.Young)
                E_type9_name[-1].set_area(self.area)
                counter+=1

        #E upper Diagonal
        E_type10_name =[]
        counter = len(E_type1_name)+len(E_type2_name)+len(E_type3_name)+len(E_type4_name)+len(E_type5_name)+len(E_type6_name)+len(E_type7_name)+len(E_type8_name)+len(E_type9_name)+1
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[num])-1):
                E_type10_name.append(e+str(counter))
                E_type10_name[-1] = Element()
                E_type10_name[-1].set_name(str(counter))
                E_type10_name[-1].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num+1][i+1])
                E_type10_name[-1].set_em(self.Young)
                E_type10_name[-1].set_area(self.area)
                counter+=1

        '''
        ==================================
        Generate Model
        ==================================
        '''
        self.model = Model()
        # Add load
        self.model.add_load(l1)
        # Add nodes
        for i in range(len(n_l_name)):
            self.model.add_node(n_l_name[i])
        for i in range(len(n_u_name)):
            self.model.add_node(n_u_name[i])
        # Add Support Conditions
        nodes_y = []
        for i in range(len(self.model.nodes)):
            nodes_y.append(self.model.nodes[i].coord[1])
        for i in range(len(nodes_y)):
            if min(nodes_y) == nodes_y[i]:
                self.model.nodes[i].set_res(1,1,1)
        # Add elements
        #E_type1
        for i in range(len(E_type1_name)):
            self.model.add_element(E_type1_name[i])
        #E_type2
        for i in range(len(E_type2_name)):
            self.model.add_element(E_type2_name[i])
        #E_type3
        for i in range(len(E_type3_name)):
            self.model.add_element(E_type3_name[i])
        #E_type4
        for i in range(len(E_type4_name)):
            self.model.add_element(E_type4_name[i])
        #E_type5
        for i in range(len(E_type5_name)):
            self.model.add_element(E_type5_name[i])
        #E_type6
        for i in range(len(E_type6_name)):
            self.model.add_element(E_type6_name[i])
        #E_type7
        for i in range(len(E_type7_name)):
            self.model.add_element(E_type7_name[i])
        #E_type8
        for i in range(len(E_type8_name)):
            self.model.add_element(E_type8_name[i])
        #E_type9
        for i in range(len(E_type9_name)):
            self.model.add_element(E_type9_name[i])
        #E_type10
        for i in range(len(E_type10_name)):
            self.model.add_element(E_type10_name[i])

        self.model.gen_all()



