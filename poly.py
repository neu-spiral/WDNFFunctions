import cvxopt
from Toplogy_gen import Problem
import math
import numpy as np
from scipy.misc import comb
from decimal import *
from random_replacement import ro_uv,generateRandomPlacement


def nCr(n,r):
    return comb(n,r,True)


class poly():
    """A class implementing a polynomial consisting of monomials with (a) negative literals and (b) integer terms """
    def __init__(self,coefficients={},sets={}):
        """ Coefficients is a dictionary containing monomial indexes as keys and 
            coefficients as values. Sets is also a dict containing monomial 
            terms as keys and sets as  values.
        """
        self.coefficients = coefficients
        self.sets = sets


    def evaluate(self,X):
        """ Given dictionary x, evaluate p(x) """
        sumsofar = 0.0
        for j in self.coefficients:
	    beta = self.coefficients[j]
            setofx = self.sets[j]
            prod = beta
	    for (v,i) in setofx:
                 prod = prod * (1.0-X[v,i])
            sumsofar = sumsofar + prod
        return sumsofar


    def product(self,p):
        """  Given a poly p, return poly self*p
        """       
        new_coefficients = dict([   ((key1,key2), self.coefficients[key1]*p.coefficients[key2])         for key1 in self.coefficients for key2 in p.coefficients])
        new_sets =   dict([   ((key1,key2), self.sets[key1].union(p.sets[key2])) for key1 in self.sets for key2 in p.sets])      
        return poly(new_coefficients,new_sets)


    def power(self,k):
        """ Return poly (self)**k. k must be greater that or equal to 1.
        """
        power_poly = self
        i = 1
        while i < k:
            power_poly = power_poly.product(self)
	    i +=1
        return power_poly


class taylor():
    """ A class computing the taylor expansion of a function"""
    def __init__(self,F,x0,m):
        self.F = F
        self.x0 = x0
        self.m = m


    def evaluate(self,x):
        """ Evaluate taylor approx at x"""
        xx = x
        out = 0.
        for i in range(self.m+1):
            centered_xk  = (xx-self.x0)**i
            terms =  self.F[i]*centered_xk/math.factorial(i)
            out  += centered_xk*terms
        return out


    def expand(self):
        """ Return the coefficients of the expanded polynomial """
        self.alpha = {}
        for i in range(self.m+1):
            self.alpha[i] = 0.
        for i in range(self.m+1):
            for j in range(i,self.m+1):
                if j-i >0:
                    self.alpha[i] += self.F[j] *nCr(j,i)/math.factorial(j) * (-self.x0)**(j-i)
		else:
		    self.alpha[i] += self.F[j] *nCr(j,i)/math.factorial(j)
        return self.alpha


    def evaluate_expanded(self,xk):
        """ Compute the expanded polynomial, using xk dictionary containing powers x^k
        """
	out = self.alpha[0]
        for i in xk:
            out = out+ xk[i]*self.alpha[i]
        return out    


def rho_uv_dicts(P):
    """Given P, problem instance, as input and generate  2 dictionaries per edge, one including the coefficients of that edge
       and one including the sets, used to describe rho_uv as a polynomial of class poly"""
    rho_uvs_coefficients = {}
    rho_uvs_sets = {}
    for edge in P.EDGE:
        rho_uvs_coefficients[edge] = {}
        rho_uvs_sets[edge] = {}

    index = 0
    for demand in P.demands:
        item = demand['item']
        path = demand['path']
        rate = demand['rate']

        nodes_so_far = []
        for i in range(len(path)-1):
                edge = (path[i],path[i+1])
                nodes_so_far.append(path[i])
                rho_uvs_coefficients[edge][index] = 1.*rate/P.EDGE[edge] #coefficient for this term is the arrival rate divided by the mu
                rho_uvs_sets[edge][index] = set(  [ (v,item)  for v in nodes_so_far]) #the set of a term contains all (v,item) pers above it in the path
        index += 1  #indices capture demands. note that each demand passes through an edge only once, so no need for an index per edge
#
    return rho_uvs_coefficients, rho_uvs_sets
#
#def Xtodict(X):
#    """Given binary matrix X, convert it to a dictionary of (row,column) as keys and X[row,column] as values
#    """
#    rows,cols = np.matrix(X).shape
#    return dict( [ ((r,c),X[r,c]) for r in range(rows) for c in range(cols)])		
#


#class rho(poly):
#    def evaluate(self,X):
#        """ Take as input a matrix, list of items corresponding to index j, and evaluate rho """
#        x= Xtodict(X)
#        return poly.evaluate(self,x)





if __name__=="__main__":

    for i in range(10):
        r_val = 0.1*i
        taylor_approx = taylor( 4*[np.exp(r_val)],r_val,3 ) 
        taylor_approx.expand()
        r = r_val+0.01
        print np.exp(r),taylor_approx.evaluate(r),taylor_approx.evaluate_expanded( dict([ (i+1,r**(i+1))    for i in range(3)]   ))




    P = Problem.unpickle_cls("problem_abilene_1000demands_300catalog_size_mincap_30maxcap_30_100_uniform")
    # Problem class is defined in Toplogy_gen.





    X = generateRandomPlacement(P)



    print X
    rhos = ro_uv(X,P)

    rho_uvs_coefficients,  rho_uvs_sets = rho_uv_dicts(P)
    print rho_uvs_coefficients[(0, 8)],rho_uvs_sets[(0,8)]
    for edge in rho_uvs_coefficients:
        rho_poly = poly(rho_uvs_coefficients[edge], rho_uvs_sets[edge])
        r_val= rho_poly.evaluate(X)



        print 'For edge',edge,'rho is',rhos[edge],'calculated via poly is',r_val


        #taylor_approx = taylor( 4*[np.exp(r_val)],r_val,3 ) 
     
        #taylor_approx.expand()
        #r = r_val+0.01
        #print np.exp(r),taylor_approx.evaluate(r),taylor_approx.evaluate_expanded( dict([ (i+1,r**(i+1))    for i in range(3)]   ))
	



        rk ={}
        i=1
        rk[i] = rho_poly
        #print rho_poly.coefficients,rho_poly.sets
        while i<2:
            i +=1
            rk[i] = rk[i-1].product(rho_poly)
            print 'For edge',edge,'rho^'+str(i),'is',rhos[edge]**i,'calculated via poly is',rk[i].evaluate(X)
	    
            #print rk[i].coefficients,rk[i].sets


    
        ##test taylor expansion with f(x) =np.exp(x).