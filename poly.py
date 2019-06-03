from Toplogy_gen import Problem
import math
import numpy as np
from scipy.misc import comb


def nCr(n,r):
    #f = math.factorial
    #return f(n) / f(r) / f(n-r)
    return comb(n,r,True)


class poly():
    """A class implementing a polynomial consisting of monomials with (a) negative literals and (b) integer terms """
    def __init__(self,coefficients,sets={}):
        """ Coefficients is a dictionary containing monomial indexes as keys and 
            coefficients as values. Sets is also a dict containing monomial 
            terms as keys and sets as  values.
        """
        
        self.coefficients = coefficients
        self.sets = sets

    def evaluate(self,X):
        """ Given matrix x, evaluate rho_e(x) """

        beta = self.coefficients
        setofx = self.sets
        prod = 1.0
        for (v,i) in setofx:
            if X[v,i] != 1:
                prod = prod * (1.0-X[v,i])
            else:
                prod = 0.0
                break
        return beta*prod

    def product(self,p):
        """  Given a poly p, return poly self*p
        """       
        new_coefficients = self.coefficients*p.coefficients
        new_sets = self.sets.union(p.sets)
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
        self.F = F # dictionary of C_e^(i)
        self.x0 = x0
        self.m = m
        self.alpha = self.expand() # generate a dictionary for alpha

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
        alpha = {}
        for i in range(self.m+1):
            alpha[i] = 0.
        for i in range(self.m+1):
            for j in range(i,self.m+1):
                if j-i >0:
                    alpha[i] += self.F[j] *nCr(j,i)/math.factorial(j) * (-self.x0)**(j-i)
                else:
                    alpha[i] += self.F[j] *nCr(j,i)/math.factorial(j)
        return alpha

    def evaluate_expanded(self,xk):
        """ Compute the expanded polynomial, using xk dictionary containing powers x^k
        """
        out = self.alpha[0] # ? do not need alpha_e^(0)
        for i in xk:
            out = out+ xk[i]*self.alpha[i]
        return out    

    def evaluate_expanded_mu(self,xk):
        out = 0.0
        for i in xk:
            out = out + xk[i]*self.alpha[i]*i
        return out

def rho_uv_dicts(demands, service_rate):
    """Given demands and service rate as input and generate  2 dictionaries per edge, one including the coefficients of that edge
       and one including the sets, used to describe rho_uv as a polynomial of class poly"""
    rho_uvs_coefficients = {}
    rho_uvs_sets = {}

    for demand in demands:
        item = demand.item
        path = demand.path
        rate = demand.rate

        nodes_so_far = []
        for i in range(len(path)-1):
            edge = (path[i],path[i+1])
            nodes_so_far.append(path[i])
            rho_coefficient = 1.*rate/service_rate[edge][demand]
            rho_set = set(  [ (v,item)  for v in nodes_so_far])
            if edge in rho_uvs_coefficients:
                rho_uvs_coefficients[edge][demand] = rho_coefficient #coefficient for this term is the arrival rate divided by the mu
                rho_uvs_sets[edge][demand] = rho_set #the set of a term contains all (v,item) pers above it in the path
            else:
                rho_uvs_coefficients[edge] = {demand: rho_coefficient}
                rho_uvs_sets[edge] = {demand: rho_set}

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



