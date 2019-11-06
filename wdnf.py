import math
import numpy as np
from scipy.misc import comb
from decimal import *


def nCr(n, r):
    return comb(n, r, True)


def merge(t1, t2):
    return tuple(sorted(set(t1).union(set(t2))))


class wdnf():
    """A class implementing a polynomial in Weighted Disjunctive Normal Form (WDNF)
    consisting of monomials with (a) negative or positive literals and (b) integer terms
    """
    def __init__(self, coefficients={}, sign=-1): #adjusted
        """ Coefficients is a dictionary containing tuples with indexes of the set
            elements as keys and coefficients as values. Sign denotes whether the WDNF
            formed with negative literals or positive literals.
        """
        self.coefficients = coefficients
        self.sign = sign


    def evaluate(self, x): #adjusted
        """ Given a dictionary x, evaluate wdnf(x) at the values x.
        """
        sumsofar = 0.0
        for i in self.coefficients:
            beta = self.coefficients[i]
            setofx = i
            prod = beta
            for j in setofx:
                if  self.sign == -1:
                    prod = prod * (1.0-x[j])
                else:
                    prod = prod * x[i]
            sumsofar = sumsofar + prod
        return sumsofar


    def __add__(self, another): #adjusted
        """ Add two polynomials in WDNF and return the resulting WDNF
        """
        if self.sign != another.sign:
            print('Two WDNF polynomials cannot be added')
            return
        new_coefficients = self.coefficients.copy() #empty dict for empty wdnf
        if not another.coefficients:
            return self
        elif not self.coefficients:
            return another
        else:
            for key in another.coefficients:
                if key in self.coefficients.keys():
                    new_coefficients[key] += another.coefficients[key]
                else:
                    new_coefficients[key] = another.coefficients[key]
        return wdnf(new_coefficients, self.sign)


    def __mul__(self, another): #adjusted
        """ Multiply two polynomials in WDNF and return the resulting WDNF
        """
        if self.sign != another.sign:
            print('Two WDNF polynomials cannot be multiplied')
            return
        new_coefficients = {}
        for key1 in self.coefficients:
            for key2 in another.coefficients:
                new_key = merge(key1, key2)
                if new_key in new_coefficients:
                    new_coefficients[new_key] += self.coefficients[key1] * another.coefficients[key2]
                else:
                    new_coefficients[new_key] = self.coefficients[key1] * another.coefficients[key2]
        return wdnf(new_coefficients, self.sign)


    def __rmul__(self, scalar): #adjusted
        """ Multiplies the coefficients of a WDNF function with a scalar
        """
        new_coefficients = self.coefficients.copy()
        for key in self.coefficients:
            new_coefficients[key] = self.coefficients[key] * scalar
        return wdnf(new_coefficients, self.sign)


    def power(self, k): #adjusted
        """ Return poly (self)**k. k must be greater that or equal to 1.
        """
        power_wdnf = self
        i = 1
        while i < k:
            power_wdnf = power_wdnf * self
            i += 1
        return power_wdnf





class taylor():
    """ A class computing the Taylor expansion of a function"""


    def __init__(self, poly_coef, center, degree):
        """
        """
        if len(poly_coef) != degree + 1:
            print('Size of the coefficients list does not match with the degree!')
        else:
            self.poly_coef = poly_coef
            self.center = center
            self.degree = degree


    def evaluate(self, x):
        """ Evaluates Taylor approx at x"""
        xx = x
        out = 0.
        for i in range(self.degree + 1):
            centered_xk  = (xx-self.center)**i
            terms =  self.poly_coef[i]*centered_xk/math.factorial(i)
            out  += centered_xk*terms
        return out


    def expand(self):
        """ Updates the coefficients of the given function so that the Taylor expansion
        is centered around zero.
        """
        if self.center == 0:
            return
        new_poly_coef = [0.0] * (self.degree + 1)
        for i in range(self.degree + 1):
            for j in range(i, self.degree + 1):
                if j-i >0:
                    new_poly_coef[i] += self.poly_coef[j] *nCr(j,i)/math.factorial(j) * (-self.center)**(j-i)
                else:
                    new_poly_coef[i] += self.poly_coef[j] *nCr(j,i)/math.factorial(j)
        self.poly_coef = new_poly_coef
        self.center = 0.0


    def evaluate_expanded(self, xk):
        """ Compute the expanded polynomial, using xk dictionary containing powers x^k
        """
        out = self.alpha[0]
        for i in xk:
            out = out+ xk[i]*self.alpha[i]
        return out


    def compose(self, my_wdnf): #new compose function
        """ Given a one-dimensional polynomial function f with degree k and coefficients
        stored in coef_list, computes f(self) and returns the result in WDNF
        """
        self.expand()
        result = wdnf({}, {}, my_wdnf.sign)
        for i in range(self.degree + 1):
            result += self.poly_coef[i] * my_wdnf.power(i)
        return result


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
    #wdnf0 = wdnf({}, {})
    wdnf1 = wdnf({(1, 3): 2.0, (2, 4): 10.0, (3, 4): 3.0})
    #print(wdnf1.coefficients)
    #print(wdnf1.sets)
    wdnf2 = wdnf({(1, 2): 4.0, (1, 3): 5.0})
    wdnf3 = wdnf1 * wdnf2
    #wdnf4 = wdnf1 + wdnf2
    #wdnf5 = 4 * wdnf1
    wdnf6 = wdnf1.power(2)
    #wdnf7 = wdnf0 + wdnf1
    #x = {1:0.5, 2:0.5, 3:0.5, 4:0.5}
    print(wdnf3.coefficients)
    print(wdnf3.sign)
    #print(wdnf2.sets.get(2))
    #print(wdnf1.sign)
    #print(wdnf1.evaluate(x))
    #print(wdnf2.coefficients)
    #print(wdnf4.coefficients)
    #print(wdnf4.sign)
    #print(wdnf5.coefficients)
    #print(wdnf5.sign)
    #print(wdnf7.coefficients)
    #print(wdnf7.sign)

    #myTaylor = taylor([1, 2, 1], 1, 2)
    #print(myTaylor.evaluate(1))
    #myTaylor.expand()
    #print(myTaylor.poly_coef)
    #print(myTaylor.center)
    #print(myTaylor.degree)
    #new_wdnf1 = myTaylor.compose(wdnf1)
    #print(new_wdnf1.coefficients)
    #print(new_wdnf1.sets)
    #print(new_wdnf1.sign)

#if __name__=="__main__":

    # for i in range(10):
    #     r_val = 0.1*i
    #     taylor_approx = taylor( 4*[np.exp(r_val)],r_val,3 )
    #     taylor_approx.expand()
    #     r = r_val+0.01
    #     print np.exp(r),taylor_approx.evaluate(r),taylor_approx.evaluate_expanded( dict([ (i+1,r**(i+1))    for i in range(3)]   ))




    # P = Problem.unpickle_cls("problem_abilene_1000demands_300catalog_size_mincap_30maxcap_30_100_uniform")
    # # Problem class is defined in Toplogy_gen.





    # X = generateRandomPlacement(P)



    # print X
    # rhos = ro_uv(X,P)

    # rho_uvs_coefficients,  rho_uvs_sets = rho_uv_dicts(P)
    # print rho_uvs_coefficients[(0, 8)],rho_uvs_sets[(0,8)]
    # for edge in rho_uvs_coefficients:
    #     rho_wdnf = wdnf(rho_uvs_coefficients[edge], rho_uvs_sets[edge])
    #     r_val= rho_wdnf.evaluate(X)



    #     print 'For edge',edge,'rho is',rhos[edge],'calculated via poly is',r_val


    #     #taylor_approx = taylor( 4*[np.exp(r_val)],r_val,3 )

    #     #taylor_approx.expand()
    #     #r = r_val+0.01
    #     #print np.exp(r),taylor_approx.evaluate(r),taylor_approx.evaluate_expanded( dict([ (i+1,r**(i+1))    for i in range(3)]   ))




    #     rk ={}
    #     i=1
    #     rk[i] = rho_wdnf
    #     #print rho_wdnf.coefficients,rho_wdnf.sets
    #     while i<2:
    #         i +=1
    #         rk[i] = rk[i-1].product(rho_wdnf)
    #         print 'For edge',edge,'rho^'+str(i),'is',rhos[edge]**i,'calculated via poly is',rk[i].evaluate(X)

    #         #print rk[i].coefficients,rk[i].sets



    #     ##test taylor expansion with f(x) =np.exp(x).
