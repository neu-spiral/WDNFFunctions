import math
import numpy as np
from scipy.misc import comb
from decimal import *


def merge(t1, t2):
    """Merge two tuples (in this context, keys) into a sorted tuple by taking
    the set union of them.
    """
    return tuple(sorted(set(t1).union(set(t2))))


class wdnf():
    """A class implementing a polynomial in Weighted Disjunctive Normal Form
    (WDNF) consisting of monomials with (a) negative or positive literals and
    (b) integer terms.
    """
    def __init__(self, coefficients={}, sign=-1):
        """ Coefficients is a dictionary containing tuples with indexes of the
        set elements as keys and coefficients as values. Sign denotes whether
        the WDNF formed with negative literals or positive literals.
        e.g: wdnf({(1, 3): 2.0, (2, 4): 10.0, (3, 4): 3.0}) =
        2.0(1-x_1)(1-x_3) + 10.0(1-x_2)(1-x_4) + 3.0(1-x_3)(1-x_4)
        """
        self.coefficients = coefficients
        self.sign = sign


    def findDependencies(self):
        dependencies = {}
        for key in self.coefficients:
            for var in key:
                if var in dependencies:
                    dependencies[var].append(key)
                else:
                    dependencies[var] = [key]
        return dependencies


    def __call__(self, x): #(old evaluate(x) function)
        """ Given a dictionary x, evaluate wdnf(x) at the values x.
        """
        sumsofar = 0.0
        for key in self.coefficients:
            beta = self.coefficients[key]
            setofx = key
            prod = beta
            if  self.sign == -1:
                for var in setofx:
                    prod = prod * (1.0-x[var])
            else:
                for var in setofx:
                    prod = prod * x[var]
            sumsofar = sumsofar + prod
        return sumsofar


    def __add__(self, other):
        """ Add two polynomials in WDNF and return the resulting WDNF
        """
        assert self.sign == other.sign, 'Two WDNF polynomials of different signs cannot be added!'
        new_coefficients = self.coefficients.copy() #empty dict for empty wdnf
        if not other.coefficients:
            return self
        elif not self.coefficients:
            return other
        else:
            for key in other.coefficients:
                if key in self.coefficients.keys():
                    new_coefficients[key] += other.coefficients[key]
                else:
                    new_coefficients[key] = other.coefficients[key]
        return wdnf(new_coefficients, self.sign)


    def __mul__(self, other):
        """ Multiply two polynomials in WDNF and return the resulting WDNF
        """
        assert self.sign == other.sign, 'Two WDNF polynomials of different signs cannot be multiplied!'
        new_coefficients = {}
        for key1 in self.coefficients:
            for key2 in other.coefficients:
                new_key = merge(key1, key2)
                if new_key in new_coefficients:
                    new_coefficients[new_key] += self.coefficients[key1] * other.coefficients[key2]
                else:
                    new_coefficients[new_key] = self.coefficients[key1] * other.coefficients[key2]
        return wdnf(new_coefficients, self.sign)


    def __rmul__(self, scalar):
        """ Multiplies the coefficients of a WDNF function with a scalar
        """
        new_coefficients = self.coefficients.copy()
        for key in self.coefficients:
            new_coefficients[key] = self.coefficients[key] * scalar
        return wdnf(new_coefficients, self.sign)


    def __pow__(self, k):
        """Calculates the kth power of a WDNF function and returns the result.
        k must be greater than or equal to 0.
        """
        if k==0:
            return wdnf({(): 1}, self.sign)
        else:
            power_wdnf = self
            for i in range(2, k + 1):
                power_wdnf *= self
            return power_wdnf


class poly():
    """A class for defining univariate polynomials with the largest degree and
    the coefficients list of size (largest degree + 1) where coefficients are
    stored as [coef_0 coef_1 ... coef_n]
    """


    def __init__(self, degree, poly_coef):
        """e.g: poly(n, [a0 a1 ... an]) defines
        f(x) = a0 + a1*x + ... + an*(x^n)
        """
        assert len(poly_coef) == (degree + 1), 'Size of the coefficients list does not match with the degree!'
        self.poly_coef = poly_coef
        self.degree = degree


    def __add__(self, other):
        """Adds two univariate polynomials and returns the sum as another poly
        object.
        """
        if self.degree >= other.degree:
            poly_coef = list(np.array(self.poly_coef) + np.array(other.poly_coef + [0] * (self.degree - other.degree)))
        else:
            return other + self
        return poly(self.degree, poly_coef)


    def __sub__(self, other):
        """Subtracts two univariate polynomials and returns the difference as
        another poly object.
        """
        return self + ((-1) * other)


    def __mul__(self, other):
        """Multiplies two polynomials and return the product as another poly
        object.
        """
        degree = self.degree + other.degree
        poly_coef = [0] * (degree + 1)
        for i in range(len(self.poly_coef)):
            for j in range(len(other.poly_coef)):
                poly_coef[i + j] += self.poly_coef[i] * other.poly_coef[j]
        return poly(degree, poly_coef)


    def __rmul__(self, scalar):
        """Multiplies a polynomial with a scalar.
        """
        return poly(self.degree, list(np.array(self.poly_coef) * scalar))


    def compose(self, my_wdnf): 
        """ Given a one-dimensional polynomial function f with degree k and coefficients
        stored in coef_list, computes f(self) and returns the result in WDNF.
        """
        wdnfSoFar = wdnf({(): 1})
        result = self.poly_coef[0] * wdnfSoFar
        for i in range(1, self.degree + 1):
            wdnfSoFar *= my_wdnf
            result += self.poly_coef[i] * wdnfSoFar
        return result


    def __call__(self, x):
        """Calculates f(x) for a given x.
        """
        output = 0.0
        for i in range(self.degree + 1):
            output += self.poly_coef[i] * (x**i)
        return output


class taylor(poly):
    """ A class computing the Taylor expansion of a function"""


    def __init__(self, degree, derivatives, center):
        """Given the calculated derivatives at the center, initializes Taylor
        expansion of a function in the standart polynomial form by expanding the
        terms using binomial expansion.
        """
        if center == 0:
            poly.__init__(self, degree, derivatives)
        else:
            poly_coef = [0.0] * (degree + 1)
            for i in range(degree + 1):
                for j in range(i, degree + 1):
                    if j-i > 0:
                        poly_coef[i] += derivatives[j] * comb(j, i, True)/math.factorial(j) * (-center)**(j-i)
                    else:
                        poly_coef[i] += derivatives[j] * comb(j, i, True)/math.factorial(j)
            poly.__init__(self, degree, poly_coef)


if __name__=="__main__":
    #wdnf0 = wdnf({}, {})
    wdnf1 = wdnf({(2, 3): 2.0, (2, 4): 10.0, (3, 4): 3.0})
    print(wdnf1.findDependencies())
    #print(wdnf1.sets)
    wdnf2 = wdnf({(1, 2): 4.0, (1, 3): 5.0})
    #wdnf3 = wdnf1 * wdnf2
    wdnf4 = wdnf1 + wdnf2
    #wdnf5 = 4 * wdnf1
    wdnf6 = wdnf1**2
    #wdnf7 = wdnf0 + wdnf1
    x = {1:0.5, 2:0.5, 3:0.5, 4:0.5}
    #print(wdnf3.coefficients)
    #print(wdnf3.sign)
    #print(wdnf2.sets.get(2))
    #print(wdnf1.sign)
    #print(wdnf1(x))
    #print(wdnf2.coefficients)
    #print(wdnf4.coefficients)
    #print(wdnf4.sign)
    #print(wdnf5.coefficients)
    #print(wdnf5.sign)
    #print(wdnf6.coefficients)
    #print(wdnf6.sign)

    poly1 = poly(2, [3, 4, 0])
    poly2 = poly(2, [8, 1, 1])
    poly3 = poly2 + poly1
    wdnf4 = poly2.compose(wdnf1)
    #print(wdnf4.coefficients)
    #print(poly3.poly_coef)
    #myTaylor = taylor(2, [1, 2, 1], 1)
    #print(myTaylor.evaluate(1))
    #myTaylor.expand()
    #print(myTaylor.poly_coef)
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
