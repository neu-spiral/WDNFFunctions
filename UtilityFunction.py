def UtilityFunction1(r, k):
    '''Utility for the first moment'''
    if k == 0:
        return r
    elif k == 1:
        return 1
    else:
        return 0

def UtilityFunction2(r, k):
    '''Utility for the second moment'''
    if k == 0:
        return r + r**2
    elif k == 1:
        return 1 + 2 * r
    elif k == 2:
        return 2
    else:
        return 0

def UtilityFunction3(r, k):
    '''Utility for the third moment'''
    if k == 0:
        return r + 3*r**2 + r**3
    elif k == 1:
        return 1 + 6*r + 3*r**2
    elif k == 2:
        return 6 + 6*r
    elif k == 3:
        return 6
    else:
        return 0

def UtilityFunction4(r, k):
    '''Utility for the forth moment'''
    if k == 0:
        return r + 7*r**2 + 6*r**3 + r**4
    elif k == 1:
        return 1 + 14*r + 18*r**2 + 4*r**3
    elif k == 2:
        return 14 + 36 * r + 12*r**2
    elif k == 3:
        return 36 + 24 * r
    elif k == 4:
        return 24
    else:
        return 0


def UtilityInfo1(r, k):
    '''Utility for the first moment'''
    if k == 0:
        return r
    elif k == 1:
        return 1
    else:
        return 0

def UtilityInfo2(r, k):
    '''Utility for the second moment'''
    if k == 0:
        return r + 2*r**2
    elif k == 1:
        return 1 + 4 * r
    elif k == 2:
        return 4
    else:
        return 0

def UtilityInfo3(r, k):
    '''Utility for the third moment'''
    if k == 0:
        return r + 6*r**2 + 6*r**3
    elif k == 1:
        return 1 + 12*r + 18*r**2
    elif k == 2:
        return 12 + 36*r
    elif k == 3:
        return 36
    else:
        return 0

def UtilityInfo4(r, k):
    '''Utility for the forth moment'''
    if k == 0:
        return r + 14*r**2 + 36*r**3 + 24*r**4
    elif k == 1:
        return 1 + 28*r + 108*r**2 + 96*r**3
    elif k == 2:
        return 28 + 216 * r + 288*r**2
    elif k == 3:
        return 216 + 576 * r
    elif k == 4:
        return 576
    else:
        return 0




