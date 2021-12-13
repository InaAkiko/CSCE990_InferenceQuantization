# Fixed Point Library
# Name: April Inamura
# Date: 11/30/21
# Converts between fixed to float and vice versa
# Implements Matrix and Single Value Basic Fixed Point Arithmetic:
#               Addition / Subtraction / Multiplication
#               Division Does not Function Properly

import torch

def to_float(x,e):
    f = x.type(torch.float32)
    for i in range(len(x[:,0])):
        for j in range(len(x[0,:])):
            c = abs(x[i,j])
            sign = 1
            if x[i,j] < 0:
                # convert back from two's complement
                c = x[i,j] - 1
                c = ~c
                sign = -1
            f[i,j] = (1.0 * c) / (2 ** e)
            f[i,j] = f[i,j] * sign
    return f

def to_fixed(f,e,bit=8):
    a = f * (2 ** e)
    b = torch.round(a)
    if (bit == 8):
        b = b.type(torch.int8)
    elif (bit == 16):
        b = b.type(torch.int16)
    for i in range(len(f[:,0])):
        for j in range(len(f[0,:])):
            if a[i,j] < 0:
                # next three lines turns b into it's 2's complement.
                c = abs(b[i,j])
                d = ~c
                b[i,j] = d + 1
    return b

def q_add(a,b,bit=8):
    tmp = a.type(torch.int32) + b.type(torch.int32)
    for i in range(len(tmp[:,0])):
        for j in range(len(tmp[0,:])):
            if(tmp[i,j] > ((2 ** bit)-1)):
                tmp[i,j] = ((2 ** bit)-1)
            if(tmp[i,j] < (-(2 ** bit))):
                tmp[i,j] = -(2 ** bit)
    if (bit == 8):
        result = tmp.type(torch.int8)
    elif (bit == 16):
        result = tmp.type(torch.int16)
    return result

def q_sub(a,b):
    return a - b

def q_mul_mat(a,b,Q,bit=8):
    tmp = torch.mm(a.type(torch.int32),b.type(torch.int32))
    tmp += (1 << (Q-1))
    tmp = tmp >> Q
    for i in range(len(tmp[:,0])):
        for j in range(len(tmp[0,:])):
            if(tmp[i,j] > ((2 ** bit)-1)):
                tmp[i,j] = (2 ** bit)-1
            elif(tmp[i,j] < (-(2 ** bit))):
                tmp[i,j] = -(2 ** bit)
            else:
                #do nothing
                nope = 0
    if (bit == 8):
        result = tmp.type(torch.int8)
    elif (bit == 16):
        result = tmp.type(torch.int16)
    return result

def q_div(a,b,Q,bit=8):
    tmp = a.type(torch.int32) << Q
    for i in range(len(tmp[:,0])):
        for j in range(len(tmp[0,:])):
            if(((tmp[i,j] >= 0) and (b[i,j] >= 0)) or ((tmp[i,j] < 0) and (b[i,j] < 0))):
                val = (b[i,j] >> 1)
                tmp[i,j] += val
            else:
                val = (b[i, j] >> 1)
                tmp[i,j] -= val
    result = tmp / b
    if (bit == 8):
        result = tmp.type(torch.int8)
    elif (bit == 16):
        result = tmp.type(torch.int16)
    return result


## SINGLE Fixed POINT
def q_adds(a,b,bit=8):
    tmp = a.type(torch.int32) + b.type(torch.int32)
    if(tmp > ((2 ** bit)-1)): #For Saturation
        tmp = (2 ** bit)-1
    if(tmp < (-(2 ** bit))):
        tmp = -(2 ** bit)
    if (bit == 8):
        result = tmp.type(torch.int8)
    elif (bit == 16):
        result = tmp.type(torch.int16)
    return result

def q_subs(a,b):
    return a - b

def q_muls(a,b,Q,bit=8):
    tmp = a.type(torch.int32) * b.type(torch.int32)
    tmp += (1 << (Q-1)) #Rounding mid values up
    tmp = tmp >> Q
    if (tmp > ((2 ** bit) - 1)):  # For Saturation
        tmp = (2 ** bit) - 1
    elif (tmp < (-(2 ** bit))):
        tmp = -(2 ** bit)
    else:
        tmp = tmp
    if(bit == 8):
        result = tmp.type(torch.int8)
    elif(bit == 16):
        result = tmp.type(torch.int16)
    return result

def q_divs(a,b,Q,bit=8):
    tmp = a.type(torch.int32) << Q
    if (((tmp >= 0) and (b >= 0)) or ((tmp < 0) and (b < 0))):
        val = (b >> 1)
        tmp += val
    else:
        val = (b >> 1)
        tmp -= val
    result = tmp / b
    if (bit == 8):
        result = tmp.type(torch.int8)
    elif (bit == 16):
        result = tmp.type(torch.int16)
    return result

def q_mul_dot(a,b,Q,bit=8):
    tmp = a.type(torch.int32) * b.type(torch.int32)
    tmp += (1 << (Q-1))
    tmp = tmp >> Q
    for i in range(len(tmp[:,0])):
        for j in range(len(tmp[0,:])):
            if(tmp[i,j] > ((2 ** bit)-1)):
                tmp[i,j] = (2 ** bit)-1
            elif(tmp[i,j] < (-(2 ** bit))):
                tmp[i,j] = -(2 ** bit)
            else:
                #do nothing
                nope = 0
    if (bit == 8):
        result = tmp.type(torch.int8)
    elif (bit == 16):
        result = tmp.type(torch.int16)
    return result