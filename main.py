# CSCE 990 - Project
# Name: April Inamura
# Date: 11/23/21
# Description: Implements a Multilayer Perceptron Neural Network (MLPNN)
#              This project focuses on post training quantization techniques for compressing the
#              input data and weights in testing. The input is quantized and the input and weights are
#              converted from a float32 to int16 fixed point using Q5.11 Notation

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import fixed as fix
import decimal as deci

def sTensor(a):
    val = torch.tensor(a)
    val = val.view(1, 1)
    return val

def gaussian_distribution(row,col,n):
    std = np.sqrt(2/n)
    w = np.random.randn(row,col)
    return w

def f_a(a):
    if (a <= -4):
        d = 0
    elif((a > -4) and (a <= 0)):
        d = ((a ** 2) / 32) + (a / 4) + 0.5
    elif((a > 0) and (a < 4)):
        d = ((-a ** 2) / 32) + (a / 4) + 0.5
    elif(a >= 4):
        d = 1
    else:
        print("Input out of bounds")
    return d

def train(input,N):
    # Parameters
    w = torch.from_numpy(gaussian_distribution(N, 2, N))  # Input Weight
    v = torch.from_numpy(gaussian_distribution(N, 1, N))  # Output Weight
    # Initalization
    d = torch.zeros(N, 1)
    pDelta_v = torch.zeros(N, 1)
    pDelta_w = torch.zeros(N, 2)
    delta_v = torch.zeros(N, 1)
    delta_w = torch.zeros(N, 2)
    e_a = torch.zeros(1, N)
    # Gain
    l_gain = 0.4
    m_gain = 0.8
    # Mean Squared Error
    epochs = 100
    MSE = torch.zeros(1, epochs)

    #Training
    for l in range(epochs):
        #Expected Output
        x_train = input[0,torch.randperm(len(input[0,:]))] #Randomized Input Each Epoch
        y_expected = 2 * (torch.square(x_train)) + 1       #Randomized Expected Output

        #FeedForward Operation
        for i in range(len(x_train)):
            X = torch.tensor([x_train[i], 1])
            X = X.view(2, 1)
            a = torch.mm(w.double(), X.double())
            for j in range(N):
                # d[j] = f_a(a[j])
                d[j] = 1/(1+math.exp(-a[j]))
            y = d * v

            #Backpropagation
            y_train = torch.sum(y)
            e_y = y_expected[i] - y_train
            MSE[0, l] = 0.5 * (e_y ** 2)
            e_d = torch.transpose(v, 0, 1) * e_y
            for k in range(N):
                e_a[0, k] = d[k, 0] * (1 - d[k, 0]) * e_d[0, k]
            # Weight Error Vectors
            e_v = e_y * torch.transpose(d, 0, 1)
            e_w = torch.mm(X, e_a)
            # Weight Updates
            delta_v = l_gain * torch.transpose(e_v, 0, 1) + m_gain * pDelta_v
            delta_w = l_gain * torch.transpose(e_w, 0, 1) + m_gain * pDelta_w
            # Weight Modification
            pDelta_v = delta_v
            pDelta_w = delta_w
            v = v + delta_v
            w = w + delta_w

    print("MSE: ")
    print(torch.mean(MSE))
    print(torch.min(MSE))

    return w,v
# End of Training

def test(input, w, v, N, qInput):
    d = torch.zeros(N, 1, dtype=torch.float32)
    y_expected = 2 * (torch.square(input)) + 1
    y_train = torch.zeros(np.shape(input), dtype=torch.float32)
    for i in range(len(input[0,:])):
        X = torch.tensor([qInput[0, i], 1],dtype=torch.float32) #Uses quantized Input
        X = X.view(2, 1)
        a = torch.mm(w.double(), X.double())
        for j in range(N):
            # d[j] = f_a(a[j])
            d[j] = 1 / (1 + math.exp(-a[j]))
        y = d * v
        y_train[0, i] = torch.sum(y)

    error = y_expected[0,100:300] - y_train[0,100:300]
    mse = 0.5*(error ** 2)
    print("Test MSE")
    print(torch.mean(mse))

    # Results Output - Graph
    print(y_train.dtype)
    plt.plot(input[0,:],y_expected[0,:])
    plt.plot(input[0,:],y_train[0,:])
    plt.title("MLPNN Result: Expected vs. Trained")
    plt.legend(['Expected','Trained'])
    plt.xlabel("Input [-1:1]")
    plt.ylabel("Output y = 2x^2 + 1")
    plt.show()

# Quantization Method 1: Uniform Quantization
#   Descrip: Takes 64 bit input and reduces to specified b bits (as low as 6 bits)
def uniform_quantization(input,b,qMin,qMax):
    L = (2 ** b)-1
    q = (qMax - qMin) / L
    Q = torch.zeros(len(input[0, :]))
    Q = Q.view(1, len(Q))
    for i in range(len(input[0, :])):
        if (input[0, i] < qMin):
            Q[0, i] = qMin + (q / 2)
        elif (input[0, i] > qMax):
            Q[0, i] = qMax - (q / 2)
        else:
            Q[0, i] = (((input[0, i] - qMin) / q) * q) + (q / 2) + qMin
    return Q



def fixed_f_a(a,Q,k):
    n4 = fix.to_fixed(sTensor(-4.0),Q,bit=k)
    zero = fix.to_fixed(sTensor(0.0),Q,bit=k)
    p4 = fix.to_fixed(sTensor(4.0),Q,bit=k)
    one = fix.to_fixed(sTensor(1.0),Q,bit=k)
    p32 = fix.to_fixed(sTensor(0.03125),Q,bit=k) #Must be 16 bit
    phalf = fix.to_fixed(sTensor(0.5),Q,bit=k)
    pquart = fix.to_fixed(sTensor(0.25),Q,bit=k)

    if (a <= n4):
        d = zero
    elif((a > n4) and (a <= zero)):
        x1 = fix.q_muls(a,a,Q,bit=k)     #(a ** 2)
        x2 = fix.q_muls(x1,p32,Q,bit=k)  #((a ** 2) * 0.03125)
        y1 = fix.q_muls(a,pquart,Q,bit=k)#(a * 0.25)
        z1 = fix.q_adds(x2,y1,bit=k)     #((a ** 2) / 32) + (a / 4)
        d = fix.q_adds(z1,phalf,bit=k)   #((a ** 2) / 32) + (a / 4) + 0.5

        # d = ((a ** 2) * 0.03125) + (a * 0.25) + 0.5
    elif((a > zero) and (a < p4)):
        x1 = fix.q_muls(a,a,Q,bit=k)     #(a ** 2)
        x2 = fix.q_muls(-x1,p32,Q,bit=k)  #(-(a ** 2) * 0.03125)
        y1 = fix.q_muls(a,pquart,Q,bit=k)#(a * 0.25)
        z1 = fix.q_adds(x2,y1,bit=k)     #((a ** 2) / 32) + (a / 4)
        d = fix.q_adds(z1,phalf,bit=k)

        # d = (-(a ** 2) / 32) + (a / 4) + 0.5
    elif(a >= p4):
        d = one
    else:
        print("Input out of bounds")
    return d

def fixed_test(input, eOutput, w, v, N, qInput, Q, k):
    d = torch.zeros(N, 1, dtype=torch.int16)
    y_train = torch.zeros(np.shape(input), dtype=torch.int16)
    one = fix.to_fixed(sTensor(1.0),Q,bit=k)
    for i in range(len(input[0,:])):
        X = torch.tensor([qInput[0, i], one],dtype=torch.int16) #Uses quantized Input
        X = X.view(2, 1)
        a = fix.q_mul_mat(w,X,Q,bit=k)
        for j in range(N):
            d[j,0] = fixed_f_a(a[j,0],Q,k)
        y = fix.q_mul_dot(d,v,Q,bit=k)
        y_train[0, i] = torch.sum(y)

    # Convert Fixed to Float for graphing
    input = fix.to_float(input, Q)
    eOutput = fix.to_float(eOutput, Q)
    y_train = fix.to_float(y_train, Q)

    error = eOutput[0,100:300] - y_train[0,100:300]
    mse = 0.5*(error ** 2)
    print("Test MSE")
    print(torch.mean(mse))

    # Results Output - Graph
    plt.plot(input[0,:],eOutput[0,:])
    plt.plot(input[0,:],y_train[0,:])
    plt.title("MLPNN Result: Expected vs. Trained")
    plt.legend(['Expected','Trained'])
    plt.xlabel("Input [-1:1]")
    plt.ylabel("Output y = 2x^2 + 1")
    plt.show()

def sgn(a):
    if(a < 0):
        sign = -1
    elif(a == 0):
        sign = 0
    elif(a > 0):
        sign = 1
    else:
        sign = 0
    return sign

def nonuniform_quantization(input,b):
    mu = (2^b)-1
    qt = torch.zeros(np.shape(input), dtype=torch.float32)
    for i in range(len(qt[0,:])):
        sign = sgn(input[0,i])
        qt[0,i] = sign * ((math.log(1 + mu*abs(input[0,i]))) / math.log(1 + mu))
    return qt


if __name__ == '__main__': #main script
    #Training Parameters
    hLayer = 5  # Hidden Layers\
    x_limit = torch.arange(-1, 1, 0.01)
    x_limit = x_limit.view(1,len(x_limit))

##### Training
    inputWeight, outputWeight = train(x_limit,hLayer)

    #Testing Parameters
    x_test = torch.arange(-2,2,0.01, dtype=torch.float32)
    x_test = x_test.view(1,len(x_test))
    y_test = 2 * (torch.square(x_test)) + 1

    # Quantization - Input
    qBit = 5
    q1_input = uniform_quantization(x_test,qBit,-1,1)

    #Reduce Bitwidth - Weights (lowest achievable is half precision : float16)
    x_float = x_test.type(torch.float16) #Used for y_expected
    in_float = inputWeight.type(torch.float16)
    out_float = outputWeight.type(torch.float16)

##### Floating Point Testing
    # test(x_test,inputWeight,outputWeight,hLayer,x_test)

    #Fixed Point Conversion
    wBits = 16
    bits = 16
    bitFrac = 11

    x_fixed = fix.to_fixed(x_test, bitFrac,bit=bits) #Q3.4 for 8 bit (1 bit assumed signed in INT)/Q3.12 for 16 bit
    y_fixed = fix.to_fixed(y_test, bitFrac,bit=bits)

    q_fixed = fix.to_fixed(q1_input, bitFrac, bit=bits)

    in_trans = torch.transpose(inputWeight, 0, 1) #for formatting reasons <- remember to return
    out_trans = torch.transpose(outputWeight,0,1)
    in_fixed = fix.to_fixed(in_trans, bitFrac,bit=wBits)
    in_fixed = torch.transpose(in_fixed,0,1) #returned to original position
    out_fixed = fix.to_fixed(out_trans, bitFrac,bit=wBits)
    out_fixed = torch.transpose(out_fixed, 0, 1)

##### Fixed Point Testing
    fixed_test(x_fixed,y_fixed, in_fixed,out_fixed,hLayer,q_fixed,bitFrac,bits) #Fixed point testing


#TESTING INDIVIDUAL FUNCTIONS
    ###Fixed Point Arithmetic Matrix Tests - All Passed
    # x_trans = torch.transpose(x_test,0,1)
    # square_x = torch.mm(x_trans, x_test)
    # div_x = in_fixed / in_fixed
    # print("Expected")
    # print(div_x)
    # print("Fixed Division")
    # dFix = q_div(in_fixed,in_fixed,bitFrac)
    # dFloat = to_float(dFix, bitFrac)
    # print(dFix.dtype)
    # print(dFloat)
    # print("Fixed Multiplication")
    # xFix_trans = torch.transpose(x_fixed,0,1)
    # aFix = q_mul(x_fixed,x_fixed,bitFrac)
    # aFloat = to_float(aFix,bitFrac)
    # print(aFix.dtype)
    # print(aFloat)
    # print("Fixed Subtraction")
    # bFix = q_sub(x_fixed,x_fixed)
    # bFloat = to_float(bFix,bitFrac)
    # print(bFix.dtype)
    # print(bFloat)
    # print("Fixed Addition")
    # cFix = q_add(x_fixed,x_fixed)
    # cFloat = to_float(cFix,bitFrac)
    # print(cFix.dtype)
    # print(cFloat)

    # Fixed Point Arithmetic Single Test
    # val1 = torch.tensor(2.0)
    # val1 = val1.view(1,1)
    # val2 = torch.tensor(4.0)
    # val2 = val2.view(1, 1)
    # val1Fix = to_fixed(val1,bitFrac)
    # val2Fix = to_fixed(val2,bitFrac)
    # valAdd = q_adds(val1Fix, val2Fix)
    # valsub = q_subs(val1Fix, val2Fix)
    # valMul = q_muls(val1Fix, val2Fix,bitFrac)
    # valDiv = q_divs(val1Fix, val2Fix,bitFrac)
    # print(to_float(valAdd,bitFrac))
    # print(to_float(valsub,bitFrac))
    # print(to_float(valMul,bitFrac))
    # print(to_float(valDiv,bitFrac))

    #Fixed Activation Function Testing
    # val1 = fix.to_fixed(sTensor(4.0),bitFrac,bit=16)
    # print(val1)
    # print(fix.to_float(val1,bitFrac))
    # print("Activation Function")
    # act1 = fixed_f_a(val1,bitFrac)
    # print(fix.to_float(act1,bitFrac))