# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:26:20 2024

@author: yvanm
"""
import numpy as np
import pandas as pd
import psychro as psy
import matplotlib.pyplot as plt

def recPreHeatVapCAV(θ0=2, φ0=0.6, θIsp=2, φIsp=0.6, UA=85, α=0.5, m=1, Qsa=500, Qla=300, ma=0.05):
    # constants
    c = 1e3         # air specific heat J/kg K
    l = 2496e3      # latent heat J/kg
    w0 = psy.w(θ0, φ0)
    wIsp = psy.w(θIsp, φIsp)
    Kt = 10e6
    Kw = 10e9
    Kpre=0
    
    
    def PreHeat():
        # Model
        A = np.zeros((14, 14))          # coefficents of unknowns
        b = np.zeros(14)                # vector of inputs
        # HC1
        A[0, 0],  b[0] = m * c, m * c * θ0 + QsHC1
        A[1, 1],  b[1] = m * l, m * l * w0       
        # mix
        A[2, 0], A[2, 2], A[2, 8], b[2] = -α * m * c, m * c, -(1 - α) * m * c, 0 
        A[3, 1], A[3, 3], A[3, 9], b[3] = -α * m * l, m * l, -(1 - α) * m * l, 0
        # HC2
        A[4, 2], A[4, 4], A[4, 12], b[4] = -m * c, m * c, -1, 0
        A[5, 3], A[5, 5], b[5] = -m * l, m * l, 0
        # VH
        A[6, 4], A[6, 6], b[6] = - m * c, m * c, 0
        A[7, 5], A[7, 7], A[7, 13], b[7] = - m * l, m * l, -1, 0
        # TZ
        A[8, 6], A[8, 8], A[8, 10], b[8] = - m * c, m * c, -1, 0
        A[9, 7], A[9, 9], A[9, 11], b[9] = - m * l,  m * l, -1, 0
        # BL
        A[10, 8], A[10, 10], b[10] = (UA + ma * c), 1, (UA + ma * c) * θ0 + Qsa
        A[11, 9], A[11, 11], b[11] = ma * l, 1, ma * l * w0 + Qla
        # Kt indoor temperature controller
        A[12, 8], A[12, 12], b[12] = Kt, 1, Kt * θIsp
        # Kw indoor hum.ratio controller
        A[13, 9], A[13, 13], b[13] = Kw, 1, Kw * wIsp
        # Solution
        x = np.linalg.solve(A, b)
        return x
    
    QsHC1 = m * c * Kpre                  # init. load on HC1

    x = PreHeat()
    if x[3] > psy.w(x[2], 1):
        # Model MX & AD
        while x[3] > psy.w(x[2], 1):
            Kpre = Kpre + 1
            QsHC1 = m * c * Kpre
            x = PreHeat()
        Kpre = Kpre + 3
        QsHC1 = m * c * Kpre
        x = PreHeat()
    
    
    A = np.array([[-1, 1, 0, 0, 0, 0],   # MX
                  [0, -1, 1, 0, 0, -1],
                  [0, 0, -1, 1, 0, 0],
                  [0, 0, 0, -1, 1, 0],
                  [0, 0, 0, 0, -1, 1]])   # AD
    θ = np.append([θ0], x[0:9:2])   # θ0, θ1, θ2, θ3, θ4, θ5
    w = np.append([w0], x[1:10:2])   # w0, w1, w2, w3, w4, w5
    psy.chartA(θ, w, A)
    return x, QsHC1