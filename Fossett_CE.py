# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:29:02 2023

@author: Will Fossett

prop nav ex: https://www.youtube.com/watch?v=CMOh2xWk_qA
3D pro nav resource: https://web.itu.edu.tr/~altilar/thesis/msc/MORAN05.pdf
MMGT: file:///C:/Users/willf/Downloads/modern_missile_guidance_theory.pdf
    used for MMGT PN, augmented PN, OGL

aim7 sparrow, max vel of Mach 2.5 and assumed accel time of 4s
target is assumed to be missile traveling ~75% of aim7 or M1.875 with some
    maneuverability as shown in the target path

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import random
import os
plt.rcParams['axes.grid'] = True  # Applies a grid to every plot
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi
RATE = 250  # Hz, interpolation rate
dt = 1/RATE  # time step size
INTERCEPT_SPEED = 857  # 857  # m/s, constant speed of pursuer
INITIAL_VELOCITY = 0
TARGET_SPEED = 643  # ~75% of pursuer velocity
CLIMB_RATE = 3  # m/s, target climb rate
ACCEL_TIME = 4  # s, time it takes for pursuer to reach max velocity
COL_THRESHOLD = 1  # m, radius within which col is said to have occurred
T_DETECT = 10  # s, time after target start that target is detected
np.random.seed(0)
random.seed(0)


# Create pursuer class to hold all pursuer properties
class pursuer:
    def __init__(self):
        t = target()
        # x, y, z: m
        # v, vx, vy, vz: m/s
        # psi (x-y) heading, theta (z-||x, y|| nose angle): rad
        PX0 = 350  # initial sim start pos, used to adjust start pos of new sim
        PY0 = 350
        PZ0 = 0
        TX0 = 503.89  # target data initial position
        TY0 = 509.16
        TZ0 = 250
        # Before interpolation, initial conditions
        RTP_VTP = np.array([PX0-TX0, PY0-TY0, PZ0-TZ0])/(15-11.747952712717765)
        PXnew = RTP_VTP[0]*(INTERCEPT_SPEED-TARGET_SPEED)+TX0
        PYnew = RTP_VTP[1]*(INTERCEPT_SPEED-TARGET_SPEED)+TY0
        PZnew = 0  # still want pursuer to start at relative ground level
        self.x = PXnew
        self.y = PYnew
        self.z = PZnew
        self.v = INTERCEPT_SPEED
        self.at = ACCEL_TIME
        if ACCEL_TIME == 0:  # if pursuer accelerates instantly
            self.a = np.inf
        else:
            self.a = self.v/self.at  # accel value: dv/dt
        i = int(T_DETECT/dt)
        RTP = [t.x[i]-PXnew, t.y[i]-PYnew, t.z[i]-PZnew]
        self.psi = np.arctan2(RTP[1], RTP[0])  # atan(y/x)
        self.theta = np.arctan2(RTP[2], np.sqrt(RTP[1]**2+RTP[0]**2))  # z/(xy)
        self.phi = 0  # roll angle, not needed
        self.vxy = self.v*np.cos(self.theta)
        self.vz = self.v*np.sin(self.theta)
        self.vx = self.vxy*np.cos(self.psi)
        self.vy = self.vxy*np.sin(self.psi)
        self.pall = []  # all positions
        self.vall = []  # all velocities
        self.Eulerall = []  # all Euler angles (navigation angles)
        self.angall = []  # all control deflection
        self.dvall = []  # all dV values
        self.D = []  # drag values
        self.Tr = []  # thrust required values


# Create target class to store kinematics of target throughout sim
class target:
    def __init__(self):
        # use nav and target tracking project 4 traj as baseline, scaled up 10x
        def store_data():
            # data: x, xdot, y, ydot, w
            path_to_truth = Path(r'./truth.csv')
            truth = []
            truth_data = pd.read_csv(path_to_truth, delimiter=',', header=None)
            for i in range(0, len(truth_data)):
                truth.append(truth_data.values[i])
            return np.array(truth)

        # inter between given data points at a higher rate to increase fidelity
        def interpolate(truth):
            x = np.linspace(0, len(truth), len(truth)+1)[0:-1]
            xvals = np.linspace(0, x[-1], RATE*(len(x)-1)+1)[0:-1]
            new = np.zeros((RATE*(len(x)-1), np.shape(truth)[1]))
            for i in range(0, np.shape(truth)[1]):
                y = truth[:, i]
                yinterp = np.interp(xvals, x, y)
                new[:, i] = yinterp
            return new

        def target_2D():
            truth = store_data()
            return interpolate(truth)

        # Generate 3D path, same as 2D path but at some nonzero climb
        def target_3D():
            # data: x, xdot, y, ydot, w, z
            truth2D = target_2D()
            c = CLIMB_RATE/RATE  # divide by new sample rate, num is in m/s
            self.vz = c*RATE
            z_0 = 100
            truth3D = np.zeros(np.shape(truth2D))
            vels = np.zeros(len(truth2D)).reshape(len(truth2D), 1)
            truth3D = np.hstack((truth3D, vels))
            truth3D[0, 0:5] = truth2D[0]
            truth3D[0, 5] = z_0
            for i in range(1, len(truth2D)):
                truth3D[i, 0:5] = truth2D[i]
                truth3D[i, 5] = truth3D[i-1, 5] + c
            return truth3D


        # if the final interpolation file exists, no need to generate
        # the data all over again. just store it and move on
        p = Path(r'./truth_final.csv')  # path to final interpolated data
        if os.path.isfile(p):
            truth = pd.read_csv(Path(r'./truth_final.csv'), delimiter=',',
                                header=None)
            self.x = truth.values[0]
            self.y = truth.values[1]
            self.z = truth.values[2]
            self.vx = truth.values[3]
            self.vy = truth.values[4]
            self.vz = truth.values[5]
            self.v = truth.values[6]
        else:
            # If previous interpolation data does not exist, generate it
            self.t2D = target_2D()
            self.t3D = target_3D()
            self.x = self.t3D[:, 0]
            self.vx = self.t3D[:, 1]
            self.y = self.t3D[:, 2]
            self.vy = self.t3D[:, 3]
            self.z = self.t3D[:, 5]
            self.vz = self.vz*np.ones(len(self.z))
            self.v = np.linalg.norm(np.vstack((self.vx, self.vy, self.vz)).T,
                                    axis=1)  # v mag at each time step
            self.v_new = self.v/np.max(self.v)*TARGET_SPEED  # scale vel mag
            self.vx_new = np.zeros(np.shape(self.vx))
            self.vy_new = np.zeros(np.shape(self.vx))
            self.vz_new = np.zeros(np.shape(self.vx))
            for i in range(0, len(self.vx)):
                r = self.vx[i]/self.vy[i]   # algebra i did to keep ratio const
                r2 = self.vz[i]/self.vy[i]  # between old vel values and new
                mag = self.v_new[i]         # this keeps new path ~= old path
                vy = np.sqrt(mag**2/(r**2+1+r2**2))
                vx = vy*r
                vz = vy*r2
                self.vx_new[i] = vx
                self.vy_new[i] = vy
                self.vz_new[i] = vz
            self.v_new = np.linalg.norm(np.vstack((self.vx_new, self.vy_new,
                                                   self.vz_new)).T, axis=1)
            self.v_new = self.v_new/np.max(self.v_new)*TARGET_SPEED
            self.x_new = np.zeros(np.shape(self.vx))
            self.y_new = np.zeros(np.shape(self.vx))
            self.z_new = np.zeros(np.shape(self.vx))
            self.x_new[0] = self.x[0]
            self.y_new[0] = self.y[0]
            self.z_new[0] = self.z[0]
            for i in range(1, len(self.x)):  # Euler kinematics to move target
                self.x_new[i] = self.x_new[i-1]+self.vx_new[i-1]*dt
                self.y_new[i] = self.y_new[i-1]+self.vy_new[i-1]*dt
                self.z_new[i] = self.z_new[i-1]+self.vz_new[i-1]*dt
            self.x = self.x_new  # above code scales the given target data
            self.y = self.y_new  # by new velocity to keep paths about the same
            self.z = self.z_new
            self.vx = self.vx_new
            self.vy = self.vy_new
            self.vz = self.vz_new
            self.v = self.v_new
            # if we entered here, the final data isn't saved. go ahead and
            # save it to save time on next run
            new = [t.x, t.y, t.z, t.vx, t.vy, t.vz, t.v]
            np.savetxt('truth_final.csv', new, delimiter=",")


# Missile geometry, modeled off of AIM7 Sparrow
class missile:
    def __init__(self):
        self.D = 0.2032  # m, rocket outer diameter
        self.L = 3.65506  # m, rocket length
        self.Ln = 19.2  # in, rocket nose length
        self.De = 3.78  # in, exhaust diameter
        self.Ae = np.pi*self.De**2/4  # in^2, exhaust area
        self.Sref = np.pi*self.D**2/4  # m^2, reference area
        self.nt = 2  # tail surfaces
        self.Stail = 0.14307*2  # m
        self.cmac_tail = 0.31242  # m
        self.tmac_tail = 0.027*self.cmac_tail  # m
        self.b_tail = 0.6096  # m
        self.LLE_tail = DEG2RAD*57  # rad, tail sweep angle
        self.dLE_tail = DEG2RAD*6.17  # rad, tail thickness angle
        self.nw = 2  # wing surfaces
        self.Swing = 0.2369*2  # m
        self.cmac_wing = 0.33782  # m
        self.tmac_wing = 0.044*self.cmac_wing  # m
        self.b_wing = 0.81788  # m
        self.LLE_wing = DEG2RAD*45  # rad
        self.dLE_wing = DEG2RAD*10.01  # rad
        self.mass = 226.8  # kg
        # from solidworks model
        self.volume = 0.1136537933  # m^3
        self.density = self.mass/self.volume
        self.cg = 1.43  # m
        self.Ixx = 1.32    # kg.m^2 roll moment
        self.Iyy = 706.01  # kg.m^2 pitch moment
        self.Izz = 706.01  # km.m^2 yaw moment
        self.CLalpha = 2*np.pi  # big assumption, 2pi is CLalpha for 2D wing
        self.a0 = 0  # deg, symmetric airfoil for wing and tial
        self.lw = 1.60+0.5*0.4623  # m, ac of wing in x direction
        self.lt = 3.271+0.5*0.3475  # m, ac of tail
        self.ac = (self.lw*self.Swing+self.lt*self.Stail) / \
            ((self.Swing+self.Stail))  # area weighted average


# Update pursuer position uisng Eulerian kinematics
def update_pursuer(p, dV, i, m):
    # dV: [dVx, dVy, dVz]
    t_elapsed = i*dt  # time that missile has accelerated
    if i == 0:
        p.angall.append((0, 0, 0, 0))
    p = control_deflection(p, dV, m)
    a = p.a
    if t_elapsed < p.at:  # if missile should still be accelerating
        vmax_i = t_elapsed*a
    else:
        vmax_i = INTERCEPT_SPEED  # hold constant velocity magnitude
        a = 0  # dont want to reset p.a here
    p.v = vmax_i  # max velocity of pursuer at this time step due to accel
    p.x += p.vx*dt  # Eulerian kinematics update, -> true as dt -> 0
    p.y += p.vy*dt
    p.z += p.vz*dt
    p.vx += dV[0]*dt
    p.vy += dV[1]*dt
    p.vz += dV[2]*dt
    # normalize velocity to max velocity, slightly deviating from optimal PN
    v = np.linalg.norm(np.vstack((p.vx, p.vy, p.vz)))  # v at this time step
    p.vx = p.vx/v*p.v  # scale velocity to keep in same direction
    p.vy = p.vy/v*p.v
    p.vz = p.vz/v*p.v
    vxy = np.sqrt(p.vx**2+p.vy**2)  # velocity mag in x-y plane
    p.theta = np.arctan2(p.vz, vxy)  # angle between z and x-y plane
    p.psi = np.arctan2(p.vy, p.vx)  # angle between x and y
    p.pall.append([p.x, p.y, p.z])  # store positions, angles, velocities
    p.vall.append([p.vx, p.vy, p.vz])
    p.Eulerall.append([p.psi, p.theta])
    p.dvall.append(dV)
    p.D.append(Drag(p, m))
    p.Tr.append(m.mass*a+Drag(p, m))  # thrust required, a = (Tr-D)/m
    return p


# Calculate required control deflection to achieve required aero angles
def control_deflection(p, dV, m):
    # assumptions: wing and tail can fully actuate
    alt = p.z*3.28084
    if alt < 36000:
        Tratio = 1-6.875*10**-6*alt
        rhoratio = Tratio**4.2561
    else:
        rhoratio = 0.2971*np.exp(-(alt-36089)/20807)
    rho = rhoratio*0.002377*515.379  # kg/m^3
    # New Time step
    vx2 = p.vx + dV[0]*dt
    vy2 = p.vy + dV[1]*dt
    vz2 = p.vz + dV[2]*dt
    v2 = np.linalg.norm(np.vstack((vx2, vy2, vz2)))
    vxy = np.sqrt(vx2**2+vy2**2)  # == to vbnw
    gamma2 = np.arctan2(vz2, vxy)  # flight path angle at new step
    sigma2 = np.arctan2(vy2, vx2)  # heading at new step
    u2 = v2*np.cos(gamma2)*np.cos(sigma2)  # u v w velocities
    vv2 = v2*np.cos(gamma2)*np.sin(sigma2)
    w2 = v2*np.sin(gamma2)
    aoa2 = np.arctan2(w2, u2)  # new missile angle of attack
    b2 = np.arcsin(vv2/v2)     # new missile beta angle
    Q = 0.5*rho*v2**2
    Lw = Q*m.Swing*m.CLalpha*aoa2
    Lt = -Lw*(m.ac-m.lw)/(m.ac-m.lt)
    alphat2 = aoa2+Lt/(Q*m.Stail*m.CLalpha)
    alphaw2 = 0
    Lw = Q*m.Swing*m.CLalpha*b2
    Lt = -Lw*(m.ac-m.lw)/(m.ac-m.lt)
    betat2 = b2+Lt/(Q*m.Stail*m.CLalpha)
    betaw2 = 0
    p.angall.append((alphaw2, alphat2*RAD2DEG, betaw2, betat2*RAD2DEG))
    return p


# Calculate high fidelity drag using eqs from 691 missile aerodynamics
def Drag(p, m):
    gamma = 1.4
    alt = p.z*3.28084  # altitude in ft
    if alt < 36000:
        Tratio = 1-6.875*10**-6*alt
        Tamb = Tratio*519
        rhoratio = Tratio**4.2561
    else:
        Tamb = 390
        rhoratio = 0.2971*np.exp(-(alt-36089)/20807)
    rhoamb = rhoratio*0.002377  # slug/ft^3
    c = np.sqrt(gamma*287*Tamb*5/9)  # speed of sound, m/s
    vft = p.v*3.28084
    q = 0.5*rhoamb*vft**2
    M = p.v/c  # mach number
    M_LLE_wing = M*np.cos(m.LLE_wing)  # wing experienced mach number
    M_LLE_tail = M*np.cos(m.LLE_tail)
    if M > 1:
        # nose length converted to meters in expression
        CD_bodywave = (1.586+1.834/M**2)*(np.arctan(0.5/(m.Ln/12/3.28084 /
                                                         m.D)))**1.69
        CD_base = 0.25/M*(1-0.00064516*m.Ae/m.Sref)  # mach pressure drag
        t1 = m.nw*(2/(gamma*M_LLE_wing**2))
        t2 = (((((gamma+1)*M_LLE_wing**2)/2)**(gamma/(gamma-1))) *
              (((gamma+1)/(2*gamma*M_LLE_wing**2-(gamma-1)))**(1/(gamma-1)))-1)
        t3 = np.sin(m.dLE_wing)**2*np.cos(m.LLE_wing)*m.tmac_wing * \
            m.b_wing/m.Swing
        CD_surface_wing = t1*t2*t3
        t1 = m.nt*(2/(gamma*M_LLE_tail**2))
        t2 = (((((gamma+1)*M_LLE_tail**2)/2)**(gamma/(gamma-1))) *
              (((gamma+1)/(2*gamma*M_LLE_tail**2-(gamma-1)))**(1/(gamma-1)))-1)
        t3 = np.sin(m.dLE_tail)**2*np.cos(m.LLE_tail)*m.tmac_tail * \
            m.b_tail/m.Stail
        CD_surface_tail = t1*t2*t3
    else:
        CD_bodywave = 0
        CD_base = 0.12+0.13*M**2*(1-0.00064516*m.Ae/m.Sref)
        CD_surface_wing = 0
        CD_surface_tail = 0
    CD_bodyfriction = 0.053*(m.L/m.D)*(M/(q*m.L*3.28084))**0.2
    CD_friction_wing = m.nw*(0.0133*(M_LLE_wing/(q*m.cmac_wing*3.28084))**0.2)\
        * (2*m.Swing/m.Sref)
    CD_friction_tail = m.nt*(0.0133*(M_LLE_tail/(q*m.cmac_tail*3.28084))**0.2)\
        * (2*m.Stail/m.Sref)
    CD_body = CD_bodywave+CD_base+CD_bodyfriction
    CD_tail = CD_friction_tail+CD_surface_tail
    CD_wing = CD_friction_wing+CD_surface_wing
    CD = CD_body+CD_wing+CD_tail
    rho_SI = 1.225*rhoratio  # kg/m^3
    D = 0.5*rho_SI*p.v**2*m.Sref*CD
    return D


# Prop Nav using Zero Effort Miss, from Ben Dickinson on YT
def pure_pn_BD(dims):
    t = target()   # create the target object
    p = pursuer()  # create the pursuer object
    m = missile()  # create the missile object
    p.v = INITIAL_VELOCITY
    N = 4  # prop nav gain, typically 3-5
    if dims == 2:  # If running 2D sim, reduce z to 0s
        t.z = np.zeros(len(t.y))
        t.vz = np.zeros(len(t.y))
    for i in range(T_DETECT*RATE, len(t.z)):
        RTP = np.array([t.x[i]-p.x, t.y[i]-p.y, t.z[i]-p.z])  # rel dist
        VTP = np.array([t.vx[i]-p.vx, t.vy[i]-p.vy, t.vz[i]-p.vz])  # rel vel
        R = np.linalg.norm(RTP)  # distance
        if np.abs(R) < COL_THRESHOLD:  # impact
            break
        if p.v == 0:  # if initial velocity is 0
            tgo = 100000  # gonna take a while to intercept
        else:
            tgo = R/p.v  # est time to intercept
        ZEM_i = RTP + VTP*tgo  # intertial ZEM distance inLOS direction
        ri = RTP/np.linalg.norm(RTP)  # line of sight unit vector
        ZEM_n = ZEM_i-(np.dot(ZEM_i, ri))*ri  # ZEM vector norm to LOS
        dV = N*ZEM_n/tgo**2  # pro nav law
        p = update_pursuer(p, dV, i-T_DETECT*RATE+1, m)  # Update pursuer state
    p.pall = np.array(p.pall)
    p.vall = np.array(p.vall)
    p.Eulerall = np.array(p.Eulerall)
    p.dvall = np.array(p.dvall)
    p.D = np.array(p.D)
    p.Tr = np.array(p.Tr)
    p.angall = np.array(p.angall)
    return p


# Prop Nav using Zero Effort Miss,from modern missile guidance theory
def pure_pn_MMGT(dims):
    t = target()   # create the target object
    p = pursuer()  # create the pursuer object
    m = missile()  # create the missile object
    p.v = INITIAL_VELOCITY
    if dims == 2:  # If running 2D sim, reduce z to 0s
        t.z = np.zeros(len(t.y))
        t.vz = np.zeros(len(t.y))
    for i in range(T_DETECT*RATE, len(t.z)):
        RTP = np.array([t.x[i]-p.x, t.y[i]-p.y, t.z[i]-p.z])  # rel dist
        VTP = np.array([t.vx[i]-p.vx, t.vy[i]-p.vy, t.vz[i]-p.vz])  # rel vel
        R = np.linalg.norm(RTP)  # distance
        if np.abs(R) < COL_THRESHOLD:  # impact
            break
        if p.v == 0:  # if initial velocity is 0
            tgo = 100000  # gonna take a while to intercept
        else:
            tgo = R/p.v  # est time to intercept
        dV = 3/tgo**2*(RTP+VTP*tgo)  # given PN law
        p = update_pursuer(p, dV, i-T_DETECT*RATE+1, m)  # Update pursuer state
    p.pall = np.array(p.pall)
    p.vall = np.array(p.vall)
    p.Eulerall = np.array(p.Eulerall)
    p.D = np.array(p.D)
    p.Tr = np.array(p.Tr)
    p.angall = np.array(p.angall)
    return p


# Augmented PN using relative accceleration
def apn_MMGT(dims):
    t = target()   # create the target object
    p = pursuer()  # create the pursuer object
    m = missile()  # create the missile object
    p.v = INITIAL_VELOCITY
    if dims == 2:  # If running 2D sim, reduce z to 0s
        t.z = np.zeros(len(t.y))
        t.vz = np.zeros(len(t.y))
    for i in range(T_DETECT*RATE, len(t.z)):
        RTP = np.array([t.x[i]-p.x, t.y[i]-p.y, t.z[i]-p.z])  # rel dist
        VTP = np.array([t.vx[i]-p.vx, t.vy[i]-p.vy, t.vz[i]-p.vz])  # rel vel
        if i > 0:
            vT1 = np.array([t.vx[i-1], t.vy[i-1], t.vz[i-1]])
            vT2 = np.array([t.vx[i], t.vy[i], t.vz[i]])
            aT = (vT2-vT1)/dt  # target acceleration
        else:
            aT = 0
        R = np.linalg.norm(RTP)  # distance
        if np.abs(R) < COL_THRESHOLD:  # impact
            break
        if p.v == 0:  # if initial velocity is 0
            tgo = 100000  # gonna take a while to intercept
        else:
            tgo = R/p.v  # est time to intercept
        dV = 3/tgo**2*(RTP+VTP*tgo+0.5*aT*tgo**2)  # augmented PN law
        p = update_pursuer(p, dV, i-T_DETECT*RATE+1, m)  # Update pursuer state
    p.pall = np.array(p.pall)
    p.vall = np.array(p.vall)
    p.Eulerall = np.array(p.Eulerall)
    p.D = np.array(p.D)
    p.Tr = np.array(p.Tr)
    p.angall = np.array(p.angall)
    return p


# Optimal Guidance Law
def OGL(dims):
    T = 0.005  # time constant of missile response, need better estimate
    t = target()   # create the target object
    p = pursuer()  # create the pursuer object
    m = missile()  # create the missile object
    p.v = INITIAL_VELOCITY
    vprev = [p.vx, p.vy, p.vz]
    if dims == 2:  # If running 2D sim, reduce z to 0s
        t.z = np.zeros(len(t.y))
        t.vz = np.zeros(len(t.y))
    for i in range(T_DETECT*RATE, len(t.z)):
        RTP = np.array([t.x[i]-p.x, t.y[i]-p.y, t.z[i]-p.z])  # rel dist
        VTP = np.array([t.vx[i]-p.vx, t.vy[i]-p.vy, t.vz[i]-p.vz])  # rel vel
        if i > 0:
            vT1 = np.array([t.vx[i-1], t.vy[i-1], t.vz[i-1]])
            vT2 = np.array([t.vx[i], t.vy[i], t.vz[i]])
            aT = (vT2-vT1)/dt  # target acceleration
            vM1 = vprev
            vM2 = np.array([p.vx, p.vy, p.vz])
            vprev = vM2
            macc = (vM2-vM1)/dt
        else:
            aT = 0
            macc = 0
        R = np.linalg.norm(RTP)  # distance
        if np.abs(R) < COL_THRESHOLD:  # impact
            break
        if p.v == 0:  # if initial velocity is 0
            tgo = 100000  # gonna take a while to intercept
        else:
            tgo = R/p.v  # est time to interceptt
        t1 = 6*(tgo/T)**2/tgo**2
        t2 = tgo/T+np.exp(-tgo/T)-1
        t3 = RTP+VTP*tgo+0.5*tgo**2*aT-T**2*(tgo/T+np.exp(-tgo/T)-1)*macc
        den = 3+6*tgo/T-6*tgo**2/T**2+2*tgo**3/T**3-12*tgo/T*np.exp(-tgo/T) - \
            3*np.exp(-2*tgo/T)
        dV = t1*t2*t3/den  # OGL law
        p = update_pursuer(p, dV, i-T_DETECT*RATE+1, m)  # Update pursuer state
    p.pall = np.array(p.pall)
    p.vall = np.array(p.vall)
    p.Eulerall = np.array(p.Eulerall)
    p.D = np.array(p.D)
    p.Tr = np.array(p.Tr)
    p.angall = np.array(p.angall)
    return p


def plot(algo):
    check = 0
    if algo == 'all':
        check = 1
    t = target()
    # TARGET PATH FOR REFERENCE
    plt.figure()
    plt.title('2D Target Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.plot(t.x, t.y, 'k')
    plt.figure(figsize=(6, 8))
    ax = plt.axes(projection='3d')
    ax.set_title('3D Target Path')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.plot3D(t.x, t.y, t.z)
    # PURE PRO NAV
    pnp = pure_pn_BD(2)
    pnp3D = pure_pn_BD(3)
    if algo == 'PN' or check == 1:
        plt.figure()
        plt.title('2D Pure Proportional Navigation')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.plot(t.x[0:len(pnp.pall[:, 0])-1+T_DETECT*RATE],
                 t.y[0:len(pnp.pall[:, 0])-1+T_DETECT*RATE], 'k',
                 label='Target')
        plt.plot(pnp.pall[:, 0], pnp.pall[:, 1], label='Pursuer')
        plt.plot(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], 'g+', markersize=10,
                 label='Detected')
        plt.plot(pnp.x, pnp.y, 'rx', markersize=10, label='Collision')
        plt.legend(loc='lower left')
        plt.figure(figsize=(6, 8))
        ax = plt.axes(projection='3d')
        ax.set_title('3D Pure Proportional Navigation')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.plot3D(t.x[0:len(pnp3D.pall[:, 0])-1+T_DETECT*RATE],
                  t.y[0:len(pnp3D.pall[:, 0])-1+T_DETECT*RATE],
                  t.z[0:len(pnp3D.pall[:, 0])-1+T_DETECT*RATE], 'k',
                  label='Target')
        ax.plot3D(pnp3D.pall[:, 0], pnp3D.pall[:, 1], pnp3D.pall[:, 2],
                  label='Pursuer')
        ax.plot3D(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], t.z[T_DETECT*RATE],
                  'g+', markersize=10, label='Detected')
        ax.plot3D(pnp3D.pall[-1, 0], pnp3D.pall[-1, 1], pnp3D.pall[-1, 2],
                  'rx', markersize=10, label='Collision')
        plt.legend()
        print('-----------------Pure Proportional Navigation-----------------')
        print('2D Intercept time (after detection at ' +
              str(T_DETECT) + ' s): ' + str(len(pnp.pall[:, 0])/RATE) + ' s')
        print('3D Intercept time (after detection at ' + str(T_DETECT) +
              ' s): ' + str(len(pnp3D.pall[:, 0])/RATE) + ' s')
        plt.figure()
        plt.title('Pure Proportional Navigation Euler Angles')
        t_span = np.linspace(0, len(t.x), len(t.x)+1)/RATE
        plt.plot(t_span[0:len(pnp.Eulerall[:, 0])], pnp.Eulerall[:, 0]*RAD2DEG,
                 label='2D Psi')
        plt.plot(t_span[0:len(pnp.Eulerall[:, 0])], pnp.Eulerall[:, 1]*RAD2DEG,
                 label='2D Theta')
        plt.plot(t_span[0:len(pnp3D.Eulerall[:, 0])],
                 pnp3D.Eulerall[:, 0]*RAD2DEG, 'k-.', label='3D Psi')
        plt.plot(t_span[0:len(pnp3D.Eulerall[:, 0])],
                 pnp3D.Eulerall[:, 1]*RAD2DEG, 'k--', label='3D Theta')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.figure()
        plt.title('3D Pure Proportional Navigation Drag and Thrust Required')
        plt.plot(t_span[0:len(pnp3D.D)], pnp3D.D, label='Drag')
        plt.plot(t_span[0:len(pnp3D.D)], pnp3D.Tr, 'k-.',
                 label='Thrust Required')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Pure PN Tail Deflections')
        plt.plot(t_span[0:len(pnp.angall[:, 0])], pnp.angall[:, 1],
                 label='2D Horizontal Tail')
        plt.plot(t_span[0:len(pnp.angall[:, 0])], pnp.angall[:, 3],
                 label='2D Vertical Tail')
        plt.plot(t_span[0:len(pnp3D.angall[:, 0])], pnp3D.angall[:, 1],
                 'k-.', label='3D Horizontal Tail')
        plt.plot(t_span[0:len(pnp3D.angall[:, 0])], pnp3D.angall[:, 3],
                 'k--', label='3D Vertical Tail')
        plt.legend()
    # PURE MMGT PRO NAV
    pnpMMGT = pure_pn_MMGT(2)
    pnp3DMMGT = pure_pn_MMGT(3)
    if algo == 'MMGT' or check == 1:
        print('--------------Pure Proportional Navigation MMGT---------------')
        plt.figure()
        plt.title('2D Pure Proportional Navigation MMGT')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.plot(t.x[0:len(pnpMMGT.pall[:, 0])-1+T_DETECT*RATE],
                 t.y[0:len(pnpMMGT.pall[:, 0])-1+T_DETECT*RATE], 'k',
                 label='Target')
        plt.plot(pnpMMGT.pall[:, 0], pnpMMGT.pall[:, 1], label='Pursuer')
        plt.plot(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], 'g+', markersize=10,
                 label='Detected')
        plt.plot(pnpMMGT.x, pnpMMGT.y, 'rx', markersize=10, label='Collision')
        plt.legend(loc='lower right')
        plt.figure(figsize=(6, 8))
        ax = plt.axes(projection='3d')
        ax.set_title('3D Pure Proportional Navigation MMGT')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.plot3D(t.x[0:len(pnp3DMMGT.pall[:, 0])-1+T_DETECT*RATE],
                  t.y[0:len(pnp3DMMGT.pall[:, 0])-1+T_DETECT*RATE],
                  t.z[0:len(pnp3DMMGT.pall[:, 0])-1+T_DETECT*RATE], 'k',
                  label='Target')
        ax.plot3D(pnp3DMMGT.pall[:, 0], pnp3DMMGT.pall[:, 1],
                  pnp3DMMGT.pall[:, 2], label='Pursuer')
        ax.plot3D(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], t.z[T_DETECT*RATE],
                  'g+', markersize=10, label='Detected')
        ax.plot3D(pnp3DMMGT.pall[-1, 0], pnp3DMMGT.pall[-1, 1],
                  pnp3DMMGT.pall[-1, 2], 'rx', markersize=10,
                  label='Collision')
        plt.legend()
        print('2D Intercept time (after detection at ' + str(T_DETECT) +
              ' s): ' + str(len(pnpMMGT.pall[:, 0])/RATE) + ' s')
        print('3D Intercept time (after detection at ' + str(T_DETECT) +
              ' s): ' + str(len(pnp3DMMGT.pall[:, 0])/RATE) + ' s')
        plt.figure()
        plt.title('Pure Proportional Navigation Euler Angles MMGT')
        t_span = np.linspace(0, len(t.x), len(t.x)+1)/RATE
        plt.plot(t_span[0:len(pnpMMGT.Eulerall[:, 0])],
                 pnpMMGT.Eulerall[:, 0]*RAD2DEG, label='2D Psi')
        plt.plot(t_span[0:len(pnpMMGT.Eulerall[:, 0])],
                 pnpMMGT.Eulerall[:, 1]*RAD2DEG, label='2D Theta')
        plt.plot(t_span[0:len(pnp3DMMGT.Eulerall[:, 0])],
                 pnp3DMMGT.Eulerall[:, 0]*RAD2DEG, 'k-.', label='3D Psi')
        plt.plot(t_span[0:len(pnp3DMMGT.Eulerall[:, 0])],
                 pnp3DMMGT.Eulerall[:, 1]*RAD2DEG, 'k--', label='3D Theta')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.figure()
        plt.title('MMGT PN Drag and Thrust Required')
        plt.plot(t_span[0:len(pnp3DMMGT.D)], pnp3DMMGT.D, label='Drag')
        plt.plot(t_span[0:len(pnp3DMMGT.D)], pnp3DMMGT.Tr, 'k-.',
                 label='Thrust Required')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('PN MMGT Tail Deflections')
        plt.plot(t_span[0:len(pnpMMGT.angall[:, 0])], pnpMMGT.angall[:, 1],
                 label='2D Horizontal Tail')
        plt.plot(t_span[0:len(pnpMMGT.angall[:, 0])], pnpMMGT.angall[:, 3],
                 label='2D Vertical Tail')
        plt.plot(t_span[0:len(pnp3DMMGT.angall[:, 0])], pnp3DMMGT.angall[:, 1],
                 'k-.', label='3D Horizontal Tail')
        plt.plot(t_span[0:len(pnp3DMMGT.angall[:, 0])], pnp3DMMGT.angall[:, 3],
                 'k--', label='3D Vertical Tail')
        plt.legend()
    # AUGMENTED PRO NAV
    apnp = apn_MMGT(2)
    apnp3D = apn_MMGT(3)
    if algo == 'APN' or check == 1:
        print('--------------Augmented Proportional Navigation---------------')
        plt.figure()
        plt.title('2D Augmented Proportional Navigation MMGT')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.plot(t.x[0:len(apnp.pall[:, 0])-1+T_DETECT*RATE],
                 t.y[0:len(apnp.pall[:, 0])-1+T_DETECT*RATE], 'k',
                 label='Target')
        plt.plot(apnp.pall[:, 0], apnp.pall[:, 1], label='Pursuer')
        plt.plot(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], 'g+', markersize=10,
                 label='Detected')
        plt.plot(apnp.x, apnp.y, 'rx', markersize=10, label='Collision')
        plt.legend(loc='lower right')
        plt.figure(figsize=(6, 8))
        ax = plt.axes(projection='3d')
        ax.set_title('3D Augmented Proportional Navigation MMGT')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.plot3D(t.x[0:len(apnp3D.pall[:, 0])-1+T_DETECT*RATE],
                  t.y[0:len(apnp3D.pall[:, 0])-1+T_DETECT*RATE],
                  t.z[0:len(apnp3D.pall[:, 0])-1+T_DETECT*RATE], 'k',
                  label='Target')
        ax.plot3D(apnp3D.pall[:, 0], apnp3D.pall[:, 1], apnp3D.pall[:, 2],
                  label='Pursuer')
        ax.plot3D(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], t.z[T_DETECT*RATE],
                  'g+', markersize=10, label='Detected')
        ax.plot3D(apnp3D.pall[-1, 0], apnp3D.pall[-1, 1], apnp3D.pall[-1, 2],
                  'rx', markersize=10, label='Collision')
        plt.legend()
        print('2D Intercept time (after detection at ' + str(T_DETECT) +
              ' s): ' + str(len(apnp.pall[:, 0])/RATE) + ' s')
        print('3D Intercept time (after detection at ' + str(T_DETECT) +
              ' s): ' + str(len(apnp3D.pall[:, 0])/RATE) + ' s')
        plt.figure()
        plt.title('Augmented Proportional Navigation Euler Angles MMGT')
        plt.plot(t_span[0:len(apnp.Eulerall[:, 0])],
                 apnp.Eulerall[:, 0]*RAD2DEG, label='2D Psi')
        plt.plot(t_span[0:len(apnp.Eulerall[:, 0])],
                 apnp.Eulerall[:, 1]*RAD2DEG, label='2D Theta')
        plt.plot(t_span[0:len(apnp3D.Eulerall[:, 0])],
                 apnp3D.Eulerall[:, 0]*RAD2DEG, 'k-.', label='3D Psi')
        plt.plot(t_span[0:len(apnp3D.Eulerall[:, 0])],
                 apnp3D.Eulerall[:, 1]*RAD2DEG, 'k--', label='3D Theta')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.figure()
        plt.title('Augmented PN Drag and Thrust Required')
        plt.plot(t_span[0:len(apnp3D.D)], apnp3D.D, label='Drag')
        plt.plot(t_span[0:len(apnp3D.D)], apnp3D.Tr, 'k-.',
                 label='Thrust Required')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Augmented Proportional Navigation Tail Deflections')
        plt.plot(t_span[0:len(apnp.angall[:, 0])], apnp.angall[:, 1],
                 label='2D Horizontal Tail')
        plt.plot(t_span[0:len(apnp.angall[:, 0])], apnp.angall[:, 3],
                 label='2D Vertical Tail')
        plt.plot(t_span[0:len(apnp3D.angall[:, 0])], apnp3D.angall[:, 1],
                 'k-.', label='3D Horizontal Tail')
        plt.plot(t_span[0:len(apnp3D.angall[:, 0])], apnp3D.angall[:, 3],
                 'k--', label='3D Vertical Tail')
        plt.legend()
        # Optimal Guidance Law
    OGLp = OGL(2)
    OGL3Dp = OGL(3)
    if algo == 'OGL' or check == 1:
        print('---------------------Optimal Guidance Law---------------------')
        plt.figure()
        plt.title('2D Optimal Guidance Law')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.plot(t.x[0:len(OGLp.pall[:, 0])-1+T_DETECT*RATE],
                 t.y[0:len(OGLp.pall[:, 0])-1+T_DETECT*RATE], 'k',
                 label='Target')
        plt.plot(OGLp.pall[:, 0], OGLp.pall[:, 1], label='Pursuer')
        plt.plot(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], 'g+', markersize=10,
                 label='Detected')
        plt.plot(OGLp.x, OGLp.y, 'rx', markersize=10, label='Collision')
        plt.legend(loc='lower right')
        plt.figure(figsize=(6, 8))
        ax = plt.axes(projection='3d')
        ax.set_title('3D Optimal Guidance Law')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.plot3D(t.x[0:len(OGL3Dp.pall[:, 0])-1+T_DETECT*RATE],
                  t.y[0:len(OGL3Dp.pall[:, 0])-1+T_DETECT*RATE],
                  t.z[0:len(OGL3Dp.pall[:, 0])-1+T_DETECT*RATE], 'k',
                  label='Target')
        ax.plot3D(OGL3Dp.pall[:, 0], OGL3Dp.pall[:, 1], OGL3Dp.pall[:, 2],
                  label='Pursuer')
        ax.plot3D(t.x[T_DETECT*RATE], t.y[T_DETECT*RATE], t.z[T_DETECT*RATE],
                  'g+', markersize=10, label='Detected')
        ax.plot3D(OGL3Dp.pall[-1, 0], OGL3Dp.pall[-1, 1], OGL3Dp.pall[-1, 2],
                  'rx', markersize=10, label='Collision')
        plt.legend()
        print('2D Intercept time (after detection at ' + str(T_DETECT) +
              ' s): ' + str(len(OGLp.pall[:, 0])/RATE) + ' s')
        print('3D Intercept time (after detection at ' + str(T_DETECT) +
              ' s): ' + str(len(OGL3Dp.pall[:, 0])/RATE) + ' s')
        plt.figure()
        plt.title('Optimal Guidance Law Euler Angles')
        t_span = np.linspace(0, len(t.x), len(t.x)+1)/RATE
        plt.plot(t_span[0:len(OGLp.Eulerall[:, 0])],
                 OGLp.Eulerall[:, 0]*RAD2DEG, label='2D Psi')
        plt.plot(t_span[0:len(OGLp.Eulerall[:, 0])],
                 OGLp.Eulerall[:, 1]*RAD2DEG, label='2D Theta')
        plt.plot(t_span[0:len(OGL3Dp.Eulerall[:, 0])],
                 OGL3Dp.Eulerall[:, 0]*RAD2DEG, 'k-.', label='3D Psi')
        plt.plot(t_span[0:len(OGL3Dp.Eulerall[:, 0])],
                 OGL3Dp.Eulerall[:, 1]*RAD2DEG, 'k--', label='3D Theta')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.figure()
        plt.title('Optimal Guidance Law Drag and Thrust Required')
        plt.plot(t_span[0:len(OGL3Dp.D)], OGL3Dp.D, label='Drag')
        plt.plot(t_span[0:len(OGL3Dp.D)], OGL3Dp.Tr, 'k-.',
                 label='Thrust Required')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('OGL Tail Deflections')
        plt.plot(t_span[0:len(OGLp.angall[:, 0])], OGLp.angall[:, 1],
                 label='2D Horizontal Tail')
        plt.plot(t_span[0:len(OGLp.angall[:, 0])], OGLp.angall[:, 3],
                 label='2D Vertical Tail')
        plt.plot(t_span[0:len(OGL3Dp.angall[:, 0])], OGL3Dp.angall[:, 1],
                 'k-.', label='3D Horizontal Tail')
        plt.plot(t_span[0:len(OGL3Dp.angall[:, 0])], OGL3Dp.angall[:, 3],
                 'k--', label='3D Vertical Tail')
        plt.legend()
    return pnp, pnp3D, pnpMMGT, pnp3DMMGT, apnp, apnp3D, OGLp, OGL3Dp


# 'PN', 'MMGT', 'APN', 'OGL', 'all'
m = missile()
t = target()
pnp, pnp3D, pnpMMGT, pnp3DMMGT, apnp, apnp3D, OGLp, OGL3Dp = plot('all')
