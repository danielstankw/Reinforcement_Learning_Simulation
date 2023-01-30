from matplotlib import pyplot as plt
from numpy import cos as cos
from numpy import sin as sin
import numpy as np
from numpy import diff
import matplotlib.pyplot as plt
from mujoco_py import (functions)
import ImpFuncs as IF


import globals
L1=globals.L1
L2=globals.L2
L3=globals.L3
SimStep=globals.SimStep
waitingTime=globals.waitingTime
sim=globals.sim
viewer=globals.viewer

#___# General navigation functions:
#full path navigation function
def Navigate(K,M,B,P,D,wantedPath, wantedTimes, waitingTime,method,methodCntrl,methodParams):
    #initialazing:
    dofs=np.shape(P[0])
    elbow=0 #can be 0/1 - choose the path of the inverse kinematics of the manipulator.
    q = invkin(wantedPath[0][0], wantedPath[0][1],wantedPath[0][2],elbow)
    sim.data.qpos[0] = q[0]
    sim.data.qpos[1] = q[1]
    sim.data.qpos[2] = q[2]

    #Impedance initital condition (for IMP only):
    xm=np.array([wantedPath[0][0],wantedPath[0][1],wantedPath[0][2]]).reshape(-1,1)
    xmDot=np.array([0,0,0]).reshape(-1,1) #specific for Robot at rest

    #Generalazied data storage vectors:
    accumulatedForce = []
    accumulatedControl = []
    accumulatedTravel = []
    accumulatedPlannedPath = []
    accumulatedJoints = []
    accumulatedJointsVel = []
    waitSim2(waitingTime)
    forcebool=0 #boolean variable that can be used to check if the endeffector is being touched.

    #General navigation loop:
    for i in range(len(wantedPath) - 1):
        nSteps = np.int32(wantedTimes[i] / SimStep)
        path = linearPathPlan([wantedPath[i][0], wantedPath[i][1],wantedPath[i][2]], [wantedPath[i + 1][0], wantedPath[i + 1][1], wantedPath[i + 1][2]], wantedTimes[i])
        [q_vec, q_dot_vec] = inversePathplan(path, wantedTimes[i], method,elbow,i)

        x_vec=np.array([path[0,:],path[2,:],path[4,:]])
        xDot_vec = np.array([path[1, :], path[3, :],path[5,:]])

        if methodCntrl == "PD": #PD navigation between 2 pts
            [travel, sensorData, controlData, jointData, jointVelData] = straightLineNav(q_vec, q_dot_vec, nSteps, P, D)
        if methodCntrl =="imp": #Imp navigation between 2 pts
            [travel, sensorData, controlData, jointData, jointVelData,xm] = IF.straightLineNavimp(x_vec,xDot_vec, nSteps,P,D,K,M,B,xm,xmDot,methodParams,dofs,elbow,q_vec, q_dot_vec,forcebool)

        #append:
        accumulatedJoints.append(jointData)
        accumulatedJointsVel.append(jointVelData)
        accumulatedForce.append(sensorData)
        accumulatedControl.append(controlData)
        accumulatedTravel.append(travel)
        accumulatedPlannedPath.append(np.transpose(path))  # path needed to be transposed in order to work


    # #finish path with step command to final target - can be applied ##
    # tmplength=len(wantedPath)-1
    # tmq=invkin(wantedPath[tmplength][0], wantedPath[tmplength][1],wantedPath[tmplength][2],elbow)
    # tmpTime=4
    # [q_vec,q_dot_vec]=stepPath(tmq[0],tmq[1],tmq[2],tmpTime)
    # nSteps = np.int32(tmpTime / SimStep)
    # [x_vec, xDot_vec] = stepPath(xm[0], xm[1],xm[2], tmpTime)
    #
    # x_vec=np.transpose(x_vec)
    # xDot_vec=np.transpose(xDot_vec)
    #
    # if methodCntrl == "PD":  # PD navigation between 2 pts
    #     [travel, sensorData, controlData, jointData, jointVelData] = straightLineNav(q_vec, q_dot_vec, nSteps, P, D)
    # if methodCntrl == "imp":  # Imp navigation between 2 pts
    #     [travel, sensorData, controlData, jointData, jointVelData,xm] = IF.straightLineNavimp(x_vec,xDot_vec, nSteps,P,D,K,M,B,xm,xmDot,methodParams,dofs,elbow,q_vec,q_dot_vec,forcebool)
    #
    # # append:
    # accumulatedJoints.append(jointData)
    # accumulatedJointsVel.append(jointVelData)
    # accumulatedForce.append(sensorData)
    # accumulatedControl.append(controlData)
    # accumulatedTravel.append(travel)
    # accumulatedPlannedPath.append(np.transpose(path))  # path needed to be transposed in order to work

    # Total outputs arrange:
    totalJoint    = outputArrange(accumulatedJoints)
    totalJointVel = outputArrange(accumulatedJointsVel)
    totalForce    = outputArrange(accumulatedForce)
    totalControl  = outputArrange(accumulatedControl)
    totalTravel   = outputArrange(accumulatedTravel)
    totalPath     = outputArrange(accumulatedPlannedPath)

    #saving csv files with all the navigation data.
    from numpy import savetxt
    savetxt('totalTravel.csv',totalTravel,delimiter=',')
    savetxt('totalPath.csv',totalPath,delimiter=',')
    savetxt('totalControl.csv', totalControl, delimiter=',')
    savetxt('totalForce.csv', totalForce, delimiter=',')
    savetxt('totalJointVel.csv', totalJointVel, delimiter=',')
    savetxt('totalJoint.csv', totalJoint, delimiter=',')

    #plotting: #plot all graphs.
    #jointPlotter(totalJoint)
    #jointVelPlotter(totalJointVel)
    #forcePlotter(totalForce)
    #controlPlotter(totalControl)
    #pathPlotter(totalTravel, totalPath)
    #plt.show()

    sim.reset()

#____#kinematics:

# inverse kinematics function. input  px,py,phi,elbow - output: q1,q2,q3
def invkin(px, py,phi,elbow):
        delta=0
        change_elbow=0
        x3 = px - L3 * cos(phi)
        y3 = py - L3 * sin(phi)
        c2 = (x3 ** 2 + y3 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2+delta)
        if c2>1:
            c2=1
        if elbow ==1:
            s2 = (1 - c2 ** 2) ** 0.5
        if elbow == 0:
            s2 = -(1 - c2 ** 2) ** 0.5
        theta2 = np.arctan2(s2, c2)
        k1 = L1 + L2 * c2
        k2 = L2 * s2
        c1 = (k1 * x3 + k2 * y3) / (k1 ** 2 + k2 ** 2 + delta)
        s1 = (-k2 * x3 + k1 * y3) / (k1 ** 2 + k2 ** 2 + delta)
        theta1 = np.arctan2(s1, c1)
        if np.abs(theta2)<0:
            change_elbow==1
        theta3 = phi - theta1 - theta2
        if change_elbow==1:
            if elbow ==1:
                elbow=0
            if elbow ==0:
                elbow=1
        q = [theta1, theta2, theta3, elbow]
        return q
# forward kinematics function. input  q1,q2,q3 output: x1,x2,phi
def forwardkin(q1,q2,q3):
    global L1
    global L2
    x=L3*(cos(q1+q2+q3))+ L2*(cos(q1)*cos(q2)-sin(q1)*sin(q2))+L1*cos(q1)
    y=L3*(sin(q1+q2+q3))+L2*(cos(q1)*sin(q2)+sin(q1)*cos(q2))+L1*sin(q1)
    phi=q1+q2+q3
    return x,y,phi


# planar path planning function. Input: current position of the end effector,
# target position of the robot, time of navigation, number of points in the path
# output: navigation vector in cartesian domain.
def linearPathPlan(curr_pos, target_pos, T):
    global SimStep
    xPath = []
    yPath = []
    phiPath = []
    x_dot_Path = []
    y_dot_Path = []
    phi_dot_Path = []
    t_list = np.linspace(0, T, num=np.int32(T / SimStep))
    # Const definitions:
    kx = (target_pos[0] - curr_pos[0]) / T ** 3
    kphi =  (target_pos[2] - curr_pos[2]) / T ** 3
    a5 = 6 / T ** 2
    a4 = -(15 / T)
    a3 = 10
    delta = 0.0001

    # Specific case when path is vertical:


    if target_pos[0] == curr_pos[0]:
        ky = (target_pos[1] - curr_pos[1]) / T ** 3
        for t in t_list:
            x = target_pos[0]
            y = ky * (a5 * (t ** 5) + a4 * (t ** 4) + a3 * (t ** 3)) + curr_pos[1]
            phi = kphi * (a5 * (t ** 5) + a4 * (t ** 4) + a3 * (t ** 3)) + curr_pos[2]

            x_dot = 0
            y_dot = ky * (a5 * 5 * (t ** 4) + a4 * 4 * (t ** 3) + 3 * a3 * (t ** 2))
            phi_dot = kphi * (a5 * 5 * (t ** 4) + a4 * 4 * (t ** 3) + 3 * a3 * (t ** 2))

            xPath.append(x)
            yPath.append(y)
            phiPath.append(phi)

            x_dot_Path.append(x_dot)
            y_dot_Path.append(y_dot)
            phi_dot_Path.append(phi_dot)

        path = np.array([xPath, x_dot_Path, yPath, y_dot_Path, phiPath, phi_dot_Path])
        return path


    # General path:

    ky = ((target_pos[1] - curr_pos[1]) / (delta + target_pos[0] - curr_pos[0]))
    ky0 = (target_pos[0] * curr_pos[1] - target_pos[1] * curr_pos[0]) / (delta + target_pos[0] - curr_pos[0])
        # General case:
    for t in t_list:
            x = kx * (a5 * (t ** 5) + a4 * (t ** 4) + a3 * (t ** 3)) + curr_pos[0]
            y = ky * x + ky0
            phi = kphi * (a5 * (t ** 5) + a4 * (t ** 4) + a3 * (t ** 3)) + curr_pos[2]
            x_dot = kx * (a5 * 5 * (t ** 4) + a4 * 4 * (t ** 3) + 3 * a3 * (t ** 2))
            y_dot = ky * x_dot
            phi_dot = kphi * (a5 * 5 * (t ** 4) + a4 * 4 * (t ** 3) + 3 * a3 * (t ** 2))
            xPath.append(x)
            yPath.append(y)
            phiPath.append(phi)
            x_dot_Path.append(x_dot)
            y_dot_Path.append(y_dot)
            phi_dot_Path.append(phi_dot)

    path = np.array([xPath, x_dot_Path, yPath, y_dot_Path, phiPath, phi_dot_Path])
    return path

# function that convert path vector to path vector in the joints domain
def inversePathplan(path, Time,method,elbow,ii):
    xPath = path[0, :]
    x_dot_Path = path[1, :]
    yPath = path[2, :]
    y_dot_Path = path[3, :]
    phiPath = path[4, :]
    phi_dot_Path = path[5, :]

    # plt.scatter(xPath, yPath) # used for debug.
    q_vec = []
    t_list = np.linspace(0, Time, np.int32(Time / SimStep))
    for i in range(len(t_list)):
        q_vec.append(invkin(xPath[i], yPath[i],phiPath[i],elbow))
    q_vec = np.array(q_vec)

    if method=='numerical': #numeric
         q_dot_vec = np.array([diff(q_vec[:, 0]) / SimStep, diff(q_vec[:, 1]) / SimStep,diff(q_vec[:, 2]) / SimStep])
         b = [0, 0, 0]
         q_dot_vec = q_dot_vec.transpose()
         q_dot_vec = np.append(q_dot_vec, b).reshape(q_dot_vec.shape[0] + 1, q_dot_vec.shape[1])

    return [q_vec, q_dot_vec]

# one point navigating function with PD control.
def pdNav2Target(q_target, q_dot_target,P,D):
    #output parameters:
    effector_pos = []
    sensorData = []
    control_effort = []
    angle=[]
    angleVel=[]
    # Control:
    error = q_target - sim.data.actuator_length
    q_dot = sim.data.actuator_velocity - q_dot_target
    #torques output:
    action = np.matmul(P, error) - np.matmul(D, q_dot) +[sim.data.qfrc_bias[0], sim.data.qfrc_bias[1], sim.data.qfrc_bias[2]] # sim.data.qfrc_bias is Gravity bias/robot dependant

    #Force collection:
    forces,torques =IF.getForces(3,'gripper')

    # collecting sensors data: particular to 3Dof Robot
    sensorData.append([forces[0],forces[1],forces[2],torques[0],torques[1],torques[2]])
    effector_pos.append([np.array(sim.data.site_xpos[0])[0], np.array(sim.data.site_xpos[0])[1], np.array(sim.data.site_xpos[0])[2]])
    control_effort.append([action[0],action[1],action[2]])
    angle.append([sim.data.actuator_length[0],sim.data.actuator_length[1],sim.data.actuator_length[2]])
    angleVel.append([sim.data.actuator_velocity[0],sim.data.actuator_velocity[1],sim.data.actuator_velocity[2]])

    return effector_pos, sensorData, control_effort, action,angle,angleVel  # collecting position of endeffector
# total navigation function - > gets all navigation points and looping for one
# point navigation function.
def straightLineNav(qvec, q_dot_vec, nPts,P,D):
    jointDataTotal = []
    jointVelDataTotal = []
    effector_pos_total = []
    sensorDataTotal = []
    control_effort_total = []

    for i in range(nPts):
        effector_pos, sensorData, control_effort, action, angle, angleVel = pdNav2Target([qvec[i, 0], qvec[i, 1], qvec[i, 2]], [q_dot_vec[i, 0], q_dot_vec[i, 1],q_dot_vec[i, 2]],P,D)

        #Output appending:
        jointDataTotal.append(angle)
        jointVelDataTotal.append(angleVel)
        effector_pos_total.append(effector_pos)
        sensorDataTotal.append(sensorData)
        control_effort_total.append(control_effort)

        # Action and rendering:
        sim.data.ctrl[:] = action
        sim.step()
        sim.forward()
        viewer.render()
    #Arrange vectors:
    full_joints_data = outputArrange(jointDataTotal)
    full_joints_vel_data = outputArrange(jointVelDataTotal)
    full_sensor_data = outputArrange(sensorDataTotal)
    full_effector_data = outputArrange(effector_pos_total)
    full_control_data = outputArrange(control_effort_total)

    return full_effector_data, full_sensor_data, full_control_data ,full_joints_data , full_joints_vel_data


#__#General purpose functions:
#function that arranges the output vector and a form that is easy to use.
#input un arranged vector -> output arranged vector
def outputArrange(vec):
    sizeF = 0
    for i in range(len(vec)):
        for j in range(len(vec[i])):
            sizeF = sizeF + 1

    vec_width=len(vec[0][0])
    full_vec = np.zeros([sizeF, vec_width])
    sizeF = 0
    for i in range(len(vec)):
        for j in range(len(vec[i])):
            full_vec[sizeF, :] = np.array(vec[i][j])
            sizeF = sizeF + 1
    return full_vec
#step input signal to the robot joints space
def stepPath(q1,q2,q3,T):
    global SimStep
    nSteps = np.int32(T / SimStep)
    path=np.zeros([nSteps,np.size(sim.data.actuator_length)])
    path_qdot=np.zeros([nSteps,np.size(sim.data.actuator_length)])
    for i in range(nSteps):
        path[i,:]=[q1,q2,q3]
        path_qdot[i,:]=[0,0,0]
    return path,path_qdot
#step input signal to the robot cartesian space
def stepPathCartesian(x,y,T):
    global SimStep
    nSteps = np.int32(T / SimStep)
    path=np.zeros([nSteps,np.size(sim.data.actuator_length)])
    path_qdot=np.zeros([nSteps,np.size(sim.data.actuator_length)])
    for i in range(nSteps):
        path[i,:]=[x,y]
        path_qdot[i,:]=[0,0]
    return path,path_qdot
#get forces and torques that acting about simulation site with name sitename


#___# all the rest:
#function that output q dot by knowitng xdot vec and q vec.
#specific to RR planar robot
def get_q_dot_from_jack(q,x_dot,y_dot):
    global waitingTime
    global L1
    global L2
    J=np.zeros([2,2])
    delta=0.1
    J[0,0]=-L2*(cos(q[0])*sin(q[1]) + cos(q[1])*sin(q[0])) -L1*sin(q[0])
    J[1,0]= L2*(cos(q[0])*cos(q[1]) - sin(q[0])*sin(q[1])) +L1*cos(q[0])
    J[0,1]= -L2*(cos(q[0])*sin(q[1]) + cos(q[1])*sin(q[0]))
    J[1,1] = L2 * (cos(q[0]) * cos(q[1]) - sin(q[0]) * sin(q[1]))
    x_vec=np.array([x_dot,y_dot]).reshape(-1,1)
    try: #try singular point
        J_inv=np.linalg.inv(J)
        qdot_vec = np.matmul(np.linalg.inv(J), x_vec)
    except Exception:
        print('robot went through a singular point')
    try: #custom solution for first step at qvec=0,0
        if sim.data.time<waitingTime*1.1:
            qdot_vec=np.transpose([0,0])
    finally:
        return (np.transpose(qdot_vec))
    return (np.transpose(qdot_vec))
#function that output qdotaim by knowing xdot vec and q vec
#specific for RRR planar robot
def get_q_dotaim(q,qd,x_dotaim):
    global waitingTime
    global L1
    global L2
    q=np.array(q)
    qd=np.array(qd)
    Jd=np.zeros([2,2])
    Jd[0,0] = -L2*(cos(q[0])*cos(q[1])*qd[0] + cos(q[0])*cos(q[1])*qd[1] - sin(q[0])*sin(q[1])*qd[0] - sin(q[0])*sin(q[1])*qd[1]) - L1*cos(q[0])*qd[0]
    Jd[1,0] = -L2*(cos(q[0])*sin(q[1])*qd[1] + cos(q[1])*sin(q[0])*qd[0] + cos(q[0])*sin(q[1])*qd[1] + cos(q[1])*sin(q[0])*qd[1]) - L1*sin(q[0])*qd[0]
    Jd[0,1] = -L2*(cos(q[0])*cos(q[1])*qd[0] + cos(q[0])*cos(q[1])*qd[1] - sin(q[0])*sin(q[1])*qd[0] - sin(q[0])*sin(q[1])*qd[1])
    Jd[1,1] = -L2*(cos(q[0])*sin(q[1])*qd[0] + cos(q[1])*sin(q[0])*qd[0] + cos(q[0])*sin(q[1])*qd[1] + cos(q[1])*sin(q[0])*qd[1])

    J=np.zeros([2,2])
    J[0,0]=-L2*(cos(q[0])*sin(q[1]) + cos(q[1])*sin(q[0])) -L1*sin(q[0])
    J[1,0]= L2*(cos(q[0])*cos(q[1]) - sin(q[0])*sin(q[1])) +L1*cos(q[0])
    J[0,1]= -L2*(cos(q[0])*sin(q[1]) + cos(q[1])*sin(q[0]))
    J[1,1] = L2 * (cos(q[0]) * cos(q[1]) - sin(q[0]) * sin(q[1]))

    try: #try singular point
        J_inv=np.linalg.inv(J)
        qdotaim_vec = np.matmul(np.linalg.inv(J),x_dotaim-np.matmul(Jd,qd))
    except Exception:
        print('robot went through a singular point')
    try: #custom solution for first step at qvec=0,0
        if sim.data.time<waitingTime*1.1:
            qdotaim_vec=([0,0])
    finally:
        return (np.array(qdotaim_vec))
    return (np.array(qdotaim_vec))

#___#Plotting functions
def pathPlotter(travel,path):
    fig = plt.figure(0)
    plt.plot(travel[:, 0], travel[:, 1], c='r')
    plt.plot(travel[0, 0], travel[0, 1], c='r', marker='X')
    plt.axis('equal')
    plt.grid()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.scatter(path[1:, 0], path[1:, 2], c='b')
    plt.scatter(path[0, 0], path[0, 2], c='b', marker='X')
def forcePlotter(force):
    fig = plt.figure(1)
    plt.plot(force[:, 0], 'green')
    plt.xlabel("Sim step")
    plt.ylabel("x force")
    plt.grid()
    plt.draw()

    fig = plt.figure(2)
    plt.plot(force[:, 1], 'blue')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("y force")
    plt.draw()

    '''''''''
    fig = plt.figure(3)
    plt.plot(force[:, 2], 'red')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("z force")
    plt.draw()

    
    fig = plt.figure(4)
    plt.plot(force[:, 3], 'green')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("torque 1")
    plt.draw()

    fig = plt.figure(5)
    plt.plot(force[:, 4], 'blue')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("torque 2")
    plt.draw()  
    '''''''''

    fig = plt.figure(3)
    plt.plot(force[:, 5], 'red')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("torque 3")
    plt.draw()

def controlPlotter(control):
    fig = plt.figure(4)
    plt.plot(control[:, 0], 'red')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("q1 control effort")
    plt.draw()

    fig = plt.figure(5)
    plt.plot(control[:, 1], 'blue')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("q2 control effort")
    plt.draw()
def jointPlotter(joint):
    fig = plt.figure(6)
    plt.plot(joint[:, 0], 'red')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("q1 [rad]]")
    plt.draw()

    fig = plt.figure(7)
    plt.plot(joint[:, 1], 'blue')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("q2 [rad]")
    plt.draw()
def jointVelPlotter(jointVel):
    fig = plt.figure(8)
    plt.plot(jointVel[:, 0], 'red')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("q1 dot [rad/sec]]")
    plt.draw()

    fig = plt.figure(9)
    plt.plot(jointVel[:, 1], 'blue')
    plt.grid()
    plt.xlabel("Sim step")
    plt.ylabel("q2 dot [rad/sec]")
    plt.draw()

#__#Waiting functions:
#waits with documentation and  action
def waitSim(T):
    global SimStep
    nSteps = np.int32(T / SimStep)
    q_vec=np.zeros([nSteps,np.size(sim.data.actuator_length)])
    q_dot_vec = np.zeros([nSteps, np.size(sim.data.actuator_length)])
    for i in range(nSteps):
        q_vec[i,:]=[sim.data.qpos[0],sim.data.qpos[1]]
        q_dot_vec[i,:]=sim.data.actuator_velocity
    [travel, sensorData, controlData] = straightLineNav(q_vec,q_dot_vec, nSteps)
    return travel, sensorData, controlData
#waits without documentationand without any action
def waitSim2(T):
    nSteps = np.int32(T / SimStep)
    for i in range(nSteps):
         sim.step()
         sim.forward()
         #viewer.render()

#get desired impedance params:
def getDesiredParamsAnalytic(xm,xmDot,xmDotaim):
    qd = invkin(xm[0],xm[1],xm[2])
    qdDot=get_q_dot_from_jack(qd, xmDot[0], xmDot[1])
    qdDot=np.array(qdDot).reshape(-1,1)
    qdDotaim=get_q_dotaim(qd,qdDot,xmDotaim)
    return qd, qdDot, qdDotaim