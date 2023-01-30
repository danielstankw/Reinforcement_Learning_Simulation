import numpy as np
from SimWrapper.Simulation import Simulator
from SimWrapper.InverseKinematics  import IK
from mujoco_py import (functions, MjSim, MjSimState)
class RHELPER:
    def __init__(self, sim):
        self.sim = sim
        self.simHelper = Simulator("UR5Gripper_only.xml")
        self.simHelper.graphics =  False
        self.gripper_site = 'gripper_finger'
        self.gripper_body = 'dummy_gripper'
        self.gripper_geom = 'gripper_geom'
        self.gripper_geom_id = self.sim.model._geom_name2id['gripper_geom']
        self.slot_id = self.sim.model._geom_name2id['Inner']
        self.sim_z_offset = -0.855
        self.last_forces_avg = [0,0,0]
    def get_xpos(self,body):
        return self.sim.data.get_body_xpos(body)

    def get_gripper_xpos(self, with_offset = True):
        gripper_offset = np.array([-0.18198229,  0.00018623, -0.10253947])
        offset = np.add(gripper_offset,np.array([0,0,with_offset*self.sim_z_offset]))
        return np.add(self.get_xpos_by_site(self.gripper_site),offset)

    def get_xpos_by_site(self,site):
        return self.sim.data.get_site_xpos(site)


    def get_gripper_vel(self):
        return self.get_vel_by_site(self.gripper_site)

    def get_vel(self,body):
        return self.sim.data.get_body_xvelp(body)

    def get_H(self):
        q_current = self.get_current_joints()
        state = self.simHelper.sim.get_state()
        state.qpos[0:6] = q_current
        self.simHelper.sim.set_state(state)
        H = np.zeros(self.simHelper.sim.model.nv * self.simHelper.sim.model.nv)
        functions.mj_fullM(self.simHelper.sim.model, H, self.simHelper.sim.data.qM)
        return H

    def get_vel_by_site(self,site):
        return self.sim.data.get_site_xvelp(site)

    def getJL(self, site):
        target_jacp = np.zeros(3 * self.sim.model.nv)
        self.sim.data.get_site_jacp(site, jacp=target_jacp)
        J_L = target_jacp.reshape((3, self.sim.model.nv))
        J_L = J_L[0:3, 0:6]
        return J_L

    def getJA(self, site):
        target_jacr = np.zeros(3 * self.sim.model.nv)
        self.sim.data.get_site_jacp(site, jacr=target_jacr)
        J_A = target_jacr.reshape((3, self.sim.model.nv))
        J_A = J_A[0:3, 0:6]
        return J_A

    def getJ(self,site):
        J_L = self.getJL(site)
        J_A = self.getJA(site)
        J = np.concatenate((J_L, J_A), axis=0)
        return J

    def getJ_T(self, site):
        J_L = self.getJL(site)
        J_A = self.getJA(site)
        J = np.concatenate((J_L, J_A), axis=0)
        return np.transpose(J)

    def get_current_joints(self):
        return self.sim.data.qpos[0:6]
    def get_current_joints_vel(self):
        return self.sim.data.qvel[0:6]

    def get_current_gripper_joints(self):
        return self.sim.data.qpos[6:9]

    def get_sensor_force(self):
        force = self.find_contacts()
        return force

    def find_contacts(self):
        ctemp =  np.zeros(6, dtype=np.float64)
        csum = np.zeros(6, dtype=np.float64)
        if self.sim.data.ncon > 1:
            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
                if (contact.geom1 ==  self.gripper_geom_id or contact.geom2 ==  self.gripper_geom_id) and (contact.geom1 !=  self.slot_id or contact.geom2 ==  self.slot_id) :
                    functions.mj_contactForce(self.sim.model,self.sim.data,i,ctemp)
                    csum +=ctemp
        gripper_orn = self.get_gripper_orn()
        force = np.dot(csum[0:3], gripper_orn)
        x = 0
        z = 1
        y = 2
        force = [force[x], force[y], force[z]]
        if ((np.abs(np.divide(force,self.last_forces_avg))) > [4,4,4]).any() and (np.abs(force) > [20,20,20]).any():
            force = self.last_forces_avg
        force = np.add(np.dot(self.last_forces_avg,2/3),np.dot(force,1/3))
        self.last_forces_avg = force
        torque  = np.dot(-csum[3:6], gripper_orn)
        torque = [torque[x], torque[y], torque[z]]
        return force


    def get_gripper_orn(self):
        return self.sim.data.get_geom_xmat(self.gripper_geom)







