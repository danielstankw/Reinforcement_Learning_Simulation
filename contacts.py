import numpy as np
from mujoco_py import (functions, MjSim, MjSimState)


def find_contacts(sim, gripper_geom_id, gripper_geom):
    TABLE = sim.model.body_name2id('OUR_TABLE')
    box = sim.model.body_name2id('box')
    ctemp = np.zeros(6, dtype=np.float64)
    csum = np.zeros(6, dtype=np.float64)
    if sim.data.ncon > 1:
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            cond1 = contact.geom1 == gripper_geom_id
            cond2 = contact.geom2 == gripper_geom_id
            if cond1:
                if contact.geom2 == 16: 
                    functions.mj_contactForce(sim.model, sim.data, i, ctemp)
                    csum += ctemp
            elif cond2:
                if contact.geom1 == 16: 
                    functions.mj_contactForce(sim.model, sim.data, i, ctemp)
                    csum += ctemp
    gripper_orn = sim.data.get_geom_xmat(gripper_geom)
    force = np.dot(csum[0:3], gripper_orn)
    x = 2
    y = 0
    z = 1

    force = np.array([force[x], force[y], force[z]])
    torque = np.dot(-csum[3:6], gripper_orn)
    torque = [torque[x], torque[y], torque[z]]
    return force
