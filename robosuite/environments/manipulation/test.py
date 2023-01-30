from robosuite.models.objects import BallObject
from robosuite.models.objects import CylinderObject, CerealObject, BreadObject, MilkObject, PlateWithHoleObject, Wire
from robosuite.utils.mjcf_utils import new_joint

from robosuite.models.arenas import TableArena

from robosuite.models import MujocoWorldBase


world = MujocoWorldBase()

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)
milk = MilkObject('milk')
board = PlateWithHoleObject('board')
wire = Wire('wire')
# Note how we don't call .get_obj()!
world.merge(wire)


model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
#viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
    sim.step()
    viewer.render()
