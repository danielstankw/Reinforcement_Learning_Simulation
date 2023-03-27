from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
# from .osc import OperationalSpaceController
# from .joint_pos import JointPositionController
# from .joint_vel import JointVelocityController
# from .joint_tor import JointTorqueController
from .impedance_near_hole import ImpedancePositionBaseControllerPartial
from .impedance_near_hole_with_spiral import ImpedanceWithSpiral
from .impedance_near_hole_with_spiral_ML import ImpedanceWithSpiralAndML
from .impedance_near_label_collection import ImpedanceSpiralMeasurements

CONTROLLER_INFO = {
    "JOINT_VELOCITY":  "Joint Velocity",
    "JOINT_TORQUE":    "Joint Torque",
    "JOINT_POSITION":  "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE":     "Operational Space Control (Position + Orientation)",
    "IK_POSE":      "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
    "IMPEDANCE_POSE_Partial": "full impedance control position based (Position+Orientation), work only near impact",
    "IMPEDANCE_WITH_SPIRAL":  "impedance with spiral",
    "IMPEDANCE_WITH_SPIRAL_AND_ML": 'impedance with spiral search and ML for overlap detection',
    "IMPEDANCE_SPIRAL_LABEL_COLLECTION": 'PD spiral search for label collection, for circle and spiral',
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
