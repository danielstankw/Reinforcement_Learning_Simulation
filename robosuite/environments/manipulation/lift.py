from collections import OrderedDict
import numpy as np
import collections
from copy import deepcopy
import random

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, new_site
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, PlateWithHoleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class Lift(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        num_via_point=0,
        dist_error=0.002,
        angle_error=0,
        tanh_value=2.0,
        r_reach_value=0.94,
        error_type='circle',
        control_spec=36,
        peg_radius=(0.0025, 0.0025),  # (0.00125, 0.00125)
        peg_length=0.12,
    ):

        #min jerk param:
        self.num_via_point = num_via_point

        # settings for table top
        self.via_point = OrderedDict()
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length

        self.dist_error = dist_error
        self.angle_error = angle_error

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            dist_error=dist_error,
            tanh_value=tanh_value,
            r_reach_value=r_reach_value,
            error_type=error_type,
            control_spec=control_spec,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 100.0 is provided if the peg is inside the plate's hole
              - Note that we enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        Un-normalized summed components if using reward shaping:

            - ????

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale

        """
        #   TODO - reward(self, action=None) - change this function
        reward = 0
        time_factor = (self.horizon - self.timestep) / self.horizon
        # Right location and angle
        if self._check_success() and self.num_via_point == 1:
            # reward = self.horizon * time_factor

            self.success += 1
            if self.success == 2:
                S = 1
            return reward

        # use a shaping reward
        if self.reward_shaping:
            # Grab relevant values
            t, d, cos = self._compute_orientation()
            # Reach a terminal state as quickly as possible

            # reaching reward
            reward += self.r_reach * 5 * cos  # * time_factor

            # Orientation reward
            reward += self.hor_dist
            # reward += 1 - np.tanh(2.0*d)
            # reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, we need to scale our sparse reward so that the max reward is identical
        # to its dense version
        else:
            reward *= 5.0

        if self.reward_scale is not None:
            reward *= self.reward_scale

        if (self.num_via_point == 1
                and ((abs(self.hole_pos[0] - self.peg_pos[0]) > 0.014
                      or abs(self.hole_pos[1] - self.peg_pos[1]) > 0.014)
                     and self.peg_pos[2] < self.table_offset[2] + 0.1)
                or self.horizon - self.timestep == 1
        ):
            reward = 0 * -self.horizon / 3
            # self.checked = (self.num_via_points-2)
            # self.switch = 0
            # self.switch_seq = 0
            # self.success = 0
            # # self.trans *= 3
            # self.reset_via_point()
            # self.built_min_jerk_traj()

        return reward

    def on_peg(self):

        res = False
        if (
                abs(self.hole_pos[0] - self.peg_pos[0]) < 0.015
                and abs(self.hole_pos[1] - self.peg_pos[1]) < 0.007
                and abs(self.hole_pos[1] - self.peg_pos[1]) + abs(self.hole_pos[0] - self.peg_pos[0]) < 0.04
                and self.peg_pos[2] < self.table_offset[2] + 0.05
        ):
            res = True
        return res

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        self.peg_radius = 0.025
        self.peg_height = 0.12
        self.peg_z_offset = 0.9
        self.rotation = None
        x_range = [-0.0, 0.0]
        y_range = [-0.1, -0.1]

        # initialize objects of interest
        self.peg = CylinderObject(name='peg',
                                  size=[self.peg_radius, self.peg_height],
                                  density=1,
                                  duplicate_collision_geoms=True,
                                  rgba=[1, 0, 0, 1], joints=None)

        # load peg object (returns extracted object in XML form)
        peg_obj = self.peg.get_obj()
        # set pegs position relative to place where it is being placed
        peg_obj.set("pos", array_to_string((0, 0, -0.04)))
        peg_obj.append(new_site(name="peg_site", pos=(0, 0, self.peg_height), size=(0.005,)))
        # append the object top the gripper (attach body to body)
        # main_eef = self.robots[0].robot_model.eef_name    # 'robot0_right_hand'
        main_eef = self.robots[0].gripper.bodies[1]     # 'gripper0_eef' body
        main_model = self.robots[0].robot_model     # <robosuite.models.robots.manipulators.ur5e_robot.UR5e at 0x7fd9ead87ca0>
        main_body = find_elements(root=main_model.worldbody, tags="body", attribs={"name": main_eef}, return_first=True)
        main_body.append(peg_obj)   # attach body to body

        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        hole_rot_set = str(np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]))
        hole_pos_set = np.array([np.random.uniform(high=x_range[0], low=x_range[1]), np.random.uniform(high=y_range[0], low=y_range[1]), 0.83])
        hole_pos_str = ' '.join(map(str, hole_pos_set))
        hole_rot_str = ' '.join(map(str, hole_rot_set))

        self.hole = PlateWithHoleObject(name='hole')
        hole_obj = self.hole.get_obj()
        hole_obj.set("quat", hole_rot_str)
        hole_obj.set("pos", hole_pos_str)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hole
        )

        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.peg)

        ## Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(self.peg)
        # else:
        #     """Object samplers use the bottom_site and top_site sites of each object in order to place objects on top of other objects,
        #     and the horizontal_radius_site site in order to ensure that objects do not collide with one another. """

        # task includes arena, robot, and objects of interest
        # self.model = ManipulationTask(
        #     mujoco_arena=mujoco_arena,
        #     mujoco_robots=[robot.robot_model for robot in self.robots],
        #     mujoco_objects=[self.peg],
        # )

    # def _load_model(self):
    #     """
    #     Loads an xml model, puts it in self.model
    #     """
    #     super()._load_model()
    #
    #     # Adjust base pose accordingly
    #     xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
    #     self.robots[0].robot_model.set_base_xpos(xpos)
    #
    #     # load model for table top workspace
    #     mujoco_arena = TableArena(
    #         table_full_size=self.table_full_size,
    #         table_friction=self.table_friction,
    #         table_offset=self.table_offset,
    #     )
    #
    #     # Arena always gets set to zero origin
    #     mujoco_arena.set_origin([0, 0, 0])
    #
    #     # initialize objects of interest
    #     tex_attrib = {
    #         "type": "cube",
    #     }
    #     mat_attrib = {
    #         "texrepeat": "1 1",
    #         "specular": "0.4",
    #         "shininess": "0.1",
    #     }
    #     redwood = CustomMaterial(
    #         texture="WoodRed",
    #         tex_name="redwood",
    #         mat_name="redwood_mat",
    #         tex_attrib=tex_attrib,
    #         mat_attrib=mat_attrib,
    #     )
    #     # self.cube = BoxObject(
    #     #     name="cube",
    #     #     size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
    #     #     size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
    #     #     rgba=[1, 0, 0, 1],
    #     #     material=redwood,
    #     # )
    #     self.cube = PlateWithHoleObject(name="cube")
    #     # Create placement initializer
    #     if self.placement_initializer is not None:
    #         self.placement_initializer.reset()
    #         self.placement_initializer.add_objects(self.cube)
    #     else:
    #         self.placement_initializer = UniformRandomSampler(
    #             name="cube",
    #             mujoco_objects=self.cube,
    #             x_range=[-0.03, 0.03],
    #             y_range=[-0.03, 0.03],
    #             rotation=None,
    #             ensure_object_boundary_in_range=True,
    #             ensure_valid_placement=True,
    #             reference_pos=self.table_offset,
    #             z_offset=0.01,
    #         )
    #
    #     self.placement_initializer.reset()
    #
    #
    #     # Add this nut to the placement initializerr
    #     self.placement_initializer.add_objects(self.cube)
    #     # task includes arena, robot, and objects of interest
    #     # self.hole = PlateWithHoleObject(name='hole',)
    #     # self.hole = PlateWith5mmHoleObject(name='peg_hole')
    #     # hole_obj = self.hole.get_obj()
    #     # hole_obj.set("quat", "0 0 0.707 0.707")
    #     # hole_obj.set("pos", "0.1 0.2 1.17")
    #
    #     self.model = ManipulationTask(
    #         mujoco_arena=mujoco_arena,
    #         mujoco_robots=[robot.robot_model for robot in self.robots],
    #         mujoco_objects=self.cube,
    #     )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # peg-related observables
            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg_body_id])

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return T.convert_quat(np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_peg_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["peg_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "peg_pos" in obs_cache else np.zeros(3)

            sensors = [peg_pos, peg_quat, gripper_to_peg_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.num_via_point = 0
        self.success = 0
        self.enter = 1
        self.t_bias = 0
        self.reset_via_point()
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        # if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            # object_placements = self.placement_initializer.sample()
            #
            # # Loop through all objects and reset their positions
            # for obj_pos, obj_quat, obj in object_placements.values():
            #     self.sim.data.set_joint_qpos(obj.joints, np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the peg.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the peg
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.peg)

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        #   TODO - _check_success(self) - change this function
        #   calculat pegs end position.
        self.r_reach = 0
        self.hor_dist = 0
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos_center = self.sim.data.body_xpos[self.peg_body_id]
        handquat = T.convert_quat(self.sim.data.get_body_xquat("robot0_right_hand"), to="xyzw")
        handDCM = T.quat2mat(handquat)
        self.peg_pos = self.sim.data.get_site_xpos(
            "peg_site")  # peg_pos_center + (handDCM @ [0, 0, 2*self.peg_length]).T

        self.hole_pos = self.sim.data.get_site_xpos("hole_middle_cylinder")
        hole_mat = self.sim.data.body_xmat[self.sim.model.body_name2id("hole_hole")]
        hole_mat.shape = (3, 3)

        dist = np.linalg.norm(self.peg_pos - self.hole_pos)
        horizon_dist = np.linalg.norm(self.peg_pos[:2] - self.hole_pos[:2])
        self.hor_dist = 1 - np.tanh(self.tanh_value * 2 * horizon_dist)
        self.r_reach = 1 - np.tanh(self.tanh_value * dist)
        self.objects_on_pegs = int(
            self.on_peg() and self.r_reach > self.r_reach_value)  # r_reach(tanh*2)=0.96, r_reach(tanh*20)=0.67

        return np.sum(self.objects_on_pegs) > 0

        # t, d, cos = self._compute_orientation()
        # if (d < 0.099 and -0.12 <= t <= 0.12 and cos > 0.95):
        #     D=1
        # return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95

        # return d < 0.099 and -0.12 <= t <= 0.08 and cos > 0.95

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        #   calculat pegs end position.
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos_center = self.sim.data.body_xpos[self.peg_body_id]
        handquat = T.convert_quat(self.sim.data.get_body_xquat("robot0_right_hand"), to="xyzw")
        handDCM = T.quat2mat(handquat)
        self.peg_pos = self.sim.data.get_site_xpos("peg_site")  # peg_pos_center + (handDCM @ [0, 0, self.peg_length]).T

        # hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.sim.model.body_name2id("hole_hole")]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = self.hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - self.peg_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, self.peg_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(
                np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)
            ),
        )

    def reset_via_point(self):

        added0 = 0.083
        added1 = 0.02

        pos_via_point_0 = deepcopy(self.sim.data.get_site_xpos("hole_middle_cylinder"))
        pos_via_point_0[2] += added0
        pos_via_point_1 = deepcopy(self.sim.data.get_site_xpos("hole_middle_cylinder"))
        pos_via_point_1[2] -= added1

        hole_angle = deepcopy(T.mat2euler(T.quat2mat(T.convert_quat(self.sim.data.get_body_xquat("hole_hole"), to="wxyz"))))
        angle_desired = hole_angle

        trans_error = [0, 0, 0]
        if self.error_type == 'fixed':
            trans_error = np.array([-60, -125, 0]) * self.dist_error  # fixed error
        if self.error_type == 'circle':
            trans_error = ((np.random.rand(3) - 0.5) * 2) * self.dist_error  # chenged error
        if self.error_type == 'ring':
            trans_error[:2] = random.choice(([random.uniform(0.0012, self.dist_error) * random.choice((-1, 1)),
                                              random.uniform(0., self.dist_error) * random.choice((-1, 1))],
                                             [random.uniform(0., self.dist_error) * random.choice((-1, 1)),
                                              random.uniform(0.0012, self.dist_error) * random.choice((-1, 1))]))
        if self.error_type == 'fixed_dir':
            x_error = random.uniform(0.002, self.dist_error) * -1.1    # -1,1,0
            y_error = random.uniform(0.0012, self.dist_error) * 0  # -1,1,0
            trans_error = np.array([x_error, y_error, 0])
            # print(trans_error)

        trans_error = self.sim.data.get_body_xmat("hole_hole") @ trans_error
        trans_error[2] = 0
        angle_error = ((np.random.rand(3) - 0.5) * 2) * (np.pi / 2) * self.angle_error

        via_point_0 = np.concatenate((pos_via_point_0 + trans_error, angle_desired), axis=-1)
        via_point_1 = np.concatenate((pos_via_point_1 + trans_error, angle_desired), axis=-1)
        # via_point_0 = np.concatenate((pos_via_point_0 + trans_error, angle_desired + angle_error), axis=-1)
        # via_point_1 = np.concatenate((pos_via_point_1 + trans_error, angle_desired + angle_error), axis=-1)

        self.via_point['p0'] = via_point_0
        self.via_point['p1'] = via_point_1

        peg_pos = self.sim.data.get_site_xpos("gripper0_grip_site")
        self.pos_in = deepcopy([peg_pos, T.mat2euler(self.sim.data.get_site_xmat("peg_site"))])
