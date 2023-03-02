from collections import OrderedDict
import numpy as np
import collections
from copy import deepcopy
import random

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, new_site
from robosuite.utils.mjcf_utils import CustomMaterial
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, PlateWithHoleSmallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class PegInHoleSmall(SingleArmEnv):
    """
    This class corresponds to the peg-in-hole task for a single robot arm.

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
            render_camera="hole_board",
            # ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand').
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            time_free=0,
            time_insertion=0,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            plot_graphs=None,
            num_via_point=0,
            dist_error=0.2,
            angle_error=0,
            tanh_value=2.0,
            r_reach_value=0.94,
            error_type='circle',
            control_spec=36,
            peg_radius=0.0021,  # (0.00125, 0.00125)
            peg_length=0.03,
            fixed_error_vec=np.zeros(3)
    ):

        self.fixed_error_vec = fixed_error_vec
        self.plot_graphs = plot_graphs
        self.error = None
        self.time_free = time_free
        self.time_insert = time_insertion
        # min jerk param:
        self.num_via_point = num_via_point
        # settings for table top
        self.via_point = OrderedDict()
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length

        self.dist_error = dist_error/1000
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
            dist_error=self.dist_error,
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
        # Right location and angle
        if self._check_success() and self.num_via_point == 1:
            self.success += 1

        # if self.robots[0].controller.stop is True:
        #     # breakpoint()
        #     self.success = -5
        #     return reward

        t, d, cos = self._compute_orientation()
        reward += self.r_reach * cos
        reward += self.hor_dist * 5
        reward += cos

        if self.reward_scale is not None:
            reward *= self.reward_scale

        if (self.num_via_point == 1
                and (abs(self.hole_pos[0] - self.peg_pos[0]) > 0.045
                     or abs(self.hole_pos[1] - self.peg_pos[1]) > 0.014
                     or self.peg_pos[2] > self.table_offset[2] + 0.1)):

            self.success -= 1

        return reward

    def on_peg(self):

        res = False
        hole_hole = deepcopy(self.sim.data.get_body_xpos("hole_hole"))
        goal_z = hole_hole[2] + 0.018# 0.015
        if (abs(self.hole_pos[0] - self.peg_pos[0]) < 0.0007
                and abs(self.hole_pos[1] - self.peg_pos[1]) < 0.0007
                and self.peg_pos[2] <= goal_z):
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

        # load model for tabletop workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        self.peg_z_offset = 0.9
        self.rotation = None
        # TODO: randomization of peg position
        # x_range = [-0.1, 0.1]
        # y_range = [-0.1, 0.1]

        x_range = [0.0, 0.0]
        y_range = [0.0, 0.0]

        # initialize objects of interest
        self.peg = CylinderObject(name='peg',
                                  size=[self.peg_radius, self.peg_length],
                                  density=1,
                                  duplicate_collision_geoms=True,
                                  rgba=[1, 0, 0, 1], joints=None)

        # load peg object (returns extracted object in XML form)
        peg_obj = self.peg.get_obj()
        # set pegs position relative to place where it is being placed
        peg_obj.set("pos", array_to_string((0, 0, 0)))
        peg_obj.append(new_site(name="peg_site", pos=(0, 0, self.peg_length), size=(0.0005,)))
        # append the object top the gripper (attach body to body)
        # main_eef = self.robots[0].robot_model.eef_name    # 'robot0_right_hand'
        main_eef = self.robots[0].gripper.bodies[1]  # 'gripper0_eef' body
        main_model = self.robots[0].robot_model  # <robosuite.models.robots.manipulators.ur5e_robot.UR5e at 0x7fd9ead87ca0>
        main_body = find_elements(root=main_model.worldbody, tags="body", attribs={"name": main_eef}, return_first=True)
        main_body.append(peg_obj)  # attach body to body

        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        hole_rot_set = str(np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]))
        hole_pos_set = np.array(
            [np.random.uniform(high=x_range[0], low=x_range[1]), np.random.uniform(high=y_range[0], low=y_range[1]),
             0.9])
        hole_pos_str = ' '.join(map(str, hole_pos_set))
        hole_rot_str = ' '.join(map(str, hole_rot_set))

        self.hole = PlateWithHoleSmallObject(name='hole')
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
        # TODO: used to save full XML
        # self.model.save_model('/home/danieln1/Desktop/latest_code/robosuite//siemens/test.xml',True)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
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

            modality = "object"

            # position and rotation of peg and hole
            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hole_body_id])

            @sensor(modality=modality)
            def hole_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.hole_body_id], to="xyzw")

            @sensor(modality=modality)
            def peg_to_hole(obs_cache):
                return obs_cache["hole_pos"] - np.array(self.sim.data.body_xpos[self.peg_body_id]) if \
                    "hole_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")

            # Relative orientation parameters
            @sensor(modality=modality)
            def angle(obs_cache):
                t, d, cos = self._compute_orientation()
                obs_cache["t"] = t
                obs_cache["d"] = d
                return cos

            @sensor(modality=modality)
            def t(obs_cache):
                return obs_cache["t"] if "t" in obs_cache else 0.0

            @sensor(modality=modality)
            def d(obs_cache):
                return obs_cache["d"] if "d" in obs_cache else 0.0

            sensors = [hole_pos]
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

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the peg.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Elad: enable visualization of environments objects sites:
        # If you dont wont to display environments site's uncomment this or set "vis_settings['env']" to "False"
        vis_settings['env'] = True
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
        #   calculate pegs end position.
        self.r_reach = 0
        self.hor_dist = 0
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        self.peg_pos = self.sim.data.get_site_xpos("peg_site")

        # TODO I changed here from Shirs "hole_middle_cylinder" > "hole_hole"
        # self.hole_pos = self.sim.data.get_site_xpos("hole_middle_cylinder")
        self.hole_pos = self.sim.data.get_body_xpos("hole_hole")
        hole_mat = self.sim.data.body_xmat[self.sim.model.body_name2id("hole_hole")]
        hole_mat.shape = (3, 3)

        dist = np.linalg.norm(self.peg_pos - self.hole_pos)
        horizon_dist = np.linalg.norm(self.peg_pos[:2] - self.hole_pos[:2])
        self.hor_dist = 1 - np.tanh(self.tanh_value * 2 * horizon_dist)
        self.r_reach = 1 - np.tanh(self.tanh_value * dist)

        self.objects_on_pegs = self.on_peg() and self.r_reach > self.r_reach_value

        return self.objects_on_pegs

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

        added0 = 3 * self.peg_length

        pos_via_point_0 = deepcopy(self.sim.data.get_body_xpos("hole_hole"))
        pos_via_point_0[2] += added0
        pos_via_point_1 = deepcopy(self.sim.data.get_body_xpos("hole_hole"))
        # pos_via_point_1[2] -= added1
        pos_via_point_1[2] += self.peg_length + 0.012

        hole_angle = deepcopy(T.quat2mat(T.convert_quat(self.sim.data.get_body_xquat("hole_hole"), to="wxyz")))
        angle_desired = hole_angle
        # touch: array([7.49693725e-04, -6.16444867e-04, 8.95993197e-01])

        trans_error = [0, 0, 0]
        if self.error_type == 'none':
            trans_error = np.array([0.0, 0, 0]) * self.dist_error * 0  # fixed error
        if self.error_type == 'fixed':
            # trans_error = np.array([7.14533037e-04, -5.01936469e-04, 8.95959506e-01])
            trans_error = self.fixed_error_vec  # fixed error
        if self.error_type == 'ring':
            r_low = 0.6 / 1000  # 3.3 / 1000
            r_high = 0.8 / 1000  # 3.9 / 1000
            r = random.uniform(r_low, r_high)
            theta = random.uniform(0, 2 * np.pi)
            x_error = r * np.cos(theta)
            y_error = r * np.sin(theta)
            # #
            trans_error[:2] = np.array([x_error, y_error])

        if self.error_type == 'fixed_dir':
            x_error = random.uniform(0.002, self.dist_error) * 5.1  # -1,1,0
            y_error = random.uniform(0.0012, self.dist_error) * 0  # -1,1,0
            trans_error = np.array([x_error, y_error, 0])
        print('Error used', np.round(trans_error, 6))
        trans_error[2] = 0
        # print()
        self.error = deepcopy(trans_error)
        # angle_error = ((np.random.rand(3) - 0.5) * 2) * (np.pi / 2) * self.angle_error

        # # daniel
        # rot_vec = np.sqrt(0.5) * np.array([np.pi, np.pi, 0.00])  # ori_init ## np.array([np.pi, 0, 0])
        # angle_desired1 = R.from_rotvec(rot_vec).as_matrix()

        self.via_point['p0'] = pos_via_point_0 + trans_error
        # add_rotation = R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        # self.via_point['o0'] = add_rotation @ angle_desired
        self.via_point['o0'] = angle_desired
        self.via_point['p1'] = pos_via_point_1 + trans_error
        self.via_point['o1'] = angle_desired
