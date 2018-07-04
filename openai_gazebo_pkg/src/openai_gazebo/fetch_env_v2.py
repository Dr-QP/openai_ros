import numpy as np

from gym.envs.robotics import rotations, robot_env_v2, utils
"""
import sys
import copy
import moveit_commander
import moveit_msgs.msg
"""
import rospy
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
#from gazebo_connection import GazeboConnection
from .gazebo_connection import GazeboConnection
#from controllers_connection import ControllersConnection
from .controllers_connection import ControllersConnection
from sensor_msgs.msg import JointState
from fetch_train.srv import EePose, EePoseRequest, EeRpy, EeRpyRequest, EeTraj, EeTrajRequest, JointTraj, JointTrajRequest


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env_v2.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        
        #rospy.init_node('fetch_gym', anonymous=True)
        
        #n_substeps = 20
        """
        initial_qpos = {
            'joint0': 0.0,
            'joint1': 0.0,
            'joint2': 0.0,
            'joint3': -1.5,
            'joint4': 0.0,
            'joint5': 1.5,
            'joint6': 0.0,
            'object': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        """
        
        #moveit_commander.roscpp_initialize(sys.argv)
        
        #robot = moveit_commander.RobotCommander()
        #scene = moveit_commander.PlanningSceneInterface()    
        #self.group = moveit_commander.MoveGroupCommander("arm")
        #display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory)
        
        JOINT_STATES_SUBSCRIBER = '/joint_states'
        
        self.joint_states_sub = rospy.Subscriber(JOINT_STATES_SUBSCRIBER, JointState, self.joint_states_callback)
        self.joint_states_data = JointState()
        
        self.ee_traj_client = rospy.ServiceProxy('/ee_traj_srv', EeTraj)
        self.joint_traj_client = rospy.ServiceProxy('/joint_traj_srv', JointTraj)
        self.ee_pose_client = rospy.ServiceProxy('/ee_pose_srv', EePose)
        self.ee_rpy_client = rospy.ServiceProxy('/ee_rpy_srv', EeRpy)
        
        self.gazebo = GazeboConnection()
        self.controllers_object = ControllersConnection(namespace="")
        
        self.init_pos = initial_qpos

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    
    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        #if self.block_gripper:
            #self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            #self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            #self.sim.forward()
        #self.execute_trajectory()
        pass

    def _set_action(self, action):
        #print "Action set_action:"
        #print action
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        #pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        #print "Action after concat"
        #print action

        # Apply action to simulation.
        #utils.ctrl_set_action(self.sim, action)
        #utils.mocap_set_action(self.sim, action)
        self.set_trajectory_ee(action)

    def _get_obs(self):
        # positions
        #grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_pos = self.get_ee_pose()
        grip_pos_array = np.array([grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z])
        #dt = self.sim.nsubsteps * self.sim.model.opt.timestep #What is this??
        #grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        grip_rpy = self.get_ee_rpy()
        #print grip_rpy
        grip_velp = np.array([grip_rpy.y, grip_rpy.y])
        robot_qpos, robot_qvel = utils.robot_get_obs(self.joint_states_data)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] #* dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos_array.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos_array, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        """
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.
        """
        pass
        
    def _render_callback(self):
        """
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()
        """
        pass

    def _reset_sim(self):
        #self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        
        #self.set_trajectory_joints(self.initial_qpos)
        #self.execute_trajectory()
        #self.gazebo.pauseSim()
        #print "Paused Sim"
        self.gazebo.resetWorld()
        #print "Reset Sim"
        #self.gazebo.unpauseSim()
        #print "Unpaused Sim"
        #self.controllers_object.reset_fetch_joint_controllers()
        #print "Reset Controllers"
        self.set_trajectory_joints(self.init_pos)
        #print "Set trajectory"
        #self.execute_trajectory()
        #print "Execute trajectory"
        
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.set_trajectory_joints(initial_qpos)
            #self.execute_trajectory()
        #utils.reset_mocap_welds(self.sim)
        #self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([0.498, 0.005, 0.431 + self.gripper_extra_height])# + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        #self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        #self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        action = np.concatenate([gripper_target, gripper_rotation])
        self.set_trajectory_ee(action)
        #self.execute_trajectory()
        #for _ in range(10):
            #self.sim.step()
            #self.step()

        # Extract information for sampling goals.
        #self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        gripper_pos = self.get_ee_pose()
        gripper_pose_array = np.array([gripper_pos.pose.position.x, gripper_pos.pose.position.y, gripper_pos.pose.position.z])
        self.initial_gripper_xpos = gripper_pose_array.copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
    
    
    def set_trajectory_ee(self, action):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        
        #print "Action at ee function"
        #print action
        """
        self.pose_target = geometry_msgs.msg.Pose()
        self.pose_target.orientation.w = 1.0
        self.pose_target.position.x = action[0]
        self.pose_target.position.y = action[1]
        self.pose_target.position.z = action[2]
        self.group.set_pose_target(self.pose_target)
        """
        ee_target = EeTrajRequest()
        ee_target.pose.orientation.w = 1.0
        ee_target.pose.position.x = action[0]
        ee_target.pose.position.y = action[1]
        ee_target.pose.position.z = action[2]
        result = self.ee_traj_client(ee_target)
        
        #return action_msg
        return True
        
    def set_trajectory_joints(self, initial_qpos):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        """
        self.group_variable_values = self.group.get_current_joint_values()

        self.group_variable_values[0] = initial_qpos["joint0"]
        self.group_variable_values[1] = initial_qpos["joint1"]
        self.group_variable_values[2] = initial_qpos["joint2"]
        self.group_variable_values[3] = initial_qpos["joint3"]
        self.group_variable_values[4] = initial_qpos["joint4"]
        self.group_variable_values[5] = initial_qpos["joint5"]
        self.group_variable_values[6] = initial_qpos["joint6"]
        self.group.set_joint_value_target(self.group_variable_values)
        """
        joint_point = JointTrajRequest()
        
        joint_point.point.positions = [None] * 7
        joint_point.point.positions[0] = initial_qpos["joint0"]
        joint_point.point.positions[1] = initial_qpos["joint1"]
        joint_point.point.positions[2] = initial_qpos["joint2"]
        joint_point.point.positions[3] = initial_qpos["joint3"]
        joint_point.point.positions[4] = initial_qpos["joint4"]
        joint_point.point.positions[5] = initial_qpos["joint5"]
        joint_point.point.positions[6] = initial_qpos["joint6"]
        
        #print "Joint Point"
        #print joint_point
        
        result = self.joint_traj_client(joint_point)
        
        return True
    
    def execute_trajectory(self):
        
        self.plan = self.group.plan()
        self.group.go(wait=True)
        
        return True
        
    def get_ee_pose(self):
        
        #gripper_pose = self.group.get_current_pose()
        gripper_pose_req = EePoseRequest()
        gripper_pose = self.ee_pose_client(gripper_pose_req)
        
        return gripper_pose
        
    def get_ee_rpy(self):
        
        #gripper_rpy = self.group.get_current_rpy()
        gripper_rpy_req = EeRpyRequest()
        gripper_rpy = self.ee_rpy_client(gripper_rpy_req)
        
        return gripper_rpy
        
    def joint_states_callback(self, data):
        self.joint_states_data = data
        #print "Joint States Data:"
        #print self.joint_states_data.position[12]
        
        
"""       
if __name__ == "__main__":
    rospy.init_node('fetch_gym')
    fetch_env = FetchEnv()
    action = np.array([0.66, 0.0, 1.0, 1.0])
    print "Action test:"
    print action
    fetch_env._reset_sim()
    fetch_env.set_trajectory_ee(action)
    fetch_env.execute_trajectory()
    action = np.array([0.76, 0.0, 1.0, 1.0])
    fetch_env.set_trajectory_ee(action)
    fetch_env.execute_trajectory()
    action = np.array([0.54, 0.0, 1.1, 1.0])
    fetch_env.set_trajectory_ee(action)
    fetch_env.execute_trajectory()
    fetch_env._reset_sim()
    action = np.array([0.5, 0.2, 1.1, 1.0])
    fetch_env.set_trajectory_ee(action)
    fetch_env.execute_trajectory()
    action = np.array([0.6, 0.2, 1.1, 1.0])
    fetch_env.set_trajectory_ee(action)
    fetch_env.execute_trajectory()
    action = np.array([0.7, 0.2, 1.1, 1.0])
    fetch_env.set_trajectory_ee(action)
    fetch_env.execute_trajectory()
    fetch_env._reset_sim()
"""