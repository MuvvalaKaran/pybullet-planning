import random
import pybullet as p
import numpy as np
import time

from pybullet_tools.ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics

from pybullet_tools.pr2_utils import DRAKE_PR2_URDF, PR2_GROUPS, SIDE_HOLDING_LEFT_ARM, REST_LEFT_ARM, rightarm_from_leftarm, \
open_gripper, get_top_grasps, get_disabled_collisions, get_gripper_link, get_arm_joints, arm_conf, open_arm, get_other_arm, set_arm_conf, get_carry_conf

from pybullet_tools.utils import connect, add_data_path, load_pybullet, set_quat, quat_from_euler, Euler, \
HideOutput, load_model, dump_body, wait_if_gui, set_point, Point, stable_z, disconnect, INFO_FROM_BODY, add_line, base_aligned_z, get_aabb, \
joint_from_name, set_joint_positions, joints_from_names, get_pose, link_from_name, get_relative_pose, unit_pose, plan_joint_motion, wait_for_duration, \
set_pose, is_placement, pairwise_collision, sub_inverse_kinematics, get_joint_positions, plan_direct_joint_motion, BodySaver

from pybullet_tools.pr2_primitives import get_grasp_gen, GripperCommand, Attach, create_trajectory, Commands, State, Pose

from pybullet_tools.utils import LockRenderer

from pybullet_tools.utils import RED, GREEN, BLACK, PI, TABLE_URDF, SMALL_BLOCK_URDF, multiply, invert

TABLE_WIDTH = 1
TABLE_LENGTH = 1.5
TABLE_HEIGHT = 0.63

EPSILON = 0.1

DEBUG: bool = True
SELF_COLLISIONS = True


class MockProblem(object):
    def __init__(self, robot, fixed=[], grasp_types=[]):
        self.robot = robot
        self.fixed = fixed
        self.grasp_types = grasp_types
        self.gripper = None


def set_pr2_base_pose(pr2_robot, base_pose) -> None:
    """
    Set the base position to base_pose
    """
    base_joints = [joint_from_name(pr2_robot, name) for name in PR2_GROUPS['base']]
    set_joint_positions(pr2_robot, base_joints, base_pose)


def set_pr2_init_pos(pr2_robot) -> None:
    """
    Initialize the arm to their desired init locations. 
    """
    left_joints = joints_from_names(pr2_robot, PR2_GROUPS['left_arm'])
    right_joints = joints_from_names(pr2_robot, PR2_GROUPS['right_arm'])
    torso_joints = joints_from_names(pr2_robot, PR2_GROUPS['torso'])

    arm = 'left'
    other_arm = get_other_arm(arm)
    grasp_type = 'top'

    # set_joint_positions(pr2_robot, left_joints, SIDE_HOLDING_LEFT_ARM)
    # set_joint_positions(pr2_robot, right_joints, rightarm_from_leftarm(REST_LEFT_ARM))
    set_arm_conf(pr2_robot, arm, get_carry_conf(arm, grasp_type))
    set_arm_conf(pr2_robot, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    set_joint_positions(pr2_robot, torso_joints, [0.2])



def draw_axes() -> None:
    add_line(Point(x=-TABLE_WIDTH/2, z=TABLE_HEIGHT), Point(x=+TABLE_WIDTH/2, z=TABLE_HEIGHT), color=RED)
    add_line(Point(y=-TABLE_LENGTH/2, z=TABLE_HEIGHT), Point(y=+TABLE_LENGTH/2, z=TABLE_HEIGHT), color=BLACK)


def test_arm_motion(pr2, left_joints, arm_goal):
    disabled_collisions = get_disabled_collisions(pr2)
    wait_if_gui('Plan Arm?')
    with LockRenderer(lock=False):
        arm_path = plan_joint_motion(pr2, left_joints, arm_goal, disabled_collisions=disabled_collisions)
    if arm_path is None:
        print('Unable to find an arm path')
        return
    print(len(arm_path))
    for q in arm_path:
        set_joint_positions(pr2, left_joints, q)
        #wait_if_gui('Continue?')
        wait_for_duration(0.01)


def grasp_fn(problem, arm, obj, pose, grasp, base_conf = None, collisions=False, teleport=False, custom_limits={}):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
    gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
    #approach_pose = multiply(grasp.approach, gripper_pose)
    approach_pose = multiply(pose.value, invert(grasp.approach))
    arm_link = get_gripper_link(robot, arm)
    arm_joints = get_arm_joints(robot, arm)
    
    default_conf = arm_conf(arm, grasp.carry)
    #sample_fn = get_sample_fn(robot, arm_joints)
    pose.assign()
    # base_conf.assign()
    open_arm(robot, arm)
    set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
    grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
                                        #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
    if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
        return None
    #approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
    #                                       upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
    approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
    if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
        #print('Approach IK failure', approach_conf)
        #wait_if_gui()
        return None
    wait_if_gui("1:Did object move?")
    approach_conf = get_joint_positions(robot, arm_joints)
    attachment = grasp.get_attachment(problem.robot, arm)
    attachments = {attachment.child: attachment}
    wait_if_gui("2:Did object move?")
    if teleport:
        path = [default_conf, approach_conf, grasp_conf]
    else:
        resolutions = 0.05**np.ones(len(arm_joints))
        grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                custom_limits=custom_limits, resolutions=resolutions/2.)
        if grasp_path is None:
            print('Grasp path failure')
            return None
        wait_if_gui("3:Did object move?")
        set_joint_positions(robot, arm_joints, default_conf)
        approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                            obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                            custom_limits=custom_limits, resolutions=resolutions,
                                            restarts=2, max_iterations=25, smooth=25)
        wait_if_gui("4:Did object move?")
        if approach_path is None:
            print('Approach path failure')
            return None
        path = approach_path + grasp_path
    mt = create_trajectory(robot, arm_joints, path)
    wait_if_gui("5:Did object move?")
    cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
    
    return mt


def init_sim():
    connect(use_gui=True)
    add_data_path()

    p.loadURDF("plane.urdf")
    # table_path = "models/table_collision/table.urdf"
    # wait_if_gui('Load Table?')
    # length = 1.5; widht = 1 and height = 0.5
    table = load_pybullet(TABLE_URDF, fixed_base=True)
    set_quat(table, quat_from_euler(Euler(yaw=PI/2)))

    # draw line
    if DEBUG:
        draw_axes()
    
    wait_if_gui('Load Block?')
    # block_path = "models/drake/objects/block_for_pick_and_place.urdf"
    block = load_pybullet(SMALL_BLOCK_URDF, fixed_base=False)
    set_point(block, Point(x=0, y=0.25, z=stable_z(block, table)))
    
    wait_if_gui('Load Robot?')

    pr2_urdf = DRAKE_PR2_URDF
    with HideOutput():
        pr2 = load_model(pr2_urdf, fixed_base=True)
    dump_body(pr2)

    wait_if_gui("Base Align Z?")
    z = base_aligned_z(pr2)
    set_pr2_base_pose(pr2, (-(TABLE_WIDTH/2 + EPSILON), 0, z))

    wait_if_gui("Initialize Arms?")
    set_pr2_init_pos(pr2_robot=pr2)

    wait_if_gui("Set to Top Grasp position?")
    ######################################################################
    # Manual get graps pose and move to that location. - need to fi robot orientation
    # block_pose = get_pose(block)
    # # print(block_pose)
    
    # open_gripper(pr2, 'left')
    
    # # tool_link = link_from_name(pr2, 'tool_link')
    # tool_link = get_gripper_link(pr2, 'left')
    # base_from_tool = get_relative_pose(pr2, tool_link)
    # #draw_pose(unit_pose(), parent=robot, parent_link=tool_link)

    # y_grasp, x_grasp = get_top_grasps(block, tool_pose=unit_pose(), grasp_length=0.03, under=False)
    # grasp = y_grasp # fingers won't collide
    # gripper_pose = multiply(block_pose, invert(grasp))
    # g = gripper_pose

    # set_pose(pr2, y_grasp)
    ######################################################################
    # use utls function
    
    problem = MockProblem(pr2, fixed=[table], grasp_types=['top'])
    # wait_if_gui("-4: Did block move?")
    grasp_gen_fn = get_grasp_gen(problem, collisions=True)
    # wait_if_gui("-3: Did block move?")
    grasps = list(grasp_gen_fn(block))
    # print('Grasps:', len(grasps))
    (g,) = random.choice(grasps)
    # wait_if_gui("-2: Did block move?")
    blockpose = Pose(block)
    

    cmd = grasp_fn(problem=problem, arm='left', obj=block, pose=blockpose, grasp=g, teleport=True)
    # for pa in  cmd.commands[0].path:
    if cmd is None:
        wait_if_gui('Failed to plan. Disconnect?')
        disconnect()
    for pa in cmd.path:
        pa.assign()
        time.sleep(0.05)
    ######################################################################

    # close_gripper = GripperCommand(pr2, 'left', g.carry, False)
    # close_gripper.apply
    # attach = Attach(problem.robot, 'left', 'top', block)

    wait_if_gui('Disconnect?')
    disconnect()


if __name__ == "__main__":
    init_sim()