import pybullet as p

from pybullet_tools.pr2_utils import DRAKE_PR2_URDF, PR2_GROUPS, SIDE_HOLDING_LEFT_ARM, REST_LEFT_ARM, rightarm_from_leftarm

from pybullet_tools.utils import connect, add_data_path, load_pybullet, set_quat, quat_from_euler, Euler, PI, \
HideOutput, load_model, dump_body, wait_if_gui, set_point, Point, stable_z, disconnect, INFO_FROM_BODY, add_line, base_aligned_z, get_aabb, \
joint_from_name, set_joint_positions, joints_from_names

from pybullet_tools.utils import RED, GREEN, BLACK

TABLE_WIDTH = 1
TABLE_LENGTH = 1.5
TABLE_HEIGHT = 0.63

EPSILON = 0.3


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

    set_joint_positions(pr2_robot, left_joints, SIDE_HOLDING_LEFT_ARM)
    set_joint_positions(pr2_robot, right_joints, rightarm_from_leftarm(REST_LEFT_ARM))
    set_joint_positions(pr2_robot, torso_joints, [0.2])


def init_sim():
    connect(use_gui=True)
    add_data_path()

    plane = p.loadURDF("plane.urdf")
    table_path = "models/table_collision/table.urdf"
    wait_if_gui('Load Table?')
    # length = 1.5; widht = 1 and height = 0.5
    table = load_pybullet(table_path, fixed_base=True)
    # print(INFO_FROM_BODY)
    set_quat(table, quat_from_euler(Euler(yaw=PI/2)))

    # draw line 
    add_line(Point(x=-TABLE_WIDTH/2, z=TABLE_HEIGHT),
                 Point(x=+TABLE_WIDTH/2, z=TABLE_HEIGHT), color=RED)
    add_line(Point(y=-TABLE_LENGTH/2, z=TABLE_HEIGHT),
                 Point(y=+TABLE_LENGTH/2, z=TABLE_HEIGHT), color=BLACK)
    # table/table.urdf, table_square/table_square.urdf, cube.urdf, block.urdf, door.urdf
    # obstacles = [plane, table]
    wait_if_gui('Load Block?')
    block_path = "models/drake/objects/block_for_pick_and_place.urdf"
    block = load_pybullet(block_path, fixed_base=False)
    set_point(block, Point(z=stable_z(block, table)))
    
    wait_if_gui('Load Robot?')

    pr2_urdf = DRAKE_PR2_URDF
    with HideOutput():
        pr2 = load_model(pr2_urdf, fixed_base=True) # TODO: suppress warnings?
    dump_body(pr2)

    wait_if_gui("Base Align Z?")
    z = base_aligned_z(pr2)
    # print(z)
    # set_point(pr2, Point(z=z))
    # print(get_aabb(pr2))
    set_pr2_base_pose(pr2, (-(TABLE_WIDTH/2 + EPSILON), 0, z))

    wait_if_gui("Initialize Arms?")
    set_pr2_init_pos(pr2_robot=pr2)

    wait_if_gui('Disconnect?')
    disconnect()