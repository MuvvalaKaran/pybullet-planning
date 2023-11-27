'''
This file contains a simple pick and place simulation using the Kuka robot
'''
import time

from pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_if_gui, disconnect, DRAKE_IIWA_URDF, wait_if_gui, update_state, disable_real_time, HideOutput


def plan(robot, block, fixed, teleport):
    grasp_gen = get_grasp_gen(robot, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    free_motion_fn = get_free_motion_gen(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen(robot, fixed=fixed, teleport=teleport)

    pose0 = BodyPose(block)
    conf0 = BodyConf(robot)
    saved_world = WorldSaver()
    for grasp, in grasp_gen(block):
        saved_world.restore()
        # compute the approach to the block
        result1 = ik_fn(block, pose0, grasp)
        if result1 is None:
            continue

        # compute the path for approach
        conf1, path2 = result1
        pose0.assign()
        
        # compute the path to the grasp position
        result2 = free_motion_fn(conf0, conf1)
        if result2 is None:
            continue
        
        path1, = result2
        # compute a path along with the obj in the hand to the org conf0 position
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        if result3 is None:
            continue
        path3, = result3
        
        # 
        result4 = holding_motion_fn(conf0, conf1, block, grasp)
        if result4 is None:
            continue
        path4, = result4
        return Command(path1.body_paths +
                          path2.body_paths +
                          path3.body_paths + path4.body_paths + path2.body_paths)
        
    return None


def init_sim(use_gui: bool = False) -> None:
    """
    The main function that initializes a simulation env, spawns a robot and objects. The key steps are control, execute and step. 
    """
    connect(use_gui=use_gui)
    disable_real_time()
    draw_global_system()

    # spawn the robot
    with HideOutput():
        robot = load_model(DRAKE_IIWA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        floor = load_model('models/short_floor.urdf')
    

    # spawn a single block
    block = load_model(BLOCK_URDF, fixed_base=False)
    set_pose(block, Pose(Point(x=0.5, z=stable_z(block, floor))))
    # set_default_camera(distance=2)
    dump_world()

    saved_world = WorldSaver()
    command = plan(robot, block, fixed=[floor], teleport=False)
    if command is None:
        print('Unable to find a plan!')
        return

    saved_world.restore()
    update_state()
    # command.step()
    command.refine(num_steps=10).execute(time_step=0.005)
    time.sleep(5.0)
    disconnect()




