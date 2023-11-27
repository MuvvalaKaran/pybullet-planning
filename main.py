from examples import test_json, test_kuka_pick, test_pr2_motion
from kuka_pick_place import simple_pick_place, pr2_pick_place


def main():
    # test_json.main(use_gui=True)
    # test_kuka_pick.main(use_gui=True)
    # simple_pick_place.init_sim(use_gui=False)
    # test_pr2_motion.main()
    pr2_pick_place.init_sim()


if __name__ == '__main__':
    main()