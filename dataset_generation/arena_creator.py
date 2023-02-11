import random as rand
import numpy as np

# Some example sizes commonly used in competition configurations
CYLINDER_SIZES = np.array(
    [(10, 10, 10), (20, 3, 3), (5, 5, 10), (4, 4, 10), (4, 4, 20)]
)
WALL_SIZES = np.array(
    [
        (10, 3, 16),
        (5, 3, 5),
        (15, 10, 1),
        (1, 5, 25),
        (2, 5, 1),
        (1.5, 2, 1),
        (20, 5, 0.5),
        (13, 10, 1),
        (15, 2, 2),
        (5, 2, 10),
    ]
)
GOAL_SIZES = np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3)])
RAMP_SIZES = np.array(
    [(4, 0.5, 4), (5, 3, 5), (5, 2, 5), (4, 1, 5), (7, 2, 4), (5, 1, 2)]
)
CARDBOX_SIZES = np.array([(1, 1, 5), (1, 1, 1)])

NEWLINE = "\n"


def insert_vector(x, y, z):
    return f"      - !Vector3 {{x: {x}, y: {y}, z: {z}}}"


def get_walls(num):
    sizes = WALL_SIZES[np.random.choice(WALL_SIZES.shape[0], num, replace=True)]
    return f"""
    - !Item
      name: Wall
      colors:
{NEWLINE.join(['      - !RGB {r: 153, g: 153, b: 153}' for _ in range(num)])}
      sizes:
{NEWLINE.join([insert_vector(*s) for s in sizes])}"""


def get_cylinders(num):
    sizes = CYLINDER_SIZES[np.random.choice(CYLINDER_SIZES.shape[0], num, replace=True)]
    return f"""
    - !Item
      name: CylinderTunnel
      colors:
{NEWLINE.join(['      - !RGB {r: 153, g: 153, b: 153}' for _ in range(num)])}
      sizes:
{NEWLINE.join([insert_vector(*s) for s in sizes])}"""


def get_ramps(num):
    sizes = RAMP_SIZES[np.random.choice(RAMP_SIZES.shape[0], num, replace=True)]
    return f"""
    - !Item
      name: Ramp
      colors:
{NEWLINE.join(['      - !RGB {r: 255, g: 0, b: 255}' for _ in range(num)])}
      sizes:
{NEWLINE.join([insert_vector(*s) for s in sizes])}"""


def get_boxes1(num):
    sizes = CARDBOX_SIZES[np.random.choice(CARDBOX_SIZES.shape[0], num, replace=True)]
    return f"""
    - !Item
      name: Cardbox1
      sizes:
{NEWLINE.join([insert_vector(*s) for s in sizes])}"""


def get_boxes2(num):
    sizes = CARDBOX_SIZES[np.random.choice(CARDBOX_SIZES.shape[0], num, replace=True)]
    return f"""
    - !Item
      name: Cardbox2
      sizes:
{NEWLINE.join([insert_vector(*s) for s in sizes])}"""


def get_multi_goals(num):
    sizes = GOAL_SIZES[
        np.random.choice(GOAL_SIZES.shape[0], num, replace=True, p=[0.9, 0.08, 0.02])
    ]
    return f"""
    - !Item
      name: GoodGoalMulti
      sizes:
{NEWLINE.join([insert_vector(-1, -1, -1) for _ in range(num)])}
      sizes:
{NEWLINE.join([insert_vector(*s) for s in sizes])}"""


def get_goals(num):
    sizes = GOAL_SIZES[
        np.random.choice(GOAL_SIZES.shape[0], num, replace=True, p=[0.9, 0.08, 0.02])
    ]
    return f"""
    - !Item
      name: GoodGoal
      sizes:
{NEWLINE.join([insert_vector(-1, -1, -1) for _ in range(num)])}
      sizes:
{NEWLINE.join([insert_vector(*s) for s in sizes])}"""


def get_agent():
    return f"""
    - !Item
      name: Agent"""


def create_rand_arena(arena_name: str, empty=False):
    with open(f"aai_dataset/new_arenas/{arena_name}.yaml", "w") as w:
        start = """!ArenaConfig
arenas:
  0: !Arena
    pass_mark: 0
    t: 250"""
        if not empty:
            start += """
    items: """
        print(start, file=w, end="")

        generate_objects = [
            (get_goals, rand.randint(1, 3)),
            (get_multi_goals, rand.randint(1, 4)),
            (get_ramps, rand.randint(0, 0)),
            (get_boxes1, rand.randint(0, 0)),
            (get_boxes2, rand.randint(0, 0)),
            (get_walls, rand.randint(0, 0)),
            (get_cylinders, rand.randint(0, 0)),
        ]

        if not empty:
            for generator, num in generate_objects:
                if num > 0:
                    print(generator(num), file=w, end="")

        w.close()
