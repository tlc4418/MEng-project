import os
import random
from datetime import datetime

import numpy as np
from animalai.envs.actions import AAIActions
from animalai.envs.environment import AnimalAIEnvironment
from arena_creator import create_rand_arena
from slot_attention_and_alignnet.src.dataloaders import DataController
from gym_unity.envs import UnityToGymWrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

training = False  # Set to false to watch the agent.
targetFrameRate = -1 if training else 60
captureFrameRate = 0 if training else 60
inference = not training

ACTIONS = AAIActions()
RAY_COUNT = 9
RAY_OBJECT_NAMES = [
    "arena",
    "immovable",
    "movable",
    "goodGoal",
    "goodGoalMulti",
    "badGoal",
]
SEQUENCE_LEN = 7  # What sequence length to collect AAI images at a time

basic_food_ahead = [
    "01-01-01",
    "01-02-01",
    "01-03-01",
    "01-01-02",
    "01-02-02",
    "01-03-02",
    "01-01-03",
    "01-02-03",
    "01-03-03",
]
basic_food_navigation = [
    "01-04-01",
    "01-05-01",
    "01-04-02",
    "01-05-02",
    "01-04-03",
    "01-05-03",
]
basic_food_variations = [
    "01-06-01",
    "01-07-01",
    "01-08-01",
    "01-09-01",
    "01-10-01",
    "01-11-01",
    "01-06-02",
    "01-07-02",
    "01-08-02",
    "01-09-02",
    "01-10-02",
    "01-11-02",
    "01-06-03",
    "01-07-03",
    "01-08-03",
    "01-09-03",
    "01-10-03",
    "01-11-03",
]
basic_exploration = [
    "01-12-01",
    "01-13-01",
    "01-14-01",
    "01-15-01",
    "01-16-01",
    "01-17-01",
    "01-12-02",
    "01-13-02",
    "01-14-02",
    "01-15-02",
    "01-16-02",
    "01-17-02",
    "01-12-03",
    "01-13-03",
    "01-14-03",
    "01-15-03",
    "01-16-03",
    "01-17-03",
]
basic_multiple_food = [
    "01-22-01",
    "01-21-01",
    "01-22-01",
    "01-20-02",
    "01-21-02",
    "01-22-02",
    "01-20-03",
    "01-21-03",
    "01-22-03",
]
basic_food_obstacles = [
    "03-01-01",
    "03-01-02",
    "03-01-03",
    "03-02-01",
    "03-02-02",
    "03-02-03",
    "03-03-01",
    "03-03-02",
    "03-03-03",
]
cylinder = []  # 03-13 to 03-15
spatial_elimination = []  # walls/cylinder/goal 05-06 to 05-10

# Total = 69 arenas (for initial dataset using competition arenas)

ARENA_MAP = {
    "Basic Food Ahead": basic_food_ahead,
    "Basic Multiple Food": basic_multiple_food,
    "Basic Food and Obstacles": basic_food_obstacles,
    "Basic Navigate to Food": basic_food_navigation,
    "Basic Food Variations": basic_food_variations,
    "Basic Exploration": basic_exploration,
}


def strfdelta(tdelta):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    if d["days"] > 0:
        return "{days} days {hours:02d}:{minutes:02d}:{seconds:02d}".format(**d)
    if d["hours"] > 0:
        return "{hours:02d}:{minutes:02d}:{seconds:02d}".format(**d)
    return "{minutes:02d}:{seconds:02d}".format(**d)


# Remove the agent from the given competition arena configuration.
# This will randomize the agent position at each env reset.
def randomize_agent_position(arena_num):
    with open(f"configs/competition/{arena_num}.yaml") as f:
        new_name = f"randomized_arena_configs/{arena_num + '-random'}.yaml"
        with open(new_name, "w+") as w:
            check = 0
            for i in f:
                if check == 0:
                    print(i, file=w, end="")
                else:
                    check -= 1
                if "name: Agent" in i:
                    check += 2
            f.close()
            return new_name


def count_objects(pair):
    return 1 if int(float(pair[1])) >= 1 else 0


def get_images_from_env(env, n_images, can_skip, writer):
    unexpected_collect_skip = 0
    min_objects = 0 # Minimum number of objects that should be visible
    image_count = 0
    skip = 0
    timeout = 400
    threshold_distance = 0.02

    while image_count < n_images:

        # Initialize the agent in a random position
        obs = env.reset()

        # Check what objects the raycasts detect
        raycast_arr_count = {}
        distance = 1
        for r in range(2 * RAY_COUNT + 1):
            start = r * 8
            for c in range(6):
                raycast_arr_count[c] = raycast_arr_count.get(c, 0) + obs[1][start + c]
                if obs[1][start] == 0:
                    distance = min(obs[1][start + 7], distance)

        if timeout < 0:
            print("Increasing threshold")
            # min_objects -= 1
            threshold_distance += 0.005
            timeout = 400
        elif timeout > 1450:
            print("Reducing threshold")
            threshold_distance -= 0.005
            threshold_distance = max(threshold_distance, 0.016)
            timeout = 400

        if distance > threshold_distance and min_objects >= 1:
            skip += 1
            timeout -= 1
            continue

        # Map object indexes to names and count total (excluding arena hits)
        objects = np.array(
            [
                (RAY_OBJECT_NAMES[i], raycast_arr_count[i])
                for i in range(1, len(RAY_OBJECT_NAMES))
                if raycast_arr_count[i] > 0
            ]
        )
        total = sum(map(lambda x: count_objects(x), objects))

        # If the agent can see an object, add the image
        if (total >= min_objects and distance < threshold_distance) or min_objects <= 0:
            image_collector = []
            image_collector.append(obs[0])
            action = random.randint(1, 8)

            done = False
            for _ in range(SEQUENCE_LEN - 1):
                try:
                    obs, _, done, _ = env.step(
                        action
                    )  # [noop, right, left, forward, forward_right, forward_left, back, back_right, back_left]
                    image_collector.append(obs[0])
                except:
                    done = True
                    break

            # If the sequence terminated successfully (error could be reaching goal for example)
            if not done:
                if image_count % 100 == 0:
                    print(
                        f"Collected images: {image_count}/{n_images}, skipped: {skip}"
                    )
                writer.write_batched_data(image_collector, SEQUENCE_LEN)
                image_count += 1

            # Check whether any objects are on the image
            if total > 0:
                min_objects = 0  # 2
                timeout += 100
            else:
                unexpected_collect_skip += 1

        else:
            skip += 1
            timeout -= 1

            # Used to collect skipped images if we need them
            if can_skip > 0:
                skipped_collector = []
                skipped_collector.append(obs[0])
                action = random.randint(1, 8)

                done = False
                for _ in range(SEQUENCE_LEN - 1):
                    try:
                        obs, _, done, _ = env.step(
                            action
                        )  # [noop, right, left, forward, forward_right, forward_left, back, back_right, back_left]
                        skipped_collector.append(obs[0])
                    except:
                        done = True
                        break

                if not done:
                    writer.write_batched_data(skipped_collector, SEQUENCE_LEN)
                    can_skip -= 1

    print(
        f"Total skipped images: {skip}. Collected skip images (no objects): {unexpected_collect_skip}"
    )
    env.close()
    return image_count, can_skip


def get_n_images_from_arena_list(arena_list, writer, n_skip):
    start = datetime.now()
    image_count = 0
    can_skip = n_skip

    # To randomly generate random arenas using the custom arena creator
    arenas = []
    for i in range(2):
        create_rand_arena(f"random_arena{i}", empty=False)
        arenas.append(f"aai_dataset/new_arenas/random_arena{i}.yaml")

    ## To use a competition arenas with random agent positions:
    # for arena_num, n_images in arena_list:
    #    randomized_arena_config = randomize_agent_position(arena_num)

    for i, arena in enumerate(arenas):
        # Create the appropriate environment for the given arena
        aai_env = AnimalAIEnvironment(
            file_name="aai_environment/env/AnimalAI",
            arenas_configurations=arena,
            play=False,
            inference=inference,
            useCamera=True,
            resolution=128,
            useRayCasts=True,
            raysPerSide=RAY_COUNT,
            rayMaxDegrees=45,
            base_port=5000 + random.randint(0, 1000),
        )
        env = UnityToGymWrapper(
            aai_env, uint8_visual=True, allow_multiple_obs=True, flatten_branched=True
        )
        print(
            f"Arena {i + 1}: {i + 1} / {len(arenas)}. Collected images: {image_count} normal, {n_skip - can_skip} skip. Elapsed: {strfdelta(datetime.now() - start)}"
        )

        # Gather N random relevant images from this arena
        image_count, can_skip = get_images_from_env(env, 2, can_skip, writer)


EXPERIMENTS = [
    ("Basic Food Ahead", 50),
    ("Basic Navigate to Food", 150),
    ("Basic Food Variations", 300),
    ("Basic Exploration", 150),
    ("Basic Multiple Food", 550),
    ("Basic Food and Obstacles", 900),
]

# Get a list of all arena numbers from the list of given arena types
arena_list = [
    (arena, count) for exp, count in EXPERIMENTS for arena in ARENA_MAP.get(exp)
]

# Number of skipped images to collect
n_skip = 0

# Run the generator
writer = DataController(
    "/media/home/thomas/data", file_name="aai_test_close_sequence", overwrite=True
)
writer.clear_data()
get_n_images_from_arena_list(arena_list, writer, n_skip)
writer.finish()
