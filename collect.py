import argparse
from pipeline import DataPipeline, DATA_DIR
from policies import ReachAndPickPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps",    type=int, default=300)
    parser.add_argument("--robot",    default="franka_panda")
    args = parser.parse_args()

    pipeline = DataPipeline(robot=args.robot, out_dir=DATA_DIR)
    policy = ReachAndPickPolicy(env=pipeline.env)

    print(f"Episodes: {args.episodes} | Steps: {args.steps}")
    pipeline.collect(policy, n_episodes=args.episodes, max_steps=args.steps)
    print("Done.")
