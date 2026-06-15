import argparse
from pipeline import DataPipeline, DATA_DIR
from policies import SinusoidalPolicy, WaypointPolicy, ReachAndPickPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",   default="reach", choices=["sinusoidal", "waypoint", "reach"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps",    type=int, default=300)
    parser.add_argument("--robot",    default="franka_panda")
    args = parser.parse_args()

    pipeline = DataPipeline(robot=args.robot, out_dir=DATA_DIR)
    nu       = pipeline.env.model.nu

    if args.policy == "sinusoidal":
        policy = SinusoidalPolicy(nu=nu, max_steps=args.steps)
    elif args.policy == "waypoint":
        policy = WaypointPolicy(steps_per_segment=args.steps // 3)
    else:
        policy = ReachAndPickPolicy(env=pipeline.env)

    print(f"Policy: {args.policy} | Episodes: {args.episodes} | Steps: {args.steps}")
    pipeline.collect(policy, n_episodes=args.episodes, max_steps=args.steps)
    print("Done.")
