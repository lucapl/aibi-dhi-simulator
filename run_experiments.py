import argparse
from environment.profiled_fogg_model import PROFILES
from pathlib import Path
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments for each profile in the jupyter notebook.')
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--profiles",
                        nargs='*',
                        default=list(PROFILES.keys()),
                        help="List of profiles to run experiments for. Default is all profiles.")
    #parser.add_argument("--behavior_threshold", type=int, default=20)
    parser.add_argument("--results_directory", type=str, default='results')
    parser.add_argument("--time_preference_update_step", type=int, default=9999999999999999)
    parser.add_argument("--habituation", action='store_true')
    parser.add_argument("--notebook_path", type=str, default='experiments.ipynb')
    args = parser.parse_args()

    for profile in args.profiles:
        res_dir = (Path(args.results_directory)/profile).as_posix()
        params = dict(runs=args.runs,
                            profile=profile,
                            #behavior_threshold=args.behavior_threshold,
                            time_preference_update_step=args.time_preference_update_step, 
                            habituation=args.habituation,
                            results_directory=res_dir)
        subprocess.run(
            ['papermill', args.notebook_path, f'experiment_profile_{profile}.ipynb'] +
            [f'-p {key} {str(value)}' for (key, value) in params.items()]
            )
