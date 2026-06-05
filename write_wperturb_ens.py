import argparse
import os

# Same SPG style as before: this writes one command per (omega, domega, seed).
# Each command internally performs many ensemble realizations, so file count stays low.

FULL_OMEGA_LIST = [0, 0.3, 0.6, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
PILOT_OMEGA_LIST = [0, 1.5, 3.0]
DEFAULT_DOMEGA_LIST = [0, 0.1, 0.2, 0.3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write SPG-style jobs for run_wperturb_ens.py."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use a small omega grid for G convergence checks.",
    )
    parser.add_argument(
        "--omega",
        nargs="*",
        type=float,
        help="Override omega values. Defaults to full grid, or pilot grid with --pilot.",
    )
    parser.add_argument(
        "--domega",
        nargs="*",
        type=float,
        default=DEFAULT_DOMEGA_LIST,
        help="dOmega values. Include 0 for fixed-omega stationary runs.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed, inclusive.",
    )
    parser.add_argument(
        "--seed-stop",
        type=int,
        default=10,
        help="Last seed, exclusive. Use 30 or 50 for higher-statistics G pilots.",
    )
    parser.add_argument(
        "--python-bin",
        default="/pds/pds21/yunsik/miniconda3/bin/python",
        help="Python executable written into each job line.",
    )
    parser.add_argument(
        "--run-script",
        default="run_wperturb_ens.py",
        help="Simulation script written into each job line.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/wperturb/ensemble2/",
        help="Output directory for npz files.",
    )
    parser.add_argument(
        "--jobs-file",
        default="jobs.txt",
        help="Job file to append to.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the number of jobs that would be added without writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the jobs file instead of appending to it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    omega_list = args.omega
    if omega_list is None:
        omega_list = PILOT_OMEGA_LIST if args.pilot else FULL_OMEGA_LIST

    seed_list = range(args.seed_start, args.seed_stop)
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    lines = []
    for omega in omega_list:
        for domega in args.domega:
            for seed in seed_list:
                filename = f"{omega:g}-{domega:g}seed{seed}.npz"
                state = os.path.join(os.getcwd(), data_dir, filename)
                if os.path.exists(state):
                    continue
                lines.append(
                    f"{args.python_bin} {args.run_script} "
                    f"{omega:f} {domega:f} {seed:d} {state}  \n"
                )

    if args.dry_run:
        print(f"would add {len(lines)} jobs to {args.jobs_file}")
        return

    mode = "w" if args.overwrite else "a"
    with open(args.jobs_file, mode) as file:
        file.writelines(lines)

    action = "wrote" if args.overwrite else "added"
    print(f"{action} {len(lines)} jobs to {args.jobs_file}")


if __name__ == "__main__":
    main()
