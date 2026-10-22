"""Compatibility entry: analytical IEKF only. Use main_ukf.py for UKF."""
from iekf.filter import IEKF
from rgbd_common.entry import run


def main(argv=None):
    return run(IEKF,argv)


if __name__ == "__main__":
    raise SystemExit(main())
