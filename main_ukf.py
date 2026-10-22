"""Independent Lie UKF entry; never imports the IEKF implementation."""
from ukf.filter import UKF
from rgbd_common.entry import run

if __name__=='__main__':raise SystemExit(run(UKF))
