"""Independent analytical IEKF entry; never imports the UKF implementation."""
from iekf.filter import IEKF
from rgbd_common.entry import run

if __name__=='__main__':raise SystemExit(run(IEKF))
