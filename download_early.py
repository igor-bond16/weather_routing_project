# download_early.py
from datetime import datetime
import reproduce_wang2019 as R
R.download_era5(datetime(2008, 1, 25), hours=240, area=R.AREA, out_prefix="era5_20080125")