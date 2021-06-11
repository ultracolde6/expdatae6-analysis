import matplotlib.pyplot as plt
from pathlib import Path
from e6dataflow.tools.tweezer_tools import auto_roi


plt.close('all')

# data_root = Path('Y:/', 'expdata-e6/data')
data_root = Path('C:/', 'Users/Justin/Desktop/Working/labdata/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(data_root, daily_subpath)
run_name = 'approach_curve'
data_dir = Path(daily_path, 'data', run_name, 'High NA Imaging')
data_prefix = 'jkam_capture'

tweezer_att_list = [3.75, 5, 7]
num_tweezer_att = len(tweezer_att_list)

mol_att_list = [2.5, 5, 7.5]
num_mol_att = len(mol_att_list)

mol_freq_list = [5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5]
num_frames = len(mol_freq_list) + 2
num_frames_auto_roi = num_frames - 1
frame_list = range(num_frames_auto_roi)

num_points = num_tweezer_att * num_mol_att

tweezer_freq_list = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120]
num_tweezers = len(tweezer_freq_list)

first_tweezer_horiz = 18
first_tweezer_vert = 82
last_tweezer_horiz = 45
last_tweezer_vert = 636

pzt_roi_dict, pzt_point_frame_dict = auto_roi(data_dir=data_dir, data_prefix=data_prefix,
                                              num_tweezers=num_tweezers, num_points=num_points,
                                              frame_list=frame_list, num_pzt=1, mode='single',
                                              first_tweezer_vert=first_tweezer_vert,
                                              first_tweezer_horiz=first_tweezer_horiz,
                                              last_tweezer_vert=last_tweezer_vert,
                                              last_tweezer_horiz=last_tweezer_horiz,
                                              vert_search_span=15, horiz_search_span=20,
                                              lock_span=True)
