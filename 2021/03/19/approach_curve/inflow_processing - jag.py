import matplotlib.pyplot as plt
from pathlib import Path
from e6dataflow.datamodel import DataModel, get_datamodel
from e6dataflow.datastream import DataStream
from e6dataflow.datafield import DataStreamDataField, H5PointDataField, DataDictShotDataField
from e6dataflow.aggregator import AvgStdAggregator
from e6dataflow.processor import CountsProcessor
from e6dataflow.reporter.reporter import Reporter
from e6dataflow.reporter.shotreporter import ImageShotReporter
from e6dataflow.tools.tweezer_tools import auto_roi, get_roi_list_by_point


plt.close('all')

"""
Directory Identification
"""
# data_root = Path('Y:/', 'expdata-e6/data')
data_root = Path('C:/', 'Users/Justin/Desktop/Working/labdata/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_data_path = Path(data_root, daily_subpath)
run_name = 'approach_curve'
data_prefix = 'jkam_capture'

"""
Run Parameters
"""
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

"""
ROI Estimation
"""
roi_data_dir = Path(daily_data_path, 'data', run_name, 'High NA Imaging')


first_tweezer_horiz = 18
first_tweezer_vert = 82
last_tweezer_horiz = 45
last_tweezer_vert = 636

pzt_roi_dict, pzt_point_frame_dict = auto_roi(data_dir=roi_data_dir, data_prefix=data_prefix,
                                              num_tweezers=num_tweezers, num_points=num_points,
                                              frame_list=frame_list, num_pzt=1, mode='single',
                                              first_tweezer_vert=first_tweezer_vert,
                                              first_tweezer_horiz=first_tweezer_horiz,
                                              last_tweezer_vert=last_tweezer_vert,
                                              last_tweezer_horiz=last_tweezer_horiz,
                                              vert_search_span=15, horiz_search_span=20,
                                              lock_span=True)

"""
Datamodel Initialization
"""

run_doc_string = (f'molasses freq = {mol_freq_list}'
                  f'tweezer_freq_list = {tweezer_freq_list}, {num_tweezers} tweezers '
                  f'mol_att_list = {mol_att_list}'
                  f'tweezer_att_list = {tweezer_att_list}'
                  f'no cavity modes, pzt_para = 5.5'
                  f't_exposure = 500 ms , t_hold = 100 ms'
                  f'photoassociation -  drop cavity ODT - hold - load cavity ODT ')

# datamodel = DataModel(run_name=run_name, num_points=num_points,
#                       run_doc_string=run_doc_string)
datamodel = get_datamodel(run_name=run_name, num_points=num_points,
                          run_doc_string=run_doc_string, reset=True)

datastream_list = []
datafield_list = []
processor_list = []
aggregator_list = []
reporter_list = []

"""
Datamodel Construction
"""

high_na_datastream = DataStream(name='high NA Imaging', daily_data_path=daily_data_path, run_name=run_name,
                                file_prefix='jkam_capture')
datastream_list.append(high_na_datastream)

for frame_num in range(num_frames):
    frame_key = f'frame-{frame_num:02d}'
    frame_datafield = DataStreamDataField(name=frame_key, datastream_name='high NA Imaging', h5_subpath=None,
                                          h5_dataset_name=frame_key, parent_names='high NA Imaging')
    datafield_list.append(frame_datafield)

    avg_datafield = H5PointDataField(name=f'{frame_key}_avg', parent_names=f'{frame_key}_avg_aggregator')
    std_datafield = H5PointDataField(name=f'{frame_key}_std', parent_names=f'{frame_key}_avg_aggregator')
    datafield_list.append(avg_datafield)
    datafield_list.append(std_datafield)
    img_avg_aggregator = AvgStdAggregator(name=f'{frame_key}_avg_aggregator', verifier_datafield_names=[],
                                          input_datafield_name=frame_key,
                                          output_mean_datafield_name=f'{frame_key}_avg',
                                          output_std_datafield_name=f'{frame_key}_std',
                                          parent_names=frame_key)
    aggregator_list.append(img_avg_aggregator)

    for tweezer_num in range(num_tweezers):
        tweezer_key = f'tweezer-{tweezer_num:02d}'
        counts_datafield_name = f'{frame_key}_{tweezer_key}_counts'
        new_counts_datafield = DataDictShotDataField(name=counts_datafield_name,
                                                     parent_names=f'{frame_key}_{tweezer_key}_counts_processor')
        datafield_list.append(new_counts_datafield)
        if frame_num != num_frames - 1:
            roi_list = get_roi_list_by_point(pzt_roi_dict, pzt_point_frame_dict,
                                             num_points, frame_num, tweezer_num)
        else:
            roi_list = get_roi_list_by_point(pzt_roi_dict, pzt_point_frame_dict,
                                             num_points, 0, tweezer_num)
        counts_processor = CountsProcessor(name=f'{frame_key}_tweezer-{tweezer_num:02d}_counts_processor',
                                           frame_datafield_name=frame_key,
                                           output_datafield_name=counts_datafield_name,
                                           roi=roi_list,
                                           parent_names=frame_key)
        processor_list.append(counts_processor)

frame_datafield_names = [f'frame-{frame_num:02d}' for frame_num in range(num_frames)]
frame_shot_reporter = ImageShotReporter(name='frame_shot_reporter', datafield_name_list=frame_datafield_names,
                                        layout=Reporter.LAYOUT_HORIZONTAL, save_data=True, roi_dict=dict(),
                                        parent_names=frame_datafield_names)
reporter_list.append(frame_shot_reporter)

"""
Datamodel Finalization
"""

datatool_list = datastream_list + datafield_list + processor_list + aggregator_list + reporter_list
for datatool in datatool_list:
    datamodel.add_datatool(datatool, overwrite=True, quiet=True)
datamodel.link_datatools()


"""
Run Datamodel
"""
datamodel.run(handler_quiet=False, save_every_shot=False)
