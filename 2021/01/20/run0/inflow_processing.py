import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from e6dataflow.datamodel import get_datamodel, DataModel
from e6dataflow.datastream import DataStream
from e6dataflow.aggregator import AvgStdAggregator
from e6dataflow.datafield import DataStreamDataField, DataDictShotDataField, DataDictPointDataField
from e6dataflow.reporter.reporter import Reporter
from e6dataflow.reporter.pointreporter import ImagePointReporter, PlotPointReporter
from e6dataflow.reporter.shotreporter import ImageShotReporter
from e6dataflow.utils import make_centered_roi
from e6dataflow.processor import MultiCountsProcessor, ThresholdProcessor

plt.close()

data_root = Path('Y:/', 'expdata-e6/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(data_root, daily_subpath)

run_name = 'run0'
num_points = 5
run_doc_string = 'release and recapture experiment. release for [5, 20, 30, 40, 60] us.'

# datamodel = get_datamodel(daily_path=daily_path, run_name=run_name, num_points=num_points,
#                           run_doc_string=run_doc_string, quiet=False)
datamodel = DataModel(daily_path=daily_path, run_name=run_name, num_points=num_points,
                      run_doc_string=run_doc_string)

datafield_list = []
processor_list = []
aggregator_list = []
reporter_list = []

high_na_datastream = DataStream(name='high NA Imaging', daily_path=daily_path, run_name=run_name,
                                file_prefix='jkam_capture')

tweezer_00_vert_center = 53
tweezer_00_horiz_center = 16
tweezer_span = 16

tweezer_vert_spacing = 28
tweezer_horiz_spacing = 1

num_tweezers = 10
tweezer_roi_list = []
for tweezer_num in range(num_tweezers):
    tweezer_roi = make_centered_roi(vert_center=tweezer_00_vert_center + tweezer_num * tweezer_vert_spacing,
                                    horiz_center=tweezer_00_horiz_center + tweezer_num * tweezer_horiz_spacing,
                                    vert_span=tweezer_span,
                                    horiz_span=tweezer_span)
    tweezer_roi_list.append(tweezer_roi)

num_frames = 3
threshold_processor_list = []
for frame_num in range(num_frames):
    frame_key = f'frame-{frame_num:02d}'
    frame_datafield = DataStreamDataField(name=frame_key, datastream_name='high NA Imaging', h5_subpath=None,
                                    h5_dataset_name=frame_key)
    datafield_list.append(frame_datafield)

    avg_datafield = DataDictPointDataField(name=f'{frame_key}_avg')
    datafield_list.append(avg_datafield)
    img_avg_aggregator = AvgStdAggregator(name=f'{frame_key}_avg_aggregator', verifier_datafield_names=[],
                                          input_datafield_name=frame_key,
                                          output_datafield_name=f'{frame_key}_avg')
    aggregator_list.append(img_avg_aggregator)

    roi_array = np.zeros([num_points, num_tweezers], dtype=object)
    multicounts_result_datafield_name_list = []
    for tweezer_num, tweezer_roi in enumerate(tweezer_roi_list):
        for point_num in range(num_points):
            roi_array[point_num, tweezer_num] = tweezer_roi
        new_counts_datafield = DataDictShotDataField(name=f'{frame_key}_tweezer-{tweezer_num:02d}_counts')
        new_counts_threshold_datafield = DataDictShotDataField(name=f'{frame_key}_tweezer-{tweezer_num:02d}_counts_verifier')
        datafield_list.append(new_counts_datafield)
        datafield_list.append(new_counts_threshold_datafield)
        multicounts_result_datafield_name_list.append(new_counts_datafield.name)
        threshold_processor = ThresholdProcessor(name=f'{frame_key}_tweezer-{tweezer_num:02d}_threshold_processor',
                                                 input_datafield_name=f'{frame_key}_tweezer-{tweezer_num:02d}_counts',
                                                 output_datafield_name=f'{frame_key}_tweezer-{tweezer_num:02d}_counts_verifier',
                                                 threshold_value=31000)
        threshold_processor_list.append(threshold_processor)
    counts_plot_reporter = PlotPointReporter(name=f'{frame_key}_counts_reporter',
                                             datafield_name_list=multicounts_result_datafield_name_list,
                                             layout=Reporter.LAYOUT_GRID, save_data=True)
    reporter_list.append(counts_plot_reporter)


    multicounts_processor = MultiCountsProcessor(name=f'{frame_key}_multicount_processor',
                                                 frame_datafield_name=frame_key,
                                                 result_datafield_name_list=multicounts_result_datafield_name_list,
                                                 roi_slice_array=roi_array)
    processor_list.append(multicounts_processor)
    processor_list += threshold_processor_list



reporter_roi_dict = dict()
for frame_num in range(num_frames):
    datafield_name = f'frame-{frame_num:02d}_avg'
    reporter_roi_dict[datafield_name] = tweezer_roi_list

roi_dict = {'frame-00_avg':[(slice(45, 65), slice(5, 25))]}
avg_img_datafield_name_list = [f'frame-{frame_num:02d}_avg' for frame_num in range(num_frames)]
image_point_reporter = ImagePointReporter(name='avg_frame_reporter',
                                          datafield_name_list=avg_img_datafield_name_list,
                                          layout=Reporter.LAYOUT_HORIZONTAL,
                                          save_data=True, roi_dict=reporter_roi_dict)

single_shot_reporter = ImageShotReporter(name=f'image_shot_reporter',
                                         datafield_name_list=[f'frame-{frame_num:02d}' for frame_num in range(num_frames)],
                                         layout=Reporter.LAYOUT_HORIZONTAL, save_data=True, roi_dict=reporter_roi_dict)
reporter_list.append(single_shot_reporter)

datastream_list = [high_na_datastream]
reporter_list += [image_point_reporter]
# reporter_list = []
datatool_list = datastream_list + datafield_list + processor_list + aggregator_list + reporter_list
for datatool in datatool_list:
    datamodel.add_datatool(datatool, overwrite=True, quiet=True)
datamodel.link_datatools()

datamodel.run_continuously(handler_quiet=True)

