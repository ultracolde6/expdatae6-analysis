import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from e6dataflow.datamodel import get_datamodel, DataModel
from e6dataflow.datastream import DataStream
from e6dataflow.aggregator import AvgStdAggregator
from e6dataflow.datafield import DataStreamDataField, DataDictShotDataField, DataDictPointDataField
from e6dataflow.reporter.reporter import Reporter
from e6dataflow.reporter.pointreporter import ImagePointReporter, PlotPointReporter
from e6dataflow.utils import make_centered_roi
from e6dataflow.processor import MultiCountsProcessor, ThresholdProcessor

import e6py.smart_gaussian2d_fit as e6fit

plt.close()

data_root = Path('Y:/', 'expdata-e6/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(data_root, daily_subpath)

run_name = 'run1'
num_points = 10
num_pzt_pos = 10
num_inner_loop = num_points//num_pzt_pos
num_tweezers = 11
num_frames = 9
run_doc_string = ('molasses freq = [4.5,5,5.5,6,6.5,7,7.5,8]'
                  'tweezer_freq_list = [108,110,112,114,116], 5 tweezers '
                  'cav_odt_att = 0 '
                  'probe att list = [0.1,1,2,3]'
                  'pzt_para_list = [5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]'
                 't_exposure = 500 ms , t_hold = 100 ms'
                 'photoassociation -  drop cavity ODT - hold - load cavity ODT ')
#
# datamodel = get_datamodel(run_name=run_name, num_points=num_points,
#                           run_doc_string=run_doc_string)
datamodel = DataModel(run_name=run_name, num_points=num_points,
                      run_doc_string=run_doc_string)

datafield_list = []
processor_list = []
aggregator_list = []
reporter_list = []

high_na_datastream = DataStream(name='high NA Imaging', daily_path=daily_path, run_name=run_name,
                                file_prefix='jkam_capture')

tweezer_00_vert_center = 231.15 - 27.8*8 + 70# starting position of tweezer 0 = 108MHz
tweezer_00_horiz_center = 28.2 - 1.3867*8
tweezer_horiz_span = 14
tweezer_vert_span = 12

tweezer_vert_spacing = 27.8*2 #2 is freq diff between neighboring tweezers on this run
tweezer_horiz_spacing = 1.3867*2

piezo_vert_shift = 0.3*0.5  #0.5 is step size of pzt_para_list on this run
piezo_horiz_shift = 1.4*0.5


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

    for tweezer_num in range(num_tweezers):
        for point_num in range(num_points):
            v_center = tweezer_00_vert_center + tweezer_num * tweezer_vert_spacing \
                                              + point_num//num_inner_loop*piezo_vert_shift
            h_center = tweezer_00_horiz_center + tweezer_num * tweezer_horiz_spacing \
                                                 + point_num//num_inner_loop*piezo_horiz_shift
            roi_array[point_num, tweezer_num] = \
                make_centered_roi(vert_center=v_center,
                                    horiz_center=h_center,
                                    vert_span=tweezer_vert_span,
                                    horiz_span=tweezer_horiz_span)
        new_counts_datafield = DataDictShotDataField(name=f'{frame_key}_tweezer-{tweezer_num:02d}_counts')
        datafield_list.append(new_counts_datafield)
        multicounts_result_datafield_name_list.append(new_counts_datafield.name)

    counts_plot_reporter = PlotPointReporter(name=f'{frame_key}_counts_reporter',
                                             datafield_name_list=multicounts_result_datafield_name_list,
                                             layout=Reporter.LAYOUT_GRID, save_data=True, close_plots=True)
    # reporter_list.append(counts_plot_reporter)
    multicounts_processor = MultiCountsProcessor(name=f'{frame_key}_multicount_processor',
                                                 frame_datafield_name=frame_key,
                                                 result_datafield_name_list=multicounts_result_datafield_name_list,
                                                 roi_slice_array=roi_array)
    processor_list.append(multicounts_processor)


avg_img_datafield_name_list = [f'frame-{frame_num:02d}_avg' for frame_num in range(num_frames)]
image_point_reporter = ImagePointReporter(name='avg_frame_reporter',
                                          datafield_name_list=avg_img_datafield_name_list,
                                          layout=Reporter.LAYOUT_GRID,
                                          save_data=True, close_plots=True, roi_slice_array=roi_array)

datastream_list = [high_na_datastream]
reporter_list += [image_point_reporter]
datatool_list = datastream_list + datafield_list + processor_list + aggregator_list + reporter_list
# datatool_list = reporter_list
for datatool in datatool_list:
    datamodel.add_datatool(datatool, overwrite=True, quiet=True)
datamodel.link_datatools()

datamodel.run(handler_quiet=True,save_every_shot=False,save_point_data=False,save_before_reporting=True)

