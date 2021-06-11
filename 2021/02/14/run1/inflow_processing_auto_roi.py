import numpy as np
import e6py.smart_gaussian2d_fit as e6fit
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from e6dataflow.datamodel import get_datamodel, DataModel
from e6dataflow.datastream import DataStream
from e6dataflow.aggregator import FrameAggregator
from e6dataflow.datafield import DataStreamDataField, DataDictShotDataField, DataDictPointDataField
from e6dataflow.reporter.reporter import Reporter
from e6dataflow.reporter.pointreporter import ImagePointReporter, PlotPointReporter
from e6dataflow.utils import make_centered_roi
from e6dataflow.processor import MultiCountsProcessor, ThresholdProcessor, AutoROIProcessor
import matplotlib.patches as patches


plt.close()

data_root = Path('Y:/', 'expdata-e6/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(Path.cwd().parent)

run_name = 'run1'
num_shots = 8320
mol_freq_list = [4.5,5,5.5,6,6.5,7,7.5,8]
pzt_para_list = [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
probe_att_list = [0.1, 3, 6, 9]
num_points = len(pzt_para_list) * len(probe_att_list)
num_pzt_pos = len(pzt_para_list)
num_inner_loop = num_points//num_pzt_pos
tweezer_freq_list = [108,110,112,114,116]
num_tweezers = len(tweezer_freq_list)
num_frames = len(mol_freq_list) + 2
run_doc_string = ('molasses freq = [4.5,5,5.5,6,6.5,7,7.5,8]'
                  'tweezer_freq_list = [108,110,112,114,116], 5 tweezers '
                  'cav_odt_att = 0 '
                  'probe att list = [0.1,1,2,3]'
                  'pzt_para_list = [5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]'
                 't_exposure = 500 ms , t_hold = 100 ms'
                 'photoassociation -  drop cavity ODT - hold - load cavity ODT ')


############# AUTO ROI #######################################################33
roi_datamodel = DataModel(run_name=run_name+'_roi', num_points=num_points,
                      run_doc_string=run_doc_string)

datafield_list = []
processor_list = []
aggregator_list = []
reporter_list = []

high_na_datastream = DataStream(name='high NA Imaging', daily_path=daily_path, run_name=run_name,
                                file_prefix='jkam_capture')

roi_frame_list = range(8)# frame to be used for ROI fit

for frame_num in roi_frame_list:
    frame_key = f'frame-{frame_num:02d}'
    frame_datafield = DataStreamDataField(name=frame_key, datastream_name='high NA Imaging', h5_subpath=None,
                                h5_dataset_name=frame_key)
    datafield_list.append(frame_datafield)

roi_datafield = DataDictPointDataField(name='roi_frames')
datafield_list.append(roi_datafield)

img_avg_aggregator = FrameAggregator(name='avg_frame_aggregator', verifier_datafield_names=[],
                                      input_datafield_name='avg',
                                      output_datafield_name='roi_frames',
                                      frames = roi_frame_list)
aggregator_list.append(img_avg_aggregator)

## Build ROI Guess Array ##
roi_guess_array = np.zeros([num_points, num_tweezers], dtype=object)

tweezer_00_vert_center_init = 81.15# starting position of tweezer 0 = 108MHz
tweezer_00_horiz_center_init = 21.2
tweezer_horiz_span_init = 20
tweezer_vert_span_init = 20

tweezer_vert_spacing_init = 27.8*2 #2 is freq diff between neighboring tweezers on this run
tweezer_horiz_spacing_init = 1.3867*2

x_centerpos = np.zeros((num_tweezers))
y_centerpos = np.zeros((num_tweezers))

for tweezer in range(num_tweezers):
        x_centerpos[tweezer] = tweezer_00_horiz_center_init + tweezer * tweezer_horiz_spacing_init
        y_centerpos[tweezer] = tweezer_00_vert_center_init + tweezer * tweezer_vert_spacing_init

        v_center = y_centerpos[tweezer]
        h_center = x_centerpos[tweezer]
        for point_num in range(num_points):
            roi_guess_array[point_num, tweezer] = \
                make_centered_roi(vert_center=v_center,
                                  horiz_center=h_center,
                                  vert_span=tweezer_vert_span_init,
                                  horiz_span=tweezer_horiz_span_init)
####

ROI_datafield = DataDictPointDataField(name='ROI_fit')
datafield_list.append(ROI_datafield)

autoROI_processor = AutoROIProcessor(name='autoROI_processor',
                                     avg_frame_datafield_name='roi_frames',
                                     result_datafield_name='ROI_fit',
                                     roi_guess_slice_array=roi_guess_array)

image_point_reporter = ImagePointReporter(name='autoROI_frame_reporter',
                                          datafield_name_list=['roi_frames'],
                                          layout=Reporter.LAYOUT_GRID,
                                          save_data=True, close_plots=True, roi_slice_array=roi_guess_array)

datastream_list = [high_na_datastream]
# reporter_list = [image_point_reporter]
postprocessor_list = [autoROI_processor]

datatool_list = datastream_list + datafield_list + processor_list + aggregator_list + reporter_list + postprocessor_list
# datatool_list = reporter_list
for datatool in datatool_list:
    print(datatool.name)
    roi_datamodel.add_datatool(datatool, overwrite=True, quiet=True)
roi_datamodel.link_datatools()

roi_datamodel.run(handler_quiet=True,save_every_shot=False,save_point_data=True,save_before_reporting=False)