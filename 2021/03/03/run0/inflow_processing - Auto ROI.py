import numpy as np
import e6py.smart_gaussian2d_fit as e6fit
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from e6dataflow.datamodel import get_datamodel, DataModel
from e6dataflow.datastream import DataStream
from e6dataflow.aggregator import AvgStdAggregator
from e6dataflow.datafield import DataStreamDataField, DataDictShotDataField, DataDictPointDataField
from e6dataflow.reporter.reporter import Reporter
from e6dataflow.reporter.pointreporter import ImagePointReporter, PlotPointReporter
from e6dataflow.utils import make_centered_roi
from e6dataflow.processor import MultiCountsProcessor, ThresholdProcessor
import matplotlib.patches as patches

plt.close()

data_root = Path('Y:/', 'expdata-e6/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(data_root, daily_subpath)

run_name = 'run0'
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

###############################################################################################

datastream_name = 'High NA Imaging'
data_path_cav = daily_path / 'data' / run_name / datastream_name

file_prefix = 'jkam_capture'
num_shots = 830
mol_freq_list = [6.5,5,5.5,6,6.5,7,7.5]
pzt_para_list = [5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
probe_att_list = [0]
num_points = len(pzt_para_list) * len(probe_att_list)

frames_array = np.zeros([len(probe_att_list), len(pzt_para_list), num_frames, 650, 70])
for shot_num in range(num_shots):

    file_name = file_prefix + '_' + str(shot_num).zfill(5) + '.h5'
    hf = h5py.File(data_path_cav / file_name, 'r')
    point = shot_num % num_points
    probe_att_counter = point % len(probe_att_list)
    probe_att = probe_att_list[probe_att_counter]
    pzt_para_counter = point // len(probe_att_list)
    pzt_para = pzt_para_list[pzt_para_counter]
    for frame_num in range(num_frames):
        photo = np.array(hf.get('frame-' + str(frame_num).zfill(2)))
        frames_array[probe_att_counter, pzt_para_counter, frame_num] += photo

    hf.close

frames_array = frames_array / num_shots * num_points



tweezer_00_vert_center_init = 79.75# starting position of tweezer 0 = 108MHz
tweezer_00_horiz_center_init = 20.1064
tweezer_horiz_span_init = 20
tweezer_vert_span_init = 14

tweezer_vert_spacing_init = 27.8*2 #2 is freq diff between neighboring tweezers on this run
tweezer_horiz_spacing_init = 1.3867*2

piezo_vert_shift_init = 0*0.3*0.5  #0.5 is step size of pzt_para_list on this run
piezo_horiz_shift_init = 0*1.4*0.5

fig, axs = plt.subplots(num_tweezers, len(pzt_para_list) ,figsize=(30, 30))
x_centerpos = np.zeros((num_tweezers,len(pzt_para_list)))
y_centerpos = np.zeros((num_tweezers,len(pzt_para_list)))
for pzt_para_counter in range(len(pzt_para_list)):
    # fig, axs = plt.subplots(1, num_frames ,figsize=(30, 10))
    # vmin = 110
    # vmax = 140
    for tweezer in range(num_tweezers):

        ROI_init = np.zeros((tweezer_vert_span_init, tweezer_horiz_span_init))
        for frame_num in range(num_frames):
            left = round(tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init + pzt_para_counter*piezo_horiz_shift_init - tweezer_horiz_span_init/2)
            right = round(tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init + pzt_para_counter*piezo_horiz_shift_init + tweezer_horiz_span_init/2)
            top = round(tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + pzt_para_counter*piezo_vert_shift_init - tweezer_vert_span_init/2)
            bottom = round(tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + pzt_para_counter*piezo_vert_shift_init + tweezer_vert_span_init/2)
            slice2d = tuple((slice(top, bottom, 1),  slice(left, right, 1)))
            photo = frames_array[0, pzt_para_counter, frame_num]
            ROI_init += photo[slice2d]
        ROI_init = ROI_init - np.amin(ROI_init)
        fit_struct = e6fit.fit_gaussian2d(ROI_init, show_plot =  False)
        x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
        sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
        axs[tweezer, pzt_para_counter].imshow(ROI_init   , norm=colors.Normalize(vmin=np.amin(ROI_init), vmax=np.amax(ROI_init)), cmap='hot', interpolation='nearest')
        rect = patches.Rectangle((x0-sx,y0-sy), 2*sx,2*sy,linewidth=1,edgecolor='blue',facecolor='none')
        axs[tweezer, pzt_para_counter].add_patch(rect)
        x_centerpos[tweezer,pzt_para_counter] = tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init + pzt_para_counter*piezo_horiz_shift_init + x0 - tweezer_horiz_span_init/2
        y_centerpos[tweezer,pzt_para_counter] = tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + pzt_para_counter*piezo_vert_shift_init + y0 - tweezer_vert_span_init/2
        print([x0,y0,sx,sy])
# fig.show()
fig.savefig('tweezers_raw_data',dpi=300)


################################################################################################


wait = input("Press enter to continue")



tweezer_horiz_span = 14
tweezer_vert_span = 12

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
            v_center = y_centerpos[tweezer_num, point_num//num_inner_loop]
            h_center = x_centerpos[tweezer_num, point_num//num_inner_loop]
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

