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
from e6dataflow.processor import MultiCountsProcessor, AvgStdProcessor, ThresholdProcessor
import matplotlib.patches as patches
import pickle


plt.close()

data_root = Path('Y:/', 'expdata-e6/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(data_root, daily_subpath)

run_name = 'run0_probe'
num_shots = 450
num_shots = 1750
save_ROI_plot = False
save_fit_plot = True



mol_freq_list = [6.5]
pzt_para_list = [5.5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
probe_att_list = [0.1,3,6]
cav_odt_att_list = [0.2]
num_points = len(probe_att_list)
num_pzt_pos = len(pzt_para_list)
# num_inner_loop = num_points//num_pzt_pos
tweezer_freq_list = [100,102,104,106,108,110,112,114,116,118,120]
num_tweezers = len(tweezer_freq_list)
num_frames = len(pzt_para_list) + 2
run_doc_string = (f'molasses freq = {mol_freq_list}'
                  f'tweezer_freq_list = {tweezer_freq_list}, {num_tweezers} tweezers '
                  f'cav_odt_att = {cav_odt_att_list}'
                  f'probe att list = {probe_att_list}'
                  f'pzt_para_list = {pzt_para_list}'
                 f't_exposure = 500 ms , t_hold = 100 ms'
                 'frames: one standard - 10 sweep pzt - one ref w/ probe - ')
#
# datamodel = get_datamodel(run_name=run_name, num_points=num_points,
#                           run_doc_string=run_doc_string)
datamodel = DataModel(run_name=run_name, num_points=num_points,
                      run_doc_string=run_doc_string)

datafield_list = []
processor_list = []
aggregator_list = []
reporter_list = []

# scope_datastream = DataStream(name='cavity_scope', daily_path=daily_path, run_name=run_name, sub_folder='data',
#                               file_prefix='cavity_scope')

high_na_datastream = DataStream(name='high NA Imaging', daily_path=daily_path, run_name=run_name,
                                file_prefix='jkam_capture')

###############################################################################################

datastream_name = 'High NA Imaging'
data_path_highNA = daily_path / 'data' / run_name / datastream_name
data_path_scope = daily_path / 'data' / run_name / 'cavity_scope' / 'data'

highNA_file_prefix = 'jkam_capture'
scope_file_prefix = 'cavity_scope'

frames_array = np.zeros([num_points, num_frames, 650, 70])
photo_array = np.zeros([num_shots, 650, 70])
# time_array = np.zeros([num_shots, 30000])
# probe_trans_array = np.zeros([num_shots, 30000])
# trap_trans_array = np.zeros([num_shots, 30000])
probe_avg_array = np.zeros(num_shots)
probe_sigma_array = np.zeros(num_shots)
trap_avg_array = np.zeros(num_shots)
trap_sigma_array = np.zeros(num_shots)
probe_avg_array[:] = np.NaN
probe_sigma_array[:] = np.NaN
trap_avg_array[:] = np.NaN
trap_sigma_array[:] = np.NaN
for shot_num in range(num_shots):

    highNA_file_name = highNA_file_prefix + '_' + str(shot_num).zfill(5) + '.h5'
    hf = h5py.File(data_path_highNA / highNA_file_name, 'r')
    point = shot_num % num_points
    for frame_num in range(num_frames):
        photo = np.array(hf.get('frame-' + str(frame_num).zfill(2)))
        if frame_num == 0:
            photo_array[shot_num] = photo
        frames_array[point, frame_num] += photo

    hf.close

    try:
        scope_file_name = scope_file_prefix + '_' + str(shot_num).zfill(5) + '.h5'
        hf = h5py.File(data_path_scope / scope_file_name, 'r')
        point = shot_num % num_points

        probe_signal = np.array(hf.get('CH2-Cav_Probe_Transmission'))
        probe_avg_array[shot_num] = np.mean(probe_signal)
        probe_sigma_array[shot_num] = np.std(probe_signal)
        trap_signal = np.array(hf.get('CH3-Cav_Trap_Transmission'))
        trap_avg_array[shot_num] = np.mean(trap_signal)
        trap_sigma_array[shot_num] = np.std(trap_signal)
        print(f'trap:{trap_avg_array[shot_num]:.03f}+/-{trap_sigma_array[shot_num]:.03f}, probe:{probe_avg_array[shot_num]:.03f}+/-{probe_sigma_array[shot_num]:.03f}')

        hf.close
    except:
        print(data_path_scope / scope_file_name)

        print(f'Error reading scope trace on shot number '+str(shot_num).zfill(5))

frames_array = frames_array / num_shots * num_points

scope_dict = {}
scope_dict["probe_avg"] = probe_avg_array
scope_dict["probe_sigma"] = probe_sigma_array
scope_dict["trap_avg"] = trap_avg_array
scope_dict["trap_sigma"] = trap_sigma_array
plt.plot(probe_sigma_array)
plt.savefig(f'trap_sigma',dpi=100)
f = open("scope_dict.pkl","wb")
pickle.dump(scope_dict,f)
f.close()
print('done')
#######################################################################################################################


tweezer_00_vert_center_init = 79.75# starting position of tweezer 0 = 108MHz
tweezer_00_horiz_center_init = 20.1064
tweezer_horiz_span_init = 24
tweezer_vert_span_init = 16

tweezer_vert_spacing_init = 27.8*2 #2 is freq diff between neighboring tweezers on this run
tweezer_horiz_spacing_init = 1.3867*2

# piezo_vert_shift_init = 0*0.3*0.5  #0.5 is step size of pzt_para_list on this run
# piezo_horiz_shift_init = 0*1.4*0.5

# x_centerpos = np.zeros((num_tweezers,num_points))
# y_centerpos = np.zeros((num_tweezers,num_points))
# x_centerpos_frame = np.zeros((num_tweezers,len(pzt_para_list),len(mol_freq_list)))
# y_centerpos_frame = np.zeros((num_tweezers,len(pzt_para_list),len(mol_freq_list)))
# x_radius_frame = np.zeros((num_tweezers,len(pzt_para_list),len(mol_freq_list)))
# y_radius_frame = np.zeros((num_tweezers,len(pzt_para_list),len(mol_freq_list)))
#
# x_centerpos_4 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list),len(mol_freq_list)))
# y_centerpos_4 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list),len(mol_freq_list)))
# x_radius_4 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list),len(mol_freq_list)))
# y_radius_4 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list),len(mol_freq_list)))

x_centerpos_3 = np.zeros((num_tweezers,num_frames,len(probe_att_list)))
y_centerpos_3 = np.zeros((num_tweezers,num_frames,len(probe_att_list)))
x_radius_3 = np.zeros((num_tweezers,num_frames,len(probe_att_list)))
y_radius_3 = np.zeros((num_tweezers,num_frames,len(probe_att_list)))

x_centerpos_2 = np.zeros((num_tweezers,num_frames))
y_centerpos_2 = np.zeros((num_tweezers,num_frames))
x_radius_2 = np.zeros((num_tweezers,num_frames))
y_radius_2 = np.zeros((num_tweezers,num_frames))

ROI_array = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list), tweezer_vert_span_init, tweezer_horiz_span_init))
ROI_array_avg = np.zeros((num_tweezers,len(pzt_para_list), tweezer_vert_span_init, tweezer_horiz_span_init))

for point in range(num_points):
    if save_ROI_plot:
        fig, axs = plt.subplots(num_tweezers, len(pzt_para_list), figsize=(4*len(pzt_para_list), 4 * num_tweezers))

    probe_att_counter = point

    for tweezer in range(num_tweezers):

        left = round(tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init - tweezer_horiz_span_init/2)
        right = round(tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init + tweezer_horiz_span_init/2)
        top = round(tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init - tweezer_vert_span_init/2)
        bottom = round(tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + tweezer_vert_span_init/2)
        slice2d = tuple((slice(top, bottom, 1),  slice(left, right, 1)))

        ROI_init = np.zeros((tweezer_vert_span_init, tweezer_horiz_span_init))
        for pzt_counter in range(len(pzt_para_list)):

            photo = frames_array[point, pzt_counter]
            ROI_temp = photo[slice2d] - np.amin(photo[slice2d])
            ROI_array[tweezer,pzt_counter,probe_att_counter] = ROI_temp
            fit_struct = e6fit.fit_gaussian2d(ROI_temp, show_plot=False)
            x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
            sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
            x_centerpos_3[tweezer, pzt_counter, probe_att_counter] = tweezer_00_horiz_center_init + tweezer * tweezer_horiz_spacing_init + x0 - tweezer_horiz_span_init / 2
            y_centerpos_3[tweezer, pzt_counter, probe_att_counter] = tweezer_00_vert_center_init + tweezer * tweezer_vert_spacing_init + y0 - tweezer_vert_span_init / 2
            x_radius_3[tweezer, pzt_counter, probe_att_counter] = sx
            y_radius_3[tweezer, pzt_counter, probe_att_counter] = sy
            if point == 0:
                photo_avg = np.mean(frames_array[:, pzt_counter],axis=0)
                ROI_avg = photo_avg[slice2d] - np.amin(photo_avg[slice2d])
                ROI_array_avg[tweezer,pzt_counter] = ROI_avg
                fit_struct = e6fit.fit_gaussian2d(ROI_avg, show_plot=False)
                x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
                sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
                x_centerpos_2[tweezer, pzt_counter] = tweezer_00_horiz_center_init + tweezer * tweezer_horiz_spacing_init + x0 - tweezer_horiz_span_init / 2
                y_centerpos_2[tweezer, pzt_counter] = tweezer_00_vert_center_init + tweezer * tweezer_vert_spacing_init + y0 - tweezer_vert_span_init / 2
                x_radius_2[tweezer, pzt_counter] = sx
                y_radius_2[tweezer, pzt_counter] = sy

            print(f"tweezer{tweezer_freq_list[tweezer]}, pzt_counter={pzt_counter}, probe_counter={probe_att_counter}"
                  f" - x0={x0:.03f},y0={y0:.03f},sx={sx:.03f},sy={sy:.03f}")
            if save_ROI_plot:
                axs[tweezer,pzt_counter].set_title(f"tweezer{tweezer}, point{point}", fontsize=10)
                axs[tweezer,pzt_counter].imshow(ROI_temp,
                                           norm=colors.Normalize(vmin=np.amin(ROI_temp), vmax=np.amax(ROI_temp)),
                                           cmap='hot', interpolation='nearest')
                rect = patches.Rectangle((x0 - sx, y0 - sy), 2 * sx, 2 * sy, linewidth=1, edgecolor='blue',
                                         facecolor='none')
                axs[tweezer,pzt_counter].add_patch(rect)

        x_centerpos_3[tweezer, num_frames-1, probe_att_counter] = x_centerpos_3[tweezer, num_frames-3, probe_att_counter]
        x_centerpos_3[tweezer, num_frames-2, probe_att_counter] = x_centerpos_3[tweezer, num_frames-3, probe_att_counter]
        y_centerpos_3[tweezer, num_frames-1, probe_att_counter] = y_centerpos_3[tweezer, num_frames-3, probe_att_counter]
        y_centerpos_3[tweezer, num_frames-2, probe_att_counter] = y_centerpos_3[tweezer, num_frames-3, probe_att_counter]
        x_centerpos_2[tweezer, num_frames-1] = x_centerpos_2[tweezer, num_frames-3]
        x_centerpos_2[tweezer, num_frames-2] = x_centerpos_2[tweezer, num_frames-3]
        y_centerpos_2[tweezer, num_frames-1] = y_centerpos_2[tweezer, num_frames-3]
        y_centerpos_2[tweezer, num_frames-2] = y_centerpos_2[tweezer, num_frames-3]

        fit_struct = e6fit.fit_gaussian2d(ROI_init, show_plot =  False)
        x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
        sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]

        print(
            f"tweezer{tweezer_freq_list[tweezer]}, pzt_counter={pzt_counter}, probe_counter={probe_att_counter}"
            f" - x0={x0:.03f},y0={y0:.03f},sx={sx:.03f},sy={sy:.03f}")
    if save_ROI_plot:
        fig.savefig(f'tweezers_raw_image_point{point}',dpi=30)


# wait = input("Press enter to continue")
if save_fit_plot:
    fig2, ax2 = plt.subplots(num_tweezers, 4, figsize=(22, 5 * num_tweezers))
    for tweezer_num in range(num_tweezers):
        ax2[tweezer_num, 0].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_center")
        ax2[tweezer_num, 1].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_radius")
        ax2[tweezer_num, 2].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_center")
        ax2[tweezer_num, 3].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_radius")

        for probe_att_counter in range(len(probe_att_list)):
            ax2[tweezer_num, 0].plot(np.array(pzt_para_list), x_centerpos_3[tweezer_num, 0:len(pzt_para_list), probe_att_counter],
                                     label=f"{probe_att_list[probe_att_counter]}")
            ax2[tweezer_num, 1].plot(np.array(pzt_para_list), x_radius_3[tweezer_num, 0:len(pzt_para_list), probe_att_counter],
                                     label=f"{probe_att_list[probe_att_counter]}")
            ax2[tweezer_num, 2].plot(np.array(pzt_para_list), y_centerpos_3[tweezer_num, 0:len(pzt_para_list), probe_att_counter],
                                     label=f"{probe_att_list[probe_att_counter]}")
            ax2[tweezer_num, 3].plot(np.array(pzt_para_list), y_radius_3[tweezer_num, 0:len(pzt_para_list), probe_att_counter],
                                     label=f"{probe_att_list[probe_att_counter]}")

        ax2[tweezer_num, 0].plot(np.array(pzt_para_list),
                                 x_centerpos_2[tweezer_num, 0:len(pzt_para_list)],
                                 label=f"avg",color="black")
        ax2[tweezer_num, 1].plot(np.array(pzt_para_list),
                                 x_radius_2[tweezer_num, 0:len(pzt_para_list)],
                                 label=f"avg",color="black")
        ax2[tweezer_num, 2].plot(np.array(pzt_para_list),
                                 y_centerpos_2[tweezer_num, 0:len(pzt_para_list)],
                                 label=f"avg",color="black")
        ax2[tweezer_num, 3].plot(np.array(pzt_para_list),
                                 y_radius_2[tweezer_num, 0:len(pzt_para_list)],
                                 label=f"avg",color="black")


        ax2[tweezer_num, 3].legend(title='probe intensity', bbox_to_anchor=(1.35, 1), loc='upper right')


    fig2.savefig('tweezers_postition_data_vs_probe_att', dpi=80)


################################################################################################


# wait = input("Press enter to continue")



tweezer_horiz_span = 20
tweezer_vert_span = 12

threshold_processor_list = []
#
# probe_trace_datafield = DataStreamDataField(name = 'probe_transmission_data',
#                                             datastream_name='cavity_scope', h5_subpath='data',
#                                             h5_dataset_name='CH2-Cav_Probe_Transmission')
# probe_transmission_datafield = DataDictShotDataField(name = 'probe_transmission')
#
# trap_trace_datafield = DataStreamDataField(name = 'trap_transmission_data',
#                                             datastream_name='cavity_scope', h5_subpath='data',
#                                             h5_dataset_name='CH3-Cav_Trap_Transmission')
# trap_transmission_datafield = DataDictShotDataField(name = 'trap_transmission')
#
# datafield_list.extend([probe_trace_datafield,trap_trace_datafield,probe_transmission_datafield,trap_transmission_datafield])
#
# probe_transmission_processor = AvgStdProcessor(name='probe_transmission_processor',
#                                                  input_datafield_name='probe_transmission_data',
#                                                  output_datafield_name='probe_transmission')
#
# trap_transmission_processor = AvgStdProcessor(name='trap_transmission_processor',
#                                                  input_datafield_name='trap_transmission_data',
#                                                  output_datafield_name='trap_transmission')
#
# processor_list.extend([probe_transmission_processor,trap_transmission_processor])


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
            # print(f"{point_num}/{num_points}")
            v_center = y_centerpos_2[tweezer_num, frame_num]
            h_center = x_centerpos_2[tweezer_num, frame_num]
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

