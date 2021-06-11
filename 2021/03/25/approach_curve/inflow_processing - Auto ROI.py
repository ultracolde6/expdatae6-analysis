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
import pickle


plt.close()

data_root = Path('Y:/', 'expdata-e6/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(data_root, daily_subpath)

run_name = 'approach_curve'
num_shots = 8910
save_ROI_plot = True
save_fit_plot = True


mol_freq_list = [5,5.5,6,6.5,7,7.5,8,8.5,9,9.25,9.5]
tweezer_att_list = [2,4,6]
mol_att_list = [2.5,5,7.5]
cav_odt_att_list = [0]
num_points = len(tweezer_att_list) * len(mol_att_list)
num_tweezer_att = len(tweezer_att_list)
num_inner_loop = num_points//num_tweezer_att
tweezer_freq_list = [108,110,112,114,116] #(110 is fake)
num_tweezers = len(tweezer_freq_list)
num_frames = len(mol_freq_list) + 2
run_doc_string = ('repump power in 3D MOT: 38uW. Max(next run) 52.5 uW'
                    f'molasses freq = {mol_freq_list}'
                  f'tweezer_freq_list = {tweezer_freq_list}, {num_tweezers} tweezers '
                  f'mol_att_list = {mol_att_list}'
                  f'tweezer_att_list = {tweezer_att_list}'
                  'no cavity modes, pzt_para = 5.5'
                 f't_exposure = 500 ms , t_hold = 100 ms'
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
data_path_highNA = daily_path / 'data' / run_name / datastream_name

highNA_file_prefix = 'jkam_capture'

frames_array = np.zeros([num_points, num_frames, 650, 70])
for shot_num in range(num_shots):

    highNA_file_name = highNA_file_prefix + '_' + str(shot_num).zfill(5) + '.h5'
    hf = h5py.File(data_path_highNA / highNA_file_name, 'r')
    point = shot_num % num_points
    for frame_num in range(num_frames):

        photo = np.array(hf.get('frame-' + str(frame_num).zfill(2)))
        frames_array[point, frame_num] += photo

    hf.close

frames_array = frames_array / num_shots * num_points




tweezer_00_vert_center_init = 79.75# starting position of tweezer 0 = 108MHz
tweezer_00_horiz_center_init = 20.1064
tweezer_horiz_span_init = 24
tweezer_vert_span_init = 16

tweezer_vert_spacing_init = 27.8*2 #2 is freq diff between neighboring tweezers on this run
tweezer_horiz_spacing_init = 1.3867*2

# piezo_vert_shift_init = 0*0.3*0.5  #0.5 is step size of tweezer_att_list on this run
# piezo_horiz_shift_init = 0*1.4*0.5

x_centerpos = np.zeros((num_tweezers,num_points))
y_centerpos = np.zeros((num_tweezers,num_points))
x_centerpos_frame = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_freq_list)))
y_centerpos_frame = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_freq_list)))
x_radius_frame = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_freq_list)))
y_radius_frame = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_freq_list)))
x_centerpos_4 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list),len(mol_freq_list)))
y_centerpos_4 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list),len(mol_freq_list)))
x_radius_4 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list),len(mol_freq_list)))
y_radius_4 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list),len(mol_freq_list)))
x_centerpos_3 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list)))
y_centerpos_3 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list)))
x_radius_3 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list)))
y_radius_3 = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list)))
x_centerpos_1 = np.zeros(num_tweezers)
y_centerpos_1 = np.zeros(num_tweezers)
x_radius_1 = np.zeros(num_tweezers)
y_radius_1 = np.zeros(num_tweezers)
ROI_init_array = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list), tweezer_vert_span_init, tweezer_horiz_span_init))
ROI_array = np.zeros((num_tweezers,len(tweezer_att_list),len(mol_att_list), len(mol_freq_list), tweezer_vert_span_init, tweezer_horiz_span_init))
if save_ROI_plot:
    fig, axs = plt.subplots(num_points, num_tweezers ,figsize=(4*num_tweezers, 4*num_points))
for point in range(num_points):
    mol_att_counter = point % len(mol_att_list)
    tweezer_att_counter = point // len(mol_att_list)
    for tweezer in range(num_tweezers):

        left = round(tweezer_00_horiz_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_horiz_spacing_init - tweezer_horiz_span_init/2)
        right = round(tweezer_00_horiz_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_horiz_spacing_init + tweezer_horiz_span_init/2)
        top = round(tweezer_00_vert_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_vert_spacing_init - tweezer_vert_span_init/2)
        bottom = round(tweezer_00_vert_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_vert_spacing_init + tweezer_vert_span_init/2)
        slice2d = tuple((slice(top, bottom, 1),  slice(left, right, 1)))

        ROI_init = np.zeros((tweezer_vert_span_init, tweezer_horiz_span_init))
        for frame_num in range(len(mol_freq_list)):
            photo = frames_array[point, frame_num]
            ROI_temp = photo[slice2d] - np.amin(photo[slice2d])
            ROI_array[tweezer,tweezer_att_counter,mol_att_counter,frame_num] = ROI_temp
            fit_struct = e6fit.fit_gaussian2d(ROI_temp, show_plot=False)
            x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
            sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
            x_centerpos_4[tweezer, tweezer_att_counter, mol_att_counter,frame_num] = left + x0
            y_centerpos_4[tweezer, tweezer_att_counter, mol_att_counter,frame_num] = top + y0
            x_radius_4[tweezer, tweezer_att_counter, mol_att_counter,frame_num] = sx
            y_radius_4[tweezer, tweezer_att_counter, mol_att_counter,frame_num] = sy

            # print(f"tweezer{tweezer_freq_list[tweezer]}, tweezer_att_counter={tweezer_att_counter}, mol_att_counter={mol_att_counter}, mol_counter={frame_num}"
            #       f" - x0={x0:.03f},y0={y0:.03f},sx={sx:.03f},sy={sy:.03f}")

            ROI_init += photo[slice2d]
        ROI_init = ROI_init - np.amin(ROI_init)
        ROI_init_array[tweezer,tweezer_att_counter,mol_att_counter] = ROI_init
        fit_struct = e6fit.fit_gaussian2d(ROI_init, show_plot = False)
        x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
        sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
        if save_ROI_plot:
            axs[point, tweezer].set_title(f"tweezer{tweezer}, point{point}", fontsize=10)
            axs[point, tweezer].imshow(ROI_init  , norm=colors.Normalize(vmin=np.amin(ROI_init), vmax=np.amax(ROI_init)), cmap='hot', interpolation='nearest')
            rect = patches.Rectangle((x0-sx,y0-sy), 2*sx,2*sy,linewidth=1,edgecolor='blue',facecolor='none')
            axs[point, tweezer].add_patch(rect)
        # x_centerpos[tweezer,point] = tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init +  x0 - tweezer_horiz_span_init/2
        # y_centerpos[tweezer,point] = tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + y0 - tweezer_vert_span_init/2

        x_centerpos_3[tweezer,tweezer_att_counter,mol_att_counter] = left + x0
        y_centerpos_3[tweezer,tweezer_att_counter,mol_att_counter] = top + y0
        x_radius_3[tweezer,tweezer_att_counter,mol_att_counter] = sx
        y_radius_3[tweezer,tweezer_att_counter,mol_att_counter] = sy

        # print(
        #     f"tweezer{tweezer_freq_list[tweezer]}, tweezer_att_counter={tweezer_att_counter}, mol_att_counter={mol_att_counter}"
        #     f" - x0={x0:.03f},y0={y0:.03f},sx={sx:.03f},sy={sy:.03f}")
if save_ROI_plot:
    fig.savefig('tweezers_raw_data',dpi=30)
for tweezer in range(num_tweezers):
    for tweezer_att_counter in range(len(tweezer_att_list)):
        for mol_freq_counter in range(len(mol_freq_list)):
            ROI_temp = np.mean(ROI_array[tweezer,tweezer_att_counter,:,mol_freq_counter],axis=0)
            fit_struct = e6fit.fit_gaussian2d(ROI_temp, show_plot=False)
            x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
            sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
            x_centerpos_frame[tweezer, tweezer_att_counter, mol_freq_counter] = round(tweezer_00_horiz_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_horiz_spacing_init - tweezer_horiz_span_init/2) + x0
            y_centerpos_frame[tweezer, tweezer_att_counter, mol_freq_counter] = round(tweezer_00_vert_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_vert_spacing_init - tweezer_vert_span_init/2) +y0
            x_radius_frame[tweezer, tweezer_att_counter, mol_freq_counter] = sx
            y_radius_frame[tweezer, tweezer_att_counter, mol_freq_counter] = sy

            # print(f"tweezer{tweezer_freq_list[tweezer]}, tweezer_att_counter={tweezer_att_counter}, mol_counter={frame_num}"
            #       f" - x0={x0:.03f},y0={y0:.03f},sx={sx:.03f},sy={sy:.03f}")

    fit_struct = e6fit.fit_gaussian2d(np.mean(ROI_init_array[tweezer,:,:],axis=(0,1)), show_plot=False)
    x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
    sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
    x_centerpos_1[tweezer] = round(tweezer_00_horiz_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_horiz_spacing_init - tweezer_horiz_span_init/2) + x0
    y_centerpos_1[tweezer] = round(tweezer_00_vert_center_init + (tweezer_freq_list[tweezer]-100)/2*tweezer_vert_spacing_init - tweezer_vert_span_init/2) +y0
    x_radius_1[tweezer] = sx
    y_radius_1[tweezer] = sy
    print(f"tweezer{tweezer_freq_list[tweezer]} - x0={x0:.03f},y0={y0:.03f},sx={sx:.03f},sy={sy:.03f}")
    for tweezer_att_counter in range(len(tweezer_att_list)):
        for inner_counter in range(num_inner_loop):
            x_centerpos[tweezer, tweezer_att_counter*num_inner_loop+inner_counter] = x_centerpos_1[tweezer]
            y_centerpos[tweezer, tweezer_att_counter*num_inner_loop+inner_counter] = y_centerpos_1[tweezer]

if save_fit_plot:
    fig1, ax1 = plt.subplots(1, 4, figsize=(20, 5))
    fig3, ax3 = plt.subplots(num_tweezers, 4, figsize=(20, 5 * num_tweezers))
    # fig4, ax4 = plt.subplots(num_tweezers, 4, figsize=(20, 5 * num_tweezers))
    for tweezer_num in range(num_tweezers):
        ax1[0].set_title(f"x_center")
        ax1[1].set_title(f"x_radius")
        ax1[2].set_title(f"y_center")
        ax1[3].set_title(f"y_radius")
        ax3[tweezer_num, 0].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_center")
        ax3[tweezer_num, 1].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_radius")
        ax3[tweezer_num, 2].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_center")
        ax3[tweezer_num, 3].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_radius")
        # ax4[tweezer_num, 0].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_center")
        # ax4[tweezer_num, 1].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_radius")
        # ax4[tweezer_num, 2].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_center")
        # ax4[tweezer_num, 3].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_radius")
        for mol_freq_counter in range(len(mol_freq_list)):
            ax3[tweezer_num, 0].plot(np.array(tweezer_att_list), x_centerpos_frame[tweezer_num, :, mol_freq_counter],
                                     label=f"{mol_freq_list[mol_freq_counter]}")
            ax3[tweezer_num, 1].plot(np.array(tweezer_att_list), x_radius_frame[tweezer_num, :, mol_freq_counter],
                                     label=f"{mol_freq_list[mol_freq_counter]}")
            ax3[tweezer_num, 2].plot(np.array(tweezer_att_list), y_centerpos_frame[tweezer_num, :, mol_freq_counter],
                                     label=f"{mol_freq_list[mol_freq_counter]}")
            ax3[tweezer_num, 3].plot(np.array(tweezer_att_list), y_radius_frame[tweezer_num, :, mol_freq_counter],
                                     label=f"{mol_freq_list[mol_freq_counter]}")

        ax1[0].plot(np.array(tweezer_freq_list), x_centerpos_1)
        ax1[1].plot(np.array(tweezer_freq_list), x_radius_1)
        ax1[2].plot(np.array(tweezer_freq_list), y_centerpos_1)
        ax1[3].plot(np.array(tweezer_freq_list), y_radius_1)
            # for mol_freq_counter in range(len(mol_freq_list)):
            #     ax4[tweezer_num, 0].plot(np.array(tweezer_att_list),
            #                              x_centerpos_4[tweezer_num, :, mol_att_counter, mol_freq_counter],
            #                              label=f"{mol_att_list[mol_att_counter]},mol_f={mol_freq_list[mol_freq_counter]}")
            #     ax4[tweezer_num, 1].plot(np.array(tweezer_att_list),
            #                              x_radius_4[tweezer_num, :, mol_att_counter, mol_freq_counter],
            #                              label=f"{mol_att_list[mol_att_counter]},mol_f={mol_freq_list[mol_freq_counter]}")
            #     ax4[tweezer_num, 2].plot(np.array(tweezer_att_list),
            #                              y_centerpos_4[tweezer_num, :, mol_att_counter, mol_freq_counter],
            #                              label=f"{mol_att_list[mol_att_counter]},mol_f={mol_freq_list[mol_freq_counter]}")
            #     ax4[tweezer_num, 3].plot(np.array(tweezer_att_list),
            #                              y_radius_4[tweezer_num, :, mol_att_counter, mol_freq_counter],
            #                              label=f"{mol_att_list[mol_att_counter]},mol_f={mol_freq_list[mol_freq_counter]}")

        # ax3[tweezer_num, 0].plot(np.array(tweezer_att_list), x_centerpos_2[tweezer_num, :], label=f"sum", c='black')
        # ax3[tweezer_num, 1].plot(np.array(tweezer_att_list), x_radius_2[tweezer_num, :], label=f"sum", c='black')
        # ax3[tweezer_num, 2].plot(np.array(tweezer_att_list), y_centerpos_2[tweezer_num, :], label=f"sum", c='black')
        # ax3[tweezer_num, 3].plot(np.array(tweezer_att_list), y_radius_2[tweezer_num, :], label=f"sum", c='black')
        # ax3[tweezer_num, 3].legend(title='mol_frequency', bbox_to_anchor=(1.35, 1), loc='upper right')
        # ax4[tweezer_num, 0].plot(np.array(tweezer_att_list), x_centerpos_2[tweezer_num, :], label=f"sum", c='black')
        # ax4[tweezer_num, 1].plot(np.array(tweezer_att_list), x_radius_2[tweezer_num, :], label=f"sum", c='black')
        # ax4[tweezer_num, 2].plot(np.array(tweezer_att_list), y_centerpos_2[tweezer_num, :], label=f"sum", c='black')
        # ax4[tweezer_num, 3].plot(np.array(tweezer_att_list), y_radius_2[tweezer_num, :], label=f"sum", c='black')
        # ax4[tweezer_num, 3].legend(title='molasses intensity', bbox_to_anchor=(1.35, 1), loc='upper right')

    fig1.savefig('tweezers_postition_data_vs_mol_att', dpi=100)
    fig3.savefig('tweezers_postition_data_vs_mol_freq', dpi=100)
    # fig4.savefig('tweezers_postition_data_vs_all', dpi=300)


################################################################################################


# wait = input("Press enter to continue")



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
            # print(f"{point_num}/{num_points}")
            v_center = y_centerpos[tweezer_num, point_num]
            h_center = x_centerpos[tweezer_num, point_num]
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

