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
from e6dataflow.utils import ROI_fit
from e6dataflow.processor import MultiCountsProcessor, ThresholdProcessor
import matplotlib.patches as patches


plt.close()

data_root = Path('Y:/', 'expdata-e6/data')
daily_subpath = Path(*Path.cwd().parent.parts[-3:])
daily_path = Path(data_root, daily_subpath)
# daily_path = Path(Path.cwd().parent)

run_name = 'run1'
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
# Load Data for Auto ROI
data_path = daily_path / 'data' / run_name / 'High NA Imaging'
frames_array = np.zeros([num_points, num_frames, 350, 50])
num_shots = 8000
for shot_num in range(num_shots):
    file_name = 'jkam_capture_' + str(shot_num).zfill(5) + '.h5'
    hf = h5py.File(data_path / file_name, 'r')
    point = shot_num % num_points
    for frame_num in range(num_frames):
        photo = np.array(hf.get('frame-' + str(frame_num).zfill(2)))
        frames_array[point, frame_num] += photo
    hf.close

    # print(shot_num)
frames_array = frames_array / num_shots * num_points
fit_frame_array = np.mean(frames_array[:,2:-2,:,:],axis=1) # averaging all frames; could also pick one, ie frames_array[:,n,:,:]


# Build ROI Guess Array
tweezer_00_vert_center_init = 81.15# starting position of tweezer 0 = 108MHz
tweezer_00_horiz_center_init = 21.2
tweezer_horiz_span_init = 20
tweezer_vert_span_init = 14
tweezer_vert_spacing_init = 27.8*2 #2 is freq diff between neighboring tweezers on this run
tweezer_horiz_spacing_init = 1.3867*2
roi_guess_array = np.zeros([num_points, num_tweezers], dtype=object)
for tweezer in range(num_tweezers):
        for point_num in range(num_points):
            roi_guess_array[point_num, tweezer] = \
                make_centered_roi(vert_center=tweezer_00_vert_center_init + tweezer * tweezer_vert_spacing_init,
                                  horiz_center=tweezer_00_horiz_center_init + tweezer * tweezer_horiz_spacing_init,
                                  vert_span=tweezer_vert_span_init,
                                  horiz_span=tweezer_horiz_span_init)

# Do fitting
result, roi_final_array = ROI_fit(fit_frame_array, roi_guess_array, iterations=2, quiet=False, roi_final_shape=(12,14))

#
# x_centerpos = np.zeros((num_tweezers,num_points))
# y_centerpos = np.zeros((num_tweezers,num_points))
# x_centerpos_3 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list)))
# y_centerpos_3 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list)))
# x_radius_3 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list)))
# y_radius_3 = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list)))
# x_centerpos_2 = np.zeros((num_tweezers,len(pzt_para_list)))
# y_centerpos_2 = np.zeros((num_tweezers,len(pzt_para_list)))
# x_radius_2 = np.zeros((num_tweezers,len(pzt_para_list)))
# y_radius_2 = np.zeros((num_tweezers,len(pzt_para_list)))
# ROI_init_array = np.zeros((num_tweezers,len(pzt_para_list),len(probe_att_list), tweezer_vert_span_init, tweezer_horiz_span_init))
# for point in range(num_points):
#     for tweezer in range(num_tweezers):
#         ROI_init = np.zeros((tweezer_vert_span_init, tweezer_horiz_span_init))
#         for frame_num in range(num_frames):
#             photo = frames_array[point, frame_num]
#             ROI_init += photo[slice2d]
#         ROI_init = ROI_init - np.amin(ROI_init)
#         ROI_init_array[tweezer,pzt_counter,probe_att_counter] = ROI_init
#         fit_struct = e6fit.fit_gaussian2d(ROI_init, show_plot =  False)
#         x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
#         sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
#         # axs[point, tweezer].set_title(f"tweezer{tweezer}, point{point}", fontsize=10)
#         # axs[point, tweezer].imshow(ROI_init  , norm=colors.Normalize(vmin=np.amin(ROI_init), vmax=np.amax(ROI_init)), cmap='hot', interpolation='nearest')
#         # rect = patches.Rectangle((x0-sx,y0-sy), 2*sx,2*sy,linewidth=1,edgecolor='blue',facecolor='none')
#         # axs[point, tweezer].add_patch(rect)
#         # x_centerpos[tweezer,point] = tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init +  x0 - tweezer_horiz_span_init/2
#         # y_centerpos[tweezer,point] = tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + y0 - tweezer_vert_span_init/2
#
#         x_centerpos_3[tweezer,pzt_counter,probe_att_counter] = tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init +  x0 - tweezer_horiz_span_init/2
#         y_centerpos_3[tweezer,pzt_counter,probe_att_counter] = tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + y0 - tweezer_vert_span_init/2
#         x_radius_3[tweezer,pzt_counter,probe_att_counter] = sx
#         y_radius_3[tweezer,pzt_counter,probe_att_counter] = sy
#         # print([x0,y0,sx,sy])
# # fig.savefig('tweezers_raw_data',dpi=300)
# for tweezer in range(num_tweezers):
#     for pzt_counter in range(len(pzt_para_list)):
#         fit_struct = e6fit.fit_gaussian2d(np.mean(ROI_init_array[tweezer,pzt_counter,:],axis=0), show_plot=False)
#         x0, y0 = [fit_struct['x0']['val'], fit_struct['y0']['val']]
#         sx, sy = [fit_struct['sx']['val'], fit_struct['sy']['val']]
#         x_centerpos_2[tweezer,pzt_counter] = tweezer_00_horiz_center_init + tweezer*tweezer_horiz_spacing_init +  x0 - tweezer_horiz_span_init/2
#         y_centerpos_2[tweezer,pzt_counter] = tweezer_00_vert_center_init + tweezer*tweezer_vert_spacing_init + y0 - tweezer_vert_span_init/2
#         x_radius_2[tweezer,pzt_counter] = sx
#         y_radius_2[tweezer,pzt_counter] = sy
#         print(f"tweezer{tweezer_freq_list[tweezer]}, pzt_counter={pzt_counter} - x0={x0:.03f},y0={y0:.03f},sx={sx:.03f},sy={sy:.03f}")
#         for inner_counter in range(num_inner_loop):
#             x_centerpos[tweezer, pzt_counter*num_inner_loop+inner_counter] = tweezer_00_horiz_center_init + tweezer * tweezer_horiz_spacing_init + x0 - tweezer_horiz_span_init / 2
#             y_centerpos[tweezer, pzt_counter*num_inner_loop+inner_counter] = tweezer_00_vert_center_init + tweezer * tweezer_vert_spacing_init + y0 - tweezer_vert_span_init / 2
#
# fig2,ax2 = plt.subplots(num_tweezers, 4 ,figsize=(20, 5*num_tweezers))
# for tweezer_num in range(num_tweezers):
#     ax2[tweezer_num, 0].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_center")
#     ax2[tweezer_num, 1].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, x_radius")
#     ax2[tweezer_num, 2].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_center")
#     ax2[tweezer_num, 3].set_title(f"tweezer{tweezer_freq_list[tweezer_num]}, y_radius")
#     for probe_att_counter in range(len(probe_att_list)):
#         ax2[tweezer_num, 0].plot(np.array(pzt_para_list), x_centerpos_3[tweezer_num,:,probe_att_counter],label = f"{probe_att_list[probe_att_counter]}")
#         ax2[tweezer_num, 1].plot(np.array(pzt_para_list), x_radius_3[tweezer_num,:,probe_att_counter],label = f"{probe_att_list[probe_att_counter]}")
#         ax2[tweezer_num, 2].plot(np.array(pzt_para_list), y_centerpos_3[tweezer_num,:,probe_att_counter],label = f"{probe_att_list[probe_att_counter]}")
#         ax2[tweezer_num, 3].plot(np.array(pzt_para_list), y_radius_3[tweezer_num,:,probe_att_counter],label = f"{probe_att_list[probe_att_counter]}")
#
#     ax2[tweezer_num, 0].plot(np.array(pzt_para_list), x_centerpos_2[tweezer_num, :],  label=f"sum", c='black')
#     ax2[tweezer_num, 1].plot(np.array(pzt_para_list), x_radius_2[tweezer_num, :],  label=f"sum", c='black')
#     ax2[tweezer_num, 2].plot(np.array(pzt_para_list), y_centerpos_2[tweezer_num, :],  label=f"sum", c='black')
#     ax2[tweezer_num, 3].plot(np.array(pzt_para_list), y_radius_2[tweezer_num, :],  label=f"sum", c='black')
#     ax2[tweezer_num, 3].legend(title='probe intensity', bbox_to_anchor=(1.35, 1), loc='upper right')
#
# fig2.savefig('tweezers_postition_data',dpi=300)
#
#
# ################################################################################################
#
#
# wait = input("Press enter to continue")
#
#
#
# tweezer_horiz_span = 14
# tweezer_vert_span = 12
#
# threshold_processor_list = []
# for frame_num in range(num_frames):
#     frame_key = f'frame-{frame_num:02d}'
#     frame_datafield = DataStreamDataField(name=frame_key, datastream_name='high NA Imaging', h5_subpath=None,
#                                     h5_dataset_name=frame_key)
#     datafield_list.append(frame_datafield)
#
#     avg_datafield = DataDictPointDataField(name=f'{frame_key}_avg')
#     datafield_list.append(avg_datafield)
#     img_avg_aggregator = AvgStdAggregator(name=f'{frame_key}_avg_aggregator', verifier_datafield_names=[],
#                                           input_datafield_name=frame_key,
#                                           output_datafield_name=f'{frame_key}_avg')
#     aggregator_list.append(img_avg_aggregator)
#
#     roi_array = np.zeros([num_points, num_tweezers], dtype=object)
#     multicounts_result_datafield_name_list = []
#
#     for tweezer_num in range(num_tweezers):
#         for point_num in range(num_points):
#             # print(f"{point_num}/{num_points}")
#             v_center = y_centerpos[tweezer_num, point_num]
#             h_center = x_centerpos[tweezer_num, point_num]
#             roi_array[point_num, tweezer_num] = \
#                 make_centered_roi(vert_center=v_center,
#                                     horiz_center=h_center,
#                                     vert_span=tweezer_vert_span,
#                                     horiz_span=tweezer_horiz_span)
#         new_counts_datafield = DataDictShotDataField(name=f'{frame_key}_tweezer-{tweezer_num:02d}_counts')
#         datafield_list.append(new_counts_datafield)
#         multicounts_result_datafield_name_list.append(new_counts_datafield.name)
#
#     counts_plot_reporter = PlotPointReporter(name=f'{frame_key}_counts_reporter',
#                                              datafield_name_list=multicounts_result_datafield_name_list,
#                                              layout=Reporter.LAYOUT_GRID, save_data=True, close_plots=True)
#     # reporter_list.append(counts_plot_reporter)
#     multicounts_processor = MultiCountsProcessor(name=f'{frame_key}_multicount_processor',
#                                                  frame_datafield_name=frame_key,
#                                                  result_datafield_name_list=multicounts_result_datafield_name_list,
#                                                  roi_slice_array=roi_array)
#     processor_list.append(multicounts_processor)
#
#
# avg_img_datafield_name_list = [f'frame-{frame_num:02d}_avg' for frame_num in range(num_frames)]
# image_point_reporter = ImagePointReporter(name='avg_frame_reporter',
#                                           datafield_name_list=avg_img_datafield_name_list,
#                                           layout=Reporter.LAYOUT_GRID,
#                                           save_data=True, close_plots=True, roi_slice_array=roi_array)
#
# datastream_list = [high_na_datastream]
# reporter_list += [image_point_reporter]
# datatool_list = datastream_list + datafield_list + processor_list + aggregator_list + reporter_list
# # datatool_list = reporter_list
# for datatool in datatool_list:
#     datamodel.add_datatool(datatool, overwrite=True, quiet=True)
# datamodel.link_datatools()
#
# datamodel.run(handler_quiet=True,save_every_shot=False,save_point_data=False,save_before_reporting=True)

