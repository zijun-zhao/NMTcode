#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import subprocess
import shutil
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
import pandas as pd
import subprocess
import shutil
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial

import random
import ast
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
import time

folder_path = 'F:\\test1Dec'


def single_brain_log(brain_regis, path, input_sub_brain):
    (b_info, ext) = os.path.splitext(input_sub_brain)
    (b_info, ext) = os.path.splitext(b_info)
    (b_info, ext) = os.path.splitext(b_info)
    b_info = b_info.split("_")[0]
    sub_path = os.path.join(path, b_info)
    exist_ccfbrain = os.path.exists(os.path.join(path, 'average_template_25_u8_xpad.v3draw'))
    if not exist_ccfbrain:
        print('Make sure the CCF standard brain template exists.')
    path_ccfbrain = os.path.join(path, 'average_template_25_u8_xpad.v3draw')

    ##
    exist_exe_affine = os.path.exists(
        os.path.join(os.path.join(path, 'second_affine'), 'main_warp_from_affine.exe'))
    if not exist_exe_affine:
        print('Make sure the .exe to perform affinement exists')
    path_exe_affine = os.path.join(os.path.join(path, 'second_affine'), 'main_warp_from_affine.exe')

    exist_manaulM = os.path.exists(os.path.join(os.path.join(path, 'second_affine'), 'Manual_marker'))
    if not exist_manaulM:
        print('Make sure the folder containing all manually-labelled marker files exists.')
    path_manaulM = os.path.join(os.path.join(path, 'second_affine'), 'Manual_marker')

    exist_ccfM = os.path.exists(os.path.join(os.path.join(path, 'second_affine'), 'CCF_marker'))
    if not exist_ccfM:
        print('Make sure the folder containing marker inside CCF exists.')
    path_ccfM = os.path.join(os.path.join(path, 'second_affine'), 'CCF_marker')

    exist_stripmove = os.path.exists(os.path.join(os.path.join(path, 'second_affine'), 'stripMove'))
    if not exist_stripmove:
        print('Make sure the folder containing CCF brain of removing strips exists.')
    path_stripm = os.path.join(os.path.join(path, 'second_affine'), 'stripMove')

    ##
    exist_exe_warp = os.path.exists(
        os.path.join(os.path.join(path, 'third_warp_swc'), 'main_warp_from_df_new.exe'))
    if not exist_exe_warp:
        print('Make sure the .exe to perform warp exists')
    path_exe_warp = os.path.join(os.path.join(path, 'third_warp_swc'), 'main_warp_from_df_new.exe')

    exist_afBrain = os.path.exists(os.path.join(os.path.join(path, 'third_warp_swc'), 'affined_brain'))
    if not exist_afBrain:
        print('Make sure the folder containing affined brains exists.')
    path_af_brain = os.path.join(os.path.join(path, 'third_warp_swc'), 'affined_brain')

    exist_auto_brain_m = os.path.exists(os.path.join(os.path.join(path, 'third_warp_swc'), 'brain_auto_marker'))
    if not exist_auto_brain_m:
        print('Make sure the folder containing auto generated marker for all brains exists.')
    path_autoBrain_m = os.path.join(os.path.join(path, 'third_warp_swc'), 'brain_auto_marker')

    exist_auto_ccf_m = os.path.exists(os.path.join(os.path.join(path, 'third_warp_swc'), 'CCF_auto_marker'))
    if not exist_auto_ccf_m:
        print('Make sure the folder containing auto generated marker for CCF brains exists.')
    path_autoCCF_m = os.path.join(os.path.join(path, 'third_warp_swc'), 'CCF_auto_marker')

    exist_ssd = os.path.exists(os.path.join(os.path.join(path, 'third_warp_swc'), 'ssd_grid'))
    if not exist_ssd:
        print('Make sure the folder containing grid. swc generated using SSD algorithm.')
    path_ssdgrid = os.path.join(os.path.join(path, 'third_warp_swc'), 'ssd_grid')


    print(" Process brain %s" % (b_info))

    folder_ori = os.path.join(os.path.join(sub_path, input_sub_brain), 'ori')
    exist_ori = os.path.exists(folder_ori)
    if not exist_ori:
        os.makedirs(folder_ori)
    for iter_check in os.listdir(os.path.join(sub_path, input_sub_brain)):
        if iter_check.endswith(("swc", "SWC")):
            shutil.move(os.path.join(os.path.join(sub_path, input_sub_brain), iter_check), folder_ori)
            print('Move ' + str(iter_check) + ' to folder ori')
    folder_affine = os.path.join(os.path.join(sub_path, input_sub_brain), 'affine')
    exist_affine = os.path.exists(folder_affine)
    if not exist_affine:
        print('Creat new folder' + str(folder_affine))
        os.makedirs(folder_affine)
    folder_resample = os.path.join(os.path.join(sub_path, input_sub_brain), 'resample')
    exist_resample = os.path.exists(folder_resample)
    if not exist_resample:
        print('Creat new folder ' + str(folder_resample))
        os.makedirs(folder_resample)

    folder_stps = os.path.join(os.path.join(sub_path, input_sub_brain), 'stps')
    exist_stps = os.path.exists(folder_stps)
    if not exist_stps:
        print('Creat new folder ' + str(folder_stps))
        os.makedirs(folder_stps)
        # specific downsample size for the input_sub_brain

    x_ds_size = brain_regis.loc[int(b_info), 'x_downsample']
    y_ds_size = brain_regis.loc[int(b_info), 'y_downsample']
    z_ds_size = brain_regis.loc[int(b_info), 'z_downsample']

    for iter_swc in os.listdir(folder_ori):
        if not iter_swc.endswith(("swc", "SWC")):
            continue
        (neuron_info, extension) = os.path.splitext(iter_swc)
        print('Neuron information: ' + str(neuron_info))
        path_old = os.path.join(folder_ori, iter_swc)
        path_new = os.path.join(folder_resample, neuron_info) + '.' + 'x' + str(int(round(x_ds_size, 0))) + 'y' + str(
            int(round(y_ds_size, 0))) + 'z' + str(int(round(z_ds_size, 0))) + str(extension)
        n_skip = 0
        with open(path_old, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    n_skip += 1
                else:
                    break
        names = ["##n", "type", "x", "y", "z", "r", "parent"]
        swc = pd.read_csv(path_old, index_col=0, skiprows=n_skip, sep=" ",
                          usecols=[0, 1, 2, 3, 4, 5, 6],
                          names=names
                          )
        swc.loc[:, 'x'] = swc.loc[:, 'x'] / round(x_ds_size, 1)
        swc.loc[:, 'y'] = swc.loc[:, 'y'] / round(y_ds_size, 1)
        swc.loc[:, 'z'] = swc.loc[:, 'z'] / round(z_ds_size, 1)
        swc = swc.round({'x': 4, 'y': 4, 'z': 4})
        assert (max(swc.loc[:, 'x'])) < brain_regis.loc[int(b_info), 'x_after'], " x has exceeds the boundary"
        assert (max(swc.loc[:, 'y'])) < brain_regis.loc[int(b_info), 'y_after'], " y has exceeds the boundary"
        assert (max(swc.loc[:, 'z'])) < brain_regis.loc[int(b_info), 'z_after'], " z has exceeds the boundary"

        swc.to_csv(path_new, sep=" ")
        print('\n-----DOWNSAMPLE-------')
        print('Saving current file to ' + str(path_new))

    print('\n-----Affinement-------')
    subtime_affine = time.time()
    print('Obtaining three files for brain ' + str(b_info) + ' for affinement:')

    for iter_s in os.listdir(path_stripm):
        if iter_s.startswith(b_info):
            path_rm_strip = os.path.join(path_stripm, iter_s)
            print('\tObtained .v3draw of strips-removed version for brain ' + str(b_info))
            break
    # Record manually-labeled file for specific brain
    # Record size
    for iter_m in os.listdir(path_manaulM):
        if iter_m.startswith(b_info):
            path_manual_m = os.path.join(path_manaulM, iter_m)
            print('\tObtained manually labelled marker file for brain ' + str(b_info))
            (brain_info, extension) = os.path.splitext(iter_m)
            affine_size = (brain_info.split('_')[-1])
            break

    # Record standard-labeled file for specific brain
    for iter_c in os.listdir(path_ccfM):
        if str(affine_size) in iter_c:
            path_ccf_m = os.path.join(path_ccfM, iter_c)
            print('\tObtained standard-labelled marker file for brain ' + str(b_info))
            break

    if os.path.exists(path_exe_affine) != 1:
        print('Make sure the .exe to perform affinement exists')
    if os.path.exists(path_ccfbrain) != 1:
        print('Make sure the CCF standard brain template exists.')
    if os.path.exists(path_rm_strip) != 1:
        print('Make sure the folder containing CCF brain of removing strips exists.')
    if os.path.exists(path_ccf_m) != 1:
        print('Make sure the folder containing marker inside CCF exists.')
    if os.path.exists(path_manual_m) != 1:
        print('Make sure the folder containing all manually-labelled marker files exists.')
    if os.path.exists(path_manual_m) != 1:
        print('Make sure the folder containing all manually-labelled marker files exists.')
    assert os.path.exists(path_exe_affine) == 1 and os.path.exists(path_ccfbrain) == 1 and os.path.exists(
        path_rm_strip) == 1 and os.path.exists(path_ccf_m) == 1 and os.path.exists(
        path_manual_m) == 1 and os.path.exists(path_manual_m) == 1, "Check the corresponding document"
    path_op_affine = os.path.join(folder_affine, '%%~ni_affine.swc')
    with open(os.path.join(os.path.join(sub_path,input_sub_brain), 'affine_swc.bat'), 'w') as OPATH:
        OPATH.writelines(['\n',
                          'set input_path=' + str(folder_resample),
                          '\n',
                          'for /r %input_path% %%i in (*.swc) do (',
                          '\n',
                          str(path_exe_affine) + ' -t ' + str(path_ccfbrain) + ' -s ' + str(path_rm_strip) +
                          ' -T ' + str(path_ccf_m) + ' -S ' + str(path_manual_m) + '  -w %%i -o ' + str(path_op_affine),
                          '\n',
                          '\n',
                          ')',
                          '\n'])
    filepath_Aff = os.path.join(os.path.join(sub_path,input_sub_brain),  'affine_swc.bat')
    print('Run code at ' + str(filepath_Aff))
    os.system(filepath_Aff)
    assert len(os.listdir(folder_affine)) == len(
        os.listdir(folder_resample)), "Note that affined swc is not equal to resampled swc!"
    sub_elapsed_affine = time.time() - subtime_affine
    print(" Time used for affine subbrain " + str(input_sub_brain) + ' is %.2f ' % sub_elapsed_affine)
    print('Have finished affinement')

    print('-----WRAP-----')

    subtime_warp = time.time()
    print('Obtaining four files for brain ' + str(b_info) + ' for warp:')
    for iter_d in os.listdir(path_af_brain):
        if iter_d.startswith(b_info):
            path_af_input_sub_brain = os.path.join(path_af_brain, iter_d)
            print('\tObtained affined version for brain ' + str(b_info))
            break
    for iter_ssd in os.listdir(path_ssdgrid):
        if iter_ssd.startswith(b_info):
            path_ssd = os.path.join(path_ssdgrid, iter_ssd)
            print('\tObtained SSD result for brain ' + str(b_info))
            break
    for iter_cm in os.listdir(path_autoCCF_m):
        if iter_cm.startswith(b_info):
            path_ccfAT_m = os.path.join(path_autoCCF_m, iter_cm)
            print('\tObtained auto integrated marker for CCF brain ')
            break
    for iter_bm in os.listdir(path_autoBrain_m):
        if iter_bm.startswith(b_info):
            path_brainAT_m = os.path.join(path_autoBrain_m, iter_bm)
            print('\tObtained auto integrated marker for brain ' + str(b_info))
            break

    path_op_warp = os.path.join(folder_stps, '%%~ni_stps.swc')
    # directory = 'C:/'
    with open(os.path.join(os.path.join(sub_path,input_sub_brain), 'warp_swc.bat'), 'w') as OPATH_2:
        OPATH_2.writelines(['\n',
                            'set input_path=' + str(folder_affine),
                            '\n',
                            'for /r %input_path% %%i in (*.swc) do (',
                            '\n',
                            str(path_exe_warp) + ' -t ' + str(path_ccfbrain) + ' -s ' + str(path_af_input_sub_brain) +
                            ' -g ' + str(path_ssd) + ' -T ' + str(path_ccfAT_m) + ' -S ' + str(
                                path_brainAT_m) + ' -w %%i  -o ' + str(path_op_warp),
                            '\n',
                            '\n',
                            ')',
                            '\n'])
    filepath_Warp = os.path.join(os.path.join(sub_path,input_sub_brain),  'warp_swc.bat')
    print('Run code at ' + str(filepath_Warp))
    #os.system(filepath_Warp)
    print(" Process brain %s Finished." % b_info)
    sub_elapsed_warp = time.time() - subtime_warp
    print(" Time used for warpping subbrain " + str(input_sub_brain) + ' is %.2f ' % sub_elapsed_warp)

    print('Have finished Warp')


if __name__ == '__main__':
    start = time.time()
    exist_table = os.path.exists(os.path.join(folder_path, 'brain_registration.xlsx'))
    if not exist_table:
        print('Make sure the table containing brain downsample information exists.')
    brain_regisTB = pd.read_excel(os.path.join(folder_path, 'brain_registration.xlsx'), index_col=[0], skiprows=[0],
                                  names=['ID', 'y_initial', 'x_initial', 'z_initial', 'y_after', 'x_after', 'z_after',
                                         'ration', 'flip', 'flip_axis'])
    brain_regisTB.loc[:, 'x_downsample'] = brain_regisTB.loc[:, 'x_initial'] / brain_regisTB.loc[:, 'x_after']
    brain_regisTB.loc[:, 'y_downsample'] = brain_regisTB.loc[:, 'y_initial'] / brain_regisTB.loc[:, 'y_after']
    brain_regisTB.loc[:, 'z_downsample'] = brain_regisTB.loc[:, 'z_initial'] / brain_regisTB.loc[:, 'z_after']
    brain_regisTB.drop_duplicates(keep='first', inplace=True)
    brain_regisTB.fillna(value='NA', inplace=True)
    brain_regisTB = brain_regisTB.loc[~brain_regisTB.index.duplicated(keep='first')]
    brain_regisTB.loc[:, 'x_initial'] = brain_regisTB.loc[:, 'x_initial'].astype(float)
    brain_regisTB.loc[:, 'y_initial'] = brain_regisTB.loc[:, 'y_initial'].astype(float)
    brain_regisTB.loc[:, 'z_initial'] = brain_regisTB.loc[:, 'z_initial'].astype(float)
    brain_folder_list = []


    for subpath in os.listdir(folder_path):
        if not (subpath) in list(map(str, brain_regisTB.index.tolist())):
            print("Skip file " + str(subpath.split('/')[-1]))
            continue
        brain_folder_list.append(subpath)
        print('Find new brain folder ' + str(subpath))
    print('ID of brains to be proceed: ' + str(brain_folder_list))
    cores = int(multiprocessing.cpu_count() * 0.6)  # multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    for iter_brain in brain_folder_list:
        ## First count how many .swc under the brain_id
        iter_f_idx = 0
        count_move = 0
        for iter_check in os.listdir(os.path.join(folder_path, iter_brain)):
            if iter_check.endswith(("swc", "SWC")):
                if count_move == 6:
                    iter_f_idx = 0
                    count_move = 0
                count_move = count_move + 1
                iter_f_idx = iter_f_idx + 1
                exist_creat = os.path.exists(
                    os.path.join(os.path.join(folder_path, iter_brain), iter_brain + '_' + str(iter_f_idx)))
                if not exist_creat:
                    print('Creat folder ' + str(os.path.join(os.path.join(folder_path, iter_brain),
                                                             iter_brain + '_' + str(iter_f_idx))) + ' under ' + str(
                        iter_brain))
                    os.makedirs(os.path.join(os.path.join(folder_path, iter_brain), iter_brain + '_' + str(iter_f_idx)))
                shutil.move(os.path.join(os.path.join(folder_path, iter_brain), iter_check),
                            os.path.join(os.path.join(folder_path, iter_brain), iter_brain + '_' + str(iter_f_idx)))
    for iter_brain in brain_folder_list:
        for iter_sub_brain in os.listdir(os.path.join(folder_path, iter_brain)):
            print('Parallel running the subfolder ' + str(iter_sub_brain))
            #pool.apply_async(single_brain_log, (brain_regisTB, folder_path, iter_sub_brain))
            single_brain_log(brain_regisTB, folder_path, iter_sub_brain)
    pool.close()
    pool.join()
    elapsed = (time.time() - start)
    print("Used %.2f to run all the brian" % elapsed)






