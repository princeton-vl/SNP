#!/bin/bash

mod='anim_0' # try the batch script
mod='anim_1' # try the batch script

dataset="TNT_Masked"
pointcloud_dir="./saved_pointclouds/TNT"
scene=$1 # command line input

crop_h=1280
crop_w=2176

if [ "$scene" == "Ignatius" ]
then
  restore_ckpt='./saved_checkpoints/TNT/r1_dTNT_Masked_full_01_Ignatius_b1_ll1_lr1e-4_lrf1e-2_lro1e-4_fo1_r1e-3_g1e-3_s2d1_pd0.5_dimf288_cnn0_so32_bsSH_sasimple_unet_knn0_snnone_fs0.01_aff0.pth'
  radius=1e-3
elif [ "$scene" == "Truck" ]
then
  restore_ckpt='./saved_checkpoints/TNT/r1_dTNT_Masked_full_01_Truck_b1_ll1_lr1e-4_lrf1e-2_lro1e-4_fo1_r5e-3_g1e-3_s2d1_pd0.5_dimf288_cnn0_so32_bsSH_sasimple_unet_knn0_snnone_fs0.01_aff0.pth'
  radius=5e-3
  # radius=1e-2
elif [ "$scene" == "Family" ]
then
  radius=1e-3
  restore_ckpt='./saved_checkpoints/TNT/r1_dTNT_Masked_full_01_Family_b1_ll1_lr1e-4_lrf1e-2_lro1e-4_fo1_r1e-3_g1e-3_s2d1_pd0.5_dimf288_cnn0_so32_bsSH_sasimple_unet_knn0_snnone_fs0.01_aff0.pth'
elif [ "$scene" == "Caterpillar" ]
then
  restore_ckpt='./saved_checkpoints/TNT/r1_dTNT_Masked_full_01_Caterpillar_b1_ll1_lr1e-4_lrf1e-2_lro1e-4_fo1_r5e-3_g1e-3_s2d1_pd0.5_dimf288_cnn0_so32_bsSH_sasimple_unet_knn0_snnone_fs0.01_aff0.pth'
  radius=5e-3
elif [ "$scene" == "Barn" ]
then
  restore_ckpt='./saved_checkpoints/TNT/r1_dTNT_Masked_full_01_Barn_b1_ll1_lr1e-4_lrf1e-2_lro1e-4_fo1_r1e-2_g1e-3_s2d1_pd0.5_dimf288_cnn0_so32_bsSH_sasimple_unet_knn0_snnone_fs0.01_aff0.pth'
  radius=1e-2
else
  echo "unsupported dataset"
  exit 1
fi

max_num_pts=1000000

rasterize_rounds=2


free_xyz=0
free_opy=1
free_rad=0
cnn_feat=0
gamma=1e-3
loss='l1' # works best
batch_size=1
do_xyz_pos_encode=0
shader_arch='simple_unet'
knn_3d_smoothing=0
shader_norm='none' # works best
feat_smooth_loss_coeff=0.01
do_random_affine=0

lr=1e-4
lr_feat=1e-2
lr_opy=1e-4

### debug
#render_scale=1
#do_2d_shading=0
#pts_dropout_rate=0.0
#num_steps=2000
#img_log_freq=50
#VAL_FREQ=5000
#dim_pointfeat=27
#shader_output_channel=3
#basis_type='SH'


# round 1. do the final training with all components
render_scale=2
do_2d_shading=1
pts_dropout_rate=0.5
num_steps=50000
img_log_freq=500
VAL_FREQ=2000
dim_pointfeat=288
shader_output_channel=32
basis_type='SH'

name_base="d${dataset}_${mod}_${scene}_b${batch_size}_l${loss}_lr${lr}_lrf${lr_feat}_lro${lr_opy}_fo${free_opy}_r${radius}_g${gamma}_s2d${do_2d_shading}_pd${pts_dropout_rate}_dimf${dim_pointfeat}_cnn${cnn_feat}_so${shader_output_channel}_bs${basis_type}_sa${shader_arch}_knn${knn_3d_smoothing}_sn${shader_norm}_fs${feat_smooth_loss_coeff}_aff${do_random_affine}"
name1="r1_${name_base}"
CUDA_VISIBLE_DEVICES=0 python TNT_Masked_pcloud_pulsar_train.py --setting "$dataset"  --crop_h $crop_h --crop_w $crop_w \
--resize_h $crop_h --resize_w $crop_w \
--name "$name1" \
--batch_size $batch_size --SUM_FREQ 100 \
--tb_log_dir ./tb \
--num_steps $num_steps --img_log_freq $img_log_freq --VAL_FREQ $VAL_FREQ \
--single "$scene" \
--HR 1 \
--pointcloud_dir $pointcloud_dir \
--fe_loss_type $loss --knn_3d_smoothing $knn_3d_smoothing --feat_smooth_loss_coeff $feat_smooth_loss_coeff --do_random_affine $do_random_affine \
--free_xyz $free_xyz --free_opy $free_opy --blend_gamma $gamma --sphere_radius $radius --lr $lr \
--render_scale $render_scale --do_2d_shading $do_2d_shading --shader_arch $shader_arch --pts_dropout_rate $pts_dropout_rate \
--dim_pointfeat $dim_pointfeat --do_xyz_pos_encode $do_xyz_pos_encode --cnn_feat $cnn_feat \
--shader_output_channel $shader_output_channel --basis_type $basis_type --shader_norm $shader_norm \
--special_args_dict "vert_feat:${lr_feat},vert_opy:${lr_opy}" \
--max_num_pts $max_num_pts --rasterize_rounds $rasterize_rounds \
--render_only 1 --restore_ckpt $restore_ckpt