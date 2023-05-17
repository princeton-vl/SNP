mod='release_00' # add noise on the model

scene=$1 # command line input

## debug setting
#render_scale=1
#do_2d_shading=0
#pts_dropout_rate=0.0
#num_steps=2000
#img_log_freq=100
#VAL_FREQ=1000
#dim_pointfeat=27
#shader_output_channel=3
#basis_type='SH'


# round0 setting
#do_2d_shading=0
#pts_dropout_rate=0.0
#num_steps=5000
#img_log_freq=50
#VAL_FREQ=500
#dim_pointfeat=27
#shader_output_channel=3
#basis_type='SH'

# round1 setting
render_scale=2
do_2d_shading=1
pts_dropout_rate=0.5
num_steps=50000
img_log_freq=500
VAL_FREQ=2000
dim_pointfeat=288
shader_output_channel=32
basis_type='SH'
val_cam_noise=0.0


# point-add setting
# --eval_only 1 --restore_ckpt '/n/fs/pvl-viewsyn/cer_mvs/saved_checkpoints/sigma08_scan48_b1_llinear_l2_lr1e-4_lrf1e-2_lro1e-4_fo1_r1.5e-3_g1e-3_s2d1_pd0.5_dimf256_cnn0.pth'


# pts_dropout_rate=0.5
gamma=1e-3
loss='l1'
batch_size=1
radius=7.5e-3
shader_arch='simple_unet'
shader_norm='none' # works best
free_opy=1
max_num_pts=500000

lr=1e-4
lr_feat=1e-2
lr_opy=1e-4

pointcloud_dir="./saved_pointclouds/SYNMVS"
restore_ckpt="./saved_checkpoints/SYNMVS/iclr_03_${scene}_b1_ll1_lr1e-4_lrf1e-2_lro1e-4_r7.5e-3_g1e-3_s2d1_pd0.5_dimf288_so32_bsSH_sasimple_unet_pts500000.pth"

CUDA_VISIBLE_DEVICES=0 python SYNMVS_pcloud_pulsar_train.py --setting SYNMVS --crop_h 800 --crop_w 800 \
--resize_h 800 --resize_w 800 \
--name "${mod}_${scene}_noise${val_cam_noise}_b${batch_size}_l${loss}_lr${lr}_lrf${lr_feat}_lro${lr_opy}_r${radius}_g${gamma}_s2d${do_2d_shading}_pd${pts_dropout_rate}_dimf${dim_pointfeat}_so${shader_output_channel}_bs${basis_type}_sa${shader_arch}_pts${max_num_pts}" \
--batch_size $batch_size --SUM_FREQ 100 \
--tb_log_dir ./tb \
--num_steps $num_steps --img_log_freq $img_log_freq --VAL_FREQ $VAL_FREQ \
--single $scene \
--HR 1 --free_opy $free_opy \
--pointcloud_dir $pointcloud_dir \
--fe_loss_type $loss --blend_gamma $gamma --sphere_radius $radius --lr $lr \
--render_scale $render_scale --do_2d_shading $do_2d_shading --shader_arch $shader_arch --pts_dropout_rate $pts_dropout_rate \
--dim_pointfeat $dim_pointfeat \
--shader_output_channel $shader_output_channel --basis_type $basis_type \
--special_args_dict "vert_feat:${lr_feat},vert_opy:${lr_opy}" \
--max_num_pts $max_num_pts \
--eval_only 1 --restore_ckpt $restore_ckpt --val_cam_noise $val_cam_noise
# --anim_only 1 --restore_ckpt $restore_ckpt
