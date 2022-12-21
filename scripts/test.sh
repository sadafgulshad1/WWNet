
function test_stl() {
    

    
    python /home/sadaf/Documents/PhD/Project_3/Builtin_robustness/test_stl10.py \
        --batch_size 32 \
        --epochs 200 \
        --optim sgd \
        --decay 0.0005 \
        --nesterov \
        --lr 0.05 \
        --lr_steps 60 80 100 150 \
        --lr_gamma 0.2 \
        --cuda \
        --test_epochs 50 100 150 200 "900|1000|20" \
        --model $1 \
        --save_model_path "/home/sadaf/Documents/PhD/Project_3/Builtin_robustness/scripts/$1_$2_$3_$4_$5_$6.pth" \
        --tag "test" \
        --data_dir="/home/sadaf/Documents/PhD/Project_3/Builtin_robustness/data" \
        --basis_alpha $2 \
        --basis_scale $3 \
        --basis_radius $4 \
        --basis_num_displacements $5 \
        --basis_mean $6 \


}

## To test a standard Resnet uncomment the line below
#test_stl "resnet152" 0 0 0 0 0 

## To test a Resnet with local elastic transform uncomment the line below 
test_stl "resnet_local_elastic_152" 1.6 1 0 4 0

## To test a Resnet with rotation scaling transform uncomment the line below 
# test_stl "resnet_rotation_scaling_152" 1.6 1 0 4 0

