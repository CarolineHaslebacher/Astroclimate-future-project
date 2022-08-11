#!/bin/bash


# model='EC-Earth'



# loop through folders
for variable in ua va zg
do
    for model in CMCC ECMWF HadGEM

    do

    cd $(echo "/home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/variables/$variable/Amon/$model")

    echo $PWD

        for forcing in /home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/variables/$variable/Amon/$model/*
            do
                cd $(echo "$forcing")
                # cd "${RED_INSTANCE_NAME}"
                echo $PWD

                for wget_file in $forcing/*.sh

                    do
                        echo $PWD
                        # screen -S $model$variable -dm ./$wget_file
                        source $wget_file
                        
                    
                    done
                cd ..
            done
    done
done