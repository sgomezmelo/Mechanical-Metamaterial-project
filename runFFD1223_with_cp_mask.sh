#/bin/bash

#basePath=/Users/santiagogomez/Downloads/realigned_p33
#dataPath=$basePath

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

basePath=/Users/blumberg/LocalDocuments
dataPath=/Users/blumberg/LocalDocuments/p33_realigned
paramPath=$basePath/eval_3
storePath=$basePath/eval_3

fmask=$dataPath/p33_mask.tif
param=$paramPath/ffdParameters.txt

#maskDir=$storePath/mask_out_s

#mkdir -p $storePath/mask_out_s
mkdir -p $storePath/elastix_out
mkdir -p $storePath/transformix_out
#mkdir -p $storePath/t1
#mkdir -p $storePath/t2

steps=(0 2 5 9)

fixed=$dataPath/p33_0um.tif

for k in 0 1 2
do
    i=${steps[k]}
    j=${steps[k+1]}
    echo i = $i um / j = $j um
    
    moving=$dataPath/p33_${j}um.tif
    outDir=$storePath/elastix_out/c${j}um
    dField=$storePath/transformix_out/c${j}um
    t0file=$storePath/elastix_out/c${i}um/TransformParameters.0.txt

    echo Fixed Image: $fixed
    echo Moving Image: $moving
    echo Fixed Mask: $fmask
    echo Parameters: $param

    rm -rf $outDir
    mkdir -p $outDir
    
    if [ $k -eq 0 ]
    then
        echo elastix -f $fixed -m $moving -fMask $fmask -p $param -out $outDir 
        elastix -f $fixed -m $moving -fMask $fmask -p $param -out $outDir
    else
        echo elastix -f $fixed -m $moving -fMask $fmask -p $param -out $outDir -t0 $t0file
        elastix -f $fixed -m $moving -fMask $fmask -p $param -out $outDir -t0 $t0file
    fi
    if [ $? -ne 0 ]
    then
        exit $?
    fi

    echo Generating deformation field and deformed mask
    rm -rf $dField
    mkdir -p $dField
    transformix  -def all -jacmat all -out $dField -tp $outDir/TransformParameters.0.txt
    #transformix  -def all -jacmat all -out $storePath/t1 -tp $outDir/TransformParameters.0.txt
    #transformix  -in $fmask -out $storePath/t2 -tp $outDir/TransformParameters.0.txt
    # transformix  -in $fmask -tp $outDir/TransformParameters.0.txt -out dField
    #transformix -in $fmask -out $dField -tp $outDir/TransformParameters.0.txt
    #python3 $SCRIPT_DIR/convert.py $dField/result.nrrd $mmask
    
    if [ ! -f $mmask ]
    then
        echo Mask creation failed
        exit 1
    fi
done


