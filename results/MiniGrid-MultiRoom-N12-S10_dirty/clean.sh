for d in torchbeast-* 
do 
if  test -f $d/model.tar
then echo 'ok'
else rm -r $d 
fi
done
