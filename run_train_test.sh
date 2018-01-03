echo
echo python train_model.py $@ --both
echo
python train_model.py $@ --both

echo
echo python test_model.py $@
echo
python test_model.py $@
