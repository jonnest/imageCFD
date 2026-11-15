import os

# dir_path = '/mnt/data/deepFlow/volunteers/vol011'
dir_path = '/mnt/data/deepFlow/IA/pat016'
# dir_path = '/mnt/data/deepFlow/PT/pat009'

# manual_0429  omelette1_0429  omelette2_0429
model = 'manual_0429'

mode = 2

img = os.path.join(dir_path, 'img')
seg = os.path.join(dir_path, 'seg')
out = os.path.join(dir_path, 'out')

print(img)
print(seg)
print(out)

print("\nPredict")
print(f"python prediction.py --ds_path {img} --out_path {out} --pretrained ./saved_models/{model} --prep_mode {mode}")

print("\nTrain")
print(f"python train.py --ds_path {img} --lb_path {seg} --prep_mode 4 --ep 10 --lr 1e-3 --outmo {out}")

print("\nAdapt")
print(f"python test_time_adaptation.py --ds_path {img} --px_path {seg} --out_path {out} --pretrained ./saved_models/{model} --prep_mode {mode} --ep 10  --lr 1e-3")