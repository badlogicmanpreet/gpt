## Setup LambdaLabs

matrices
gaussian processes
ditributions
sampling


export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH

Host 152.69.203.83
  HostName 152.69.203.83
  User ubuntu
  IdentityFile **.pem

1. Create an instance on https://cloud.lambdalabs.com/instances
2. Get/Store ssh pem key, and host, user details
3. Locally vi ~/.ssh/config
   - Host <>
     HostName <>
     User ubuntu
     IdentityFile <path/key>.pem 
4. Install remote-ssh in vscode
5. Goto cmd-shift-p and open Remote-SSH: Settings
   - configure the path to config file <~/.ssh/config> , this should be aboslute path
6. Make sure the pem file has permission of chmod 400
7. Goto Remote explorer in vscode and connect
8. File gets uploaded automatically


## Output

ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.25.2
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
Traceback (most recent call last):
  File "/home/ubuntu/train_gpt_with_experiments.py", line 333, in <module>
    import tiktoken
ModuleNotFoundError: No module named 'tiktoken'
ubuntu@129-146-181-133:~$ pip install tiktoken
Defaulting to user installation because normal site-packages is not writeable
Collecting tiktoken
  Downloading tiktoken-0.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
Collecting regex>=2022.1.18 (from tiktoken)
  Downloading regex-2024.9.11-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.5/40.5 kB 1.8 MB/s eta 0:00:00
Requirement already satisfied: requests>=2.26.0 in ./.local/lib/python3.10/site-packages (from tiktoken) (2.31.0)
Requirement already satisfied: charset-normalizer<4,>=2 in ./.local/lib/python3.10/site-packages (from requests>=2.26.0->tiktoken) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests>=2.26.0->tiktoken) (3.3)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/lib/python3/dist-packages (from requests>=2.26.0->tiktoken) (1.26.5)
Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests>=2.26.0->tiktoken) (2020.6.20)
Downloading tiktoken-0.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 24.9 MB/s eta 0:00:00
Downloading regex-2024.9.11-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (782 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 782.7/782.7 kB 74.9 MB/s eta 0:00:00
DEPRECATION: flatbuffers 1.12.1-git20200711.33e2d80-dfsg1-0.6 has a non-standard version number. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of flatbuffers or contact the author to suggest that they release a version with a conforming version number. Discussion can be found at https://github.com/pypa/pip/issues/12063
Installing collected packages: regex, tiktoken
Successfully installed regex-2024.9.11 tiktoken-0.7.0

[notice] A new release of pip is available: 23.3.1 -> 24.2
[notice] To update, run: python3 -m pip install --upgrade pip
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.25.2
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
step: 0, loss: 10.909722328186035, dt: 1793.06ms, tok/sec: 9137.44
step: 1, loss: 9.463696479797363, dt: 1149.43ms, tok/sec: 14253.96
step: 2, loss: 9.207608222961426, dt: 1149.41ms, tok/sec: 14254.28
step: 3, loss: 9.012910842895508, dt: 1149.14ms, tok/sec: 14257.61
step: 4, loss: 8.80670166015625, dt: 1149.90ms, tok/sec: 14248.19
step: 5, loss: 8.670883178710938, dt: 1148.40ms, tok/sec: 14266.75
step: 6, loss: 8.428380966186523, dt: 1148.26ms, tok/sec: 14268.52
step: 7, loss: 8.157726287841797, dt: 1147.89ms, tok/sec: 14273.20
step: 8, loss: 7.881133079528809, dt: 1147.83ms, tok/sec: 14273.85
step: 9, loss: 7.674062728881836, dt: 1147.21ms, tok/sec: 14281.66
step: 10, loss: 7.5191545486450195, dt: 1148.72ms, tok/sec: 14262.78
step: 11, loss: 7.386804103851318, dt: 1150.31ms, tok/sec: 14243.08
step: 12, loss: 7.192424774169922, dt: 1147.96ms, tok/sec: 14272.27
step: 13, loss: 7.097297668457031, dt: 1147.68ms, tok/sec: 14275.73
step: 14, loss: 7.043957233428955, dt: 1147.49ms, tok/sec: 14278.08
step: 15, loss: 6.906817436218262, dt: 1148.21ms, tok/sec: 14269.16
step: 16, loss: 6.884222030639648, dt: 1147.80ms, tok/sec: 14274.27
step: 17, loss: 6.849422454833984, dt: 1147.02ms, tok/sec: 14283.95
step: 18, loss: 6.833486557006836, dt: 1147.00ms, tok/sec: 14284.25
step: 19, loss: 6.672182559967041, dt: 1147.55ms, tok/sec: 14277.40
step: 20, loss: 6.573649883270264, dt: 1147.21ms, tok/sec: 14281.55
step: 21, loss: 6.373497009277344, dt: 1147.32ms, tok/sec: 14280.29
step: 22, loss: 6.427977085113525, dt: 1147.23ms, tok/sec: 14281.37
step: 23, loss: 6.3945841789245605, dt: 1148.63ms, tok/sec: 14263.93
step: 24, loss: 6.348397731781006, dt: 1146.90ms, tok/sec: 14285.53
step: 25, loss: 6.569319248199463, dt: 1146.83ms, tok/sec: 14286.29
step: 26, loss: 6.648755073547363, dt: 1146.89ms, tok/sec: 14285.59
step: 27, loss: 6.542632579803467, dt: 1149.36ms, tok/sec: 14254.95
step: 28, loss: 6.483856201171875, dt: 1147.35ms, tok/sec: 14279.90
step: 29, loss: 6.3740997314453125, dt: 1147.28ms, tok/sec: 14280.70
step: 30, loss: 6.416515350341797, dt: 1147.23ms, tok/sec: 14281.40
step: 31, loss: 6.461634159088135, dt: 1147.46ms, tok/sec: 14278.51
step: 32, loss: 6.411356449127197, dt: 1147.40ms, tok/sec: 14279.21
step: 33, loss: 6.443024635314941, dt: 1148.83ms, tok/sec: 14261.46
step: 34, loss: 6.528824806213379, dt: 1148.16ms, tok/sec: 14269.82
step: 35, loss: 6.406240940093994, dt: 1147.39ms, tok/sec: 14279.32
step: 36, loss: 6.42188024520874, dt: 1148.10ms, tok/sec: 14270.54
step: 37, loss: 6.443517684936523, dt: 1147.97ms, tok/sec: 14272.10
step: 38, loss: 6.447713851928711, dt: 1147.98ms, tok/sec: 14272.03
step: 39, loss: 6.314174175262451, dt: 1147.25ms, tok/sec: 14281.12
step: 40, loss: 6.414916038513184, dt: 1149.29ms, tok/sec: 14255.72
step: 41, loss: 6.1900787353515625, dt: 1148.29ms, tok/sec: 14268.14
step: 42, loss: 6.307947635650635, dt: 1147.81ms, tok/sec: 14274.10
step: 43, loss: 6.260335445404053, dt: 1147.37ms, tok/sec: 14279.61
step: 44, loss: 6.212643146514893, dt: 1147.17ms, tok/sec: 14282.05
step: 45, loss: 6.4190545082092285, dt: 1149.93ms, tok/sec: 14247.78
step: 46, loss: 6.519870758056641, dt: 1148.11ms, tok/sec: 14270.36
step: 47, loss: 6.385984897613525, dt: 1147.50ms, tok/sec: 14278.03
step: 48, loss: 6.327324390411377, dt: 1147.44ms, tok/sec: 14278.80
step: 49, loss: 6.222888469696045, dt: 1147.39ms, tok/sec: 14279.31
tensor(6.2229, device='cuda:0', grad_fn=<NllLossBackward0>)
ubuntu@129-146-181-133:~$ nvidia-smi 
Mon Sep 30 04:39:56 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              47W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.25.2
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
step: 0, loss: 10.909676551818848, dt: 1067.45ms, tok/sec: 15348.78
step: 1, loss: 9.463589668273926, dt: 496.98ms, tok/sec: 32966.89
step: 2, loss: 9.207433700561523, dt: 490.16ms, tok/sec: 33425.97
step: 3, loss: 9.012932777404785, dt: 489.87ms, tok/sec: 33445.62
step: 4, loss: 8.806645393371582, dt: 490.51ms, tok/sec: 33401.84
step: 5, loss: 8.670784950256348, dt: 489.63ms, tok/sec: 33462.00
step: 6, loss: 8.428265571594238, dt: 488.77ms, tok/sec: 33520.63
step: 7, loss: 8.15764045715332, dt: 488.79ms, tok/sec: 33519.36
step: 8, loss: 7.881072998046875, dt: 488.83ms, tok/sec: 33516.87
step: 9, loss: 7.674021244049072, dt: 488.35ms, tok/sec: 33549.63
step: 10, loss: 7.519104480743408, dt: 489.29ms, tok/sec: 33485.16
step: 11, loss: 7.386782646179199, dt: 489.39ms, tok/sec: 33478.16
step: 12, loss: 7.192378997802734, dt: 488.95ms, tok/sec: 33508.55
step: 13, loss: 7.097264289855957, dt: 491.48ms, tok/sec: 33335.89
step: 14, loss: 7.0439581871032715, dt: 488.86ms, tok/sec: 33514.81
step: 15, loss: 6.906820774078369, dt: 488.75ms, tok/sec: 33521.96
step: 16, loss: 6.884228706359863, dt: 491.95ms, tok/sec: 33304.31
step: 17, loss: 6.849427700042725, dt: 488.75ms, tok/sec: 33522.35
step: 18, loss: 6.8335089683532715, dt: 488.64ms, tok/sec: 33529.61
step: 19, loss: 6.672179222106934, dt: 491.24ms, tok/sec: 33352.56
step: 20, loss: 6.573631286621094, dt: 489.51ms, tok/sec: 33470.51
step: 21, loss: 6.373477458953857, dt: 489.70ms, tok/sec: 33457.20
step: 22, loss: 6.427975177764893, dt: 489.03ms, tok/sec: 33502.85
step: 23, loss: 6.394591331481934, dt: 487.99ms, tok/sec: 33574.47
step: 24, loss: 6.348391056060791, dt: 488.40ms, tok/sec: 33546.52
step: 25, loss: 6.569315433502197, dt: 488.64ms, tok/sec: 33529.56
step: 26, loss: 6.648756980895996, dt: 488.01ms, tok/sec: 33573.01
step: 27, loss: 6.542604923248291, dt: 488.40ms, tok/sec: 33546.54
step: 28, loss: 6.483847141265869, dt: 488.32ms, tok/sec: 33551.45
step: 29, loss: 6.374086380004883, dt: 488.30ms, tok/sec: 33553.43
step: 30, loss: 6.4165167808532715, dt: 488.32ms, tok/sec: 33551.47
step: 31, loss: 6.461633205413818, dt: 488.67ms, tok/sec: 33528.01
step: 32, loss: 6.411349296569824, dt: 488.51ms, tok/sec: 33538.83
step: 33, loss: 6.44301700592041, dt: 488.71ms, tok/sec: 33525.15
step: 34, loss: 6.528808116912842, dt: 488.86ms, tok/sec: 33514.60
step: 35, loss: 6.4062299728393555, dt: 491.41ms, tok/sec: 33340.78
step: 36, loss: 6.421882629394531, dt: 489.04ms, tok/sec: 33502.64
step: 37, loss: 6.4435200691223145, dt: 488.61ms, tok/sec: 33531.74
step: 38, loss: 6.447725296020508, dt: 491.06ms, tok/sec: 33364.30
step: 39, loss: 6.314180850982666, dt: 488.45ms, tok/sec: 33542.93
step: 40, loss: 6.414908409118652, dt: 488.25ms, tok/sec: 33556.74
step: 41, loss: 6.190071105957031, dt: 490.79ms, tok/sec: 33382.66
step: 42, loss: 6.307929515838623, dt: 489.22ms, tok/sec: 33490.00
step: 43, loss: 6.260313510894775, dt: 488.30ms, tok/sec: 33552.91
step: 44, loss: 6.212624549865723, dt: 488.21ms, tok/sec: 33559.17
step: 45, loss: 6.419023513793945, dt: 488.25ms, tok/sec: 33556.82
step: 46, loss: 6.519842147827148, dt: 488.23ms, tok/sec: 33557.66
step: 47, loss: 6.385951042175293, dt: 490.95ms, tok/sec: 33372.14
step: 48, loss: 6.3272857666015625, dt: 488.60ms, tok/sec: 33532.75
step: 49, loss: 6.222859859466553, dt: 488.51ms, tok/sec: 33538.68
tensor(6.2229, device='cuda:0', grad_fn=<NllLossBackward0>)
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.25.2
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> logits.dtype
torch.bfloat16
>>> model.transformer.wte.weight
Parameter containing:
tensor([[ 9.9743e-05,  1.6065e-03,  1.6118e-02,  ..., -2.3506e-02,
         -9.5092e-03,  8.6977e-04],
        [ 5.6103e-03, -8.9315e-04,  3.3858e-02,  ...,  1.8497e-02,
         -1.2024e-02,  4.2558e-03],
        [ 1.2853e-02,  8.0832e-03,  1.8367e-02,  ..., -2.2407e-02,
         -1.2174e-02, -1.2083e-02],
        ...,
        [ 6.8869e-03,  1.8946e-02,  2.7229e-02,  ..., -9.2498e-03,
         -1.6403e-02,  1.1806e-02],
        [-6.4153e-03,  6.4614e-03, -1.8471e-02,  ...,  3.3779e-04,
          8.5628e-03, -4.6225e-03],
        [ 4.5271e-03, -2.1883e-02,  2.6784e-02,  ..., -4.7267e-03,
         -1.2253e-02,  2.1918e-02]], device='cuda:0', requires_grad=True)
>>> model.transformer.wte.weight.dtype
torch.float32
>>> exit()
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.25.2
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
step: 0, loss: 10.909770965576172, dt: 1103.20ms, tok/sec: 14851.37
step: 1, loss: 9.463577270507812, dt: 445.86ms, tok/sec: 36747.20
step: 2, loss: 9.20934009552002, dt: 448.15ms, tok/sec: 36559.35
step: 3, loss: 9.011079788208008, dt: 444.84ms, tok/sec: 36831.07
step: 4, loss: 8.805235862731934, dt: 445.32ms, tok/sec: 36791.47
step: 5, loss: 8.671212196350098, dt: 445.20ms, tok/sec: 36801.14
step: 6, loss: 8.428132057189941, dt: 445.25ms, tok/sec: 36797.72
step: 7, loss: 8.156357765197754, dt: 448.78ms, tok/sec: 36507.92
step: 8, loss: 7.880721092224121, dt: 445.07ms, tok/sec: 36811.89
step: 9, loss: 7.676907539367676, dt: 445.88ms, tok/sec: 36745.38
step: 10, loss: 7.5155439376831055, dt: 444.73ms, tok/sec: 36840.13
step: 11, loss: 7.389228343963623, dt: 445.31ms, tok/sec: 36792.67
step: 12, loss: 7.192400932312012, dt: 445.18ms, tok/sec: 36802.78
step: 13, loss: 7.096391677856445, dt: 444.76ms, tok/sec: 36837.66
step: 14, loss: 7.046197891235352, dt: 444.76ms, tok/sec: 36837.70
step: 15, loss: 6.9066877365112305, dt: 444.13ms, tok/sec: 36890.14
step: 16, loss: 6.881811141967773, dt: 444.84ms, tok/sec: 36831.62
step: 17, loss: 6.850700378417969, dt: 444.64ms, tok/sec: 36847.54
step: 18, loss: 6.836300849914551, dt: 444.50ms, tok/sec: 36859.04
step: 19, loss: 6.668896675109863, dt: 456.32ms, tok/sec: 35904.66
step: 20, loss: 6.574646949768066, dt: 444.73ms, tok/sec: 36840.05
step: 21, loss: 6.372208595275879, dt: 444.16ms, tok/sec: 36887.57
step: 22, loss: 6.429536819458008, dt: 444.38ms, tok/sec: 36869.54
step: 23, loss: 6.396400451660156, dt: 445.33ms, tok/sec: 36790.74
step: 24, loss: 6.348324298858643, dt: 443.84ms, tok/sec: 36914.56
step: 25, loss: 6.568527698516846, dt: 444.14ms, tok/sec: 36889.59
step: 26, loss: 6.646594524383545, dt: 443.51ms, tok/sec: 36941.98
step: 27, loss: 6.542299747467041, dt: 450.16ms, tok/sec: 36396.16
step: 28, loss: 6.480898857116699, dt: 444.27ms, tok/sec: 36878.74
step: 29, loss: 6.376673698425293, dt: 444.27ms, tok/sec: 36878.84
step: 30, loss: 6.417859077453613, dt: 444.47ms, tok/sec: 36861.53
step: 31, loss: 6.462467193603516, dt: 444.46ms, tok/sec: 36862.72
step: 32, loss: 6.408759117126465, dt: 445.15ms, tok/sec: 36805.17
step: 33, loss: 6.44106388092041, dt: 444.08ms, tok/sec: 36894.58
step: 34, loss: 6.531723976135254, dt: 443.97ms, tok/sec: 36903.36
step: 35, loss: 6.407721519470215, dt: 445.60ms, tok/sec: 36768.77
step: 36, loss: 6.420440673828125, dt: 444.03ms, tok/sec: 36898.05
step: 37, loss: 6.443750858306885, dt: 443.84ms, tok/sec: 36914.30
step: 38, loss: 6.448118209838867, dt: 444.54ms, tok/sec: 36856.03
step: 39, loss: 6.314005374908447, dt: 443.90ms, tok/sec: 36908.95
step: 40, loss: 6.413841724395752, dt: 443.87ms, tok/sec: 36911.88
step: 41, loss: 6.1905837059021, dt: 444.28ms, tok/sec: 36877.69
step: 42, loss: 6.30879545211792, dt: 444.07ms, tok/sec: 36895.33
step: 43, loss: 6.258794784545898, dt: 445.17ms, tok/sec: 36803.57
step: 44, loss: 6.21610689163208, dt: 443.64ms, tok/sec: 36931.12
step: 45, loss: 6.4190473556518555, dt: 443.62ms, tok/sec: 36932.73
step: 46, loss: 6.522340774536133, dt: 444.60ms, tok/sec: 36850.88
step: 47, loss: 6.384696006774902, dt: 444.29ms, tok/sec: 36876.82
step: 48, loss: 6.324511528015137, dt: 444.68ms, tok/sec: 36844.73
step: 49, loss: 6.223577976226807, dt: 445.48ms, tok/sec: 36778.38
tensor(6.2236, device='cuda:0', grad_fn=<NllLossBackward0>)
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.25.2
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/torch/_dynamo/output_graph.py", line 670, in call_user_compiler
    compiled_fn = compiler_fn(gm, self.fake_example_inputs())
  File "/usr/lib/python3/dist-packages/torch/_dynamo/debug_utils.py", line 1055, in debug_wrapper
    compiled_gm = compiler_fn(gm, example_inputs)
  File "/usr/lib/python3/dist-packages/torch/__init__.py", line 1390, in __call__
    return compile_fx(model_, inputs_, config_patches=self.config)
  File "/usr/lib/python3/dist-packages/torch/_inductor/compile_fx.py", line 455, in compile_fx
    return aot_autograd(
  File "/usr/lib/python3/dist-packages/torch/_dynamo/backends/common.py", line 48, in compiler_fn
    cg = aot_module_simplified(gm, example_inputs, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/_functorch/aot_autograd.py", line 2822, in aot_module_simplified
    compiled_fn = create_aot_dispatcher_function(
  File "/usr/lib/python3/dist-packages/torch/_dynamo/utils.py", line 163, in time_wrapper
    r = func(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/_functorch/aot_autograd.py", line 2515, in create_aot_dispatcher_function
    compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config)
  File "/usr/lib/python3/dist-packages/torch/_functorch/aot_autograd.py", line 1715, in aot_wrapper_dedupe
    return compiler_fn(flat_fn, leaf_flat_args, aot_config)
  File "/usr/lib/python3/dist-packages/torch/_functorch/aot_autograd.py", line 2131, in aot_dispatch_autograd
    fw_module, bw_module = aot_config.partition_fn(
  File "/usr/lib/python3/dist-packages/torch/_functorch/partitioners.py", line 305, in min_cut_rematerialization_partition
    import networkx as nx
  File "/usr/lib/python3/dist-packages/networkx/__init__.py", line 115, in <module>
    import networkx.readwrite
  File "/usr/lib/python3/dist-packages/networkx/readwrite/__init__.py", line 15, in <module>
    from networkx.readwrite.graphml import *
  File "/usr/lib/python3/dist-packages/networkx/readwrite/graphml.py", line 314, in <module>
    class GraphML(object):
  File "/usr/lib/python3/dist-packages/networkx/readwrite/graphml.py", line 346, in GraphML
    (np.int, "int"), (np.int8, "int"),
  File "/home/ubuntu/.local/lib/python3.10/site-packages/numpy/__init__.py", line 319, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'inf'?

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/train_gpt_with_experiments.py", line 449, in <module>
    logits, loss = model(x, y) # forward pass
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/eval_frame.py", line 82, in forward
    return self.dynamo_ctx(self._orig_mod.forward)(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/eval_frame.py", line 209, in _fn
    return fn(*args, **kwargs)
  File "/home/ubuntu/train_gpt_with_experiments.py", line 209, in forward
    logger.debug(f"training model...")
  File "/home/ubuntu/train_gpt_with_experiments.py", line 236, in <graph break in forward>
    x = block(x)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/train_gpt_with_experiments.py", line 144, in forward
    logger.debug(f"block (forward section): Started... Block")
  File "/home/ubuntu/train_gpt_with_experiments.py", line 145, in <graph break in forward>
    logger.debug(f"block (forward section): x size: {x.size()}")
  File "/usr/lib/python3/dist-packages/torch/_dynamo/eval_frame.py", line 337, in catch_errors
    return callback(frame, cache_size, hooks)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/convert_frame.py", line 404, in _convert_frame
    result = inner_convert(frame, cache_size, hooks)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/convert_frame.py", line 104, in _fn
    return fn(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/convert_frame.py", line 262, in _convert_frame_assert
    return _compile(
  File "/usr/lib/python3/dist-packages/torch/_dynamo/utils.py", line 163, in time_wrapper
    r = func(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/convert_frame.py", line 324, in _compile
    out_code = transform_code_object(code, transform)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/bytecode_transformation.py", line 445, in transform_code_object
    transformations(instructions, code_options)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/convert_frame.py", line 311, in transform
    tracer.run()
  File "/usr/lib/python3/dist-packages/torch/_dynamo/symbolic_convert.py", line 1726, in run
    super().run()
  File "/usr/lib/python3/dist-packages/torch/_dynamo/symbolic_convert.py", line 576, in run
    and self.step()
  File "/usr/lib/python3/dist-packages/torch/_dynamo/symbolic_convert.py", line 540, in step
    getattr(self, inst.opname)(inst)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/symbolic_convert.py", line 372, in wrapper
    self.output.compile_subgraph(self, reason=reason)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/output_graph.py", line 541, in compile_subgraph
    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/output_graph.py", line 588, in compile_and_call_fx_graph
    compiled_fn = self.call_user_compiler(gm)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/utils.py", line 163, in time_wrapper
    r = func(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/_dynamo/output_graph.py", line 675, in call_user_compiler
    raise BackendCompilerFailed(self.compiler_fn, e) from e
torch._dynamo.exc.BackendCompilerFailed: debug_wrapper raised AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

Set torch._dynamo.config.verbose=True for more information


You can suppress this exception and fall back to eager by setting:
    torch._dynamo.config.suppress_errors = True

ubuntu@129-146-181-133:~$ pip install numpy==1.23
Defaulting to user installation because normal site-packages is not writeable
Collecting numpy==1.23
  Downloading numpy-1.23.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.2 kB)
Downloading numpy-1.23.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.0/17.0 MB 45.9 MB/s eta 0:00:00
DEPRECATION: flatbuffers 1.12.1-git20200711.33e2d80-dfsg1-0.6 has a non-standard version number. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of flatbuffers or contact the author to suggest that they release a version with a conforming version number. Discussion can be found at https://github.com/pypa/pip/issues/12063
Installing collected packages: numpy
  Attempting uninstall: numpy
    Found existing installation: numpy 1.25.2
    Uninstalling numpy-1.25.2:
      Successfully uninstalled numpy-1.25.2
Successfully installed numpy-1.23.0

[notice] A new release of pip is available: 23.3.1 -> 24.2
[notice] To update, run: python3 -m pip install --upgrade pip
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
step: 0, loss: 10.909587860107422, dt: 21162.68ms, tok/sec: 774.19
step: 1, loss: 9.463571548461914, dt: 501.83ms, tok/sec: 32648.22
step: 2, loss: 9.209184646606445, dt: 485.75ms, tok/sec: 33729.11
step: 3, loss: 9.01120376586914, dt: 485.21ms, tok/sec: 33766.59
step: 4, loss: 8.80517864227295, dt: 486.30ms, tok/sec: 33691.15
step: 5, loss: 8.67113208770752, dt: 484.10ms, tok/sec: 33844.27
step: 6, loss: 8.428162574768066, dt: 484.20ms, tok/sec: 33837.26
step: 7, loss: 8.15641975402832, dt: 485.24ms, tok/sec: 33764.67
step: 8, loss: 7.88065242767334, dt: 484.40ms, tok/sec: 33823.20
step: 9, loss: 7.676858425140381, dt: 484.47ms, tok/sec: 33818.09
step: 10, loss: 7.515589714050293, dt: 484.38ms, tok/sec: 33824.72
step: 11, loss: 7.389187335968018, dt: 483.96ms, tok/sec: 33853.93
step: 12, loss: 7.192559242248535, dt: 484.44ms, tok/sec: 33820.49
step: 13, loss: 7.0963592529296875, dt: 484.43ms, tok/sec: 33821.47
step: 14, loss: 7.046241760253906, dt: 485.22ms, tok/sec: 33766.03
step: 15, loss: 6.906777381896973, dt: 484.15ms, tok/sec: 33840.94
step: 16, loss: 6.881681442260742, dt: 484.29ms, tok/sec: 33831.01
step: 17, loss: 6.850545883178711, dt: 483.81ms, tok/sec: 33864.52
step: 18, loss: 6.836298942565918, dt: 484.96ms, tok/sec: 33784.02
step: 19, loss: 6.6685686111450195, dt: 483.76ms, tok/sec: 33868.24
step: 20, loss: 6.57454776763916, dt: 484.14ms, tok/sec: 33841.66
step: 21, loss: 6.3721418380737305, dt: 484.48ms, tok/sec: 33817.48
step: 22, loss: 6.4294843673706055, dt: 484.35ms, tok/sec: 33826.60
step: 23, loss: 6.396289825439453, dt: 483.65ms, tok/sec: 33875.81
step: 24, loss: 6.348189353942871, dt: 483.94ms, tok/sec: 33855.36
step: 25, loss: 6.568576335906982, dt: 484.08ms, tok/sec: 33845.99
step: 26, loss: 6.646596908569336, dt: 483.58ms, tok/sec: 33880.87
step: 27, loss: 6.542257785797119, dt: 484.57ms, tok/sec: 33811.65
step: 28, loss: 6.481026649475098, dt: 484.13ms, tok/sec: 33842.09
step: 29, loss: 6.376506805419922, dt: 491.87ms, tok/sec: 33309.94
step: 30, loss: 6.417791366577148, dt: 483.99ms, tok/sec: 33851.71
step: 31, loss: 6.462449073791504, dt: 485.37ms, tok/sec: 33755.40
step: 32, loss: 6.408792495727539, dt: 484.25ms, tok/sec: 33833.76
step: 33, loss: 6.440635681152344, dt: 484.55ms, tok/sec: 33812.98
step: 34, loss: 6.531564712524414, dt: 484.34ms, tok/sec: 33827.18
step: 35, loss: 6.407556533813477, dt: 484.72ms, tok/sec: 33800.84
step: 36, loss: 6.420320987701416, dt: 483.74ms, tok/sec: 33869.51
step: 37, loss: 6.443763732910156, dt: 483.64ms, tok/sec: 33876.21
step: 38, loss: 6.448014736175537, dt: 483.74ms, tok/sec: 33869.16
step: 39, loss: 6.313876628875732, dt: 483.95ms, tok/sec: 33854.49
step: 40, loss: 6.413727283477783, dt: 484.61ms, tok/sec: 33808.97
step: 41, loss: 6.190462112426758, dt: 484.31ms, tok/sec: 33829.23
step: 42, loss: 6.308787822723389, dt: 487.49ms, tok/sec: 33608.55
step: 43, loss: 6.258730888366699, dt: 484.39ms, tok/sec: 33824.32
step: 44, loss: 6.216023921966553, dt: 484.09ms, tok/sec: 33844.79
step: 45, loss: 6.418927192687988, dt: 486.08ms, tok/sec: 33706.72
step: 46, loss: 6.521891117095947, dt: 484.76ms, tok/sec: 33798.03
step: 47, loss: 6.384504795074463, dt: 484.05ms, tok/sec: 33847.61
step: 48, loss: 6.324315071105957, dt: 483.91ms, tok/sec: 33857.60
step: 49, loss: 6.223581790924072, dt: 484.10ms, tok/sec: 33843.96
tensor(6.2236, device='cuda:0', grad_fn=<NllLossBackward0>)
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
step: 0, loss: 10.909587860107422, dt: 17201.09ms, tok/sec: 952.50
step: 1, loss: 9.463571548461914, dt: 500.98ms, tok/sec: 32703.70
step: 2, loss: 9.209184646606445, dt: 488.05ms, tok/sec: 33570.41
step: 3, loss: 9.01120376586914, dt: 485.99ms, tok/sec: 33712.86
step: 4, loss: 8.80517864227295, dt: 486.57ms, tok/sec: 33672.12
step: 5, loss: 8.67113208770752, dt: 486.34ms, tok/sec: 33688.59
step: 6, loss: 8.428162574768066, dt: 485.92ms, tok/sec: 33717.14
step: 7, loss: 8.15641975402832, dt: 485.58ms, tok/sec: 33741.00
step: 8, loss: 7.88065242767334, dt: 485.82ms, tok/sec: 33724.17
step: 9, loss: 7.676858425140381, dt: 485.59ms, tok/sec: 33740.07
step: 10, loss: 7.515589714050293, dt: 489.28ms, tok/sec: 33485.78
step: 11, loss: 7.389187335968018, dt: 485.99ms, tok/sec: 33712.43
step: 12, loss: 7.192559242248535, dt: 486.12ms, tok/sec: 33703.40
step: 13, loss: 7.0963592529296875, dt: 486.11ms, tok/sec: 33704.08
step: 14, loss: 7.046241760253906, dt: 486.06ms, tok/sec: 33707.58
step: 15, loss: 6.906777381896973, dt: 486.91ms, tok/sec: 33648.97
step: 16, loss: 6.881681442260742, dt: 485.95ms, tok/sec: 33715.07
step: 17, loss: 6.850545883178711, dt: 485.94ms, tok/sec: 33716.08
step: 18, loss: 6.836298942565918, dt: 485.63ms, tok/sec: 33737.40
step: 19, loss: 6.6685686111450195, dt: 485.70ms, tok/sec: 33732.47
step: 20, loss: 6.57454776763916, dt: 486.38ms, tok/sec: 33685.69
step: 21, loss: 6.3721418380737305, dt: 485.90ms, tok/sec: 33719.14
step: 22, loss: 6.4294843673706055, dt: 486.20ms, tok/sec: 33698.01
step: 23, loss: 6.396289825439453, dt: 485.57ms, tok/sec: 33741.81
step: 24, loss: 6.348189353942871, dt: 485.62ms, tok/sec: 33738.33
step: 25, loss: 6.568576335906982, dt: 486.03ms, tok/sec: 33709.71
step: 26, loss: 6.646596908569336, dt: 485.60ms, tok/sec: 33739.90
step: 27, loss: 6.542257785797119, dt: 485.82ms, tok/sec: 33724.54
step: 28, loss: 6.481026649475098, dt: 485.87ms, tok/sec: 33720.90
step: 29, loss: 6.376506805419922, dt: 488.02ms, tok/sec: 33572.25
step: 30, loss: 6.417791366577148, dt: 485.69ms, tok/sec: 33733.64
step: 31, loss: 6.462449073791504, dt: 486.23ms, tok/sec: 33695.68
step: 32, loss: 6.408792495727539, dt: 486.09ms, tok/sec: 33705.76
step: 33, loss: 6.440635681152344, dt: 486.20ms, tok/sec: 33698.16
step: 34, loss: 6.531564712524414, dt: 486.39ms, tok/sec: 33684.90
step: 35, loss: 6.407556533813477, dt: 486.16ms, tok/sec: 33700.67
step: 36, loss: 6.420320987701416, dt: 485.79ms, tok/sec: 33726.34
step: 37, loss: 6.443763732910156, dt: 485.63ms, tok/sec: 33737.32
step: 38, loss: 6.448014736175537, dt: 485.75ms, tok/sec: 33729.24
step: 39, loss: 6.313876628875732, dt: 489.75ms, tok/sec: 33453.76
step: 40, loss: 6.413727283477783, dt: 486.18ms, tok/sec: 33699.61
step: 41, loss: 6.190462112426758, dt: 486.67ms, tok/sec: 33665.65
step: 42, loss: 6.308787822723389, dt: 486.07ms, tok/sec: 33706.92
step: 43, loss: 6.258730888366699, dt: 485.15ms, tok/sec: 33770.79
step: 44, loss: 6.216023921966553, dt: 485.05ms, tok/sec: 33778.23
step: 45, loss: 6.418927192687988, dt: 485.16ms, tok/sec: 33770.08
step: 46, loss: 6.521891117095947, dt: 485.49ms, tok/sec: 33747.67
step: 47, loss: 6.384504795074463, dt: 485.32ms, tok/sec: 33758.91
step: 48, loss: 6.324315071105957, dt: 485.22ms, tok/sec: 33765.96
step: 49, loss: 6.223581790924072, dt: 485.80ms, tok/sec: 33725.48
tensor(6.2236, device='cuda:0', grad_fn=<NllLossBackward0>)
ubuntu@129-146-181-133:~$ /usr/bin/python3 /home/ubuntu/train_gpt_with_experiments.py
using device: cuda...
loaded 338025 tokens
1 epoch = 20 batches
step: 0, loss: 10.909587860107422, dt: 17068.33ms, tok/sec: 959.91
step: 1, loss: 9.463571548461914, dt: 499.39ms, tok/sec: 32808.24
step: 2, loss: 9.209184646606445, dt: 485.84ms, tok/sec: 33722.95
step: 3, loss: 9.01120376586914, dt: 485.08ms, tok/sec: 33775.87
step: 4, loss: 8.80517864227295, dt: 485.40ms, tok/sec: 33753.76
step: 5, loss: 8.67113208770752, dt: 485.25ms, tok/sec: 33764.09
step: 6, loss: 8.428162574768066, dt: 484.80ms, tok/sec: 33795.36
step: 7, loss: 8.15641975402832, dt: 486.03ms, tok/sec: 33709.88
step: 8, loss: 7.88065242767334, dt: 485.70ms, tok/sec: 33733.10
step: 9, loss: 7.676858425140381, dt: 484.68ms, tok/sec: 33803.87
step: 10, loss: 7.515589714050293, dt: 484.79ms, tok/sec: 33796.25
step: 11, loss: 7.389187335968018, dt: 486.33ms, tok/sec: 33689.04
step: 12, loss: 7.192559242248535, dt: 484.97ms, tok/sec: 33783.71
step: 13, loss: 7.0963592529296875, dt: 484.81ms, tok/sec: 33794.99
step: 14, loss: 7.046241760253906, dt: 484.71ms, tok/sec: 33801.44
step: 15, loss: 6.906777381896973, dt: 484.76ms, tok/sec: 33797.97
step: 16, loss: 6.881681442260742, dt: 484.91ms, tok/sec: 33787.60
step: 17, loss: 6.850545883178711, dt: 484.94ms, tok/sec: 33785.70
step: 18, loss: 6.836298942565918, dt: 488.61ms, tok/sec: 33531.90
step: 19, loss: 6.6685686111450195, dt: 484.78ms, tok/sec: 33796.52
step: 20, loss: 6.57454776763916, dt: 485.26ms, tok/sec: 33763.16
step: 21, loss: 6.3721418380737305, dt: 484.87ms, tok/sec: 33790.62
step: 22, loss: 6.4294843673706055, dt: 484.71ms, tok/sec: 33801.34
step: 23, loss: 6.396289825439453, dt: 484.06ms, tok/sec: 33847.24
step: 24, loss: 6.348189353942871, dt: 484.19ms, tok/sec: 33838.16
step: 25, loss: 6.568576335906982, dt: 485.75ms, tok/sec: 33729.30
step: 26, loss: 6.646596908569336, dt: 487.70ms, tok/sec: 33594.13
step: 27, loss: 6.542257785797119, dt: 488.23ms, tok/sec: 33558.02
step: 28, loss: 6.481026649475098, dt: 484.39ms, tok/sec: 33824.08
step: 29, loss: 6.376506805419922, dt: 484.26ms, tok/sec: 33832.79
step: 30, loss: 6.417791366577148, dt: 484.41ms, tok/sec: 33822.93
step: 31, loss: 6.462449073791504, dt: 484.08ms, tok/sec: 33845.76
step: 32, loss: 6.408792495727539, dt: 483.89ms, tok/sec: 33858.76
step: 33, loss: 6.440635681152344, dt: 484.55ms, tok/sec: 33813.10
step: 34, loss: 6.531564712524414, dt: 484.55ms, tok/sec: 33812.82
step: 35, loss: 6.407556533813477, dt: 484.43ms, tok/sec: 33821.49
step: 36, loss: 6.420320987701416, dt: 484.19ms, tok/sec: 33838.02
step: 37, loss: 6.443763732910156, dt: 484.15ms, tok/sec: 33841.01
step: 38, loss: 6.448014736175537, dt: 484.90ms, tok/sec: 33788.26
step: 39, loss: 6.313876628875732, dt: 484.72ms, tok/sec: 33800.71
step: 40, loss: 6.413727283477783, dt: 484.21ms, tok/sec: 33836.57
step: 41, loss: 6.190462112426758, dt: 484.37ms, tok/sec: 33825.60
step: 42, loss: 6.308787822723389, dt: 484.11ms, tok/sec: 33843.39
step: 43, loss: 6.258730888366699, dt: 484.11ms, tok/sec: 33843.36
step: 44, loss: 6.216023921966553, dt: 484.04ms, tok/sec: 33848.36
step: 45, loss: 6.418927192687988, dt: 483.94ms, tok/sec: 33855.48
step: 46, loss: 6.521891117095947, dt: 486.52ms, tok/sec: 33675.68
step: 47, loss: 6.384504795074463, dt: 484.55ms, tok/sec: 33812.52
step: 48, loss: 6.324315071105957, dt: 484.34ms, tok/sec: 33827.71
step: 49, loss: 6.223581790924072, dt: 484.58ms, tok/sec: 33810.77
tensor(6.2236, device='cuda:0', grad_fn=<NllLossBackward0>)
ubuntu@129-146-181-133:~$ ls -ltr
total 1152
-rw-rw-r-- 1 ubuntu ubuntu 1115394 Sep 30 04:34 input.txt
-rw-rw-r-- 1 ubuntu ubuntu   28234 Sep 30 05:02 train_gpt_with_experiments.py
drwxrwxr-x 2 ubuntu ubuntu    4096 Sep 30 05:08 logs
-rw-rw-r-- 1 ubuntu ubuntu   26019 Sep 30 05:08 train.log
ubuntu@129-146-181-133:~$ vi train_gpt_with_experiments.py 
ubuntu@129-146-181-133:~$ nvidia-sim -q
Command 'nvidia-sim' not found, did you mean:
  command 'nvidia-smi' from deb nvidia-utils-390 (390.157-0ubuntu0.22.04.2)
  command 'nvidia-smi' from deb nvidia-utils-418-server (418.226.00-0ubuntu5~0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-450-server (450.248.02-0ubuntu0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-470 (470.223.02-0ubuntu0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-470-server (470.223.02-0ubuntu0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-525 (525.147.05-0ubuntu0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-525-server (525.147.05-0ubuntu0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-535 (535.129.03-0ubuntu0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-535-server (535.129.03-0ubuntu0.22.04.1)
  command 'nvidia-smi' from deb nvidia-utils-510 (510.60.02-0ubuntu1)
  command 'nvidia-smi' from deb nvidia-utils-510-server (510.47.03-0ubuntu3)
Try: sudo apt install <deb name>
ubuntu@129-146-181-133:~$ nvidia-smi --query-gpu=driver_version,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
driver_version, name, memory.total [MiB], memory.used [MiB], memory.free [MiB], utilization.gpu [%], temperature.gpu
535.129.03, NVIDIA A100-SXM4-40GB, 40960 MiB, 4 MiB, 40334 MiB, 0 %, 31
ubuntu@129-146-181-133:~$ 

### nvidia-smi monitoring

|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:09 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              66W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:10 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              66W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:11 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              66W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:12 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:14 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:15 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:16 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:17 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |      7MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:19 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   32C    P0              49W / 400W |      7MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:20 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              70W / 400W |     31MiB / 40960MiB |      1%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                             22MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:21 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |    977MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                            964MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:22 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |    999MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                            986MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:24 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |   1329MiB / 40960MiB |      2%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                           1316MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:25 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |   3731MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                           3718MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:26 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              70W / 400W |   5653MiB / 40960MiB |      1%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                           5640MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:27 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              52W / 400W |   9111MiB / 40960MiB |      4%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                           9098MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:28 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |  10649MiB / 40960MiB |      1%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          10636MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:30 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |  14107MiB / 40960MiB |      4%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          14094MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:31 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   34C    P0              67W / 400W |  16027MiB / 40960MiB |      1%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          16014MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:32 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |  19103MiB / 40960MiB |      4%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          19090MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:33 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   33C    P0              67W / 400W |  21023MiB / 40960MiB |      1%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          21010MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:35 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   34C    P0              75W / 400W |  31999MiB / 40960MiB |      4%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          31986MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:36 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   34C    P0              78W / 400W |  31999MiB / 40960MiB |      7%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          31986MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:37 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   34C    P0              68W / 400W |  31999MiB / 40960MiB |      7%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          31986MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:38 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   36C    P0             188W / 400W |  31999MiB / 40960MiB |     11%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          31986MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:40 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   40C    P0             220W / 400W |  35143MiB / 40960MiB |     77%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:41 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   40C    P0             187W / 400W |  35143MiB / 40960MiB |     88%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:42 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   41C    P0             178W / 400W |  35143MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:44 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   42C    P0             244W / 400W |  35143MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:45 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   42C    P0             263W / 400W |  35143MiB / 40960MiB |     94%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:47 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   44C    P0             260W / 400W |  35143MiB / 40960MiB |     84%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:48 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   45C    P0             242W / 400W |  35143MiB / 40960MiB |     75%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:49 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   44C    P0             165W / 400W |  35143MiB / 40960MiB |     85%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:51 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   44C    P0             176W / 400W |  35143MiB / 40960MiB |     98%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:52 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   44C    P0             145W / 400W |  35143MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:53 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   45C    P0             265W / 400W |  35143MiB / 40960MiB |     94%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:55 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   46C    P0             261W / 400W |  35143MiB / 40960MiB |     84%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:56 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   46C    P0             230W / 400W |  35143MiB / 40960MiB |     76%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:57 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   46C    P0             187W / 400W |  35143MiB / 40960MiB |     66%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:27:59 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   45C    P0             183W / 400W |  35143MiB / 40960MiB |     95%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:00 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   46C    P0             179W / 400W |  35143MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:02 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   46C    P0             267W / 400W |  35143MiB / 40960MiB |     96%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:03 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   41C    P0              72W / 400W |  35143MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      7170      C   /usr/bin/python3                          35130MiB |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:04 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   40C    P0              74W / 400W |      4MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:06 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   39C    P0              71W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:07 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   39C    P0              70W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:08 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   39C    P0              70W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:09 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   38C    P0              70W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:10 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   38C    P0              70W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:12 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   38C    P0              70W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Mon Sep 30 05:28:13 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   38C    P0              70W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+

# Use bfloat16 for FLOPs

# Torch.compile

![Tan GELU](tangelu.png)

![GPU Memory Travel](GPU-Mem-Travel.png)
The way the python interpretor handles this is... operation by operation. Taking each operation to gpu cores from HBM and calculating (utilizes local registers and caches) and then writing back the result to HBM. This travel time is killing, this is where the torch.compile ensures that full set of operations are done in the gpu and only final result is written back. This is kernel fusion, reduces the round trips.

## GPUs

![Inside GPUs](inside-gpu.png)

Left image is a zoom in of the GPU, it has SM (serial M), L2 cache, L1 chache, some amount of memory on the chip. Also connected to HBM (High bandwidth memory) which is connected outside. Extreme left image shows a single SM and details.

![GPU Memory](gpu-mem.png)

## FlashAttention

Apply this

# Look for ugly numbers
replace everything 2 to the power, cuds loves 2 to the power.
Increase the vocab size to 50304
The dt will decrease, computation time is better with 2 to the power numbers

# GPT3 paper https://arxiv.org/pdf/2005.14165v4
For hyperameter tuning (goto appendix)
To train all versions of GPT-3, we use Adam with β1 = 0.9, β2 = 0.95, and  = 10−8
, we clip the global norm of the
gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260
billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375
million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over
the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during
training (until an epoch boundary is reached) to minimize overfitting. All models use weight decay of 0.1 to provide a
small amount of regularization [LH17].
During training we always train on sequences of the full nctx = 2048 token context window, packing multiple
documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency.
Sequences with multiple documents are not masked in any special way but instead documents within a sequence
are delimited with a special end of text token, giving the language model the information necessary to infer that
context separated by the end of text token is unrelated. This allows for efficient training without need for any special
sequence-specific masking.

# 0.5 batch size using gradient accumulation

# Use distribute data parallel (torch)
   - launch 8 process with 8 gpus
   - each processing different parts of the data
   - we will now run our script with torch.run, to run with 8 of them

# Bigger dataset
   - https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
   - https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

# HellaSwag

----------------------------------------------------------------------------------------------------------------

ubuntu@152-69-203-83:~$ /usr/bin/python3 /home/ubuntu/msgpt/fineweb.py
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23.2k/23.2k [00:00<00:00, 184MB/s]
Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1630/1630 [00:04<00:00, 340.08it/s]
000_00000.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [00:09<00:00, 237MB/s]
001_00000.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [00:08<00:00, 242MB/s]
002_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [01:29<00:00, 24.1MB/s]
003_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [00:58<00:00, 36.7MB/s]
004_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [00:59<00:00, 36.0MB/s]
005_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [01:01<00:00, 35.2MB/s]
006_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [01:03<00:00, 33.9MB/s]
007_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [01:56<00:00, 18.4MB/s]
008_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [00:58<00:00, 36.8MB/s]
009_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [00:59<00:00, 36.3MB/s]
010_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [00:58<00:00, 36.6MB/s]
011_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [01:16<00:00, 28.1MB/s]
012_00000.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.15G/2.15G [01:31<00:00, 23.5MB/s]
013_00000.parquet: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541M/541M [00:15<00:00, 34.0MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 9672101/9672101 [02:45<00:00, 58334.17 examples/s]
Loading dataset shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 98/98 [00:00<00:00, 334.17it/s]
Shard 0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 100000000/100000000 [00:10<00:00, 9446255.12tokens/s]
Shard 1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997871/100000000 [00:14<00:00, 6702184.89tokens/s]
Shard 2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999982/100000000 [00:17<00:00, 5587307.81tokens/s]
Shard 3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999934/100000000 [00:20<00:00, 4886692.54tokens/s]
Shard 4: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999796/100000000 [00:19<00:00, 5043283.87tokens/s]
Shard 5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998855/100000000 [00:20<00:00, 4945634.45tokens/s]
Shard 6: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997594/100000000 [00:20<00:00, 4847692.88tokens/s]
Shard 7: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99996910/100000000 [00:18<00:00, 5291454.34tokens/s]
Shard 8: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999293/100000000 [00:18<00:00, 5341875.70tokens/s]
Shard 9: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997859/100000000 [00:20<00:00, 4960472.53tokens/s]
Shard 10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999807/100000000 [00:18<00:00, 5372597.49tokens/s]
Shard 11: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999526/100000000 [00:19<00:00, 5174294.13tokens/s]
Shard 12: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99996983/100000000 [00:20<00:00, 4830563.47tokens/s]
Shard 13: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99995675/100000000 [00:20<00:00, 4980598.91tokens/s]
Shard 14: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99992321/100000000 [00:23<00:00, 4200379.10tokens/s]
Shard 15: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99994498/100000000 [00:18<00:00, 5323752.34tokens/s]
Shard 16: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999908/100000000 [00:22<00:00, 4433287.97tokens/s]
Shard 17: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999237/100000000 [00:20<00:00, 4942419.48tokens/s]
Shard 18: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999699/100000000 [00:20<00:00, 4894172.19tokens/s]
Shard 19: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99996354/100000000 [00:20<00:00, 4936630.87tokens/s]
Shard 20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997912/100000000 [00:24<00:00, 4033879.44tokens/s]
Shard 21: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999786/100000000 [00:19<00:00, 5042057.07tokens/s]
Shard 22: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997168/100000000 [00:22<00:00, 4436087.65tokens/s]
Shard 23: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999515/100000000 [00:18<00:00, 5340470.22tokens/s]
Shard 24: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99994716/100000000 [00:21<00:00, 4741783.61tokens/s]
Shard 25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998669/100000000 [00:24<00:00, 4144959.47tokens/s]
Shard 26: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99994587/100000000 [00:23<00:00, 4304736.85tokens/s]
Shard 27: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999173/100000000 [00:21<00:00, 4617370.47tokens/s]
Shard 28: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999394/100000000 [00:23<00:00, 4246257.53tokens/s]
Shard 29: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999350/100000000 [00:20<00:00, 4852512.49tokens/s]
Shard 30: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999935/100000000 [00:22<00:00, 4462363.91tokens/s]
Shard 31: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999984/100000000 [00:20<00:00, 4895374.91tokens/s]
Shard 32: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99966920/100000000 [00:21<00:00, 4665725.36tokens/s]
Shard 33: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999906/100000000 [00:20<00:00, 4829720.11tokens/s]
Shard 34: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999699/100000000 [00:17<00:00, 5657161.95tokens/s]
Shard 35: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999462/100000000 [00:16<00:00, 6062507.77tokens/s]
Shard 36: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99993819/100000000 [00:20<00:00, 4868753.78tokens/s]
Shard 37: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999986/100000000 [00:19<00:00, 5061171.81tokens/s]
Shard 38: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999506/100000000 [00:19<00:00, 5061363.96tokens/s]
Shard 39: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998491/100000000 [00:19<00:00, 5120591.32tokens/s]
Shard 40: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999686/100000000 [00:19<00:00, 5084391.57tokens/s]
Shard 41: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999210/100000000 [00:18<00:00, 5424959.56tokens/s]
Shard 42: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99982614/100000000 [00:20<00:00, 4836125.87tokens/s]
Shard 43: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999919/100000000 [00:21<00:00, 4680441.02tokens/s]
Shard 44: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999958/100000000 [00:22<00:00, 4396717.71tokens/s]
Shard 45: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999884/100000000 [00:21<00:00, 4589613.67tokens/s]
Shard 46: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999708/100000000 [00:23<00:00, 4344788.85tokens/s]
Shard 47: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99992492/100000000 [00:22<00:00, 4438196.88tokens/s]
Shard 48: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99996591/100000000 [00:21<00:00, 4572207.15tokens/s]
Shard 49: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99996813/100000000 [00:22<00:00, 4451233.94tokens/s]
Shard 50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997449/100000000 [00:17<00:00, 5681567.52tokens/s]
Shard 51: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999842/100000000 [00:17<00:00, 5720794.78tokens/s]
Shard 52: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999850/100000000 [00:00<00:00, 248291428.02tokens/s]
Shard 53: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997806/100000000 [00:00<00:00, 249572581.07tokens/s]
Shard 54: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99984585/100000000 [00:00<00:00, 255008923.49tokens/s]
Shard 55: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999298/100000000 [00:00<00:00, 244428288.89tokens/s]
Shard 56: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999282/100000000 [00:00<00:00, 238332818.22tokens/s]
Shard 57: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999855/100000000 [00:00<00:00, 250328879.08tokens/s]
Shard 58: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999276/100000000 [00:00<00:00, 253046961.66tokens/s]
Shard 59: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999625/100000000 [00:00<00:00, 246628540.34tokens/s]
Shard 60: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999919/100000000 [00:00<00:00, 252615144.42tokens/s]
Shard 61: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999599/100000000 [00:00<00:00, 249020206.47tokens/s]
Shard 62: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998880/100000000 [00:00<00:00, 248549448.10tokens/s]
Shard 63: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999237/100000000 [00:00<00:00, 249887069.42tokens/s]
Shard 64: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999879/100000000 [00:00<00:00, 245769591.02tokens/s]
Shard 65: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999637/100000000 [00:00<00:00, 247030349.23tokens/s]
Shard 66: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99996361/100000000 [00:00<00:00, 247920555.34tokens/s]
Shard 67: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999579/100000000 [00:00<00:00, 249842524.30tokens/s]
Shard 68: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999974/100000000 [00:00<00:00, 250945189.10tokens/s]
Shard 69: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999062/100000000 [00:00<00:00, 249359231.06tokens/s]
Shard 70: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999505/100000000 [00:00<00:00, 255807314.48tokens/s]
Shard 71: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99995068/100000000 [00:00<00:00, 249688916.30tokens/s]
Shard 72: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999256/100000000 [00:00<00:00, 248609459.84tokens/s]
Shard 73: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998978/100000000 [00:00<00:00, 250795639.65tokens/s]
Shard 74: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999526/100000000 [00:00<00:00, 247011598.81tokens/s]
Shard 75: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999968/100000000 [00:00<00:00, 248216782.71tokens/s]
Shard 76: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997615/100000000 [00:00<00:00, 246891285.16tokens/s]
Shard 77: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999376/100000000 [00:00<00:00, 247754779.63tokens/s]
Shard 78: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999868/100000000 [00:00<00:00, 246986988.68tokens/s]
Shard 79: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99952079/100000000 [00:00<00:00, 251541818.45tokens/s]
Shard 80: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999706/100000000 [00:00<00:00, 245252677.40tokens/s]
Shard 81: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998256/100000000 [00:00<00:00, 246694647.26tokens/s]
Shard 82: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998705/100000000 [00:00<00:00, 245240041.08tokens/s]
Shard 83: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999959/100000000 [00:00<00:00, 238719673.74tokens/s]
Shard 84: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99990999/100000000 [00:00<00:00, 245991509.84tokens/s]
Shard 85: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999894/100000000 [00:00<00:00, 248607214.67tokens/s]
Shard 86: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998828/100000000 [00:00<00:00, 247436254.98tokens/s]
Shard 87: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999100/100000000 [00:00<00:00, 247020729.89tokens/s]
Shard 88: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99997157/100000000 [00:00<00:00, 246071897.23tokens/s]
Shard 89: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999271/100000000 [00:00<00:00, 247518100.78tokens/s]
Shard 90: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999897/100000000 [00:00<00:00, 245912866.05tokens/s]
Shard 91: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99998966/100000000 [00:00<00:00, 248266012.33tokens/s]
Shard 92: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999502/100000000 [00:00<00:00, 241762981.77tokens/s]
Shard 93: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999683/100000000 [00:00<00:00, 245212617.11tokens/s]
Shard 94: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999562/100000000 [00:00<00:00, 247554320.95tokens/s]
Shard 95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999367/100000000 [00:00<00:00, 245426926.25tokens/s]
Shard 96: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99945054/100000000 [00:00<00:00, 247536556.10tokens/s]
Shard 97: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999910/100000000 [00:00<00:00, 244775846.84tokens/s]
Shard 98: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████▉| 99999499/100000000 [00:00<00:00, 247735767.91tokens/s]
Shard 99:  54%|██████████████████████████████████████████████████████▌                                              | 53989101/100000000 [00:01<00:01, 33339154.06tokens/s]
ubuntu@152-69-203-83:~$ pip install transformers
Defaulting to user installation because normal site-packages is not writeable
Collecting transformers
  Downloading transformers-4.45.1-py3-none-any.whl.metadata (44 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.4/44.4 kB 1.4 MB/s eta 0:00:00
Requirement already satisfied: filelock in /usr/lib/python3/dist-packages (from transformers) (3.6.0)
Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in ./.local/lib/python3.10/site-packages (from transformers) (0.25.1)
Requirement already satisfied: numpy>=1.17 in ./.local/lib/python3.10/site-packages (from transformers) (1.23.0)
Requirement already satisfied: packaging>=20.0 in /usr/lib/python3/dist-packages (from transformers) (21.3)
Requirement already satisfied: pyyaml>=5.1 in /usr/lib/python3/dist-packages (from transformers) (5.4.1)
Requirement already satisfied: regex!=2019.12.17 in ./.local/lib/python3.10/site-packages (from transformers) (2024.9.11)
Requirement already satisfied: requests in ./.local/lib/python3.10/site-packages (from transformers) (2.32.3)
Collecting safetensors>=0.4.1 (from transformers)
  Downloading safetensors-0.4.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
Collecting tokenizers<0.21,>=0.20 (from transformers)
  Downloading tokenizers-0.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
Requirement already satisfied: tqdm>=4.27 in ./.local/lib/python3.10/site-packages (from transformers) (4.66.5)
Requirement already satisfied: fsspec>=2023.5.0 in ./.local/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.6.1)
Requirement already satisfied: typing-extensions>=3.7.4.3 in ./.local/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.8.0)
Requirement already satisfied: charset-normalizer<4,>=2 in ./.local/lib/python3.10/site-packages (from requests->transformers) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests->transformers) (3.3)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/lib/python3/dist-packages (from requests->transformers) (1.26.5)
Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests->transformers) (2020.6.20)
Downloading transformers-4.45.1-py3-none-any.whl (9.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.9/9.9 MB 121.2 MB/s eta 0:00:00
Downloading safetensors-0.4.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (435 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 435.0/435.0 kB 66.5 MB/s eta 0:00:00
Downloading tokenizers-0.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 137.8 MB/s eta 0:00:00
DEPRECATION: flatbuffers 1.12.1-git20200711.33e2d80-dfsg1-0.6 has a non-standard version number. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of flatbuffers or contact the author to suggest that they release a version with a conforming version number. Discussion can be found at https://github.com/pypa/pip/issues/12063
Installing collected packages: safetensors, tokenizers, transformers
Successfully installed safetensors-0.4.5 tokenizers-0.20.0 transformers-4.45.1

[notice] A new release of pip is available: 23.3.1 -> 24.2
[notice] To update, run: python3 -m pip install --upgrade pip
ubuntu@152-69-203-83:~$ 


ubuntu@152-69-203-83:~/msgpt$ torchrun --standalone --nproc_per_node=8 train_gpt_10BTokens.py 
master_addr is only used for static rdzv_backend and when rdzv_endpoint is not specified.
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
True
True
True
True
True
total desired batch size: 524288
==> calculated gradient accumulation steps: 1
found 99 shards for train split
True
True
True
found 1 shards for val split
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 453, in <module>
    model.to(device) # move the model to the GPU on Cloudbox
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 1145, in to
    return self._apply(convert)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 820, in _apply
    param_applied = fn(param)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 1143, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 39.39 GiB total capacity; 297.94 MiB already allocated; 18.44 MiB free; 310.00 MiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    return dist._verify_params_across_processes(process_group, tensors, logger)
    RuntimeErrorreturn dist._verify_params_across_processes(process_group, tensors, logger): 
[4] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
RuntimeError: [5] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [2] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [6] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [3] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [7] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [1] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 60857) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/usr/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.0.1', 'console_scripts', 'torchrun')())
  File "/usr/lib/python3/dist-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 344, in wrapper
    return f(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/distributed/run.py", line 794, in main
    run(args)
  File "/usr/lib/python3/dist-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/usr/lib/python3/dist-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/usr/lib/python3/dist-packages/torch/distributed/launcher/api.py", line 248, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train_gpt_10BTokens.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 60858)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 60859)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 60860)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[4]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 4 (local_rank: 4)
  exitcode  : 1 (pid: 60861)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[5]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 5 (local_rank: 5)
  exitcode  : 1 (pid: 60862)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[6]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 6 (local_rank: 6)
  exitcode  : 1 (pid: 60863)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[7]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 7 (local_rank: 7)
  exitcode  : 1 (pid: 60864)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-10-04_21:10:25
  host      : 152-69-203-83
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 60857)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
ubuntu@152-69-203-83:~/msgpt$ torchrun --standalone --nproc_per_node=8 train_gpt_10BTokens.py 
master_addr is only used for static rdzv_backend and when rdzv_endpoint is not specified.
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[W socket.cpp:601] [c10d] The client socket has failed to connect to [152-69-203-83]:41933 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [152-69-203-83]:41933 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [152-69-203-83]:41933 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [152-69-203-83]:41933 (errno: 22 - Invalid argument).
True
total desired batch size: 524288
==> calculated gradient accumulation steps: 2
found 99 shards for train split
True
True
True
True
True
True
True
found 1 shards for val split
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 453, in <module>
    model.to(device) # move the model to the GPU on Cloudbox
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 1145, in to
    return self._apply(convert)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 820, in _apply
    param_applied = fn(param)
  File "/usr/lib/python3/dist-packages/torch/nn/modules/module.py", line 1143, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 39.39 GiB total capacity; 297.94 MiB already allocated; 18.44 MiB free; 310.00 MiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [1] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [7] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [5] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
Traceback (most recent call last):
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>

  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [4] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [3] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [2] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
Traceback (most recent call last):
  File "/home/ubuntu/msgpt/train_gpt_10BTokens.py", line 462, in <module>
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
  File "/usr/lib/python3/dist-packages/torch/nn/parallel/distributed.py", line 674, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3/dist-packages/torch/distributed/utils.py", line 118, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: [6] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer. This may indicate a possible application crash on rank 0 or a network set up issue.
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 61145) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/usr/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.0.1', 'console_scripts', 'torchrun')())
  File "/usr/lib/python3/dist-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 344, in wrapper
    return f(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/torch/distributed/run.py", line 794, in main
    run(args)
  File "/usr/lib/python3/dist-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/usr/lib/python3/dist-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/usr/lib/python3/dist-packages/torch/distributed/launcher/api.py", line 248, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train_gpt_10BTokens.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 61146)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 61147)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 61148)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[4]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 4 (local_rank: 4)
  exitcode  : 1 (pid: 61149)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[5]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 5 (local_rank: 5)
  exitcode  : 1 (pid: 61150)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[6]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 6 (local_rank: 6)
  exitcode  : 1 (pid: 61151)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[7]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 7 (local_rank: 7)
  exitcode  : 1 (pid: 61152)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-10-04_21:11:47
  host      : 152-69-203-83
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 61145)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
ubuntu@152-69-203-83:~/msgpt$ 


pip install tiktoken
pip install datasets
pip install transformers
pip install numpy==1.23

-----------------------------------------------

ubuntu@164-152-26-183:~/msgpt$ torchrun --standalone --nproc_per_node=8 train_gpt_10BTokens.py 
master_addr is only used for static rdzv_backend and when rdzv_endpoint is not specified.
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:55069 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:55069 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:55069 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:55069 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:55069 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:55069 (errno: 22 - Invalid argument).
total desired batch size: 524288
==> calculated gradient accumulation steps: 1
found 99 shards for train split
found 1 shards for val split
Model size: 0.23 GB
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
using fused AdamW optimizer
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
num decay parameter tensors: 50, with 124,354,560 parametersnum decay parameter tensors: 50, with 124,354,560 parameters

num no decay parameter tensors: 98, with 121,344 parametersnum no decay parameter tensors: 98, with 121,344 parameters

num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
using fused AdamW optimizer
using fused AdamW optimizer
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
validation loss: 10.9846
HellaSwag accuracy: 2442/10042=0.2432
step:    0 | loss: 10.984997 | lr: 8.3916e-07 | norm: 5.6500, dt: 12588.70ms, tok/sec: 41647.51
step:    1 | loss: 10.966601 | lr: 1.6783e-06 | norm: 5.5017, dt: 439.95ms, tok/sec: 1191704.25
step:    2 | loss: 10.930476 | lr: 2.5175e-06 | norm: 5.5983, dt: 440.78ms, tok/sec: 1189442.39
step:    3 | loss: 10.867207 | lr: 3.3566e-06 | norm: 5.6312, dt: 441.97ms, tok/sec: 1186240.01
step:    4 | loss: 10.796509 | lr: 4.1958e-06 | norm: 5.3467, dt: 441.08ms, tok/sec: 1188640.02
step:    5 | loss: 10.708714 | lr: 5.0350e-06 | norm: 5.0610, dt: 441.21ms, tok/sec: 1188298.95
step:    6 | loss: 10.614870 | lr: 5.8741e-06 | norm: 4.5635, dt: 441.69ms, tok/sec: 1187016.72
step:    7 | loss: 10.521290 | lr: 6.7133e-06 | norm: 4.0480, dt: 441.53ms, tok/sec: 1187439.12
step:    8 | loss: 10.421499 | lr: 7.5524e-06 | norm: 3.7063, dt: 440.85ms, tok/sec: 1189260.35
step:    9 | loss: 10.351803 | lr: 8.3916e-06 | norm: 3.3809, dt: 441.03ms, tok/sec: 1188792.95
step:   10 | loss: 10.262074 | lr: 9.2308e-06 | norm: 3.1972, dt: 443.28ms, tok/sec: 1182746.21
step:   11 | loss: 10.197437 | lr: 1.0070e-05 | norm: 3.0131, dt: 441.98ms, tok/sec: 1186219.53
step:   12 | loss: 10.127195 | lr: 1.0909e-05 | norm: 2.8488, dt: 441.68ms, tok/sec: 1187030.18
step:   13 | loss: 10.065224 | lr: 1.1748e-05 | norm: 2.7092, dt: 441.57ms, tok/sec: 1187337.18
step:   14 | loss: 9.985319 | lr: 1.2587e-05 | norm: 2.6439, dt: 442.84ms, tok/sec: 1183928.07
step:   15 | loss: 9.934050 | lr: 1.3427e-05 | norm: 2.5374, dt: 441.67ms, tok/sec: 1187058.37
step:   16 | loss: 9.885630 | lr: 1.4266e-05 | norm: 2.4145, dt: 441.59ms, tok/sec: 1187268.59
step:   17 | loss: 9.841616 | lr: 1.5105e-05 | norm: 2.3405, dt: 441.53ms, tok/sec: 1187447.46
step:   18 | loss: 9.846683 | lr: 1.5944e-05 | norm: 2.1963, dt: 442.05ms, tok/sec: 1186041.67
step:   19 | loss: 9.825327 | lr: 1.6783e-05 | norm: 2.2202, dt: 442.62ms, tok/sec: 1184497.55
step:   20 | loss: 9.723330 | lr: 1.7622e-05 | norm: 2.1832, dt: 441.47ms, tok/sec: 1187585.33
step:   21 | loss: 9.676581 | lr: 1.8462e-05 | norm: 2.1861, dt: 441.09ms, tok/sec: 1188611.75
step:   22 | loss: 9.655930 | lr: 1.9301e-05 | norm: 2.1420, dt: 441.35ms, tok/sec: 1187925.35
step:   23 | loss: 9.639503 | lr: 2.0140e-05 | norm: 2.1051, dt: 441.48ms, tok/sec: 1187561.60
step:   24 | loss: 9.586742 | lr: 2.0979e-05 | norm: 2.1296, dt: 441.52ms, tok/sec: 1187473.75






ubuntu@164-152-26-183:~/msgpt$ torchrun --standalone --nproc_per_node=8 train_gpt_10BTokens.py
master_addr is only used for static rdzv_backend and when rdzv_endpoint is not specified.
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:54095 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:54095 (errno: 22 - Invalid argument).
[W socket.cpp:601] [c10d] The client socket has failed to connect to [164-152-26-183]:54095 (errno: 22 - Invalid argument).
total desired batch size: 524288
==> calculated gradient accumulation steps: 1
found 99 shards for train split
found 1 shards for val split
Model size: 0.23 GB
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
using fused AdamW optimizer
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
using fused AdamW optimizer
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
num decay parameter tensors: 50, with 124,354,560 parameters
num no decay parameter tensors: 98, with 121,344 parameters
using fused AdamW optimizer
using fused AdamW optimizer
using fused AdamW optimizer
using fused AdamW optimizer
validation loss: 10.9846
HellaSwag accuracy: 2441/10042=0.2431
step:    0 | loss: 10.984997 | lr: 8.3916e-07 | norm: 5.6500, dt: 12496.13ms, tok/sec: 41956.04
step:    1 | loss: 10.966601 | lr: 1.6783e-06 | norm: 5.5017, dt: 442.30ms, tok/sec: 1185378.68
step:    2 | loss: 10.930476 | lr: 2.5175e-06 | norm: 5.5983, dt: 442.48ms, tok/sec: 1184888.79
step:    3 | loss: 10.867207 | lr: 3.3566e-06 | norm: 5.6312, dt: 441.59ms, tok/sec: 1187265.38
step:    4 | loss: 10.796509 | lr: 4.1958e-06 | norm: 5.3467, dt: 440.60ms, tok/sec: 1189951.51
step:    5 | loss: 10.708714 | lr: 5.0350e-06 | norm: 5.0610, dt: 441.13ms, tok/sec: 1188503.82
step:    6 | loss: 10.614870 | lr: 5.8741e-06 | norm: 4.5635, dt: 441.87ms, tok/sec: 1186517.79
step:    7 | loss: 10.521290 | lr: 6.7133e-06 | norm: 4.0480, dt: 441.90ms, tok/sec: 1186443.53
step:    8 | loss: 10.421499 | lr: 7.5524e-06 | norm: 3.7063, dt: 441.05ms, tok/sec: 1188715.19
step:    9 | loss: 10.351803 | lr: 8.3916e-06 | norm: 3.3809, dt: 441.93ms, tok/sec: 1186346.88
step:   10 | loss: 10.262074 | lr: 9.2308e-06 | norm: 3.1972, dt: 441.37ms, tok/sec: 1187875.94
step:   11 | loss: 10.197437 | lr: 1.0070e-05 | norm: 3.0131, dt: 441.55ms, tok/sec: 1187386.54
step:   12 | loss: 10.127195 | lr: 1.0909e-05 | norm: 2.8488, dt: 440.78ms, tok/sec: 1189452.69
step:   13 | loss: 10.065224 | lr: 1.1748e-05 | norm: 2.7092, dt: 441.25ms, tok/sec: 1188198.14
step:   14 | loss: 9.985319 | lr: 1.2587e-05 | norm: 2.6439, dt: 441.59ms, tok/sec: 1187264.74
step:   15 | loss: 9.934050 | lr: 1.3427e-05 | norm: 2.5374, dt: 441.44ms, tok/sec: 1187664.22
step:   16 | loss: 9.885630 | lr: 1.4266e-05 | norm: 2.4145, dt: 441.23ms, tok/sec: 1188246.30
step:   17 | loss: 9.841616 | lr: 1.5105e-05 | norm: 2.3405, dt: 440.86ms, tok/sec: 1189252.63
step:   18 | loss: 9.846683 | lr: 1.5944e-05 | norm: 2.1963, dt: 441.50ms, tok/sec: 1187525.69
step:   19 | loss: 9.825327 | lr: 1.6783e-05 | norm: 2.2202, dt: 441.82ms, tok/sec: 1186663.78
step:   20 | loss: 9.723330 | lr: 1.7622e-05 | norm: 2.1832, dt: 442.12ms, tok/sec: 1185859.39
step:   21 | loss: 9.676581 | lr: 1.8462e-05 | norm: 2.1861, dt: 442.12ms, tok/sec: 1185858.11
step:   22 | loss: 9.655930 | lr: 1.9301e-05 | norm: 2.1420, dt: 441.23ms, tok/sec: 1188234.74
step:   23 | loss: 9.639503 | lr: 2.0140e-05 | norm: 2.1051, dt: 442.16ms, tok/sec: 1185732.78
step:   24 | loss: 9.586742 | lr: 2.0979e-05 | norm: 2.1296, dt: 442.19ms, tok/sec: 1185661.82
step:   25 | loss: 9.570937 | lr: 2.1818e-05 | norm: 2.1170, dt: 441.58ms, tok/sec: 1187308.33
step:   26 | loss: 9.549131 | lr: 2.2657e-05 | norm: 2.0815, dt: 441.74ms, tok/sec: 1186864.88
step:   27 | loss: 9.519006 | lr: 2.3497e-05 | norm: 2.1041, dt: 441.43ms, tok/sec: 1187691.16
step:   28 | loss: 9.500747 | lr: 2.4336e-05 | norm: 2.1007, dt: 442.09ms, tok/sec: 1185929.09
step:   29 | loss: 9.470724 | lr: 2.5175e-05 | norm: 2.1240, dt: 441.58ms, tok/sec: 1187292.94
step:   30 | loss: 9.491784 | lr: 2.6014e-05 | norm: 2.0402, dt: 441.23ms, tok/sec: 1188244.37
step:   31 | loss: 9.440409 | lr: 2.6853e-05 | norm: 2.0940, dt: 442.66ms, tok/sec: 1184412.06
step:   32 | loss: 9.423861 | lr: 2.7692e-05 | norm: 2.0612, dt: 442.07ms, tok/sec: 1185989.22
step:   33 | loss: 9.393246 | lr: 2.8531e-05 | norm: 2.0798, dt: 441.71ms, tok/sec: 1186943.04
step:   34 | loss: 9.317858 | lr: 2.9371e-05 | norm: 2.1475, dt: 441.45ms, tok/sec: 1187643.06
step:   35 | loss: 9.332920 | lr: 3.0210e-05 | norm: 2.0384, dt: 441.87ms, tok/sec: 1186525.47
step:   36 | loss: 9.322514 | lr: 3.1049e-05 | norm: 2.0034, dt: 444.42ms, tok/sec: 1179717.05
step:   37 | loss: 9.269900 | lr: 3.1888e-05 | norm: 2.0367, dt: 442.78ms, tok/sec: 1184087.44
step:   38 | loss: 9.239744 | lr: 3.2727e-05 | norm: 2.0255, dt: 441.85ms, tok/sec: 1186574.77
step:   39 | loss: 9.233005 | lr: 3.3566e-05 | norm: 1.9830, dt: 442.05ms, tok/sec: 1186033.99
step:   40 | loss: 9.209680 | lr: 3.4406e-05 | norm: 1.9550, dt: 441.99ms, tok/sec: 1186189.46
step:   41 | loss: 9.137701 | lr: 3.5245e-05 | norm: 2.0455, dt: 442.20ms, tok/sec: 1185645.83
step:   42 | loss: 9.111229 | lr: 3.6084e-05 | norm: 1.9953, dt: 441.95ms, tok/sec: 1186306.56
step:   43 | loss: 9.085827 | lr: 3.6923e-05 | norm: 2.0010, dt: 442.58ms, tok/sec: 1184609.85
step:   44 | loss: 9.066192 | lr: 3.7762e-05 | norm: 1.9445, dt: 441.74ms, tok/sec: 1186870.65
step:   45 | loss: 9.046094 | lr: 3.8601e-05 | norm: 1.9200, dt: 442.00ms, tok/sec: 1186174.10
step:   46 | loss: 8.989664 | lr: 3.9441e-05 | norm: 1.9263, dt: 442.64ms, tok/sec: 1184446.51
step:   47 | loss: 8.958224 | lr: 4.0280e-05 | norm: 1.9253, dt: 442.77ms, tok/sec: 1184105.93
step:   48 | loss: 8.926687 | lr: 4.1119e-05 | norm: 1.8769, dt: 442.60ms, tok/sec: 1184551.78
step:   49 | loss: 8.858720 | lr: 4.1958e-05 | norm: 1.8764, dt: 442.31ms, tok/sec: 1185349.29
step:   50 | loss: 8.860399 | lr: 4.2797e-05 | norm: 1.7767, dt: 442.57ms, tok/sec: 1184655.16
step:   51 | loss: 8.809139 | lr: 4.3636e-05 | norm: 1.7640, dt: 443.23ms, tok/sec: 1182890.63
step:   52 | loss: 8.777260 | lr: 4.4476e-05 | norm: 1.7381, dt: 444.88ms, tok/sec: 1178505.69
step:   53 | loss: 8.732130 | lr: 4.5315e-05 | norm: 1.7334, dt: 443.85ms, tok/sec: 1181238.57
step:   54 | loss: 8.725812 | lr: 4.6154e-05 | norm: 1.6780, dt: 442.64ms, tok/sec: 1184468.20
step:   55 | loss: 8.680664 | lr: 4.6993e-05 | norm: 1.6565, dt: 444.13ms, tok/sec: 1180495.38
step:   56 | loss: 8.646110 | lr: 4.7832e-05 | norm: 1.6341, dt: 444.92ms, tok/sec: 1178385.06
step:   57 | loss: 8.618370 | lr: 4.8671e-05 | norm: 1.6466, dt: 444.34ms, tok/sec: 1179913.28
step:   58 | loss: 8.564542 | lr: 4.9510e-05 | norm: 1.6071, dt: 444.99ms, tok/sec: 1178191.24
step:   59 | loss: 8.527041 | lr: 5.0350e-05 | norm: 1.5452, dt: 444.69ms, tok/sec: 1178995.37
step:   60 | loss: 8.501081 | lr: 5.1189e-05 | norm: 1.5322, dt: 444.06ms, tok/sec: 1180656.37
step:   61 | loss: 8.486146 | lr: 5.2028e-05 | norm: 1.5199, dt: 443.04ms, tok/sec: 1183388.42
step:   62 | loss: 8.401423 | lr: 5.2867e-05 | norm: 1.5363, dt: 445.70ms, tok/sec: 1176335.76
step:   63 | loss: 8.361864 | lr: 5.3706e-05 | norm: 1.4624, dt: 443.94ms, tok/sec: 1180981.01
step:   64 | loss: 8.321377 | lr: 5.4545e-05 | norm: 1.4525, dt: 444.87ms, tok/sec: 1178515.16
step:   65 | loss: 8.305117 | lr: 5.5385e-05 | norm: 1.4362, dt: 445.07ms, tok/sec: 1178000.63
step:   66 | loss: 8.247305 | lr: 5.6224e-05 | norm: 1.4315, dt: 445.23ms, tok/sec: 1177572.94
step:   67 | loss: 8.229324 | lr: 5.7063e-05 | norm: 1.3292, dt: 444.68ms, tok/sec: 1179016.23
step:   68 | loss: 8.195562 | lr: 5.7902e-05 | norm: 1.3425, dt: 444.78ms, tok/sec: 1178745.11
step:   69 | loss: 8.107881 | lr: 5.8741e-05 | norm: 1.4203, dt: 444.38ms, tok/sec: 1179821.49
step:   70 | loss: 8.059123 | lr: 5.9580e-05 | norm: 1.3845, dt: 445.42ms, tok/sec: 1177064.27
step:   71 | loss: 8.066184 | lr: 6.0420e-05 | norm: 1.2527, dt: 445.09ms, tok/sec: 1177943.84
step:   72 | loss: 7.991982 | lr: 6.1259e-05 | norm: 1.2850, dt: 445.20ms, tok/sec: 1177655.55
step:   73 | loss: 7.958554 | lr: 6.2098e-05 | norm: 1.1963, dt: 445.29ms, tok/sec: 1177416.57
step:   74 | loss: 7.904431 | lr: 6.2937e-05 | norm: 1.1167, dt: 444.38ms, tok/sec: 1179817.06
step:   75 | loss: 7.906116 | lr: 6.3776e-05 | norm: 1.1039, dt: 445.11ms, tok/sec: 1177877.59
step:   76 | loss: 7.841115 | lr: 6.4615e-05 | norm: 1.1806, dt: 445.23ms, tok/sec: 1177554.02
step:   77 | loss: 7.845302 | lr: 6.5455e-05 | norm: 1.0617, dt: 444.91ms, tok/sec: 1178419.16
step:   78 | loss: 7.802566 | lr: 6.6294e-05 | norm: 1.1025, dt: 445.34ms, tok/sec: 1177286.72
step:   79 | loss: 7.752333 | lr: 6.7133e-05 | norm: 1.2820, dt: 445.06ms, tok/sec: 1178029.03
step:   80 | loss: 7.720936 | lr: 6.7972e-05 | norm: 1.0814, dt: 446.18ms, tok/sec: 1175057.23
step:   81 | loss: 7.717269 | lr: 6.8811e-05 | norm: 0.8792, dt: 447.54ms, tok/sec: 1171491.59
step:   82 | loss: 7.660481 | lr: 6.9650e-05 | norm: 1.0291, dt: 446.15ms, tok/sec: 1175140.12
step:   83 | loss: 7.612842 | lr: 7.0490e-05 | norm: 1.0907, dt: 444.87ms, tok/sec: 1178527.16
step:   84 | loss: 7.600497 | lr: 7.1329e-05 | norm: 0.8359, dt: 445.47ms, tok/sec: 1176929.46
step:   85 | loss: 7.558265 | lr: 7.2168e-05 | norm: 0.8811, dt: 445.90ms, tok/sec: 1175794.85
step:   86 | loss: 7.541884 | lr: 7.3007e-05 | norm: 0.8609, dt: 446.03ms, tok/sec: 1175455.45
step:   87 | loss: 7.537360 | lr: 7.3846e-05 | norm: 0.7605, dt: 445.81ms, tok/sec: 1176040.08
step:   88 | loss: 7.517060 | lr: 7.4685e-05 | norm: 0.8456, dt: 447.28ms, tok/sec: 1172169.74
step:   89 | loss: 7.543760 | lr: 7.5524e-05 | norm: 0.6485, dt: 445.68ms, tok/sec: 1176368.49
step:   90 | loss: 7.462922 | lr: 7.6364e-05 | norm: 0.7967, dt: 446.35ms, tok/sec: 1174624.77
step:   91 | loss: 7.447705 | lr: 7.7203e-05 | norm: 0.6629, dt: 446.96ms, tok/sec: 1172998.83
step:   92 | loss: 7.467535 | lr: 7.8042e-05 | norm: 0.7131, dt: 446.81ms, tok/sec: 1173405.05
step:   93 | loss: 7.519574 | lr: 7.8881e-05 | norm: 0.5366, dt: 445.90ms, tok/sec: 1175808.68
step:   94 | loss: 7.520703 | lr: 7.9720e-05 | norm: 0.5810, dt: 446.64ms, tok/sec: 1173854.79
step:   95 | loss: 7.505235 | lr: 8.0559e-05 | norm: 0.4679, dt: 445.69ms, tok/sec: 1176357.79
step:   96 | loss: 7.460219 | lr: 8.1399e-05 | norm: 0.6015, dt: 446.58ms, tok/sec: 1174018.35
step:   97 | loss: 7.416075 | lr: 8.2238e-05 | norm: 0.4880, dt: 447.08ms, tok/sec: 1172687.32
step:   98 | loss: 7.439541 | lr: 8.3077e-05 | norm: 0.5034, dt: 446.45ms, tok/sec: 1174338.10
step:   99 | loss: 7.383360 | lr: 8.3916e-05 | norm: 0.5162, dt: 446.22ms, tok/sec: 1174957.41
step:  100 | loss: 7.438282 | lr: 8.4755e-05 | norm: 0.6728, dt: 447.28ms, tok/sec: 1172175.36
step:  101 | loss: 7.362479 | lr: 8.5594e-05 | norm: 0.6511, dt: 449.64ms, tok/sec: 1166021.49
step:  102 | loss: 7.347324 | lr: 8.6434e-05 | norm: 0.4418, dt: 446.83ms, tok/sec: 1173346.82
step:  103 | loss: 7.235550 | lr: 8.7273e-05 | norm: 0.5553, dt: 447.50ms, tok/sec: 1171595.19
step:  104 | loss: 7.320236 | lr: 8.8112e-05 | norm: 0.6486, dt: 447.92ms, tok/sec: 1170491.40
step:  105 | loss: 7.331547 | lr: 8.8951e-05 | norm: 0.5998, dt: 446.73ms, tok/sec: 1173601.69
step:  106 | loss: 7.279395 | lr: 8.9790e-05 | norm: 0.5306, dt: 446.80ms, tok/sec: 1173430.72
step:  107 | loss: 7.222516 | lr: 9.0629e-05 | norm: 0.5213, dt: 447.26ms, tok/sec: 1172214.73
step:  108 | loss: 7.281051 | lr: 9.1469e-05 | norm: 0.4255, dt: 447.76ms, tok/sec: 1170903.36
step:  109 | loss: 7.297616 | lr: 9.2308e-05 | norm: 0.4192, dt: 448.47ms, tok/sec: 1169062.05
step:  110 | loss: 7.224943 | lr: 9.3147e-05 | norm: 0.5073, dt: 447.34ms, tok/sec: 1172016.68
step:  111 | loss: 7.205843 | lr: 9.3986e-05 | norm: 0.7830, dt: 448.04ms, tok/sec: 1170183.08
step:  112 | loss: 7.175447 | lr: 9.4825e-05 | norm: 0.8159, dt: 448.33ms, tok/sec: 1169417.66
step:  113 | loss: 7.236903 | lr: 9.5664e-05 | norm: 1.1234, dt: 447.75ms, tok/sec: 1170936.41
step:  114 | loss: 7.227196 | lr: 9.6503e-05 | norm: 1.0139, dt: 447.51ms, tok/sec: 1171572.10
step:  115 | loss: 7.185486 | lr: 9.7343e-05 | norm: 0.7300, dt: 450.97ms, tok/sec: 1162578.00
step:  116 | loss: 7.115022 | lr: 9.8182e-05 | norm: 0.9093, dt: 449.01ms, tok/sec: 1167658.51
step:  117 | loss: 7.028243 | lr: 9.9021e-05 | norm: 0.6225, dt: 448.07ms, tok/sec: 1170108.98
step:  118 | loss: 7.055861 | lr: 9.9860e-05 | norm: 0.6153, dt: 448.46ms, tok/sec: 1169080.70
step:  119 | loss: 7.100358 | lr: 1.0070e-04 | norm: 0.5763, dt: 448.11ms, tok/sec: 1170010.62
step:  120 | loss: 7.055558 | lr: 1.0154e-04 | norm: 0.6547, dt: 448.07ms, tok/sec: 1170104.62
step:  121 | loss: 7.043574 | lr: 1.0238e-04 | norm: 0.6398, dt: 450.64ms, tok/sec: 1163439.73
step:  122 | loss: 7.018792 | lr: 1.0322e-04 | norm: 0.4199, dt: 448.49ms, tok/sec: 1168996.80
step:  123 | loss: 7.009007 | lr: 1.0406e-04 | norm: 0.6196, dt: 448.19ms, tok/sec: 1169799.00
step:  124 | loss: 7.025376 | lr: 1.0490e-04 | norm: 0.6824, dt: 449.64ms, tok/sec: 1166008.51
step:  125 | loss: 7.009961 | lr: 1.0573e-04 | norm: 0.5455, dt: 449.12ms, tok/sec: 1167378.33
step:  126 | loss: 6.996265 | lr: 1.0657e-04 | norm: 0.7052, dt: 447.82ms, tok/sec: 1170763.72
step:  127 | loss: 6.947212 | lr: 1.0741e-04 | norm: 0.5965, dt: 449.64ms, tok/sec: 1166013.46
step:  128 | loss: 6.917241 | lr: 1.0825e-04 | norm: 0.5550, dt: 448.85ms, tok/sec: 1168064.76
step:  129 | loss: 6.894266 | lr: 1.0909e-04 | norm: 0.4855, dt: 449.87ms, tok/sec: 1165427.01
step:  130 | loss: 6.950639 | lr: 1.0993e-04 | norm: 0.4155, dt: 449.38ms, tok/sec: 1166691.47
step:  131 | loss: 6.903189 | lr: 1.1077e-04 | norm: 0.4674, dt: 448.26ms, tok/sec: 1169612.34
step:  132 | loss: 6.905964 | lr: 1.1161e-04 | norm: 0.5491, dt: 448.89ms, tok/sec: 1167977.91
step:  133 | loss: 6.806884 | lr: 1.1245e-04 | norm: 0.4110, dt: 449.34ms, tok/sec: 1166801.04
step:  134 | loss: 6.800524 | lr: 1.1329e-04 | norm: 0.4785, dt: 448.74ms, tok/sec: 1168363.28
step:  135 | loss: 6.876669 | lr: 1.1413e-04 | norm: 0.5397, dt: 449.03ms, tok/sec: 1167594.66
step:  136 | loss: 6.845569 | lr: 1.1497e-04 | norm: 0.4287, dt: 449.09ms, tok/sec: 1167443.41
step:  137 | loss: 6.864476 | lr: 1.1580e-04 | norm: 0.3743, dt: 449.32ms, tok/sec: 1166857.39
step:  138 | loss: 6.791529 | lr: 1.1664e-04 | norm: 0.6621, dt: 449.60ms, tok/sec: 1166117.33
step:  139 | loss: 6.889002 | lr: 1.1748e-04 | norm: 1.2161, dt: 449.35ms, tok/sec: 1166765.76
step:  140 | loss: 6.994197 | lr: 1.1832e-04 | norm: 1.3236, dt: 448.54ms, tok/sec: 1168866.31
step:  141 | loss: 6.864680 | lr: 1.1916e-04 | norm: 0.5406, dt: 448.28ms, tok/sec: 1169544.54
step:  142 | loss: 6.927751 | lr: 1.2000e-04 | norm: 0.6296, dt: 449.03ms, tok/sec: 1167612.63
step:  143 | loss: 6.866826 | lr: 1.2084e-04 | norm: 1.2847, dt: 448.95ms, tok/sec: 1167803.62
step:  144 | loss: 6.864555 | lr: 1.2168e-04 | norm: 0.8253, dt: 449.42ms, tok/sec: 1166589.35
step:  145 | loss: 6.857577 | lr: 1.2252e-04 | norm: 0.6977, dt: 450.92ms, tok/sec: 1162704.01
step:  146 | loss: 6.862476 | lr: 1.2336e-04 | norm: 0.8974, dt: 448.90ms, tok/sec: 1167946.27
step:  147 | loss: 6.847271 | lr: 1.2420e-04 | norm: 0.5530, dt: 450.07ms, tok/sec: 1164910.89
step:  148 | loss: 6.842546 | lr: 1.2503e-04 | norm: 0.5699, dt: 450.34ms, tok/sec: 1164212.14
step:  149 | loss: 6.835727 | lr: 1.2587e-04 | norm: 0.5309, dt: 450.33ms, tok/sec: 1164219.53
step:  150 | loss: 6.815480 | lr: 1.2671e-04 | norm: 0.4247, dt: 449.58ms, tok/sec: 1166168.66
step:  151 | loss: 6.730823 | lr: 1.2755e-04 | norm: 0.5400, dt: 450.37ms, tok/sec: 1164120.92
step:  152 | loss: 6.746531 | lr: 1.2839e-04 | norm: 0.3429, dt: 450.15ms, tok/sec: 1164686.31
step:  153 | loss: 6.815625 | lr: 1.2923e-04 | norm: 0.5078, dt: 449.51ms, tok/sec: 1166341.85
step:  154 | loss: 6.714005 | lr: 1.3007e-04 | norm: 0.4115, dt: 450.05ms, tok/sec: 1164961.49
step:  155 | loss: 6.680983 | lr: 1.3091e-04 | norm: 0.5853, dt: 448.57ms, tok/sec: 1168797.97
step:  156 | loss: 6.748111 | lr: 1.3175e-04 | norm: 0.5987, dt: 449.14ms, tok/sec: 1167316.99
step:  157 | loss: 6.792259 | lr: 1.3259e-04 | norm: 0.6736, dt: 449.99ms, tok/sec: 1165122.59
step:  158 | loss: 6.770561 | lr: 1.3343e-04 | norm: 0.5589, dt: 449.59ms, tok/sec: 1166157.53
step:  159 | loss: 6.692471 | lr: 1.3427e-04 | norm: 0.6203, dt: 449.84ms, tok/sec: 1165500.52
step:  160 | loss: 6.634858 | lr: 1.3510e-04 | norm: 0.5755, dt: 450.40ms, tok/sec: 1164046.36
step:  161 | loss: 6.714288 | lr: 1.3594e-04 | norm: 0.5431, dt: 449.72ms, tok/sec: 1165797.72
step:  162 | loss: 6.607517 | lr: 1.3678e-04 | norm: 0.4319, dt: 450.23ms, tok/sec: 1164498.20
step:  163 | loss: 6.566184 | lr: 1.3762e-04 | norm: 0.5186, dt: 449.99ms, tok/sec: 1165121.98
step:  164 | loss: 6.600638 | lr: 1.3846e-04 | norm: 0.4589, dt: 450.34ms, tok/sec: 1164197.96
step:  165 | loss: 6.629689 | lr: 1.3930e-04 | norm: 0.5563, dt: 450.08ms, tok/sec: 1164885.59
step:  166 | loss: 6.473983 | lr: 1.4014e-04 | norm: 0.6979, dt: 449.65ms, tok/sec: 1166000.47
step:  167 | loss: 6.572123 | lr: 1.4098e-04 | norm: 0.7365, dt: 449.05ms, tok/sec: 1167559.94
step:  168 | loss: 6.539640 | lr: 1.4182e-04 | norm: 0.7272, dt: 450.76ms, tok/sec: 1163114.20
step:  169 | loss: 6.573174 | lr: 1.4266e-04 | norm: 0.6441, dt: 450.29ms, tok/sec: 1164336.04
step:  170 | loss: 6.538308 | lr: 1.4350e-04 | norm: 0.5181, dt: 449.62ms, tok/sec: 1166068.48
step:  171 | loss: 6.563982 | lr: 1.4434e-04 | norm: 0.5153, dt: 450.80ms, tok/sec: 1163017.01
step:  172 | loss: 6.551649 | lr: 1.4517e-04 | norm: 0.6497, dt: 451.06ms, tok/sec: 1162343.87
step:  173 | loss: 6.503856 | lr: 1.4601e-04 | norm: 1.1364, dt: 450.71ms, tok/sec: 1163243.41
step:  174 | loss: 6.488745 | lr: 1.4685e-04 | norm: 1.4928, dt: 449.64ms, tok/sec: 1166014.07
step:  175 | loss: 6.539491 | lr: 1.4769e-04 | norm: 0.6952, dt: 449.55ms, tok/sec: 1166263.29
step:  176 | loss: 6.504408 | lr: 1.4853e-04 | norm: 1.8715, dt: 449.18ms, tok/sec: 1167212.89
step:  177 | loss: 6.516397 | lr: 1.4937e-04 | norm: 0.8497, dt: 451.50ms, tok/sec: 1161211.44
step:  178 | loss: 6.460094 | lr: 1.5021e-04 | norm: 1.2074, dt: 450.71ms, tok/sec: 1163252.02
step:  179 | loss: 6.437253 | lr: 1.5105e-04 | norm: 0.9735, dt: 450.58ms, tok/sec: 1163582.56
step:  180 | loss: 6.423006 | lr: 1.5189e-04 | norm: 0.8232, dt: 449.25ms, tok/sec: 1167025.82
step:  181 | loss: 6.456751 | lr: 1.5273e-04 | norm: 0.8804, dt: 451.19ms, tok/sec: 1161999.30
step:  182 | loss: 6.446209 | lr: 1.5357e-04 | norm: 0.8058, dt: 450.52ms, tok/sec: 1163748.20
step:  183 | loss: 6.432818 | lr: 1.5441e-04 | norm: 0.6058, dt: 449.09ms, tok/sec: 1167455.18
step:  184 | loss: 6.455407 | lr: 1.5524e-04 | norm: 0.6054, dt: 449.36ms, tok/sec: 1166752.76
step:  185 | loss: 6.505716 | lr: 1.5608e-04 | norm: 0.5919, dt: 449.85ms, tok/sec: 1165483.22
step:  186 | loss: 6.522611 | lr: 1.5692e-04 | norm: 0.5636, dt: 450.42ms, tok/sec: 1163992.14
step:  187 | loss: 6.586942 | lr: 1.5776e-04 | norm: 0.5942, dt: 449.46ms, tok/sec: 1166489.72
step:  188 | loss: 6.533071 | lr: 1.5860e-04 | norm: 0.4262, dt: 1106.71ms, tok/sec: 473733.85
step:  189 | loss: 6.480268 | lr: 1.5944e-04 | norm: 0.5042, dt: 1210.80ms, tok/sec: 433007.90
step:  190 | loss: 6.522179 | lr: 1.6028e-04 | norm: 0.4689, dt: 450.91ms, tok/sec: 1162722.45
step:  191 | loss: 6.523131 | lr: 1.6112e-04 | norm: 0.5065, dt: 448.20ms, tok/sec: 1169764.15
step:  192 | loss: 6.491390 | lr: 1.6196e-04 | norm: 0.4789, dt: 449.69ms, tok/sec: 1165883.02
step:  193 | loss: 6.452076 | lr: 1.6280e-04 | norm: 0.4098, dt: 449.98ms, tok/sec: 1165143.58
step:  194 | loss: 6.443089 | lr: 1.6364e-04 | norm: 0.4042, dt: 449.07ms, tok/sec: 1167502.29
step:  195 | loss: 6.407020 | lr: 1.6448e-04 | norm: 0.4913, dt: 448.60ms, tok/sec: 1168716.60
step:  196 | loss: 6.465550 | lr: 1.6531e-04 | norm: 0.5459, dt: 449.06ms, tok/sec: 1167519.03
step:  197 | loss: 6.486465 | lr: 1.6615e-04 | norm: 0.5827, dt: 448.14ms, tok/sec: 1169915.38
step:  198 | loss: 6.408715 | lr: 1.6699e-04 | norm: 0.4461, dt: 449.18ms, tok/sec: 1167221.57
step:  199 | loss: 6.392152 | lr: 1.6783e-04 | norm: 0.4673, dt: 449.62ms, tok/sec: 1166066.01
step:  200 | loss: 6.430371 | lr: 1.6867e-04 | norm: 0.5517, dt: 450.64ms, tok/sec: 1163434.81
step:  201 | loss: 6.407748 | lr: 1.6951e-04 | norm: 0.4543, dt: 449.67ms, tok/sec: 1165933.71
step:  202 | loss: 6.403425 | lr: 1.7035e-04 | norm: 0.6153, dt: 449.32ms, tok/sec: 1166857.39
step:  203 | loss: 6.378550 | lr: 1.7119e-04 | norm: 0.6690, dt: 449.08ms, tok/sec: 1167468.20
step:  204 | loss: 6.457418 | lr: 1.7203e-04 | norm: 0.9377, dt: 449.74ms, tok/sec: 1165753.84
step:  205 | loss: 6.421860 | lr: 1.7287e-04 | norm: 1.0300, dt: 449.12ms, tok/sec: 1167367.18
step:  206 | loss: 6.379951 | lr: 1.7371e-04 | norm: 0.8000, dt: 450.33ms, tok/sec: 1164231.24
step:  207 | loss: 6.383246 | lr: 1.7455e-04 | norm: 0.6885, dt: 450.06ms, tok/sec: 1164928.17
step:  208 | loss: 6.315172 | lr: 1.7538e-04 | norm: 0.8582, dt: 448.23ms, tok/sec: 1169691.98
step:  209 | loss: 6.256432 | lr: 1.7622e-04 | norm: 0.7116, dt: 449.52ms, tok/sec: 1166336.90
step:  210 | loss: 6.352299 | lr: 1.7706e-04 | norm: 0.6799, dt: 450.13ms, tok/sec: 1164759.72
step:  211 | loss: 6.355441 | lr: 1.7790e-04 | norm: 0.6947, dt: 449.02ms, tok/sec: 1167623.79
step:  212 | loss: 6.386889 | lr: 1.7874e-04 | norm: 0.6490, dt: 449.50ms, tok/sec: 1166392.58
step:  213 | loss: 6.382277 | lr: 1.7958e-04 | norm: 0.7221, dt: 450.64ms, tok/sec: 1163424.34
step:  214 | loss: 6.311139 | lr: 1.8042e-04 | norm: 0.4978, dt: 449.15ms, tok/sec: 1167279.19
step:  215 | loss: 6.305889 | lr: 1.8126e-04 | norm: 0.5537, dt: 450.35ms, tok/sec: 1164185.02
step:  216 | loss: 6.295021 | lr: 1.8210e-04 | norm: 0.5754, dt: 450.15ms, tok/sec: 1164699.88
step:  217 | loss: 6.287257 | lr: 1.8294e-04 | norm: 0.4908, dt: 448.34ms, tok/sec: 1169385.32
step:  218 | loss: 6.267263 | lr: 1.8378e-04 | norm: 0.4726, dt: 449.41ms, tok/sec: 1166615.34
step:  219 | loss: 6.258142 | lr: 1.8462e-04 | norm: 0.4753, dt: 449.71ms, tok/sec: 1165829.86
step:  220 | loss: 6.222856 | lr: 1.8545e-04 | norm: 0.5685, dt: 450.21ms, tok/sec: 1164549.38
step:  221 | loss: 6.245907 | lr: 1.8629e-04 | norm: 0.6145, dt: 449.40ms, tok/sec: 1166633.91
step:  222 | loss: 6.191341 | lr: 1.8713e-04 | norm: 0.5752, dt: 450.34ms, tok/sec: 1164212.14
step:  223 | loss: 6.184554 | lr: 1.8797e-04 | norm: 0.5481, dt: 449.29ms, tok/sec: 1166916.83
step:  224 | loss: 6.207159 | lr: 1.8881e-04 | norm: 0.5876, dt: 449.91ms, tok/sec: 1165309.06
step:  225 | loss: 6.180538 | lr: 1.8965e-04 | norm: 0.7537, dt: 450.12ms, tok/sec: 1164770.21
step:  226 | loss: 6.206604 | lr: 1.9049e-04 | norm: 0.7657, dt: 449.34ms, tok/sec: 1166804.76
step:  227 | loss: 6.224407 | lr: 1.9133e-04 | norm: 0.8259, dt: 450.09ms, tok/sec: 1164861.52
step:  228 | loss: 6.189745 | lr: 1.9217e-04 | norm: 0.8158, dt: 448.80ms, tok/sec: 1168205.00
step:  229 | loss: 6.180536 | lr: 1.9301e-04 | norm: 0.8734, dt: 450.25ms, tok/sec: 1164435.30
step:  230 | loss: 6.200702 | lr: 1.9385e-04 | norm: 0.8801, dt: 449.97ms, tok/sec: 1165152.84
step:  231 | loss: 6.280072 | lr: 1.9469e-04 | norm: 1.0266, dt: 450.22ms, tok/sec: 1164512.38
step:  232 | loss: 6.342929 | lr: 1.9552e-04 | norm: 0.7994, dt: 449.69ms, tok/sec: 1165877.45
step:  233 | loss: 6.332344 | lr: 1.9636e-04 | norm: 0.7861, dt: 449.11ms, tok/sec: 1167394.45
step:  234 | loss: 6.342448 | lr: 1.9720e-04 | norm: 0.8185, dt: 449.80ms, tok/sec: 1165602.45
step:  235 | loss: 6.329099 | lr: 1.9804e-04 | norm: 0.7627, dt: 449.43ms, tok/sec: 1166569.54
step:  236 | loss: 6.295857 | lr: 1.9888e-04 | norm: 0.6337, dt: 451.45ms, tok/sec: 1161348.81
step:  237 | loss: 6.312985 | lr: 1.9972e-04 | norm: 0.5449, dt: 449.76ms, tok/sec: 1165694.52
step:  238 | loss: 6.288090 | lr: 2.0056e-04 | norm: 0.5759, dt: 449.66ms, tok/sec: 1165974.51
step:  239 | loss: 6.269328 | lr: 2.0140e-04 | norm: 0.5547, dt: 449.24ms, tok/sec: 1167054.93
step:  240 | loss: 6.295012 | lr: 2.0224e-04 | norm: 0.5505, dt: 449.91ms, tok/sec: 1165313.38
step:  241 | loss: 6.285894 | lr: 2.0308e-04 | norm: 0.5919, dt: 449.47ms, tok/sec: 1166455.69
step:  242 | loss: 6.268569 | lr: 2.0392e-04 | norm: 0.6770, dt: 448.76ms, tok/sec: 1168301.82
step:  243 | loss: 6.215553 | lr: 2.0476e-04 | norm: 0.7256, dt: 450.29ms, tok/sec: 1164336.65
step:  244 | loss: 6.176208 | lr: 2.0559e-04 | norm: 0.6462, dt: 449.95ms, tok/sec: 1165207.79
step:  245 | loss: 6.197804 | lr: 2.0643e-04 | norm: 0.5104, dt: 452.09ms, tok/sec: 1159693.33
step:  246 | loss: 6.255145 | lr: 2.0727e-04 | norm: 0.4342, dt: 449.95ms, tok/sec: 1165211.50
step:  247 | loss: 6.252660 | lr: 2.0811e-04 | norm: 0.7076, dt: 450.09ms, tok/sec: 1164856.59
step:  248 | loss: 6.167966 | lr: 2.0895e-04 | norm: 0.8970, dt: 449.61ms, tok/sec: 1166083.32
step:  249 | loss: 6.222438 | lr: 2.0979e-04 | norm: 0.9220, dt: 450.61ms, tok/sec: 1163515.45
validation loss: 6.1953
HellaSwag accuracy: 2455/10042=0.2445
rank 4 sample 0: Hello, I'm a Manpreet, the "C. The 18D. L., N. (16
B. S. U. L.
rank 4 sample 1: Hello, I'm a Manpreet, so the first that he would get my wife, we go.
How to have you.
" it might
rank 4 sample 2: Hello, I'm a Manpreet, "4. So a I am you I didn't not, for a new ways in other other types of how
rank 4 sample 3: Hello, I'm a Manpreet, I was always not like this. After this way on us how we feel and I will have some things. A
rank 6 sample 0: Hello, I'm a Manpreet, “I” or those the first time: I am as I would all of her.’s
rank 6 sample 1: Hello, I'm a Manpreet, we see we have some that they do all of this.
We want to do you can use a time."
rank 6 sample 2: Hello, I'm a Manpreet, a man of a person that we don’s a lot of a week.
-I has also are
rank 6 sample 3: Hello, I'm a Manpreet, that they must have to the two-18-1-1-1) and it:
-10-
rank 7 sample 0: Hello, I'm a Manpreet, I’s the.
To look of your school students to their women must know for his people to the
rank 7 sample 1: Hello, I'm a Manpreet, my good time, I, you make something.
the, I have a result for I've’re
rank 7 sample 2: Hello, I'm a Manpreet, H. The G.
2): E. However, P.org.
The first first in a year
rank 7 sample 3: Hello, I'm a Manpreet, at a number of a year. One: p.
After a time, and the United States was a small
rank 0 sample 0: Hello, I'm a Manpreet, I am like the first to have. One’s a "’s, to get it and I
rank 0 sample 1: Hello, I'm a Manpreet, an unold, and the I should be an very part. This happens for a very difficult. As it is
rank 0 sample 2: Hello, I'm a Manpreet, the "-4-6-14- "- "C
- "- "- "- ""
rank 0 sample 3: Hello, I'm a Manpreet, we be a day.
-C, R.
-5/10 days to that the following the right
rank 5 sample 0: Hello, I'm a Manpreet, a good way of I’s most people about I’t do you’t the “
rank 3 sample 0: Hello, I'm a Manpreet, I am I’, then she have to “R. Now something to” but one I did
rank 5 sample 1: Hello, I'm a Manpreet, Sado (4 (7.
T) The (or)|4. (P).S.
rank 3 sample 1: Hello, I'm a Manpreet, I have been a good school or be the name of your child, be taken to be about to the person'srank 5 sample 2: Hello, I'm a Manpreet, was the United Kingdom during the country on the same of the "and the year. If the
The first year

rank 5 sample 3: Hello, I'm a Manpreet, J.
T.S.
The R. (R, a much for this, of a good.
rank 3 sample 2: Hello, I'm a Manpreet, I’s new way: I’s important things. I I could I. Then, I were
rank 3 sample 3: Hello, I'm a Manpreet, I need to have a variety of the same of their way. This study, when she are used; I am
rank 2 sample 0: Hello, I'm a Manpreet, if it. (The problem of the same of the second of the day, I want to be a great.
rank 2 sample 1: Hello, I'm a Manpreet, then a much more about a result it you have what you can’t know that it is so you know
rank 2 sample 2: Hello, I'm a Manpreet, the "the I I was we don (1) I are not we can you, "a--e
rank 1 sample 0: Hello, I'm a Manpreet, is also are not so that in several weeks are in those different.
’m is in the same,
rank 2 sample 3: Hello, I'm a Manpreet, "1<|endoftext|>181925, I is very, and
The "What's is on? The first,
rank 1 sample 1: Hello, I'm a Manpreet, and the government from the first-The last of the University of the state that and and most successful in the most
rank 1 sample 2: Hello, I'm a Manpreet, we go in the first few of the first time of I is not in the first of what he had to that
rank 1 sample 3: Hello, I'm a Manpreet, I knew that was the book. This that I is like her with I also a student should do I get me
step:  250 | loss: 6.214639 | lr: 2.1063e-04 | norm: 0.7168, dt: 12218.66ms, tok/sec: 42908.79
step:  251 | loss: 6.186831 | lr: 2.1147e-04 | norm: 0.5827, dt: 448.74ms, tok/sec: 1168357.07
step:  252 | loss: 6.214961 | lr: 2.1231e-04 | norm: 0.4408, dt: 448.72ms, tok/sec: 1168394.94
step:  253 | loss: 6.158668 | lr: 2.1315e-04 | norm: 0.5365, dt: 450.22ms, tok/sec: 1164508.06
step:  254 | loss: 6.167080 | lr: 2.1399e-04 | norm: 0.4292, dt: 449.97ms, tok/sec: 1165159.63
step:  255 | loss: 6.120817 | lr: 2.1483e-04 | norm: 0.5135, dt: 448.00ms, tok/sec: 1170280.23
step:  256 | loss: 6.114298 | lr: 2.1566e-04 | norm: 0.6367, dt: 449.08ms, tok/sec: 1167462.62
step:  257 | loss: 6.124767 | lr: 2.1650e-04 | norm: 0.6068, dt: 451.50ms, tok/sec: 1161202.85
step:  258 | loss: 6.144757 | lr: 2.1734e-04 | norm: 0.8023, dt: 449.85ms, tok/sec: 1165483.84
step:  259 | loss: 6.061673 | lr: 2.1818e-04 | norm: 1.0474, dt: 449.81ms, tok/sec: 1165578.98
step:  260 | loss: 6.141784 | lr: 2.1902e-04 | norm: 1.4084, dt: 449.55ms, tok/sec: 1166252.78
step:  261 | loss: 6.075083 | lr: 2.1986e-04 | norm: 0.6517, dt: 450.11ms, tok/sec: 1164793.04
step:  262 | loss: 6.079259 | lr: 2.2070e-04 | norm: 0.8155, dt: 450.07ms, tok/sec: 1164896.08
step:  263 | loss: 6.062269 | lr: 2.2154e-04 | norm: 0.7763, dt: 449.01ms, tok/sec: 1167652.93
step:  264 | loss: 6.097523 | lr: 2.2238e-04 | norm: 0.5781, dt: 449.67ms, tok/sec: 1165949.78
step:  265 | loss: 6.099622 | lr: 2.2322e-04 | norm: 0.5609, dt: 449.31ms, tok/sec: 1166862.34
step:  266 | loss: 6.059387 | lr: 2.2406e-04 | norm: 0.5327, dt: 449.97ms, tok/sec: 1165168.28
step:  267 | loss: 6.078262 | lr: 2.2490e-04 | norm: 0.5518, dt: 450.03ms, tok/sec: 1165004.70
step:  268 | loss: 6.086834 | lr: 2.2573e-04 | norm: 0.7170, dt: 449.72ms, tok/sec: 1165821.82
step:  269 | loss: 6.012804 | lr: 2.2657e-04 | norm: 0.8193, dt: 449.28ms, tok/sec: 1166951.51
step:  270 | loss: 6.004620 | lr: 2.2741e-04 | norm: 0.8708, dt: 450.63ms, tok/sec: 1163452.66
step:  271 | loss: 5.979101 | lr: 2.2825e-04 | norm: 0.7366, dt: 451.24ms, tok/sec: 1161875.28
step:  272 | loss: 6.010499 | lr: 2.2909e-04 | norm: 0.6463, dt: 450.82ms, tok/sec: 1162956.73
step:  273 | loss: 5.984761 | lr: 2.2993e-04 | norm: 0.7423, dt: 449.98ms, tok/sec: 1165123.83
step:  274 | loss: 6.008135 | lr: 2.3077e-04 | norm: 0.8351, dt: 450.51ms, tok/sec: 1163774.68
step:  275 | loss: 6.082451 | lr: 2.3161e-04 | norm: 0.9280, dt: 454.19ms, tok/sec: 1154344.13
step:  276 | loss: 6.070482 | lr: 2.3245e-04 | norm: 0.9948, dt: 450.61ms, tok/sec: 1163506.83
step:  277 | loss: 6.078722 | lr: 2.3329e-04 | norm: 0.7361, dt: 451.04ms, tok/sec: 1162397.94
step:  278 | loss: 6.109946 | lr: 2.3413e-04 | norm: 0.6787, dt: 450.14ms, tok/sec: 1164731.96
step:  279 | loss: 6.136664 | lr: 2.3497e-04 | norm: 0.6818, dt: 450.07ms, tok/sec: 1164891.14
step:  280 | loss: 6.106078 | lr: 2.3580e-04 | norm: 0.9221, dt: 450.84ms, tok/sec: 1162923.52
step:  281 | loss: 6.101063 | lr: 2.3664e-04 | norm: 1.0005, dt: 450.81ms, tok/sec: 1162996.10
step:  282 | loss: 6.167770 | lr: 2.3748e-04 | norm: 0.8571, dt: 449.54ms, tok/sec: 1166281.85
step:  283 | loss: 6.100859 | lr: 2.3832e-04 | norm: 0.7064, dt: 449.53ms, tok/sec: 1166298.55
step:  284 | loss: 6.095455 | lr: 2.3916e-04 | norm: 0.7158, dt: 450.23ms, tok/sec: 1164492.03
step:  285 | loss: 6.095938 | lr: 2.4000e-04 | norm: 0.6747, dt: 450.77ms, tok/sec: 1163092.67
step:  286 | loss: 6.056768 | lr: 2.4084e-04 | norm: 0.6997, dt: 451.65ms, tok/sec: 1160837.51
step:  287 | loss: 6.073696 | lr: 2.4168e-04 | norm: 0.4459, dt: 449.00ms, tok/sec: 1167680.21
step:  288 | loss: 6.016847 | lr: 2.4252e-04 | norm: 0.5218, dt: 450.96ms, tok/sec: 1162596.44
step:  289 | loss: 6.033051 | lr: 2.4336e-04 | norm: 0.4929, dt: 450.03ms, tok/sec: 1165009.63
step:  290 | loss: 6.042095 | lr: 2.4420e-04 | norm: 0.5193, dt: 450.92ms, tok/sec: 1162697.86
step:  291 | loss: 5.977726 | lr: 2.4503e-04 | norm: 0.7977, dt: 449.45ms, tok/sec: 1166516.94
step:  292 | loss: 6.036140 | lr: 2.4587e-04 | norm: 0.7354, dt: 450.24ms, tok/sec: 1164459.35
step:  293 | loss: 6.050822 | lr: 2.4671e-04 | norm: 0.7241, dt: 450.59ms, tok/sec: 1163561.62
step:  294 | loss: 6.039846 | lr: 2.4755e-04 | norm: 0.6847, dt: 450.53ms, tok/sec: 1163721.72
step:  295 | loss: 6.026638 | lr: 2.4839e-04 | norm: 0.6354, dt: 450.76ms, tok/sec: 1163130.20
step:  296 | loss: 5.994185 | lr: 2.4923e-04 | norm: 0.7376, dt: 447.91ms, tok/sec: 1170515.70
step:  297 | loss: 6.034417 | lr: 2.5007e-04 | norm: 0.6218, dt: 455.18ms, tok/sec: 1151814.37
step:  298 | loss: 6.000319 | lr: 2.5091e-04 | norm: 0.5385, dt: 449.44ms, tok/sec: 1166523.75
step:  299 | loss: 5.994640 | lr: 2.5175e-04 | norm: 0.5864, dt: 450.00ms, tok/sec: 1165084.94
step:  300 | loss: 6.014772 | lr: 2.5259e-04 | norm: 0.6851, dt: 449.05ms, tok/sec: 1167556.22
step:  301 | loss: 5.931314 | lr: 2.5343e-04 | norm: 0.7851, dt: 449.90ms, tok/sec: 1165350.43
step:  302 | loss: 5.977562 | lr: 2.5427e-04 | norm: 1.1005, dt: 449.64ms, tok/sec: 1166006.66
step:  303 | loss: 5.962909 | lr: 2.5510e-04 | norm: 1.1393, dt: 448.68ms, tok/sec: 1168502.96
step:  304 | loss: 5.930360 | lr: 2.5594e-04 | norm: 0.7235, dt: 449.67ms, tok/sec: 1165939.89
step:  305 | loss: 5.897799 | lr: 2.5678e-04 | norm: 0.8101, dt: 449.62ms, tok/sec: 1166070.96
step:  306 | loss: 5.911919 | lr: 2.5762e-04 | norm: 0.8616, dt: 450.58ms, tok/sec: 1163586.87
step:  307 | loss: 5.904377 | lr: 2.5846e-04 | norm: 0.7761, dt: 449.66ms, tok/sec: 1165974.51
step:  308 | loss: 5.929042 | lr: 2.5930e-04 | norm: 0.7307, dt: 449.61ms, tok/sec: 1166094.46
step:  309 | loss: 5.871301 | lr: 2.6014e-04 | norm: 0.8316, dt: 449.31ms, tok/sec: 1166877.20
step:  310 | loss: 5.911944 | lr: 2.6098e-04 | norm: 0.8192, dt: 450.37ms, tok/sec: 1164133.25
step:  311 | loss: 5.862546 | lr: 2.6182e-04 | norm: 0.8082, dt: 450.11ms, tok/sec: 1164793.04
step:  312 | loss: 5.845728 | lr: 2.6266e-04 | norm: 0.6714, dt: 450.44ms, tok/sec: 1163942.23
step:  313 | loss: 5.866456 | lr: 2.6350e-04 | norm: 0.7175, dt: 450.13ms, tok/sec: 1164750.47
step:  314 | loss: 5.840581 | lr: 2.6434e-04 | norm: 0.7739, dt: 451.03ms, tok/sec: 1162424.36
step:  315 | loss: 5.849180 | lr: 2.6517e-04 | norm: 0.6568, dt: 451.37ms, tok/sec: 1161540.81
step:  316 | loss: 5.877315 | lr: 2.6601e-04 | norm: 0.8295, dt: 450.52ms, tok/sec: 1163750.66
step:  317 | loss: 5.847209 | lr: 2.6685e-04 | norm: 0.6657, dt: 450.53ms, tok/sec: 1163719.26
step:  318 | loss: 5.809249 | lr: 2.6769e-04 | norm: 0.6853, dt: 450.66ms, tok/sec: 1163367.10
step:  319 | loss: 5.845904 | lr: 2.6853e-04 | norm: 0.5875, dt: 451.70ms, tok/sec: 1160696.59
step:  320 | loss: 5.753056 | lr: 2.6937e-04 | norm: 0.5082, dt: 450.20ms, tok/sec: 1164577.13
step:  321 | loss: 5.787788 | lr: 2.7021e-04 | norm: 0.5563, dt: 452.82ms, tok/sec: 1157831.00
step:  322 | loss: 5.837470 | lr: 2.7105e-04 | norm: 0.6570, dt: 450.14ms, tok/sec: 1164718.39
step:  323 | loss: 5.891701 | lr: 2.7189e-04 | norm: 0.9148, dt: 450.63ms, tok/sec: 1163463.74
step:  324 | loss: 5.959124 | lr: 2.7273e-04 | norm: 1.1377, dt: 453.55ms, tok/sec: 1155972.81
step:  325 | loss: 5.982431 | lr: 2.7357e-04 | norm: 0.7852, dt: 451.01ms, tok/sec: 1162487.04
step:  326 | loss: 5.991827 | lr: 2.7441e-04 | norm: 1.1098, dt: 450.49ms, tok/sec: 1163823.34
step:  327 | loss: 5.990618 | lr: 2.7524e-04 | norm: 1.2544, dt: 452.08ms, tok/sec: 1159734.92
step:  328 | loss: 5.954941 | lr: 2.7608e-04 | norm: 0.9867, dt: 450.80ms, tok/sec: 1163025.00
step:  329 | loss: 5.952481 | lr: 2.7692e-04 | norm: 0.7412, dt: 450.24ms, tok/sec: 1164475.38
step:  330 | loss: 5.973281 | lr: 2.7776e-04 | norm: 0.6696, dt: 451.39ms, tok/sec: 1161498.48
step:  331 | loss: 5.939539 | lr: 2.7860e-04 | norm: 0.7399, dt: 451.39ms, tok/sec: 1161485.60
step:  332 | loss: 5.946920 | lr: 2.7944e-04 | norm: 0.6814, dt: 451.15ms, tok/sec: 1162121.50
step:  333 | loss: 5.977661 | lr: 2.8028e-04 | norm: 0.7488, dt: 450.56ms, tok/sec: 1163639.20
step:  334 | loss: 5.918220 | lr: 2.8112e-04 | norm: 0.7229, dt: 451.46ms, tok/sec: 1161308.33
step:  335 | loss: 5.854396 | lr: 2.8196e-04 | norm: 0.6178, dt: 450.91ms, tok/sec: 1162722.45
step:  336 | loss: 5.887157 | lr: 2.8280e-04 | norm: 0.6254, dt: 451.58ms, tok/sec: 1161002.99
step:  337 | loss: 5.918372 | lr: 2.8364e-04 | norm: 0.6665, dt: 451.33ms, tok/sec: 1161658.01
step:  338 | loss: 5.921431 | lr: 2.8448e-04 | norm: 0.7039, dt: 452.17ms, tok/sec: 1159486.65
step:  339 | loss: 5.853133 | lr: 2.8531e-04 | norm: 0.7266, dt: 450.50ms, tok/sec: 1163803.02
step:  340 | loss: 5.867404 | lr: 2.8615e-04 | norm: 0.7868, dt: 451.10ms, tok/sec: 1162237.59
step:  341 | loss: 5.887032 | lr: 2.8699e-04 | norm: 0.8791, dt: 451.68ms, tok/sec: 1160756.02
step:  342 | loss: 5.872311 | lr: 2.8783e-04 | norm: 0.8367, dt: 451.33ms, tok/sec: 1161650.03
step:  343 | loss: 5.796939 | lr: 2.8867e-04 | norm: 0.8462, dt: 451.03ms, tok/sec: 1162436.65
step:  344 | loss: 5.825594 | lr: 2.8951e-04 | norm: 0.8498, dt: 451.57ms, tok/sec: 1161042.22
step:  345 | loss: 5.864815 | lr: 2.9035e-04 | norm: 0.8698, dt: 450.82ms, tok/sec: 1162961.65
step:  346 | loss: 5.777358 | lr: 2.9119e-04 | norm: 0.8296, dt: 450.82ms, tok/sec: 1162973.34
step:  347 | loss: 5.795688 | lr: 2.9203e-04 | norm: 0.9352, dt: 451.55ms, tok/sec: 1161077.16
step:  348 | loss: 5.798475 | lr: 2.9287e-04 | norm: 1.2148, dt: 451.18ms, tok/sec: 1162025.71
step:  349 | loss: 5.808562 | lr: 2.9371e-04 | norm: 0.7880, dt: 451.66ms, tok/sec: 1160798.30
step:  350 | loss: 5.766810 | lr: 2.9455e-04 | norm: 0.8349, dt: 451.81ms, tok/sec: 1160406.27
step:  351 | loss: 5.771683 | lr: 2.9538e-04 | norm: 0.7897, dt: 451.09ms, tok/sec: 1162272.61
step:  352 | loss: 5.732916 | lr: 2.9622e-04 | norm: 0.7615, dt: 450.05ms, tok/sec: 1164965.20
step:  353 | loss: 5.781081 | lr: 2.9706e-04 | norm: 0.7323, dt: 451.77ms, tok/sec: 1160517.72
step:  354 | loss: 5.766120 | lr: 2.9790e-04 | norm: 0.6561, dt: 452.30ms, tok/sec: 1159168.83
step:  355 | loss: 5.754228 | lr: 2.9874e-04 | norm: 0.7058, dt: 451.51ms, tok/sec: 1161188.14
step:  356 | loss: 5.732021 | lr: 2.9958e-04 | norm: 0.5620, dt: 450.64ms, tok/sec: 1163422.50
step:  357 | loss: 5.699888 | lr: 3.0042e-04 | norm: 0.7321, dt: 450.67ms, tok/sec: 1163352.33
step:  358 | loss: 5.701073 | lr: 3.0126e-04 | norm: 0.7928, dt: 450.84ms, tok/sec: 1162922.29
step:  359 | loss: 5.701853 | lr: 3.0210e-04 | norm: 0.7868, dt: 450.38ms, tok/sec: 1164113.53
step:  360 | loss: 5.698860 | lr: 3.0294e-04 | norm: 0.7722, dt: 451.53ms, tok/sec: 1161127.44
step:  361 | loss: 5.722428 | lr: 3.0378e-04 | norm: 1.0117, dt: 450.97ms, tok/sec: 1162568.78
step:  362 | loss: 5.715956 | lr: 3.0462e-04 | norm: 1.0087, dt: 450.95ms, tok/sec: 1162641.31
step:  363 | loss: 5.664105 | lr: 3.0545e-04 | norm: 0.8599, dt: 450.41ms, tok/sec: 1164021.09
step:  364 | loss: 5.639720 | lr: 3.0629e-04 | norm: 0.8163, dt: 452.29ms, tok/sec: 1159193.27
step:  365 | loss: 5.657108 | lr: 3.0713e-04 | norm: 0.9989, dt: 451.01ms, tok/sec: 1162473.52
step:  366 | loss: 5.760880 | lr: 3.0797e-04 | norm: 1.1454, dt: 450.49ms, tok/sec: 1163808.56
step:  367 | loss: 5.629283 | lr: 3.0881e-04 | norm: 0.9936, dt: 452.44ms, tok/sec: 1158790.12
step:  368 | loss: 5.675241 | lr: 3.0965e-04 | norm: 1.0883, dt: 449.95ms, tok/sec: 1165207.17
step:  369 | loss: 5.682134 | lr: 3.1049e-04 | norm: 0.8596, dt: 451.70ms, tok/sec: 1160698.43
step:  370 | loss: 5.807151 | lr: 3.1133e-04 | norm: 1.0392, dt: 451.26ms, tok/sec: 1161843.98
step:  371 | loss: 5.791483 | lr: 3.1217e-04 | norm: 0.9032, dt: 450.91ms, tok/sec: 1162724.91
step:  372 | loss: 5.811588 | lr: 3.1301e-04 | norm: 0.7696, dt: 451.97ms, tok/sec: 1159999.21
step:  373 | loss: 5.738242 | lr: 3.1385e-04 | norm: 0.8590, dt: 450.90ms, tok/sec: 1162750.12
step:  374 | loss: 5.819611 | lr: 3.1469e-04 | norm: 0.8636, dt: 451.52ms, tok/sec: 1161166.68
step:  375 | loss: 5.821373 | lr: 3.1552e-04 | norm: 1.0915, dt: 451.62ms, tok/sec: 1160901.86
step:  376 | loss: 5.839181 | lr: 3.1636e-04 | norm: 1.2390, dt: 450.69ms, tok/sec: 1163290.17
step:  377 | loss: 5.814862 | lr: 3.1720e-04 | norm: 0.9996, dt: 1224.63ms, tok/sec: 428120.06
step:  378 | loss: 5.757415 | lr: 3.1804e-04 | norm: 1.1817, dt: 449.77ms, tok/sec: 1165671.65
step:  379 | loss: 5.745970 | lr: 3.1888e-04 | norm: 0.8145, dt: 1220.34ms, tok/sec: 429625.79
step:  380 | loss: 5.762936 | lr: 3.1972e-04 | norm: 0.8127, dt: 449.42ms, tok/sec: 1166585.64
step:  381 | loss: 5.698739 | lr: 3.2056e-04 | norm: 0.7948, dt: 449.61ms, tok/sec: 1166103.11
step:  382 | loss: 5.726965 | lr: 3.2140e-04 | norm: 0.7206, dt: 451.31ms, tok/sec: 1161701.58
step:  383 | loss: 5.719723 | lr: 3.2224e-04 | norm: 0.7268, dt: 450.01ms, tok/sec: 1165062.72
step:  384 | loss: 5.752025 | lr: 3.2308e-04 | norm: 0.7198, dt: 449.43ms, tok/sec: 1166568.93
step:  385 | loss: 5.833965 | lr: 3.2392e-04 | norm: 0.8220, dt: 449.36ms, tok/sec: 1166732.95
step:  386 | loss: 5.684060 | lr: 3.2476e-04 | norm: 0.9928, dt: 452.75ms, tok/sec: 1158015.13
step:  387 | loss: 5.735903 | lr: 3.2559e-04 | norm: 0.9841, dt: 450.87ms, tok/sec: 1162841.12
step:  388 | loss: 5.681014 | lr: 3.2643e-04 | norm: 0.9506, dt: 450.38ms, tok/sec: 1164091.96
step:  389 | loss: 5.710708 | lr: 3.2727e-04 | norm: 0.9674, dt: 450.57ms, tok/sec: 1163600.41
step:  390 | loss: 5.669311 | lr: 3.2811e-04 | norm: 1.0228, dt: 450.48ms, tok/sec: 1163831.35
step:  391 | loss: 5.664381 | lr: 3.2895e-04 | norm: 0.8800, dt: 450.84ms, tok/sec: 1162911.22
step:  392 | loss: 5.682292 | lr: 3.2979e-04 | norm: 0.9028, dt: 452.07ms, tok/sec: 1159747.16
step:  393 | loss: 5.604332 | lr: 3.3063e-04 | norm: 0.9015, dt: 450.38ms, tok/sec: 1164096.27
step:  394 | loss: 5.594863 | lr: 3.3147e-04 | norm: 0.7609, dt: 450.55ms, tok/sec: 1163663.22
step:  395 | loss: 5.635665 | lr: 3.3231e-04 | norm: 0.8825, dt: 452.29ms, tok/sec: 1159185.94
step:  396 | loss: 5.615592 | lr: 3.3315e-04 | norm: 0.8013, dt: 451.76ms, tok/sec: 1160539.77
step:  397 | loss: 5.587329 | lr: 3.3399e-04 | norm: 0.7075, dt: 451.80ms, tok/sec: 1160431.99
step:  398 | loss: 5.615745 | lr: 3.3483e-04 | norm: 0.7338, dt: 450.18ms, tok/sec: 1164619.07
step:  399 | loss: 5.575627 | lr: 3.3566e-04 | norm: 0.6913, dt: 451.63ms, tok/sec: 1160886.54
step:  400 | loss: 5.613231 | lr: 3.3650e-04 | norm: 0.6932, dt: 451.67ms, tok/sec: 1160769.50
step:  401 | loss: 5.573400 | lr: 3.3734e-04 | norm: 0.9003, dt: 450.65ms, tok/sec: 1163404.03
step:  402 | loss: 5.603078 | lr: 3.3818e-04 | norm: 0.9312, dt: 449.72ms, tok/sec: 1165819.35
step:  403 | loss: 5.536586 | lr: 3.3902e-04 | norm: 0.9383, dt: 451.99ms, tok/sec: 1159955.76
step:  404 | loss: 5.540977 | lr: 3.3986e-04 | norm: 1.1147, dt: 450.85ms, tok/sec: 1162890.31
step:  405 | loss: 5.508604 | lr: 3.4070e-04 | norm: 0.7824, dt: 450.77ms, tok/sec: 1163088.98
step:  406 | loss: 5.505768 | lr: 3.4154e-04 | norm: 0.8309, dt: 451.16ms, tok/sec: 1162098.78
step:  407 | loss: 5.508634 | lr: 3.4238e-04 | norm: 0.9747, dt: 450.94ms, tok/sec: 1162649.91
step:  408 | loss: 5.544557 | lr: 3.4322e-04 | norm: 1.2454, dt: 451.37ms, tok/sec: 1161540.81
step:  409 | loss: 5.543612 | lr: 3.4406e-04 | norm: 1.1819, dt: 451.69ms, tok/sec: 1160728.45
step:  410 | loss: 5.554426 | lr: 3.4490e-04 | norm: 0.9769, dt: 451.16ms, tok/sec: 1162093.25
step:  411 | loss: 5.565089 | lr: 3.4573e-04 | norm: 1.1033, dt: 451.00ms, tok/sec: 1162493.80
step:  412 | loss: 5.498644 | lr: 3.4657e-04 | norm: 1.0151, dt: 450.70ms, tok/sec: 1163272.33
step:  413 | loss: 5.580599 | lr: 3.4741e-04 | norm: 1.1057, dt: 451.19ms, tok/sec: 1162005.44
step:  414 | loss: 5.554599 | lr: 3.4825e-04 | norm: 1.2544, dt: 452.70ms, tok/sec: 1158130.40
step:  415 | loss: 5.639246 | lr: 3.4909e-04 | norm: 0.8868, dt: 450.82ms, tok/sec: 1162959.19
step:  416 | loss: 5.704820 | lr: 3.4993e-04 | norm: 1.2068, dt: 451.27ms, tok/sec: 1161816.35
step:  417 | loss: 5.709803 | lr: 3.5077e-04 | norm: 1.2227, dt: 450.60ms, tok/sec: 1163526.53
step:  418 | loss: 5.672858 | lr: 3.5161e-04 | norm: 1.0530, dt: 451.16ms, tok/sec: 1162084.04
step:  419 | loss: 5.672499 | lr: 3.5245e-04 | norm: 1.2488, dt: 451.01ms, tok/sec: 1162481.51
step:  420 | loss: 5.696769 | lr: 3.5329e-04 | norm: 1.0212, dt: 450.94ms, tok/sec: 1162661.59
step:  421 | loss: 5.613521 | lr: 3.5413e-04 | norm: 1.1244, dt: 450.56ms, tok/sec: 1163631.81
step:  422 | loss: 5.700728 | lr: 3.5497e-04 | norm: 0.8356, dt: 451.26ms, tok/sec: 1161831.09
step:  423 | loss: 5.646099 | lr: 3.5580e-04 | norm: 0.9345, dt: 450.91ms, tok/sec: 1162737.82
step:  424 | loss: 5.597819 | lr: 3.5664e-04 | norm: 0.9221, dt: 449.85ms, tok/sec: 1165480.75
step:  425 | loss: 5.610747 | lr: 3.5748e-04 | norm: 0.9562, dt: 451.20ms, tok/sec: 1161983.95
step:  426 | loss: 5.616501 | lr: 3.5832e-04 | norm: 0.6879, dt: 451.36ms, tok/sec: 1161571.49
step:  427 | loss: 5.594351 | lr: 3.5916e-04 | norm: 0.7341, dt: 450.68ms, tok/sec: 1163320.94
step:  428 | loss: 5.663035 | lr: 3.6000e-04 | norm: 0.7131, dt: 450.84ms, tok/sec: 1162923.52
step:  429 | loss: 5.606853 | lr: 3.6084e-04 | norm: 0.7456, dt: 450.86ms, tok/sec: 1162861.41
step:  430 | loss: 5.556004 | lr: 3.6168e-04 | norm: 0.7284, dt: 451.82ms, tok/sec: 1160385.45
step:  431 | loss: 5.583633 | lr: 3.6252e-04 | norm: 0.8414, dt: 450.78ms, tok/sec: 1163076.68
step:  432 | loss: 5.637104 | lr: 3.6336e-04 | norm: 0.8565, dt: 450.91ms, tok/sec: 1162729.22
step:  433 | loss: 5.545121 | lr: 3.6420e-04 | norm: 0.8338, dt: 450.72ms, tok/sec: 1163215.72
step:  434 | loss: 5.577405 | lr: 3.6503e-04 | norm: 0.9026, dt: 451.93ms, tok/sec: 1160098.34
step:  435 | loss: 5.555623 | lr: 3.6587e-04 | norm: 1.0107, dt: 450.54ms, tok/sec: 1163699.55
step:  436 | loss: 5.676847 | lr: 3.6671e-04 | norm: 1.1117, dt: 451.05ms, tok/sec: 1162362.30
step:  437 | loss: 5.560984 | lr: 3.6755e-04 | norm: 1.1578, dt: 450.82ms, tok/sec: 1162964.11
step:  438 | loss: 5.542531 | lr: 3.6839e-04 | norm: 0.9811, dt: 452.07ms, tok/sec: 1159757.55
step:  439 | loss: 5.443342 | lr: 3.6923e-04 | norm: 1.3759, dt: 450.83ms, tok/sec: 1162951.81
step:  440 | loss: 5.502358 | lr: 3.7007e-04 | norm: 1.1391, dt: 450.66ms, tok/sec: 1163385.57
step:  441 | loss: 5.488263 | lr: 3.7091e-04 | norm: 1.0395, dt: 449.95ms, tok/sec: 1165216.43
step:  442 | loss: 5.487529 | lr: 3.7175e-04 | norm: 0.9587, dt: 450.92ms, tok/sec: 1162704.01
step:  443 | loss: 5.530531 | lr: 3.7259e-04 | norm: 0.9174, dt: 451.24ms, tok/sec: 1161889.40
step:  444 | loss: 5.499425 | lr: 3.7343e-04 | norm: 0.9686, dt: 450.75ms, tok/sec: 1163144.35
step:  445 | loss: 5.496019 | lr: 3.7427e-04 | norm: 1.0058, dt: 451.59ms, tok/sec: 1160993.80
step:  446 | loss: 5.514693 | lr: 3.7510e-04 | norm: 1.2254, dt: 451.20ms, tok/sec: 1161977.20
step:  447 | loss: 5.507476 | lr: 3.7594e-04 | norm: 0.8538, dt: 450.28ms, tok/sec: 1164369.33
step:  448 | loss: 5.451132 | lr: 3.7678e-04 | norm: 0.7847, dt: 450.86ms, tok/sec: 1162862.03
step:  449 | loss: 5.469124 | lr: 3.7762e-04 | norm: 0.7428, dt: 451.93ms, tok/sec: 1160101.40
step:  450 | loss: 5.448326 | lr: 3.7846e-04 | norm: 0.7632, dt: 451.25ms, tok/sec: 1161860.55
step:  451 | loss: 5.378790 | lr: 3.7930e-04 | norm: 0.9439, dt: 450.95ms, tok/sec: 1162640.69
step:  452 | loss: 5.439177 | lr: 3.8014e-04 | norm: 1.4160, dt: 451.83ms, tok/sec: 1160365.85
step:  453 | loss: 5.457942 | lr: 3.8098e-04 | norm: 1.0395, dt: 451.98ms, tok/sec: 1159973.51
step:  454 | loss: 5.388053 | lr: 3.8182e-04 | norm: 0.9757, dt: 450.33ms, tok/sec: 1164225.70
step:  455 | loss: 5.450621 | lr: 3.8266e-04 | norm: 1.0219, dt: 452.34ms, tok/sec: 1159053.97
step:  456 | loss: 5.375077 | lr: 3.8350e-04 | norm: 0.9776, dt: 452.06ms, tok/sec: 1159764.28
step:  457 | loss: 5.391158 | lr: 3.8434e-04 | norm: 1.0195, dt: 452.03ms, tok/sec: 1159854.81
step:  458 | loss: 5.389266 | lr: 3.8517e-04 | norm: 1.0195, dt: 453.40ms, tok/sec: 1156356.37
step:  459 | loss: 5.373677 | lr: 3.8601e-04 | norm: 0.9525, dt: 450.38ms, tok/sec: 1164109.21
step:  460 | loss: 5.368550 | lr: 3.8685e-04 | norm: 0.8782, dt: 450.69ms, tok/sec: 1163292.02
step:  461 | loss: 5.337828 | lr: 3.8769e-04 | norm: 0.8321, dt: 451.70ms, tok/sec: 1160711.90
step:  462 | loss: 5.446158 | lr: 3.8853e-04 | norm: 1.0132, dt: 451.23ms, tok/sec: 1161915.19
step:  463 | loss: 5.532671 | lr: 3.8937e-04 | norm: 1.3394, dt: 449.88ms, tok/sec: 1165395.52
step:  464 | loss: 5.531384 | lr: 3.9021e-04 | norm: 1.1671, dt: 451.21ms, tok/sec: 1161950.18
step:  465 | loss: 5.593490 | lr: 3.9105e-04 | norm: 1.1726, dt: 451.41ms, tok/sec: 1161448.79
step:  466 | loss: 5.506946 | lr: 3.9189e-04 | norm: 1.2026, dt: 451.33ms, tok/sec: 1161643.28
step:  467 | loss: 5.490635 | lr: 3.9273e-04 | norm: 0.8810, dt: 450.75ms, tok/sec: 1163145.58
step:  468 | loss: 5.496211 | lr: 3.9357e-04 | norm: 0.9814, dt: 453.12ms, tok/sec: 1157063.38
step:  469 | loss: 5.489705 | lr: 3.9441e-04 | norm: 1.1769, dt: 451.53ms, tok/sec: 1161139.08
step:  470 | loss: 5.493035 | lr: 3.9524e-04 | norm: 1.1036, dt: 450.17ms, tok/sec: 1164638.20
step:  471 | loss: 5.521268 | lr: 3.9608e-04 | norm: 0.8645, dt: 451.52ms, tok/sec: 1161166.68
step:  472 | loss: 5.488312 | lr: 3.9692e-04 | norm: 0.8456, dt: 450.00ms, tok/sec: 1165074.44
step:  473 | loss: 5.515857 | lr: 3.9776e-04 | norm: 0.9602, dt: 453.45ms, tok/sec: 1156215.32
step:  474 | loss: 5.467071 | lr: 3.9860e-04 | norm: 0.9527, dt: 449.82ms, tok/sec: 1165550.56
step:  475 | loss: 5.441284 | lr: 3.9944e-04 | norm: 1.1269, dt: 451.24ms, tok/sec: 1161873.44
step:  476 | loss: 5.475091 | lr: 4.0028e-04 | norm: 0.9133, dt: 449.74ms, tok/sec: 1165754.46
step:  477 | loss: 5.487018 | lr: 4.0112e-04 | norm: 0.9719, dt: 450.60ms, tok/sec: 1163534.53
step:  478 | loss: 5.456959 | lr: 4.0196e-04 | norm: 1.0718, dt: 451.31ms, tok/sec: 1161697.90
step:  479 | loss: 5.468352 | lr: 4.0280e-04 | norm: 0.9141, dt: 451.02ms, tok/sec: 1162446.48
step:  480 | loss: 5.438668 | lr: 4.0364e-04 | norm: 0.8564, dt: 449.72ms, tok/sec: 1165806.37
step:  481 | loss: 5.463982 | lr: 4.0448e-04 | norm: 0.9143, dt: 450.27ms, tok/sec: 1164374.88
step:  482 | loss: 5.447021 | lr: 4.0531e-04 | norm: 0.9981, dt: 451.04ms, tok/sec: 1162390.56
step:  483 | loss: 5.477362 | lr: 4.0615e-04 | norm: 1.1173, dt: 450.99ms, tok/sec: 1162521.45
step:  484 | loss: 5.400808 | lr: 4.0699e-04 | norm: 0.9611, dt: 451.21ms, tok/sec: 1161963.69
step:  485 | loss: 5.413534 | lr: 4.0783e-04 | norm: 1.0446, dt: 449.79ms, tok/sec: 1165627.17
step:  486 | loss: 5.426154 | lr: 4.0867e-04 | norm: 0.9821, dt: 451.54ms, tok/sec: 1161108.43
step:  487 | loss: 5.343286 | lr: 4.0951e-04 | norm: 1.0943, dt: 450.30ms, tok/sec: 1164297.82
step:  488 | loss: 5.350983 | lr: 4.1035e-04 | norm: 1.0020, dt: 450.90ms, tok/sec: 1162746.43
step:  489 | loss: 5.392688 | lr: 4.1119e-04 | norm: 1.2761, dt: 449.67ms, tok/sec: 1165936.18
step:  490 | loss: 5.349108 | lr: 4.1203e-04 | norm: 0.9643, dt: 451.38ms, tok/sec: 1161533.45
step:  491 | loss: 5.374433 | lr: 4.1287e-04 | norm: 0.9807, dt: 451.18ms, tok/sec: 1162030.62
step:  492 | loss: 5.348203 | lr: 4.1371e-04 | norm: 0.8637, dt: 450.67ms, tok/sec: 1163364.03
step:  493 | loss: 5.362193 | lr: 4.1455e-04 | norm: 0.9438, dt: 449.97ms, tok/sec: 1165166.43
step:  494 | loss: 5.311892 | lr: 4.1538e-04 | norm: 0.9653, dt: 450.39ms, tok/sec: 1164064.84
step:  495 | loss: 5.334944 | lr: 4.1622e-04 | norm: 1.0359, dt: 450.54ms, tok/sec: 1163693.39
step:  496 | loss: 5.324602 | lr: 4.1706e-04 | norm: 0.8971, dt: 451.36ms, tok/sec: 1161581.92
step:  497 | loss: 5.281355 | lr: 4.1790e-04 | norm: 1.0776, dt: 451.21ms, tok/sec: 1161960.62
step:  498 | loss: 5.255301 | lr: 4.1874e-04 | norm: 1.1696, dt: 450.05ms, tok/sec: 1164944.21
step:  499 | loss: 5.329250 | lr: 4.1958e-04 | norm: 0.9349, dt: 450.03ms, tok/sec: 1165005.93
validation loss: 5.3899
HellaSwag accuracy: 2379/10042=0.2369
rank 4 sample 0: Hello, I'm a Manpreet, and I was a family that I am and my grandfather, I am and I like. When I would get to
rank 6 sample 0: Hello, I'm a Manpreet, A1,82 1+9,0003-0-0 (N1)
-3-rank 4 sample 1: Hello, I'm a Manpreet, if we have read here! That my! I’m in her to make up with my heart and give

rank 4 sample 2: Hello, I'm a Manpreet, the author – We will take a new name of the history to my understanding the study that in this activity, whichrank 6 sample 1: Hello, I'm a Manpreet, who may say that is the name is found in that the name of Jesus is a word that we have seen in

rank 4 sample 3: Hello, I'm a Manpreet, I'm here with me I'm pretty a look from the bottom. But the answer is a pretty much more commonrank 6 sample 2: Hello, I'm a Manpreet, a member of a few people in the U.S. I'm going back to write a story you are working

rank 6 sample 3: Hello, I'm a Manpreet, I. I haven’t you! I am I am for my husband, and are my grandparents who should
rank 0 sample 0: Hello, I'm a Manpreet, I am it on my father who has learned some ways I can take two months to go by me and how I
rank 0 sample 1: Hello, I'm a Manpreet, there?
I don’t be the story: I think from a story?
- So if yourank 5 sample 0: Hello, I'm a Manpreet, a girl.
So, I’s like the children with children. It’s a woman who

rank 0 sample 2: Hello, I'm a Manpreet, and I recommend the top 51010. And I am able to use a list, but I am I have
rank 2 sample 0: Hello, I'm a Manpreet, one who is, the I’m, I don’t know to I’t take arank 5 sample 1: Hello, I'm a Manpreet, just two weeks to say you are you are right to see, who don't go on your car in their cat

rank 0 sample 3: Hello, I'm a Manpreet, who we live in the time, which is something that we start your life? Here if one, my baby�
rank 5 sample 2: Hello, I'm a Manpreet, for a long time every day a room for the day.
For a little-known, I’s
rank 5 sample 3: Hello, I'm a Manpreet, “Try” is a good challenge with your work at the number’s for a ‘
rank 2 sample 1: Hello, I'm a Manpreet,
- Make your friend and do they choose to look at a week that is a good for you may get a
rank 2 sample 2: Hello, I'm a Manpreet, and I have been the first place as I'm an e-oldt and is I'd get I'm �
rank 2 sample 3: Hello, I'm a Manpreet, a group you can find the most popular style of worksheet on the “A” because it doesn't
rank 7 sample 0: Hello, I'm a Manpreet, I am sure I don't have a way to me see an hour you the phone in the image on your phone
rank 7 sample 1: Hello, I'm a Manpreet, Bard is a very few months since it is too proud to the time. However, if you need to do
rank 7 sample 2: Hello, I'm a Manpreet, P. I need a lot of any information about by I am my advice. I have a lot of the right
rank 7 sample 3: Hello, I'm a Manpreet, No, No, Dink, V, K, L. A. A. No. M. A.
rank 1 sample 0: Hello, I'm a Manpreet, The “I can” here are more common because their children are being used. It was a great deal
rank 1 sample 1: Hello, I'm a Manpreet, or a day you need to make them get to your veterinarian. Then, let you, tell the best way to
rank 1 sample 2: Hello, I'm a Manpreet, a common word in the word, or a person, the second or two may be a condition in any other cases
rank 1 sample 3: Hello, I'm a Manpreet, I am I was I don't want back to your health at: I may be so I think I think I
rank 3 sample 0: Hello, I'm a Manpreet, I am I have I’m the first and I know my writing when I don’m my head
rank 3 sample 1: Hello, I'm a Manpreet, I have an example of Soguro in the United States.
L. I am I am I can
rank 3 sample 2: Hello, I'm a Manpreet, I’m as I feel I I would add me at my own? I am I were my I felt
rank 3 sample 3: Hello, I'm a Manpreet, I see the word on the scene of a good and is the name, and any of I see at a name
step:  500 | loss: 5.297225 | lr: 4.2042e-04 | norm: 1.0650, dt: 12084.58ms, tok/sec: 43384.88
step:  501 | loss: 5.255619 | lr: 4.2126e-04 | norm: 1.0036, dt: 448.70ms, tok/sec: 1168451.43
step:  502 | loss: 5.242509 | lr: 4.2210e-04 | norm: 1.0750, dt: 449.56ms, tok/sec: 1166237.31
step:  503 | loss: 5.322710 | lr: 4.2294e-04 | norm: 1.0572, dt: 449.43ms, tok/sec: 1166552.84
step:  504 | loss: 5.233439 | lr: 4.2378e-04 | norm: 0.8838, dt: 449.46ms, tok/sec: 1166485.39
step:  505 | loss: 5.324159 | lr: 4.2462e-04 | norm: 0.7903, dt: 451.50ms, tok/sec: 1161220.63
step:  506 | loss: 5.194259 | lr: 4.2545e-04 | norm: 0.8204, dt: 450.08ms, tok/sec: 1164875.72
step:  507 | loss: 5.239083 | lr: 4.2629e-04 | norm: 0.8443, dt: 450.15ms, tok/sec: 1164703.58
step:  508 | loss: 5.307143 | lr: 4.2713e-04 | norm: 0.8156, dt: 448.83ms, tok/sec: 1168113.16
step:  509 | loss: 5.419031 | lr: 4.2797e-04 | norm: 0.9919, dt: 450.59ms, tok/sec: 1163553.00
step:  510 | loss: 5.447724 | lr: 4.2881e-04 | norm: 1.2587, dt: 450.46ms, tok/sec: 1163897.88
step:  511 | loss: 5.419936 | lr: 4.2965e-04 | norm: 0.9754, dt: 450.73ms, tok/sec: 1163201.57
step:  512 | loss: 5.508165 | lr: 4.3049e-04 | norm: 1.0157, dt: 449.63ms, tok/sec: 1166045.61
step:  513 | loss: 5.404527 | lr: 4.3133e-04 | norm: 1.1525, dt: 450.45ms, tok/sec: 1163929.91
step:  514 | loss: 5.375427 | lr: 4.3217e-04 | norm: 0.9902, dt: 451.65ms, tok/sec: 1160824.03
step:  515 | loss: 5.416192 | lr: 4.3301e-04 | norm: 0.9048, dt: 450.40ms, tok/sec: 1164060.53
step:  516 | loss: 5.453724 | lr: 4.3385e-04 | norm: 0.8743, dt: 451.24ms, tok/sec: 1161873.44
step:  517 | loss: 5.383505 | lr: 4.3469e-04 | norm: 1.1267, dt: 450.82ms, tok/sec: 1162970.26
step:  518 | loss: 5.443062 | lr: 4.3552e-04 | norm: 1.1288, dt: 450.43ms, tok/sec: 1163981.05
step:  519 | loss: 5.379546 | lr: 4.3636e-04 | norm: 1.0350, dt: 450.10ms, tok/sec: 1164835.61
step:  520 | loss: 5.350798 | lr: 4.3720e-04 | norm: 1.0402, dt: 450.01ms, tok/sec: 1165068.27
step:  521 | loss: 5.349977 | lr: 4.3804e-04 | norm: 0.9136, dt: 451.69ms, tok/sec: 1160719.87
step:  522 | loss: 5.344882 | lr: 4.3888e-04 | norm: 0.8487, dt: 450.88ms, tok/sec: 1162818.37
step:  523 | loss: 5.340988 | lr: 4.3972e-04 | norm: 0.9443, dt: 451.15ms, tok/sec: 1162120.28
step:  524 | loss: 5.302733 | lr: 4.4056e-04 | norm: 0.8437, dt: 450.22ms, tok/sec: 1164514.85
step:  525 | loss: 5.335336 | lr: 4.4140e-04 | norm: 0.9802, dt: 450.62ms, tok/sec: 1163480.36
step:  526 | loss: 5.297252 | lr: 4.4224e-04 | norm: 1.0038, dt: 449.43ms, tok/sec: 1166563.36
step:  527 | loss: 5.333380 | lr: 4.4308e-04 | norm: 0.8546, dt: 451.24ms, tok/sec: 1161893.09
step:  528 | loss: 5.304715 | lr: 4.4392e-04 | norm: 0.8433, dt: 451.12ms, tok/sec: 1162197.66
step:  529 | loss: 5.300472 | lr: 4.4476e-04 | norm: 0.8575, dt: 450.28ms, tok/sec: 1164370.56
step:  530 | loss: 5.262229 | lr: 4.4559e-04 | norm: 0.7776, dt: 450.28ms, tok/sec: 1164358.85
step:  531 | loss: 5.297285 | lr: 4.4643e-04 | norm: 0.7353, dt: 450.44ms, tok/sec: 1163950.86
step:  532 | loss: 5.261681 | lr: 4.4727e-04 | norm: 0.8066, dt: 450.11ms, tok/sec: 1164802.91
step:  533 | loss: 5.256494 | lr: 4.4811e-04 | norm: 0.7652, dt: 450.99ms, tok/sec: 1162521.45
step:  534 | loss: 5.227607 | lr: 4.4895e-04 | norm: 0.7855, dt: 450.68ms, tok/sec: 1163336.95
step:  535 | loss: 5.310398 | lr: 4.4979e-04 | norm: 0.8726, dt: 449.49ms, tok/sec: 1166403.71
step:  536 | loss: 5.251180 | lr: 4.5063e-04 | norm: 1.2113, dt: 450.33ms, tok/sec: 1164225.70
step:  537 | loss: 5.222036 | lr: 4.5147e-04 | norm: 1.1137, dt: 450.64ms, tok/sec: 1163418.80
step:  538 | loss: 5.274092 | lr: 4.5231e-04 | norm: 1.1591, dt: 450.25ms, tok/sec: 1164439.62
step:  539 | loss: 5.194966 | lr: 4.5315e-04 | norm: 1.3044, dt: 451.49ms, tok/sec: 1161251.29
step:  540 | loss: 5.223688 | lr: 4.5399e-04 | norm: 1.0320, dt: 450.52ms, tok/sec: 1163739.58
step:  541 | loss: 5.245652 | lr: 4.5483e-04 | norm: 1.0130, dt: 450.57ms, tok/sec: 1163615.80
step:  542 | loss: 5.240453 | lr: 4.5566e-04 | norm: 1.2302, dt: 450.21ms, tok/sec: 1164535.20
step:  543 | loss: 5.228573 | lr: 4.5650e-04 | norm: 1.0810, dt: 450.10ms, tok/sec: 1164837.46
step:  544 | loss: 5.173733 | lr: 4.5734e-04 | norm: 1.1923, dt: 449.81ms, tok/sec: 1165582.06
step:  545 | loss: 5.122792 | lr: 4.5818e-04 | norm: 1.0529, dt: 451.05ms, tok/sec: 1162383.81
step:  546 | loss: 5.146041 | lr: 4.5902e-04 | norm: 0.9708, dt: 449.40ms, tok/sec: 1166637.62
step:  547 | loss: 5.187994 | lr: 4.5986e-04 | norm: 0.9669, dt: 450.21ms, tok/sec: 1164537.66
step:  548 | loss: 5.153467 | lr: 4.6070e-04 | norm: 0.9254, dt: 450.64ms, tok/sec: 1163436.04
step:  549 | loss: 5.243633 | lr: 4.6154e-04 | norm: 1.2015, dt: 451.58ms, tok/sec: 1161003.60
step:  550 | loss: 5.128854 | lr: 4.6238e-04 | norm: 0.9047, dt: 449.61ms, tok/sec: 1166098.78
step:  551 | loss: 5.073302 | lr: 4.6322e-04 | norm: 0.9187, dt: 450.06ms, tok/sec: 1164918.91
step:  552 | loss: 5.207730 | lr: 4.6406e-04 | norm: 1.1619, dt: 450.63ms, tok/sec: 1163449.58
step:  553 | loss: 5.172976 | lr: 4.6490e-04 | norm: 1.0000, dt: 450.93ms, tok/sec: 1162686.18
step:  554 | loss: 5.174593 | lr: 4.6573e-04 | norm: 0.9681, dt: 451.02ms, tok/sec: 1162439.72
step:  555 | loss: 5.183406 | lr: 4.6657e-04 | norm: 0.9789, dt: 450.24ms, tok/sec: 1164453.80
step:  556 | loss: 5.320887 | lr: 4.6741e-04 | norm: 0.9735, dt: 451.69ms, tok/sec: 1160725.38
step:  557 | loss: 5.312648 | lr: 4.6825e-04 | norm: 1.0633, dt: 450.06ms, tok/sec: 1164941.75
step:  558 | loss: 5.224999 | lr: 4.6909e-04 | norm: 0.9253, dt: 451.48ms, tok/sec: 1161258.04
step:  559 | loss: 5.270197 | lr: 4.6993e-04 | norm: 0.9259, dt: 450.18ms, tok/sec: 1164612.29
step:  560 | loss: 5.273290 | lr: 4.7077e-04 | norm: 1.0577, dt: 451.32ms, tok/sec: 1161668.44
step:  561 | loss: 5.241816 | lr: 4.7161e-04 | norm: 0.8536, dt: 451.29ms, tok/sec: 1161743.31
step:  562 | loss: 5.278222 | lr: 4.7245e-04 | norm: 0.9410, dt: 450.45ms, tok/sec: 1163908.35
step:  563 | loss: 5.299146 | lr: 4.7329e-04 | norm: 1.2300, dt: 450.45ms, tok/sec: 1163922.52
step:  564 | loss: 5.296222 | lr: 4.7413e-04 | norm: 1.0367, dt: 450.53ms, tok/sec: 1163710.63
step:  565 | loss: 5.261301 | lr: 4.7497e-04 | norm: 1.1144, dt: 450.71ms, tok/sec: 1163252.02
step:  566 | loss: 5.243072 | lr: 4.7580e-04 | norm: 1.1105, dt: 1155.46ms, tok/sec: 453750.21
step:  567 | loss: 5.229466 | lr: 4.7664e-04 | norm: 0.8619, dt: 451.21ms, tok/sec: 1161970.44
step:  568 | loss: 5.220088 | lr: 4.7748e-04 | norm: 0.7609, dt: 449.50ms, tok/sec: 1166382.68
step:  569 | loss: 5.199427 | lr: 4.7832e-04 | norm: 0.7100, dt: 1204.49ms, tok/sec: 435276.30
step:  570 | loss: 5.173608 | lr: 4.7916e-04 | norm: 0.7423, dt: 449.33ms, tok/sec: 1166823.33
step:  571 | loss: 5.210699 | lr: 4.8000e-04 | norm: 0.7105, dt: 448.58ms, tok/sec: 1168773.12
step:  572 | loss: 5.184916 | lr: 4.8084e-04 | norm: 0.6491, dt: 450.82ms, tok/sec: 1162969.65
step:  573 | loss: 5.166655 | lr: 4.8168e-04 | norm: 0.6708, dt: 449.84ms, tok/sec: 1165486.31
step:  574 | loss: 5.196191 | lr: 4.8252e-04 | norm: 0.7081, dt: 449.75ms, tok/sec: 1165730.36
step:  575 | loss: 5.157238 | lr: 4.8336e-04 | norm: 0.7112, dt: 449.49ms, tok/sec: 1166408.05
step:  576 | loss: 5.161129 | lr: 4.8420e-04 | norm: 0.8000, dt: 451.01ms, tok/sec: 1162474.13
step:  577 | loss: 5.169188 | lr: 4.8503e-04 | norm: 1.0340, dt: 450.79ms, tok/sec: 1163038.54
step:  578 | loss: 5.149042 | lr: 4.8587e-04 | norm: 1.1868, dt: 450.01ms, tok/sec: 1165054.69
step:  579 | loss: 5.141571 | lr: 4.8671e-04 | norm: 0.8924, dt: 449.34ms, tok/sec: 1166799.19
step:  580 | loss: 5.053495 | lr: 4.8755e-04 | norm: 1.2460, dt: 450.58ms, tok/sec: 1163586.25
step:  581 | loss: 5.072748 | lr: 4.8839e-04 | norm: 0.8116, dt: 451.02ms, tok/sec: 1162455.08
step:  582 | loss: 5.117496 | lr: 4.8923e-04 | norm: 1.0050, dt: 450.08ms, tok/sec: 1164881.89
step:  583 | loss: 5.130450 | lr: 4.9007e-04 | norm: 1.0577, dt: 450.53ms, tok/sec: 1163709.40
step:  584 | loss: 5.121805 | lr: 4.9091e-04 | norm: 1.1925, dt: 453.10ms, tok/sec: 1157116.96
step:  585 | loss: 5.182517 | lr: 4.9175e-04 | norm: 1.0247, dt: 450.46ms, tok/sec: 1163894.80
step:  586 | loss: 5.129673 | lr: 4.9259e-04 | norm: 1.0516, dt: 450.16ms, tok/sec: 1164661.63
step:  587 | loss: 5.125637 | lr: 4.9343e-04 | norm: 1.2616, dt: 449.90ms, tok/sec: 1165332.52
step:  588 | loss: 5.144878 | lr: 4.9427e-04 | norm: 1.1245, dt: 452.14ms, tok/sec: 1159582.65
step:  589 | loss: 5.153380 | lr: 4.9510e-04 | norm: 1.1552, dt: 450.93ms, tok/sec: 1162676.35
step:  590 | loss: 5.029729 | lr: 4.9594e-04 | norm: 0.9445, dt: 450.54ms, tok/sec: 1163698.32
step:  591 | loss: 5.036118 | lr: 4.9678e-04 | norm: 1.0730, dt: 451.26ms, tok/sec: 1161836.00
step:  592 | loss: 5.049754 | lr: 4.9762e-04 | norm: 1.1290, dt: 452.25ms, tok/sec: 1159296.55
step:  593 | loss: 5.066182 | lr: 4.9846e-04 | norm: 1.0446, dt: 450.78ms, tok/sec: 1163077.29
step:  594 | loss: 5.001928 | lr: 4.9930e-04 | norm: 0.9351, dt: 453.18ms, tok/sec: 1156912.42
step:  595 | loss: 5.012881 | lr: 5.0014e-04 | norm: 0.8800, dt: 450.61ms, tok/sec: 1163504.37
step:  596 | loss: 5.068063 | lr: 5.0098e-04 | norm: 0.8648, dt: 450.66ms, tok/sec: 1163373.87
step:  597 | loss: 5.008821 | lr: 5.0182e-04 | norm: 0.8179, dt: 451.20ms, tok/sec: 1161995.62
step:  598 | loss: 5.070797 | lr: 5.0266e-04 | norm: 0.8850, dt: 450.42ms, tok/sec: 1163988.44
step:  599 | loss: 5.034027 | lr: 5.0350e-04 | norm: 0.8304, dt: 450.65ms, tok/sec: 1163408.96
step:  600 | loss: 5.016186 | lr: 5.0434e-04 | norm: 0.6412, dt: 450.34ms, tok/sec: 1164196.73
step:  601 | loss: 5.118967 | lr: 5.0517e-04 | norm: 0.6289, dt: 454.79ms, tok/sec: 1152815.51
step:  602 | loss: 5.144393 | lr: 5.0601e-04 | norm: 0.7043, dt: 451.04ms, tok/sec: 1162401.01
step:  603 | loss: 5.179273 | lr: 5.0685e-04 | norm: 0.8092, dt: 450.46ms, tok/sec: 1163891.10
step:  604 | loss: 5.135999 | lr: 5.0769e-04 | norm: 0.8773, dt: 450.88ms, tok/sec: 1162801.77
step:  605 | loss: 5.136438 | lr: 5.0853e-04 | norm: 0.8940, dt: 450.99ms, tok/sec: 1162525.14
step:  606 | loss: 5.160385 | lr: 5.0937e-04 | norm: 1.0216, dt: 451.15ms, tok/sec: 1162109.84
step:  607 | loss: 5.169281 | lr: 5.1021e-04 | norm: 1.1879, dt: 450.65ms, tok/sec: 1163409.57
step:  608 | loss: 5.119873 | lr: 5.1105e-04 | norm: 0.8632, dt: 450.99ms, tok/sec: 1162515.92
step:  609 | loss: 5.186512 | lr: 5.1189e-04 | norm: 1.1110, dt: 450.35ms, tok/sec: 1164186.86
step:  610 | loss: 5.143051 | lr: 5.1273e-04 | norm: 1.0733, dt: 450.68ms, tok/sec: 1163336.33
step:  611 | loss: 5.121382 | lr: 5.1357e-04 | norm: 1.0093, dt: 450.36ms, tok/sec: 1164146.80
step:  612 | loss: 5.149398 | lr: 5.1441e-04 | norm: 1.0024, dt: 450.52ms, tok/sec: 1163729.73
step:  613 | loss: 5.154481 | lr: 5.1524e-04 | norm: 0.9584, dt: 450.10ms, tok/sec: 1164827.59
step:  614 | loss: 5.084535 | lr: 5.1608e-04 | norm: 0.7911, dt: 452.79ms, tok/sec: 1157905.98
step:  615 | loss: 5.111565 | lr: 5.1692e-04 | norm: 0.7817, dt: 450.83ms, tok/sec: 1162947.51
step:  616 | loss: 5.084018 | lr: 5.1776e-04 | norm: 0.8180, dt: 450.17ms, tok/sec: 1164633.26
step:  617 | loss: 5.093588 | lr: 5.1860e-04 | norm: 0.7334, dt: 450.01ms, tok/sec: 1165067.04
step:  618 | loss: 5.108469 | lr: 5.1944e-04 | norm: 0.7324, dt: 450.51ms, tok/sec: 1163754.98
step:  619 | loss: 5.065288 | lr: 5.2028e-04 | norm: 0.7723, dt: 450.88ms, tok/sec: 1162798.69
step:  620 | loss: 5.071309 | lr: 5.2112e-04 | norm: 0.7817, dt: 450.29ms, tok/sec: 1164329.87
step:  621 | loss: 5.057457 | lr: 5.2196e-04 | norm: 0.7741, dt: 449.81ms, tok/sec: 1165568.47
step:  622 | loss: 5.061941 | lr: 5.2280e-04 | norm: 0.7704, dt: 450.61ms, tok/sec: 1163511.14
step:  623 | loss: 5.034254 | lr: 5.2364e-04 | norm: 0.7142, dt: 452.20ms, tok/sec: 1159412.68
step:  624 | loss: 5.115631 | lr: 5.2448e-04 | norm: 0.8089, dt: 449.55ms, tok/sec: 1166249.68
step:  625 | loss: 4.960752 | lr: 5.2531e-04 | norm: 0.8376, dt: 450.32ms, tok/sec: 1164248.50
step:  626 | loss: 5.040748 | lr: 5.2615e-04 | norm: 0.8287, dt: 450.52ms, tok/sec: 1163729.73
step:  627 | loss: 4.993439 | lr: 5.2699e-04 | norm: 0.7954, dt: 449.90ms, tok/sec: 1165331.90
step:  628 | loss: 5.031451 | lr: 5.2783e-04 | norm: 0.8419, dt: 449.56ms, tok/sec: 1166235.46
step:  629 | loss: 5.040916 | lr: 5.2867e-04 | norm: 0.7777, dt: 451.12ms, tok/sec: 1162200.12
step:  630 | loss: 5.000244 | lr: 5.2951e-04 | norm: 0.8058, dt: 450.55ms, tok/sec: 1163661.37
step:  631 | loss: 5.080201 | lr: 5.3035e-04 | norm: 0.9596, dt: 449.97ms, tok/sec: 1165166.43
step:  632 | loss: 5.067458 | lr: 5.3119e-04 | norm: 1.0581, dt: 450.71ms, tok/sec: 1163238.48
step:  633 | loss: 5.012858 | lr: 5.3203e-04 | norm: 0.9249, dt: 449.74ms, tok/sec: 1165749.52
step:  634 | loss: 5.038227 | lr: 5.3287e-04 | norm: 0.9803, dt: 450.56ms, tok/sec: 1163647.21
step:  635 | loss: 5.236688 | lr: 5.3371e-04 | norm: 1.0454, dt: 449.84ms, tok/sec: 1165499.90
step:  636 | loss: 4.966404 | lr: 5.3455e-04 | norm: 1.0065, dt: 449.28ms, tok/sec: 1166949.03
step:  637 | loss: 4.990733 | lr: 5.3538e-04 | norm: 1.2511, dt: 450.96ms, tok/sec: 1162600.12
step:  638 | loss: 4.834529 | lr: 5.3622e-04 | norm: 0.8528, dt: 449.63ms, tok/sec: 1166052.41
step:  639 | loss: 4.968249 | lr: 5.3706e-04 | norm: 0.8231, dt: 451.19ms, tok/sec: 1162015.27
step:  640 | loss: 4.941356 | lr: 5.3790e-04 | norm: 0.8636, dt: 450.24ms, tok/sec: 1164474.15
step:  641 | loss: 4.926757 | lr: 5.3874e-04 | norm: 0.9435, dt: 451.77ms, tok/sec: 1160516.50
step:  642 | loss: 4.954450 | lr: 5.3958e-04 | norm: 1.0738, dt: 450.20ms, tok/sec: 1164559.25
step:  643 | loss: 4.926229 | lr: 5.4042e-04 | norm: 0.9436, dt: 450.38ms, tok/sec: 1164108.60
step:  644 | loss: 4.919380 | lr: 5.4126e-04 | norm: 0.9851, dt: 451.05ms, tok/sec: 1162382.58
step:  645 | loss: 4.964190 | lr: 5.4210e-04 | norm: 0.8736, dt: 450.78ms, tok/sec: 1163074.22
step:  646 | loss: 4.955375 | lr: 5.4294e-04 | norm: 0.7722, dt: 450.62ms, tok/sec: 1163487.13
step:  647 | loss: 4.914682 | lr: 5.4378e-04 | norm: 0.7388, dt: 449.69ms, tok/sec: 1165885.49
step:  648 | loss: 5.054055 | lr: 5.4462e-04 | norm: 0.6492, dt: 450.32ms, tok/sec: 1164248.50
step:  649 | loss: 5.051454 | lr: 5.4545e-04 | norm: 0.7183, dt: 450.06ms, tok/sec: 1164921.38
step:  650 | loss: 5.142298 | lr: 5.4629e-04 | norm: 0.7661, dt: 450.54ms, tok/sec: 1163681.69
step:  651 | loss: 5.043602 | lr: 5.4713e-04 | norm: 0.8658, dt: 450.38ms, tok/sec: 1164091.96
step:  652 | loss: 5.022565 | lr: 5.4797e-04 | norm: 0.9053, dt: 450.82ms, tok/sec: 1162965.96
step:  653 | loss: 5.083238 | lr: 5.4881e-04 | norm: 0.9761, dt: 450.50ms, tok/sec: 1163780.84
step:  654 | loss: 5.046685 | lr: 5.4965e-04 | norm: 0.9764, dt: 450.87ms, tok/sec: 1162841.12
step:  655 | loss: 5.013186 | lr: 5.5049e-04 | norm: 0.8754, dt: 450.24ms, tok/sec: 1164463.66
step:  656 | loss: 4.992725 | lr: 5.5133e-04 | norm: 0.7582, dt: 450.77ms, tok/sec: 1163106.82
step:  657 | loss: 5.054976 | lr: 5.5217e-04 | norm: 0.6870, dt: 450.58ms, tok/sec: 1163573.32
step:  658 | loss: 5.025474 | lr: 5.5301e-04 | norm: 0.6538, dt: 450.68ms, tok/sec: 1163338.79
step:  659 | loss: 5.036510 | lr: 5.5385e-04 | norm: 0.6164, dt: 452.12ms, tok/sec: 1159626.06
step:  660 | loss: 4.968155 | lr: 5.5469e-04 | norm: 0.6724, dt: 450.84ms, tok/sec: 1162913.68
step:  661 | loss: 4.980232 | lr: 5.5552e-04 | norm: 0.8244, dt: 449.78ms, tok/sec: 1165659.91
step:  662 | loss: 4.942608 | lr: 5.5636e-04 | norm: 1.0247, dt: 449.55ms, tok/sec: 1166254.01
step:  663 | loss: 4.937282 | lr: 5.5720e-04 | norm: 0.9642, dt: 452.21ms, tok/sec: 1159398.01
step:  664 | loss: 4.973715 | lr: 5.5804e-04 | norm: 0.9964, dt: 451.03ms, tok/sec: 1162418.21
step:  665 | loss: 4.972162 | lr: 5.5888e-04 | norm: 1.2179, dt: 450.64ms, tok/sec: 1163431.11
step:  666 | loss: 4.950792 | lr: 5.5972e-04 | norm: 0.7450, dt: 449.90ms, tok/sec: 1165339.32
step:  667 | loss: 4.930778 | lr: 5.6056e-04 | norm: 0.8718, dt: 450.61ms, tok/sec: 1163509.91
step:  668 | loss: 4.909274 | lr: 5.6140e-04 | norm: 0.7674, dt: 451.02ms, tok/sec: 1162444.64
step:  669 | loss: 4.953950 | lr: 5.6224e-04 | norm: 0.7410, dt: 450.22ms, tok/sec: 1164520.40
step:  670 | loss: 4.999726 | lr: 5.6308e-04 | norm: 0.8256, dt: 450.08ms, tok/sec: 1164876.95
step:  671 | loss: 4.963212 | lr: 5.6392e-04 | norm: 0.8708, dt: 450.00ms, tok/sec: 1165084.32
step:  672 | loss: 4.891076 | lr: 5.6476e-04 | norm: 0.8653, dt: 451.16ms, tok/sec: 1162088.96
step:  673 | loss: 4.890080 | lr: 5.6559e-04 | norm: 0.8707, dt: 449.64ms, tok/sec: 1166009.13
step:  674 | loss: 4.913070 | lr: 5.6643e-04 | norm: 0.9400, dt: 451.29ms, tok/sec: 1161741.47
step:  675 | loss: 4.947227 | lr: 5.6727e-04 | norm: 0.8127, dt: 450.66ms, tok/sec: 1163382.49
step:  676 | loss: 4.933978 | lr: 5.6811e-04 | norm: 0.7417, dt: 450.15ms, tok/sec: 1164685.69
step:  677 | loss: 4.891995 | lr: 5.6895e-04 | norm: 0.7169, dt: 450.67ms, tok/sec: 1163346.18
step:  678 | loss: 4.896741 | lr: 5.6979e-04 | norm: 0.7436, dt: 450.12ms, tok/sec: 1164785.63
step:  679 | loss: 4.849129 | lr: 5.7063e-04 | norm: 0.6892, dt: 450.68ms, tok/sec: 1163337.56
step:  680 | loss: 4.880952 | lr: 5.7147e-04 | norm: 0.6261, dt: 449.92ms, tok/sec: 1165303.50
step:  681 | loss: 4.857063 | lr: 5.7231e-04 | norm: 0.5440, dt: 449.98ms, tok/sec: 1165124.45
step:  682 | loss: 4.835935 | lr: 5.7315e-04 | norm: 0.5607, dt: 450.55ms, tok/sec: 1163655.21
step:  683 | loss: 4.768756 | lr: 5.7399e-04 | norm: 0.6226, dt: 451.40ms, tok/sec: 1161461.67
step:  684 | loss: 4.775902 | lr: 5.7483e-04 | norm: 0.7877, dt: 449.67ms, tok/sec: 1165950.40
step:  685 | loss: 4.819441 | lr: 5.7566e-04 | norm: 1.0536, dt: 449.70ms, tok/sec: 1165855.82
step:  686 | loss: 4.826666 | lr: 5.7650e-04 | norm: 1.1272, dt: 451.69ms, tok/sec: 1160719.87
step:  687 | loss: 4.805628 | lr: 5.7734e-04 | norm: 1.0618, dt: 449.99ms, tok/sec: 1165115.19
step:  688 | loss: 4.870863 | lr: 5.7818e-04 | norm: 0.8689, dt: 450.34ms, tok/sec: 1164201.66
step:  689 | loss: 4.814578 | lr: 5.7902e-04 | norm: 0.9874, dt: 452.24ms, tok/sec: 1159311.83
step:  690 | loss: 4.864911 | lr: 5.7986e-04 | norm: 0.9500, dt: 450.61ms, tok/sec: 1163508.68
step:  691 | loss: 4.820231 | lr: 5.8070e-04 | norm: 0.9642, dt: 451.09ms, tok/sec: 1162271.99
step:  692 | loss: 5.029173 | lr: 5.8154e-04 | norm: 1.1109, dt: 449.98ms, tok/sec: 1165134.32
step:  693 | loss: 4.819975 | lr: 5.8238e-04 | norm: 0.9629, dt: 451.76ms, tok/sec: 1160553.86
step:  694 | loss: 4.967187 | lr: 5.8322e-04 | norm: 0.9138, dt: 451.27ms, tok/sec: 1161815.74
step:  695 | loss: 4.993787 | lr: 5.8406e-04 | norm: 0.8621, dt: 450.38ms, tok/sec: 1164114.14
step:  696 | loss: 4.928793 | lr: 5.8490e-04 | norm: 0.7629, dt: 450.05ms, tok/sec: 1164951.62
step:  697 | loss: 4.948499 | lr: 5.8573e-04 | norm: 0.6987, dt: 451.08ms, tok/sec: 1162299.02
step:  698 | loss: 4.930549 | lr: 5.8657e-04 | norm: 0.8034, dt: 451.11ms, tok/sec: 1162207.49
step:  699 | loss: 4.909345 | lr: 5.8741e-04 | norm: 0.7358, dt: 449.56ms, tok/sec: 1166226.80
step:  700 | loss: 4.857543 | lr: 5.8825e-04 | norm: 0.7435, dt: 450.60ms, tok/sec: 1163544.38
step:  701 | loss: 4.960154 | lr: 5.8909e-04 | norm: 0.8290, dt: 450.61ms, tok/sec: 1163495.75
step:  702 | loss: 4.948364 | lr: 5.8993e-04 | norm: 1.0217, dt: 451.28ms, tok/sec: 1161786.89
step:  703 | loss: 4.889576 | lr: 5.9077e-04 | norm: 0.9127, dt: 452.92ms, tok/sec: 1157568.31
step:  704 | loss: 4.933463 | lr: 5.9161e-04 | norm: 0.9264, dt: 450.65ms, tok/sec: 1163412.03
step:  705 | loss: 4.958347 | lr: 5.9245e-04 | norm: 0.9986, dt: 450.23ms, tok/sec: 1164479.70
step:  706 | loss: 4.948677 | lr: 5.9329e-04 | norm: 0.9061, dt: 450.16ms, tok/sec: 1164670.89
step:  707 | loss: 4.946730 | lr: 5.9413e-04 | norm: 0.7794, dt: 451.60ms, tok/sec: 1160947.83
step:  708 | loss: 4.901402 | lr: 5.9497e-04 | norm: 0.7714, dt: 449.85ms, tok/sec: 1165469.63
step:  709 | loss: 4.915551 | lr: 5.9580e-04 | norm: 0.6721, dt: 450.87ms, tok/sec: 1162825.75
step:  710 | loss: 4.823151 | lr: 5.9664e-04 | norm: 0.6354, dt: 450.15ms, tok/sec: 1164693.71
step:  711 | loss: 4.880397 | lr: 5.9748e-04 | norm: 0.6277, dt: 451.13ms, tok/sec: 1162165.73
step:  712 | loss: 4.798316 | lr: 5.9832e-04 | norm: 0.6900, dt: 450.96ms, tok/sec: 1162598.89
step:  713 | loss: 4.903032 | lr: 5.9916e-04 | norm: 0.7639, dt: 449.79ms, tok/sec: 1165639.52
step:  714 | loss: 4.863691 | lr: 6.0000e-04 | norm: 0.7391, dt: 452.32ms, tok/sec: 1159099.18
step:  715 | loss: 4.843947 | lr: 6.0000e-04 | norm: 0.7897, dt: 449.92ms, tok/sec: 1165283.74
step:  716 | loss: 4.894820 | lr: 6.0000e-04 | norm: 0.6996, dt: 450.76ms, tok/sec: 1163125.89
step:  717 | loss: 4.838642 | lr: 6.0000e-04 | norm: 0.6874, dt: 450.35ms, tok/sec: 1164180.09
step:  718 | loss: 4.852530 | lr: 6.0000e-04 | norm: 0.6497, dt: 450.54ms, tok/sec: 1163690.31
step:  719 | loss: 4.775919 | lr: 6.0000e-04 | norm: 0.6617, dt: 450.97ms, tok/sec: 1162583.53
step:  720 | loss: 4.825980 | lr: 6.0000e-04 | norm: 0.7479, dt: 450.19ms, tok/sec: 1164586.39
step:  721 | loss: 4.791979 | lr: 6.0000e-04 | norm: 0.7230, dt: 451.15ms, tok/sec: 1162108.61
step:  722 | loss: 4.805150 | lr: 6.0000e-04 | norm: 0.7237, dt: 450.67ms, tok/sec: 1163364.03
step:  723 | loss: 4.776428 | lr: 6.0000e-04 | norm: 0.7553, dt: 450.24ms, tok/sec: 1164467.36
step:  724 | loss: 4.875051 | lr: 6.0000e-04 | norm: 0.8075, dt: 454.17ms, tok/sec: 1154395.64
step:  725 | loss: 4.784839 | lr: 6.0000e-04 | norm: 0.8898, dt: 450.83ms, tok/sec: 1162937.67
step:  726 | loss: 4.780155 | lr: 6.0000e-04 | norm: 1.2078, dt: 450.71ms, tok/sec: 1163252.64
step:  727 | loss: 4.782218 | lr: 6.0000e-04 | norm: 0.8495, dt: 450.46ms, tok/sec: 1163894.80
step:  728 | loss: 4.733937 | lr: 6.0000e-04 | norm: 0.7616, dt: 449.88ms, tok/sec: 1165405.40
step:  729 | loss: 4.801769 | lr: 6.0000e-04 | norm: 0.8738, dt: 450.77ms, tok/sec: 1163087.13
step:  730 | loss: 4.714267 | lr: 6.0000e-04 | norm: 0.7956, dt: 450.48ms, tok/sec: 1163845.52
step:  731 | loss: 4.716516 | lr: 6.0000e-04 | norm: 0.7497, dt: 450.89ms, tok/sec: 1162797.46
step:  732 | loss: 4.693797 | lr: 6.0000e-04 | norm: 0.7671, dt: 449.55ms, tok/sec: 1166241.02
step:  733 | loss: 4.669572 | lr: 6.0000e-04 | norm: 0.7685, dt: 451.25ms, tok/sec: 1161854.41
step:  734 | loss: 4.681342 | lr: 6.0000e-04 | norm: 0.7727, dt: 449.98ms, tok/sec: 1165132.47
step:  735 | loss: 4.691572 | lr: 6.0000e-04 | norm: 0.7775, dt: 450.42ms, tok/sec: 1164008.77
step:  736 | loss: 4.709432 | lr: 6.0000e-04 | norm: 0.7282, dt: 450.58ms, tok/sec: 1163586.25
step:  737 | loss: 4.657310 | lr: 6.0000e-04 | norm: 0.6970, dt: 450.30ms, tok/sec: 1164296.58
step:  738 | loss: 4.689465 | lr: 6.0000e-04 | norm: 0.7862, dt: 450.71ms, tok/sec: 1163240.95
step:  739 | loss: 4.701566 | lr: 6.0000e-04 | norm: 0.8481, dt: 450.16ms, tok/sec: 1164664.10
step:  740 | loss: 4.669748 | lr: 6.0000e-04 | norm: 0.7111, dt: 449.91ms, tok/sec: 1165328.20
step:  741 | loss: 4.729314 | lr: 6.0000e-04 | norm: 0.6682, dt: 449.89ms, tok/sec: 1165362.16
step:  742 | loss: 4.823674 | lr: 6.0000e-04 | norm: 0.6858, dt: 450.97ms, tok/sec: 1162567.55
step:  743 | loss: 4.790393 | lr: 6.0000e-04 | norm: 0.8256, dt: 451.18ms, tok/sec: 1162042.29
step:  744 | loss: 4.842621 | lr: 6.0000e-04 | norm: 0.8397, dt: 451.04ms, tok/sec: 1162406.54
step:  745 | loss: 4.847676 | lr: 6.0000e-04 | norm: 0.7939, dt: 450.39ms, tok/sec: 1164085.79
step:  746 | loss: 4.838341 | lr: 6.0000e-04 | norm: 0.8694, dt: 450.69ms, tok/sec: 1163300.64
step:  747 | loss: 4.830745 | lr: 6.0000e-04 | norm: 0.8122, dt: 451.41ms, tok/sec: 1161440.20
step:  748 | loss: 4.823099 | lr: 6.0000e-04 | norm: 0.7751, dt: 452.13ms, tok/sec: 1159603.44
step:  749 | loss: 4.884480 | lr: 6.0000e-04 | norm: 0.8847, dt: 450.29ms, tok/sec: 1164332.95
validation loss: 4.7895
HellaSwag accuracy: 2457/10042=0.2447
rank 4 sample 0: Hello, I'm a Manpreet, I'm gonna be so old.
That is okay to have a hard disk so far and much I want to
rank 6 sample 0: Hello, I'm a Manpreet, to help you help us today, and to help people with disabilities, but not necessarily a doctor, but it is
rank 4 sample 1: Hello, I'm a Manpreet, at firstname who will be married during the end of four months. You are ready to be married at a two
rank 2 sample 0: Hello, I'm a Manpreet, there has a history of knowledge and skills that has its power.
And the problem is that we have no idearank 6 sample 1: Hello, I'm a Manpreet, or who wants to give you some time!
"I'm in all, and I don't want a more

rank 4 sample 2: Hello, I'm a Manpreet, who wanted to leave their school. And when I was a good part of the following I know that that I could
rank 6 sample 2: Hello, I'm a Manpreet, a woman. I was born, and had two children, with all three weeks, and I was sick. After
rank 4 sample 3: Hello, I'm a Manpreet, and you live and live, and see you and I even seen in more detail.
The more I have made
rank 2 sample 1: Hello, I'm a Manpreet, that is another, it is a kind of that
one, in one way, and that it does
to
rank 6 sample 3: Hello, I'm a Manpreet, who lives, lives.
It is for us to be a professor of life and the brain. Instead this kind
rank 2 sample 2: Hello, I'm a Manpreet, I'm not the one of our history because I was as one, my daughter is not the best time, how
rank 0 sample 0: Hello, I'm a Manpreet, and I know what is it’s and you need to work! We need this job! I know that
rank 2 sample 3: Hello, I'm a Manpreet, a New York, a public school teacher has a responsibility of their students to go and go ahead as a school school
rank 0 sample 1: Hello, I'm a Manpreet, whose memory is the most efficient. Since the first days old as he may find his name as he became known as
rank 0 sample 2: Hello, I'm a Manpreet, I'm to find more to understand, and understand what is in.
- You want to know what is that
rank 0 sample 3: Hello, I'm a Manpreet, or just a little bit of it, but here was a thing and we've finished in your course.
-
rank 5 sample 0: Hello, I'm a Manpreet, a couple, and two, and a young man from a high boy with a little chance of her baby and having
rank 5 sample 1: Hello, I'm a Manpreet, just by being an enemy to vote on the government. The American state of British history, and there was about the
rank 7 sample 0: Hello, I'm a Manpreet, and I think I'm sure to let his children and grandchildren enjoy the idea of learning.
Why can you know
rank 5 sample 2: Hello, I'm a Manpreet, we've worked to with the best for the next year.
As a teenager has helped the city and the city
rank 7 sample 1: Hello, I'm a Manpreet, has said:
The I don're getting to start
to give me a picture of a friend with my mother
rank 5 sample 3: Hello, I'm a Manpreet, because you can start it down and you're trying to make sure that you need is going on right, but I
rank 7 sample 2: Hello, I'm a Manpreet, so I don't leave me the world’s first time. There are a lot of fun activities. There
rank 7 sample 3: Hello, I'm a Manpreet, when you don't like anything. All right do you've got a way to say you'd for your kids.
rank 3 sample 0: Hello, I'm a Manpreet, and I'm a man and a mother for the world,
the time goes by us to end him, and
rank 3 sample 1: Hello, I'm a Manpreet, and it was a great fact I won't see anything so I can't see it.
"I guess if
rank 3 sample 2: Hello, I'm a Manpreet, and the two three letters of you can only send me the best way. You've got your back in. Your
rank 3 sample 3: Hello, I'm a Manpreet, and, and a daughter. (p. 18: 17). I am a female (r. 42). (
rank 1 sample 0: Hello, I'm a Manpreet, where the big thing to say, one can read about you?
"One of our greatest mysteries, that's
rank 1 sample 1: Hello, I'm a Manpreet, the people, have a lot of money in the world."
We have some books of New York and New York
rank 1 sample 2: Hello, I'm a Manpreet, a Japanese-based Japanese, and the Japanese Japanese, or “the Japanese”. This Japanese Japanese and
rank 1 sample 3: Hello, I'm a Manpreet, and I'm looking to show up your ability to read these sounds before he would be talking without having to him.
step:  750 | loss: 4.807343 | lr: 6.0000e-04 | norm: 0.8735, dt: 12080.68ms, tok/sec: 43398.87
step:  751 | loss: 4.848520 | lr: 5.9999e-04 | norm: 0.8644, dt: 448.73ms, tok/sec: 1168394.31
step:  752 | loss: 4.829801 | lr: 5.9999e-04 | norm: 0.8145, dt: 449.66ms, tok/sec: 1165975.74
step:  753 | loss: 4.812675 | lr: 5.9999e-04 | norm: 0.8377, dt: 449.53ms, tok/sec: 1166296.69
step:  754 | loss: 4.820474 | lr: 5.9999e-04 | norm: 0.6827, dt: 449.37ms, tok/sec: 1166705.09
step:  755 | loss: 4.721570 | lr: 5.9999e-04 | norm: 0.6415, dt: 1121.43ms, tok/sec: 467516.41
step:  756 | loss: 4.766013 | lr: 5.9999e-04 | norm: 0.5993, dt: 449.59ms, tok/sec: 1166151.35
step:  757 | loss: 4.791905 | lr: 5.9999e-04 | norm: 0.6086, dt: 449.34ms, tok/sec: 1166794.85
step:  758 | loss: 4.725642 | lr: 5.9999e-04 | norm: 0.6019, dt: 449.14ms, tok/sec: 1167312.65
step:  759 | loss: 4.805218 | lr: 5.9999e-04 | norm: 0.6636, dt: 1201.78ms, tok/sec: 436258.83
step:  760 | loss: 4.755946 | lr: 5.9999e-04 | norm: 0.7700, dt: 449.87ms, tok/sec: 1165433.81
step:  761 | loss: 4.784189 | lr: 5.9999e-04 | norm: 0.9345, dt: 449.85ms, tok/sec: 1165480.75
step:  762 | loss: 4.746758 | lr: 5.9999e-04 | norm: 0.9564, dt: 449.95ms, tok/sec: 1165204.70
step:  763 | loss: 4.785253 | lr: 5.9999e-04 | norm: 0.9431, dt: 449.79ms, tok/sec: 1165630.26
step:  764 | loss: 4.764233 | lr: 5.9999e-04 | norm: 0.8630, dt: 450.59ms, tok/sec: 1163561.62
step:  765 | loss: 4.718877 | lr: 5.9999e-04 | norm: 0.7471, dt: 451.06ms, tok/sec: 1162343.87
step:  766 | loss: 4.691868 | lr: 5.9999e-04 | norm: 0.8045, dt: 450.62ms, tok/sec: 1163481.59
step:  767 | loss: 4.732925 | lr: 5.9999e-04 | norm: 0.8203, dt: 450.53ms, tok/sec: 1163703.24
step:  768 | loss: 4.685675 | lr: 5.9999e-04 | norm: 0.8180, dt: 450.69ms, tok/sec: 1163313.56
step:  769 | loss: 4.687204 | lr: 5.9999e-04 | norm: 0.7465, dt: 449.55ms, tok/sec: 1166249.68
step:  770 | loss: 4.713746 | lr: 5.9999e-04 | norm: 0.6616, dt: 451.95ms, tok/sec: 1160063.46
step:  771 | loss: 4.705406 | lr: 5.9999e-04 | norm: 0.6999, dt: 450.73ms, tok/sec: 1163196.03
step:  772 | loss: 4.677678 | lr: 5.9999e-04 | norm: 0.5810, dt: 450.21ms, tok/sec: 1164551.85
step:  773 | loss: 4.672338 | lr: 5.9999e-04 | norm: 0.5937, dt: 450.24ms, tok/sec: 1164457.50
step:  774 | loss: 4.610915 | lr: 5.9999e-04 | norm: 0.6106, dt: 451.88ms, tok/sec: 1160243.41
step:  775 | loss: 4.575705 | lr: 5.9999e-04 | norm: 0.6325, dt: 450.41ms, tok/sec: 1164031.57
step:  776 | loss: 4.587792 | lr: 5.9999e-04 | norm: 0.7374, dt: 450.04ms, tok/sec: 1164972.60
step:  777 | loss: 4.583910 | lr: 5.9998e-04 | norm: 0.8703, dt: 450.89ms, tok/sec: 1162780.86
step:  778 | loss: 4.691227 | lr: 5.9998e-04 | norm: 0.8826, dt: 450.84ms, tok/sec: 1162919.83
step:  779 | loss: 4.554300 | lr: 5.9998e-04 | norm: 0.8058, dt: 453.20ms, tok/sec: 1156867.99
step:  780 | loss: 4.554441 | lr: 5.9998e-04 | norm: 0.7192, dt: 450.79ms, tok/sec: 1163049.61
step:  781 | loss: 4.566776 | lr: 5.9998e-04 | norm: 0.7807, dt: 450.45ms, tok/sec: 1163929.29
step:  782 | loss: 4.574511 | lr: 5.9998e-04 | norm: 0.7381, dt: 451.25ms, tok/sec: 1161847.05
step:  783 | loss: 4.606620 | lr: 5.9998e-04 | norm: 0.7471, dt: 450.95ms, tok/sec: 1162640.69
step:  784 | loss: 4.602698 | lr: 5.9998e-04 | norm: 0.8042, dt: 450.43ms, tok/sec: 1163976.73
step:  785 | loss: 4.616574 | lr: 5.9998e-04 | norm: 0.8342, dt: 450.29ms, tok/sec: 1164338.50
step:  786 | loss: 4.591324 | lr: 5.9998e-04 | norm: 0.6930, dt: 451.06ms, tok/sec: 1162350.63
step:  787 | loss: 4.679798 | lr: 5.9998e-04 | norm: 0.6511, dt: 450.64ms, tok/sec: 1163433.58
step:  788 | loss: 4.740541 | lr: 5.9998e-04 | norm: 0.7431, dt: 450.65ms, tok/sec: 1163398.49
step:  789 | loss: 4.696736 | lr: 5.9998e-04 | norm: 0.7556, dt: 450.72ms, tok/sec: 1163236.02
step:  790 | loss: 4.702076 | lr: 5.9998e-04 | norm: 0.7606, dt: 450.82ms, tok/sec: 1162973.95
step:  791 | loss: 4.706477 | lr: 5.9998e-04 | norm: 0.8413, dt: 450.34ms, tok/sec: 1164199.81
step:  792 | loss: 4.673226 | lr: 5.9998e-04 | norm: 0.9736, dt: 449.67ms, tok/sec: 1165929.38

step: 4421 | loss: 3.359128 | lr: 5.4750e-04 | norm: 0.3041, dt: 451.03ms, tok/sec: 1162417.60
step: 4422 | loss: 3.384596 | lr: 5.4747e-04 | norm: 0.2924, dt: 451.71ms, tok/sec: 1160665.34
step: 4423 | loss: 3.392948 | lr: 5.4744e-04 | norm: 0.2704, dt: 451.01ms, tok/sec: 1162467.99
step: 4424 | loss: 3.445918 | lr: 5.4741e-04 | norm: 0.2784, dt: 450.96ms, tok/sec: 1162606.27
step: 4425 | loss: 3.430316 | lr: 5.4739e-04 | norm: 0.2504, dt: 451.59ms, tok/sec: 1160983.99
step: 4426 | loss: 3.382772 | lr: 5.4736e-04 | norm: 0.2565, dt: 450.91ms, tok/sec: 1162721.84
step: 4427 | loss: 3.376990 | lr: 5.4733e-04 | norm: 0.2603, dt: 451.27ms, tok/sec: 1161803.46
step: 4428 | loss: 3.418103 | lr: 5.4730e-04 | norm: 0.2912, dt: 451.89ms, tok/sec: 1160205.46
step: 4429 | loss: 3.400937 | lr: 5.4728e-04 | norm: 0.2625, dt: 451.26ms, tok/sec: 1161830.47
step: 4430 | loss: 3.405616 | lr: 5.4725e-04 | norm: 0.2450, dt: 450.83ms, tok/sec: 1162949.97
step: 4431 | loss: 3.366644 | lr: 5.4722e-04 | norm: 0.2622, dt: 451.93ms, tok/sec: 1160100.18
step: 4432 | loss: 3.317298 | lr: 5.4720e-04 | norm: 0.2818, dt: 451.38ms, tok/sec: 1161521.79
step: 4433 | loss: 3.343770 | lr: 5.4717e-04 | norm: 0.2755, dt: 450.82ms, tok/sec: 1162957.96
step: 4434 | loss: 3.409106 | lr: 5.4714e-04 | norm: 0.2934, dt: 450.37ms, tok/sec: 1164117.84
step: 4435 | loss: 3.361298 | lr: 5.4711e-04 | norm: 0.2734, dt: 452.01ms, tok/sec: 1159912.32
step: 4436 | loss: 3.294379 | lr: 5.4709e-04 | norm: 0.2980, dt: 451.10ms, tok/sec: 1162238.82
step: 4437 | loss: 3.385554 | lr: 5.4706e-04 | norm: 0.2727, dt: 450.73ms, tok/sec: 1163205.87
step: 4438 | loss: 3.339437 | lr: 5.4703e-04 | norm: 0.2549, dt: 451.11ms, tok/sec: 1162224.69
step: 4439 | loss: 3.341364 | lr: 5.4700e-04 | norm: 0.2750, dt: 451.79ms, tok/sec: 1160461.38
step: 4440 | loss: 3.379750 | lr: 5.4698e-04 | norm: 0.2667, dt: 450.41ms, tok/sec: 1164026.64
step: 4441 | loss: 3.353417 | lr: 5.4695e-04 | norm: 0.2728, dt: 449.01ms, tok/sec: 1167651.07
step: 4442 | loss: 3.343123 | lr: 5.4692e-04 | norm: 0.2605, dt: 450.74ms, tok/sec: 1163167.73
step: 4443 | loss: 3.321782 | lr: 5.4689e-04 | norm: 0.2526, dt: 451.01ms, tok/sec: 1162464.30
step: 4444 | loss: 3.354173 | lr: 5.4687e-04 | norm: 0.2662, dt: 452.69ms, tok/sec: 1158149.92
step: 4445 | loss: 3.361459 | lr: 5.4684e-04 | norm: 0.2694, dt: 451.62ms, tok/sec: 1160896.34
step: 4446 | loss: 3.410709 | lr: 5.4681e-04 | norm: 0.2621, dt: 451.12ms, tok/sec: 1162200.74
step: 4447 | loss: 3.336343 | lr: 5.4678e-04 | norm: 0.2645, dt: 451.06ms, tok/sec: 1162348.78
step: 4448 | loss: 3.403448 | lr: 5.4676e-04 | norm: 0.3022, dt: 451.23ms, tok/sec: 1161901.68
step: 4449 | loss: 3.334867 | lr: 5.4673e-04 | norm: 0.3055, dt: 450.74ms, tok/sec: 1163167.11
step: 4450 | loss: 3.407516 | lr: 5.4670e-04 | norm: 0.2660, dt: 452.19ms, tok/sec: 1159437.75
step: 4451 | loss: 3.395393 | lr: 5.4667e-04 | norm: 0.2758, dt: 451.09ms, tok/sec: 1162279.98
step: 4452 | loss: 3.357027 | lr: 5.4664e-04 | norm: 0.2872, dt: 450.65ms, tok/sec: 1163392.95
step: 4453 | loss: 3.460746 | lr: 5.4662e-04 | norm: 0.2988, dt: 452.20ms, tok/sec: 1159424.91
step: 4454 | loss: 3.342000 | lr: 5.4659e-04 | norm: 0.2529, dt: 451.14ms, tok/sec: 1162150.37
step: 4455 | loss: 3.239975 | lr: 5.4656e-04 | norm: 0.2489, dt: 450.89ms, tok/sec: 1162781.47
step: 4456 | loss: 3.206433 | lr: 5.4653e-04 | norm: 0.2745, dt: 451.00ms, tok/sec: 1162490.11
step: 4457 | loss: 3.209413 | lr: 5.4651e-04 | norm: 0.3022, dt: 451.89ms, tok/sec: 1160220.76
step: 4458 | loss: 3.237169 | lr: 5.4648e-04 | norm: 0.3205, dt: 450.26ms, tok/sec: 1164413.72
step: 4459 | loss: 3.291021 | lr: 5.4645e-04 | norm: 0.3064, dt: 451.42ms, tok/sec: 1161423.64
step: 4460 | loss: 3.260573 | lr: 5.4642e-04 | norm: 0.2810, dt: 450.92ms, tok/sec: 1162697.86
step: 4461 | loss: 3.246778 | lr: 5.4640e-04 | norm: 0.2626, dt: 451.54ms, tok/sec: 1161102.30
step: 4462 | loss: 3.431473 | lr: 5.4637e-04 | norm: 0.3582, dt: 450.83ms, tok/sec: 1162931.52
step: 4463 | loss: 3.233631 | lr: 5.4634e-04 | norm: 0.3204, dt: 450.69ms, tok/sec: 1163304.33
step: 4464 | loss: 3.282382 | lr: 5.4631e-04 | norm: 0.3423, dt: 450.87ms, tok/sec: 1162839.89
step: 4465 | loss: 3.264042 | lr: 5.4629e-04 | norm: 0.2854, dt: 450.90ms, tok/sec: 1162747.66
step: 4466 | loss: 3.401390 | lr: 5.4626e-04 | norm: 0.2921, dt: 452.37ms, tok/sec: 1158990.44
step: 4467 | loss: 3.423584 | lr: 5.4623e-04 | norm: 0.3360, dt: 451.73ms, tok/sec: 1160610.21
step: 4468 | loss: 3.401810 | lr: 5.4620e-04 | norm: 0.2929, dt: 451.09ms, tok/sec: 1162262.16
step: 4469 | loss: 3.403140 | lr: 5.4618e-04 | norm: 0.2821, dt: 451.15ms, tok/sec: 1162110.45
step: 4470 | loss: 3.426635 | lr: 5.4615e-04 | norm: 0.3051, dt: 451.54ms, tok/sec: 1161107.82
step: 4471 | loss: 3.469340 | lr: 5.4612e-04 | norm: 0.4506, dt: 450.73ms, tok/sec: 1163200.95
step: 4472 | loss: 3.464427 | lr: 5.4609e-04 | norm: 0.3641, dt: 453.18ms, tok/sec: 1156912.42
step: 4473 | loss: 3.355725 | lr: 5.4606e-04 | norm: 0.3218, dt: 451.45ms, tok/sec: 1161334.70
step: 4474 | loss: 3.469188 | lr: 5.4604e-04 | norm: 0.3117, dt: 450.98ms, tok/sec: 1162558.94
step: 4475 | loss: 3.423909 | lr: 5.4601e-04 | norm: 0.2996, dt: 451.53ms, tok/sec: 1161142.76
step: 4476 | loss: 3.425230 | lr: 5.4598e-04 | norm: 0.3051, dt: 451.13ms, tok/sec: 1162171.87
step: 4477 | loss: 3.381702 | lr: 5.4595e-04 | norm: 0.2789, dt: 451.03ms, tok/sec: 1162419.44
step: 4478 | loss: 3.374624 | lr: 5.4593e-04 | norm: 0.2720, dt: 451.16ms, tok/sec: 1162084.66
step: 4479 | loss: 3.307718 | lr: 5.4590e-04 | norm: 0.2594, dt: 452.63ms, tok/sec: 1158302.43
step: 4480 | loss: 3.364757 | lr: 5.4587e-04 | norm: 0.2776, dt: 451.15ms, tok/sec: 1162119.05
step: 4481 | loss: 3.351280 | lr: 5.4584e-04 | norm: 0.2528, dt: 451.70ms, tok/sec: 1160700.88
step: 4482 | loss: 3.284569 | lr: 5.4581e-04 | norm: 0.2595, dt: 450.10ms, tok/sec: 1164822.03
step: 4483 | loss: 3.327097 | lr: 5.4579e-04 | norm: 0.2545, dt: 451.53ms, tok/sec: 1161130.50
step: 4484 | loss: 3.380925 | lr: 5.4576e-04 | norm: 0.2330, dt: 451.10ms, tok/sec: 1162247.42
step: 4485 | loss: 3.312778 | lr: 5.4573e-04 | norm: 0.2507, dt: 451.34ms, tok/sec: 1161632.85
step: 4486 | loss: 3.359381 | lr: 5.4570e-04 | norm: 0.2721, dt: 451.83ms, tok/sec: 1160367.69
step: 4487 | loss: 3.388181 | lr: 5.4568e-04 | norm: 0.2410, dt: 451.26ms, tok/sec: 1161837.22
step: 4488 | loss: 3.437197 | lr: 5.4565e-04 | norm: 0.2746, dt: 450.87ms, tok/sec: 1162825.13
step: 4489 | loss: 3.341065 | lr: 5.4562e-04 | norm: 0.2732, dt: 451.35ms, tok/sec: 1161597.87
step: 4490 | loss: 3.384256 | lr: 5.4559e-04 | norm: 0.2911, dt: 450.46ms, tok/sec: 1163894.18
step: 4491 | loss: 3.361495 | lr: 5.4556e-04 | norm: 0.2373, dt: 451.27ms, tok/sec: 1161813.90
step: 4492 | loss: 3.349765 | lr: 5.4554e-04 | norm: 0.2672, dt: 452.55ms, tok/sec: 1158519.06
step: 4493 | loss: 3.368674 | lr: 5.4551e-04 | norm: 0.2760, dt: 451.10ms, tok/sec: 1162247.42
step: 4494 | loss: 3.345582 | lr: 5.4548e-04 | norm: 0.2548, dt: 449.95ms, tok/sec: 1165213.97
step: 4495 | loss: 3.405331 | lr: 5.4545e-04 | norm: 0.2521, dt: 451.62ms, tok/sec: 1160910.44
step: 4496 | loss: 3.365353 | lr: 5.4543e-04 | norm: 0.2625, dt: 452.08ms, tok/sec: 1159727.58
step: 4497 | loss: 3.421246 | lr: 5.4540e-04 | norm: 0.3289, dt: 450.84ms, tok/sec: 1162916.14
step: 4498 | loss: 3.386518 | lr: 5.4537e-04 | norm: 0.3975, dt: 451.77ms, tok/sec: 1160514.05
step: 4499 | loss: 3.333818 | lr: 5.4534e-04 | norm: 0.3263, dt: 451.52ms, tok/sec: 1161171.58
validation loss: 3.4094
HellaSwag accuracy: 2733/10042=0.2722
rank 0 sample 0: Hello, I'm a Manpreet, and I don't want to use 'gett' because it'll help to get it a good look by using
rank 0 sample 1: Hello, I'm a Manpreet, this word is a noun. The one that I're seeing here is actually a man, a woman, a man
rank 4 sample 0: Hello, I'm a Manpreet, I'm Manpreet, I'm you. You're in the process of thinking how about making a friend?
rank 0 sample 2: Hello, I'm a Manpreet, I'm at work because I'm on the train, I don't know what you're doing. I'm not
rank 4 sample 1: Hello, I'm a Manpreet,” he will do that when I have to add any other name: a-na, a-na,
rank 0 sample 3: Hello, I'm a Manpreet, and they're in the middle. This makes both of them visible to the eye
It's not hard to find
rank 4 sample 2: Hello, I'm a Manpreet, but have you mentioned a bit of history and then I'm not worried about that when I did? Would you consider
rank 2 sample 0: Hello, I'm a Manpreet, where my mother works as a teacher to the students. We're going to go and collect the books and take themrank 4 sample 3: Hello, I'm a Manpreet, and the difference has to be in first case it seems like so. Why? Because I am really just a man

rank 2 sample 1: Hello, I'm a Manpreet, that's his job. I'm interested (and also am) a Teacher, and I'm interested to read that
rank 2 sample 2: Hello, I'm a Manpreet, I'm just an assistant, you get the name of the teacher in charge of this lesson, if you don't
rank 2 sample 3: Hello, I'm a Manpreet, so let me know if I'm not your boss (and, if you wanna talk about more I'll talk to
rank 6 sample 0: Hello, I'm a Manpreet, or maybe you like this thing, right.
In my opinion, the more obvious thing about the name "Man
rank 6 sample 1: Hello, I'm a Manpreet,
a little bit older.
I really like how the characters are
used to be. I think I'm
rank 6 sample 2: Hello, I'm a Manpreet, so my mom can read me, "My kid is a good guy!" It's about the same and a better
rank 6 sample 3: Hello, I'm a Manpreet, and my friends,
I don't want to be here. I'm here, I are here, the boy
rank 5 sample 0: Hello, I'm a Manpreet, so the next time you're taking a closer look at this part of my English language, you know that it's
rank 5 sample 1: Hello, I'm a Manpreet, here you'll learn different styles of English grammar on the web.
Here, a little bit of fun on grammar
rank 5 sample 2: Hello, I'm a Manpreet, is the language that creates the right words. It's a language I've always been learning, but I'm not
rank 5 sample 3: Hello, I'm a Manpreet, "Do you think that your child is going to go on a social media screen today?", "Be a ManP
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going to do some interesting things about this subject which is at night in Australia.
A few months
rank 3 sample 1: Hello, I'm a Manpreet, and that means I'm out. I'm looking forward to the day when I'm going to meet the local Aboriginal
rank 3 sample 2: Hello, I'm a Manpreet, and you've got a wonderful feeling. A girl has to learn to speak and use a variety of language, a
rank 7 sample 0: Hello, I'm a Manpreet, and I'm a Manping.
From the time I'm here when I visit the "how" page,
rank 3 sample 3: Hello, I'm a Manpreet, and as I said, the difference between a man and woman in terms of gender is to take the subject and identify
rank 7 sample 1: Hello, I'm a Manpreet, how big does it all have to be?
2) What is the difference between the man and the cat?
rank 7 sample 2: Hello, I'm a Manpreet, thanks for your explanation of the manping job. (And, that's what I'm doing here.)
If
rank 7 sample 3: Hello, I'm a Manpreet, now you see that it's not for "Taming" the name of the company.
How to use the
rank 1 sample 0: Hello, I'm a Manpreet, as a man. As a man a Man's house
or a house near the school or university.
I
rank 1 sample 1: Hello, I'm a Manpreet, a Manpreet, a Manpreet, a Kreet, a Kreet - something you've got to
rank 1 sample 2: Hello, I'm a Manpreet, so they'll go to the next table.
First I can see how the table is filled by an object called
rank 1 sample 3: Hello, I'm a Manpreet, and I'm in a Bitty (of course, is my head really getting a huffy appearance).
If
step: 4500 | loss: 3.400841 | lr: 5.4531e-04 | norm: 0.2606, dt: 12094.49ms, tok/sec: 43349.31
step: 4501 | loss: 3.232046 | lr: 5.4529e-04 | norm: 0.2920, dt: 451.48ms, tok/sec: 1161268.47
step: 4502 | loss: 3.273896 | lr: 5.4526e-04 | norm: 0.2747, dt: 451.08ms, tok/sec: 1162306.39
step: 4503 | loss: 3.203296 | lr: 5.4523e-04 | norm: 0.2476, dt: 449.64ms, tok/sec: 1166012.84
step: 4504 | loss: 3.189549 | lr: 5.4520e-04 | norm: 0.2714, dt: 450.28ms, tok/sec: 1164353.30
step: 4505 | loss: 3.288268 | lr: 5.4517e-04 | norm: 0.2629, dt: 450.01ms, tok/sec: 1165059.63
step: 4506 | loss: 3.290722 | lr: 5.4515e-04 | norm: 0.3047, dt: 452.04ms, tok/sec: 1159820.56
step: 4507 | loss: 3.258672 | lr: 5.4512e-04 | norm: 0.3203, dt: 450.69ms, tok/sec: 1163301.25
step: 4508 | loss: 3.283618 | lr: 5.4509e-04 | norm: 0.3389, dt: 450.34ms, tok/sec: 1164194.26
step: 4509 | loss: 3.262981 | lr: 5.4506e-04 | norm: 0.3054, dt: 450.47ms, tok/sec: 1163867.08
step: 4510 | loss: 3.393291 | lr: 5.4503e-04 | norm: 0.3057, dt: 450.91ms, tok/sec: 1162725.53
step: 4511 | loss: 3.326857 | lr: 5.4501e-04 | norm: 0.2564, dt: 450.49ms, tok/sec: 1163812.87
step: 4512 | loss: 3.271555 | lr: 5.4498e-04 | norm: 0.2658, dt: 450.56ms, tok/sec: 1163639.20
step: 4513 | loss: 3.325122 | lr: 5.4495e-04 | norm: 0.3010, dt: 450.08ms, tok/sec: 1164875.72
step: 4514 | loss: 3.357470 | lr: 5.4492e-04 | norm: 0.2593, dt: 450.94ms, tok/sec: 1162643.15
step: 4515 | loss: 3.418589 | lr: 5.4489e-04 | norm: 0.2773, dt: 452.72ms, tok/sec: 1158071.85
step: 4516 | loss: 3.442409 | lr: 5.4487e-04 | norm: 0.2460, dt: 451.29ms, tok/sec: 1161744.54
step: 4517 | loss: 3.413930 | lr: 5.4484e-04 | norm: 0.2592, dt: 451.13ms, tok/sec: 1162160.81
step: 4518 | loss: 3.362976 | lr: 5.4481e-04 | norm: 0.2847, dt: 451.07ms, tok/sec: 1162324.82
step: 4519 | loss: 3.408874 | lr: 5.4478e-04 | norm: 0.2319, dt: 452.05ms, tok/sec: 1159807.71
step: 4520 | loss: 3.384815 | lr: 5.4476e-04 | norm: 0.2689, dt: 450.37ms, tok/sec: 1164135.10
step: 4521 | loss: 3.405663 | lr: 5.4473e-04 | norm: 0.2500, dt: 451.21ms, tok/sec: 1161965.53
step: 4522 | loss: 3.398885 | lr: 5.4470e-04 | norm: 0.2590, dt: 451.14ms, tok/sec: 1162130.10
step: 4523 | loss: 3.377404 | lr: 5.4467e-04 | norm: 0.2510, dt: 451.42ms, tok/sec: 1161421.18
step: 4524 | loss: 3.378654 | lr: 5.4464e-04 | norm: 0.2735, dt: 451.53ms, tok/sec: 1161133.57
step: 4525 | loss: 3.359033 | lr: 5.4461e-04 | norm: 0.2655, dt: 451.81ms, tok/sec: 1160425.25
step: 4526 | loss: 3.372179 | lr: 5.4459e-04 | norm: 0.2910, dt: 450.48ms, tok/sec: 1163836.28
step: 4527 | loss: 3.322074 | lr: 5.4456e-04 | norm: 0.2591, dt: 452.27ms, tok/sec: 1159243.38
step: 4528 | loss: 3.346156 | lr: 5.4453e-04 | norm: 0.2494, dt: 451.89ms, tok/sec: 1160223.21
step: 4529 | loss: 3.317671 | lr: 5.4450e-04 | norm: 0.2717, dt: 450.95ms, tok/sec: 1162621.64
step: 4530 | loss: 3.376853 | lr: 5.4447e-04 | norm: 0.2562, dt: 450.76ms, tok/sec: 1163127.74
step: 4531 | loss: 3.368129 | lr: 5.4445e-04 | norm: 0.2695, dt: 451.50ms, tok/sec: 1161205.30
step: 4532 | loss: 3.293994 | lr: 5.4442e-04 | norm: 0.2695, dt: 456.02ms, tok/sec: 1149711.48
step: 4533 | loss: 3.348350 | lr: 5.4439e-04 | norm: 0.2652, dt: 450.83ms, tok/sec: 1162934.59
step: 4534 | loss: 3.324881 | lr: 5.4436e-04 | norm: 0.2626, dt: 451.62ms, tok/sec: 1160903.09
step: 4535 | loss: 3.330404 | lr: 5.4433e-04 | norm: 0.2545, dt: 1150.97ms, tok/sec: 455516.98
step: 4536 | loss: 3.410071 | lr: 5.4431e-04 | norm: 0.2931, dt: 450.48ms, tok/sec: 1163846.13
step: 4537 | loss: 3.413493 | lr: 5.4428e-04 | norm: 0.2539, dt: 449.52ms, tok/sec: 1166330.10
step: 4538 | loss: 3.356327 | lr: 5.4425e-04 | norm: 0.2652, dt: 450.51ms, tok/sec: 1163759.29
step: 4539 | loss: 3.330953 | lr: 5.4422e-04 | norm: 0.2661, dt: 453.01ms, tok/sec: 1157335.59
step: 4540 | loss: 3.394309 | lr: 5.4419e-04 | norm: 0.2494, dt: 450.12ms, tok/sec: 1164766.51
step: 4541 | loss: 3.360179 | lr: 5.4417e-04 | norm: 0.2618, dt: 451.12ms, tok/sec: 1162194.59
step: 4542 | loss: 3.371463 | lr: 5.4414e-04 | norm: 0.2536, dt: 450.36ms, tok/sec: 1164141.26
step: 4543 | loss: 3.359180 | lr: 5.4411e-04 | norm: 0.2590, dt: 450.88ms, tok/sec: 1162805.45
step: 4544 | loss: 3.336830 | lr: 5.4408e-04 | norm: 0.2719, dt: 450.26ms, tok/sec: 1164399.54
step: 4545 | loss: 3.424152 | lr: 5.4405e-04 | norm: 0.2436, dt: 450.32ms, tok/sec: 1164248.50
step: 4546 | loss: 3.303814 | lr: 5.4402e-04 | norm: 0.2417, dt: 451.18ms, tok/sec: 1162025.09
step: 4547 | loss: 3.380021 | lr: 5.4400e-04 | norm: 0.2634, dt: 451.66ms, tok/sec: 1160794.01
step: 4548 | loss: 3.300802 | lr: 5.4397e-04 | norm: 0.2628, dt: 454.06ms, tok/sec: 1154664.77
step: 4549 | loss: 3.238016 | lr: 5.4394e-04 | norm: 0.2628, dt: 451.66ms, tok/sec: 1160792.17
step: 4550 | loss: 3.283939 | lr: 5.4391e-04 | norm: 0.2520, dt: 451.45ms, tok/sec: 1161348.81
step: 4551 | loss: 3.276675 | lr: 5.4388e-04 | norm: 0.2611, dt: 450.31ms, tok/sec: 1164292.88
step: 4552 | loss: 3.340871 | lr: 5.4386e-04 | norm: 0.2518, dt: 451.45ms, tok/sec: 1161332.25
step: 4553 | loss: 3.175738 | lr: 5.4383e-04 | norm: 0.2613, dt: 451.35ms, tok/sec: 1161597.87
step: 4554 | loss: 3.238377 | lr: 5.4380e-04 | norm: 0.3377, dt: 450.47ms, tok/sec: 1163879.40
step: 4555 | loss: 3.172327 | lr: 5.4377e-04 | norm: 0.3376, dt: 451.79ms, tok/sec: 1160468.73
step: 4556 | loss: 3.306664 | lr: 5.4374e-04 | norm: 0.2689, dt: 453.25ms, tok/sec: 1156733.50
step: 4557 | loss: 3.253670 | lr: 5.4371e-04 | norm: 0.2710, dt: 451.73ms, tok/sec: 1160625.53
step: 4558 | loss: 3.175811 | lr: 5.4369e-04 | norm: 0.3057, dt: 451.51ms, tok/sec: 1161187.52
step: 4559 | loss: 3.208519 | lr: 5.4366e-04 | norm: 0.2993, dt: 1216.70ms, tok/sec: 430908.54
step: 4560 | loss: 3.404714 | lr: 5.4363e-04 | norm: 0.2977, dt: 450.73ms, tok/sec: 1163197.87
step: 4561 | loss: 3.371064 | lr: 5.4360e-04 | norm: 0.3050, dt: 450.33ms, tok/sec: 1164229.39
step: 4562 | loss: 3.353976 | lr: 5.4357e-04 | norm: 0.3028, dt: 450.07ms, tok/sec: 1164890.53
step: 4563 | loss: 3.362903 | lr: 5.4355e-04 | norm: 0.2883, dt: 450.95ms, tok/sec: 1162617.95
step: 4564 | loss: 3.449541 | lr: 5.4352e-04 | norm: 0.3051, dt: 451.62ms, tok/sec: 1160902.47
step: 4565 | loss: 3.366327 | lr: 5.4349e-04 | norm: 0.2737, dt: 450.65ms, tok/sec: 1163398.49
step: 4566 | loss: 3.394696 | lr: 5.4346e-04 | norm: 0.3015, dt: 450.98ms, tok/sec: 1162547.27
step: 4567 | loss: 3.400715 | lr: 5.4343e-04 | norm: 0.3035, dt: 451.32ms, tok/sec: 1161673.96
step: 4568 | loss: 3.368906 | lr: 5.4340e-04 | norm: 0.3418, dt: 451.67ms, tok/sec: 1160770.11
step: 4569 | loss: 3.425979 | lr: 5.4338e-04 | norm: 0.3124, dt: 450.25ms, tok/sec: 1164447.02
step: 4570 | loss: 3.309004 | lr: 5.4335e-04 | norm: 0.2692, dt: 450.73ms, tok/sec: 1163200.95
step: 4571 | loss: 3.359340 | lr: 5.4332e-04 | norm: 0.2961, dt: 450.74ms, tok/sec: 1163168.34
step: 4572 | loss: 3.356426 | lr: 5.4329e-04 | norm: 0.2839, dt: 450.31ms, tok/sec: 1164289.80
step: 4573 | loss: 3.372448 | lr: 5.4326e-04 | norm: 0.2620, dt: 450.97ms, tok/sec: 1162590.29
step: 4574 | loss: 3.300297 | lr: 5.4323e-04 | norm: 0.2818, dt: 451.54ms, tok/sec: 1161102.30
step: 4575 | loss: 3.297628 | lr: 5.4321e-04 | norm: 0.2564, dt: 451.47ms, tok/sec: 1161296.68
step: 4576 | loss: 3.323240 | lr: 5.4318e-04 | norm: 0.2731, dt: 452.01ms, tok/sec: 1159903.76
step: 4577 | loss: 3.429079 | lr: 5.4315e-04 | norm: 0.2603, dt: 450.92ms, tok/sec: 1162716.92
step: 4578 | loss: 3.368082 | lr: 5.4312e-04 | norm: 0.2715, dt: 450.59ms, tok/sec: 1163568.40
step: 4579 | loss: 3.302544 | lr: 5.4309e-04 | norm: 0.2894, dt: 452.17ms, tok/sec: 1159504.38
step: 4580 | loss: 3.384990 | lr: 5.4306e-04 | norm: 0.2749, dt: 450.81ms, tok/sec: 1162990.56
step: 4581 | loss: 3.340959 | lr: 5.4304e-04 | norm: 0.2962, dt: 450.74ms, tok/sec: 1163180.65
step: 4582 | loss: 3.409802 | lr: 5.4301e-04 | norm: 0.2944, dt: 451.48ms, tok/sec: 1161268.47
step: 4583 | loss: 3.391138 | lr: 5.4298e-04 | norm: 0.2643, dt: 451.44ms, tok/sec: 1161367.21
step: 4584 | loss: 3.317817 | lr: 5.4295e-04 | norm: 0.2439, dt: 450.36ms, tok/sec: 1164149.89
step: 4585 | loss: 3.419344 | lr: 5.4292e-04 | norm: 0.3395, dt: 451.17ms, tok/sec: 1162063.78
step: 4586 | loss: 3.379602 | lr: 5.4289e-04 | norm: 0.3026, dt: 451.22ms, tok/sec: 1161934.83
step: 4587 | loss: 3.395834 | lr: 5.4286e-04 | norm: 0.3017, dt: 450.51ms, tok/sec: 1163778.38
step: 4588 | loss: 3.377592 | lr: 5.4284e-04 | norm: 0.3118, dt: 451.14ms, tok/sec: 1162148.53
step: 4589 | loss: 3.385577 | lr: 5.4281e-04 | norm: 0.3057, dt: 451.34ms, tok/sec: 1161622.42
step: 4590 | loss: 3.297474 | lr: 5.4278e-04 | norm: 0.2706, dt: 450.64ms, tok/sec: 1163420.04
step: 4591 | loss: 3.390070 | lr: 5.4275e-04 | norm: 0.3234, dt: 450.73ms, tok/sec: 1163197.26
step: 4592 | loss: 3.375704 | lr: 5.4272e-04 | norm: 0.2897, dt: 450.08ms, tok/sec: 1164883.74
step: 4593 | loss: 3.337165 | lr: 5.4269e-04 | norm: 0.3400, dt: 450.59ms, tok/sec: 1163552.39
step: 4594 | loss: 3.475777 | lr: 5.4267e-04 | norm: 0.3134, dt: 450.44ms, tok/sec: 1163958.25
step: 4595 | loss: 3.260075 | lr: 5.4264e-04 | norm: 0.3088, dt: 451.11ms, tok/sec: 1162216.09
step: 4596 | loss: 3.280767 | lr: 5.4261e-04 | norm: 0.2892, dt: 451.00ms, tok/sec: 1162490.73
step: 4597 | loss: 3.219220 | lr: 5.4258e-04 | norm: 0.3273, dt: 450.94ms, tok/sec: 1162666.51
step: 4598 | loss: 3.278830 | lr: 5.4255e-04 | norm: 0.3022, dt: 450.32ms, tok/sec: 1164247.89
step: 4599 | loss: 3.270060 | lr: 5.4252e-04 | norm: 0.2469, dt: 451.77ms, tok/sec: 1160522.01
step: 4600 | loss: 3.235199 | lr: 5.4249e-04 | norm: 0.2448, dt: 452.93ms, tok/sec: 1157536.01
step: 4601 | loss: 3.263751 | lr: 5.4247e-04 | norm: 0.2545, dt: 450.79ms, tok/sec: 1163048.38
step: 4602 | loss: 3.251817 | lr: 5.4244e-04 | norm: 0.2503, dt: 450.83ms, tok/sec: 1162933.36
step: 4603 | loss: 3.242966 | lr: 5.4241e-04 | norm: 0.2548, dt: 450.96ms, tok/sec: 1162595.82
step: 4604 | loss: 3.248774 | lr: 5.4238e-04 | norm: 0.2756, dt: 450.77ms, tok/sec: 1163104.97
step: 4605 | loss: 3.221117 | lr: 5.4235e-04 | norm: 0.2689, dt: 451.30ms, tok/sec: 1161725.51
step: 4606 | loss: 3.319160 | lr: 5.4232e-04 | norm: 0.2541, dt: 451.30ms, tok/sec: 1161723.06
step: 4607 | loss: 3.358713 | lr: 5.4229e-04 | norm: 0.2620, dt: 450.92ms, tok/sec: 1162710.16
step: 4608 | loss: 3.443849 | lr: 5.4227e-04 | norm: 0.2576, dt: 451.26ms, tok/sec: 1161834.15
step: 4609 | loss: 3.378918 | lr: 5.4224e-04 | norm: 0.2641, dt: 451.62ms, tok/sec: 1160893.89
step: 4610 | loss: 3.415211 | lr: 5.4221e-04 | norm: 0.2831, dt: 449.96ms, tok/sec: 1165179.39
step: 4611 | loss: 3.379066 | lr: 5.4218e-04 | norm: 0.2746, dt: 450.62ms, tok/sec: 1163482.82
step: 4612 | loss: 3.369564 | lr: 5.4215e-04 | norm: 0.2696, dt: 450.84ms, tok/sec: 1162904.46
step: 4613 | loss: 3.420230 | lr: 5.4212e-04 | norm: 0.2821, dt: 451.68ms, tok/sec: 1160738.25
step: 4614 | loss: 3.387102 | lr: 5.4209e-04 | norm: 0.2827, dt: 449.85ms, tok/sec: 1165475.19
step: 4615 | loss: 3.324452 | lr: 5.4207e-04 | norm: 0.2823, dt: 451.01ms, tok/sec: 1162475.98
step: 4616 | loss: 3.416223 | lr: 5.4204e-04 | norm: 0.2605, dt: 451.07ms, tok/sec: 1162324.82
step: 4617 | loss: 3.342790 | lr: 5.4201e-04 | norm: 0.2721, dt: 450.57ms, tok/sec: 1163607.80
step: 4618 | loss: 3.343413 | lr: 5.4198e-04 | norm: 0.2707, dt: 450.40ms, tok/sec: 1164056.83
step: 4619 | loss: 3.353612 | lr: 5.4195e-04 | norm: 0.2829, dt: 449.76ms, tok/sec: 1165708.11
step: 4620 | loss: 3.354636 | lr: 5.4192e-04 | norm: 0.2573, dt: 450.93ms, tok/sec: 1162678.80
step: 4621 | loss: 3.336735 | lr: 5.4189e-04 | norm: 0.2656, dt: 451.66ms, tok/sec: 1160790.94
step: 4622 | loss: 3.308258 | lr: 5.4187e-04 | norm: 0.2651, dt: 452.21ms, tok/sec: 1159387.01
step: 4623 | loss: 3.373296 | lr: 5.4184e-04 | norm: 0.2348, dt: 450.97ms, tok/sec: 1162575.54
step: 4624 | loss: 3.355006 | lr: 5.4181e-04 | norm: 0.2794, dt: 450.61ms, tok/sec: 1163516.06
step: 4625 | loss: 3.346431 | lr: 5.4178e-04 | norm: 0.2688, dt: 450.79ms, tok/sec: 1163050.84
step: 4626 | loss: 3.290238 | lr: 5.4175e-04 | norm: 0.2695, dt: 450.58ms, tok/sec: 1163591.18
step: 4627 | loss: 3.359290 | lr: 5.4172e-04 | norm: 0.2691, dt: 451.14ms, tok/sec: 1162130.10
step: 4628 | loss: 3.363311 | lr: 5.4169e-04 | norm: 0.2662, dt: 451.03ms, tok/sec: 1162428.66
step: 4629 | loss: 3.392439 | lr: 5.4167e-04 | norm: 0.2553, dt: 451.12ms, tok/sec: 1162195.82
step: 4630 | loss: 3.369504 | lr: 5.4164e-04 | norm: 0.2492, dt: 451.05ms, tok/sec: 1162380.73
step: 4631 | loss: 3.352671 | lr: 5.4161e-04 | norm: 0.2699, dt: 451.25ms, tok/sec: 1161864.23
step: 4632 | loss: 3.318968 | lr: 5.4158e-04 | norm: 0.2406, dt: 451.45ms, tok/sec: 1161346.97
step: 4633 | loss: 3.319849 | lr: 5.4155e-04 | norm: 0.2293, dt: 451.18ms, tok/sec: 1162027.55
step: 4634 | loss: 3.387199 | lr: 5.4152e-04 | norm: 0.2273, dt: 450.73ms, tok/sec: 1163204.03
step: 4635 | loss: 3.389752 | lr: 5.4149e-04 | norm: 0.2510, dt: 450.64ms, tok/sec: 1163427.42
step: 4636 | loss: 3.357938 | lr: 5.4146e-04 | norm: 0.2645, dt: 450.87ms, tok/sec: 1162826.98
step: 4637 | loss: 3.377839 | lr: 5.4144e-04 | norm: 0.2826, dt: 451.45ms, tok/sec: 1161341.45
step: 4638 | loss: 3.386773 | lr: 5.4141e-04 | norm: 0.2953, dt: 450.68ms, tok/sec: 1163324.02
step: 4639 | loss: 3.365202 | lr: 5.4138e-04 | norm: 0.2834, dt: 451.01ms, tok/sec: 1162464.30
step: 4640 | loss: 3.361318 | lr: 5.4135e-04 | norm: 0.2657, dt: 451.06ms, tok/sec: 1162359.23
step: 4641 | loss: 3.243459 | lr: 5.4132e-04 | norm: 0.2833, dt: 450.51ms, tok/sec: 1163766.68
step: 4642 | loss: 3.160370 | lr: 5.4129e-04 | norm: 0.3087, dt: 450.53ms, tok/sec: 1163714.95
step: 4643 | loss: 3.209622 | lr: 5.4126e-04 | norm: 0.3007, dt: 450.78ms, tok/sec: 1163060.68
step: 4644 | loss: 3.206955 | lr: 5.4123e-04 | norm: 0.2935, dt: 450.90ms, tok/sec: 1162771.02
step: 4645 | loss: 3.228441 | lr: 5.4121e-04 | norm: 0.2684, dt: 450.77ms, tok/sec: 1163095.13
step: 4646 | loss: 3.238655 | lr: 5.4118e-04 | norm: 0.2556, dt: 450.29ms, tok/sec: 1164329.87
step: 4647 | loss: 3.238352 | lr: 5.4115e-04 | norm: 0.2424, dt: 451.06ms, tok/sec: 1162356.16
step: 4648 | loss: 3.250336 | lr: 5.4112e-04 | norm: 0.2617, dt: 450.19ms, tok/sec: 1164588.85
step: 4649 | loss: 3.227957 | lr: 5.4109e-04 | norm: 0.2584, dt: 450.41ms, tok/sec: 1164014.32
step: 4650 | loss: 3.287172 | lr: 5.4106e-04 | norm: 0.2655, dt: 452.17ms, tok/sec: 1159483.60
step: 4651 | loss: 3.246869 | lr: 5.4103e-04 | norm: 0.2460, dt: 450.72ms, tok/sec: 1163216.33
step: 4652 | loss: 3.108431 | lr: 5.4100e-04 | norm: 0.2636, dt: 454.18ms, tok/sec: 1154367.76
step: 4653 | loss: 3.363163 | lr: 5.4098e-04 | norm: 0.2892, dt: 451.48ms, tok/sec: 1161273.37
step: 4654 | loss: 3.364444 | lr: 5.4095e-04 | norm: 0.2538, dt: 450.53ms, tok/sec: 1163706.94
step: 4655 | loss: 3.390184 | lr: 5.4092e-04 | norm: 0.2949, dt: 450.53ms, tok/sec: 1163717.41
step: 4656 | loss: 3.390004 | lr: 5.4089e-04 | norm: 0.2843, dt: 451.26ms, tok/sec: 1161842.75
step: 4657 | loss: 3.372612 | lr: 5.4086e-04 | norm: 0.2886, dt: 451.31ms, tok/sec: 1161708.94
step: 4658 | loss: 3.380628 | lr: 5.4083e-04 | norm: 0.2987, dt: 451.35ms, tok/sec: 1161601.55
step: 4659 | loss: 3.381191 | lr: 5.4080e-04 | norm: 0.3018, dt: 451.16ms, tok/sec: 1162081.59
step: 4660 | loss: 3.373625 | lr: 5.4077e-04 | norm: 0.2827, dt: 450.67ms, tok/sec: 1163351.10
step: 4661 | loss: 3.452779 | lr: 5.4074e-04 | norm: 0.2890, dt: 450.87ms, tok/sec: 1162844.19
step: 4662 | loss: 3.373923 | lr: 5.4072e-04 | norm: 0.3211, dt: 451.17ms, tok/sec: 1162058.25
step: 4663 | loss: 3.287832 | lr: 5.4069e-04 | norm: 0.3151, dt: 449.85ms, tok/sec: 1165481.99
step: 4664 | loss: 3.287143 | lr: 5.4066e-04 | norm: 0.3067, dt: 450.73ms, tok/sec: 1163196.03
step: 4665 | loss: 3.320447 | lr: 5.4063e-04 | norm: 0.2869, dt: 450.29ms, tok/sec: 1164325.56
step: 4666 | loss: 3.320110 | lr: 5.4060e-04 | norm: 0.2878, dt: 450.77ms, tok/sec: 1163093.29
step: 4667 | loss: 3.346676 | lr: 5.4057e-04 | norm: 0.2885, dt: 450.98ms, tok/sec: 1162562.02
step: 4668 | loss: 3.312972 | lr: 5.4054e-04 | norm: 0.2618, dt: 450.50ms, tok/sec: 1163786.39
step: 4669 | loss: 3.321085 | lr: 5.4051e-04 | norm: 0.2449, dt: 450.65ms, tok/sec: 1163402.80
step: 4670 | loss: 3.326267 | lr: 5.4048e-04 | norm: 0.2346, dt: 451.54ms, tok/sec: 1161119.47
step: 4671 | loss: 3.330895 | lr: 5.4046e-04 | norm: 0.2314, dt: 450.98ms, tok/sec: 1162551.57
step: 4672 | loss: 3.366170 | lr: 5.4043e-04 | norm: 0.2282, dt: 450.64ms, tok/sec: 1163439.12
step: 4673 | loss: 3.329588 | lr: 5.4040e-04 | norm: 0.2334, dt: 450.82ms, tok/sec: 1162965.96
step: 4674 | loss: 3.329599 | lr: 5.4037e-04 | norm: 0.2286, dt: 450.92ms, tok/sec: 1162716.30
step: 4675 | loss: 3.392375 | lr: 5.4034e-04 | norm: 0.2438, dt: 450.74ms, tok/sec: 1163171.42
step: 4676 | loss: 3.309520 | lr: 5.4031e-04 | norm: 0.2332, dt: 450.58ms, tok/sec: 1163579.48
step: 4677 | loss: 3.391044 | lr: 5.4028e-04 | norm: 0.2183, dt: 452.34ms, tok/sec: 1159048.47
step: 4678 | loss: 3.345771 | lr: 5.4025e-04 | norm: 0.2751, dt: 450.81ms, tok/sec: 1162994.25
step: 4679 | loss: 3.288398 | lr: 5.4022e-04 | norm: 0.3161, dt: 450.14ms, tok/sec: 1164720.24
step: 4680 | loss: 3.335353 | lr: 5.4019e-04 | norm: 0.2691, dt: 450.83ms, tok/sec: 1162942.59
step: 4681 | loss: 3.366947 | lr: 5.4017e-04 | norm: 0.2820, dt: 450.44ms, tok/sec: 1163952.71
step: 4682 | loss: 3.410518 | lr: 5.4014e-04 | norm: 0.3046, dt: 450.78ms, tok/sec: 1163063.14
step: 4683 | loss: 3.357643 | lr: 5.4011e-04 | norm: 0.2779, dt: 449.82ms, tok/sec: 1165541.29
step: 4684 | loss: 3.375822 | lr: 5.4008e-04 | norm: 0.3079, dt: 449.78ms, tok/sec: 1165665.47
step: 4685 | loss: 3.349295 | lr: 5.4005e-04 | norm: 0.2905, dt: 451.15ms, tok/sec: 1162126.42
step: 4686 | loss: 3.320603 | lr: 5.4002e-04 | norm: 0.2872, dt: 451.73ms, tok/sec: 1160623.69
step: 4687 | loss: 3.331274 | lr: 5.3999e-04 | norm: 0.2820, dt: 450.06ms, tok/sec: 1164926.93
step: 4688 | loss: 3.293044 | lr: 5.3996e-04 | norm: 0.2757, dt: 450.42ms, tok/sec: 1164004.46
step: 4689 | loss: 3.271291 | lr: 5.3993e-04 | norm: 0.2782, dt: 450.81ms, tok/sec: 1162985.64
step: 4690 | loss: 3.276254 | lr: 5.3990e-04 | norm: 0.3518, dt: 451.30ms, tok/sec: 1161725.51
step: 4691 | loss: 3.189820 | lr: 5.3987e-04 | norm: 0.2814, dt: 449.39ms, tok/sec: 1166667.95
step: 4692 | loss: 3.275333 | lr: 5.3985e-04 | norm: 0.2684, dt: 451.18ms, tok/sec: 1162036.76
step: 4693 | loss: 3.271168 | lr: 5.3982e-04 | norm: 0.2942, dt: 451.56ms, tok/sec: 1161059.39
step: 4694 | loss: 3.269635 | lr: 5.3979e-04 | norm: 0.2675, dt: 452.07ms, tok/sec: 1159737.98
step: 4695 | loss: 3.317080 | lr: 5.3976e-04 | norm: 0.2681, dt: 450.42ms, tok/sec: 1163998.91
step: 4696 | loss: 3.268780 | lr: 5.3973e-04 | norm: 0.2697, dt: 451.20ms, tok/sec: 1161979.04
step: 4697 | loss: 3.253542 | lr: 5.3970e-04 | norm: 0.2436, dt: 451.17ms, tok/sec: 1162058.25
step: 4698 | loss: 3.188108 | lr: 5.3967e-04 | norm: 0.2534, dt: 451.08ms, tok/sec: 1162296.56
step: 4699 | loss: 3.324830 | lr: 5.3964e-04 | norm: 0.2812, dt: 451.31ms, tok/sec: 1161696.67
step: 4700 | loss: 3.356902 | lr: 5.3961e-04 | norm: 0.2791, dt: 451.43ms, tok/sec: 1161404.62
step: 4701 | loss: 3.432822 | lr: 5.3958e-04 | norm: 0.2718, dt: 451.04ms, tok/sec: 1162393.64
step: 4702 | loss: 3.360350 | lr: 5.3955e-04 | norm: 0.2671, dt: 450.88ms, tok/sec: 1162806.07
step: 4703 | loss: 3.447093 | lr: 5.3953e-04 | norm: 0.2762, dt: 451.14ms, tok/sec: 1162152.21
step: 4704 | loss: 3.440922 | lr: 5.3950e-04 | norm: 0.2971, dt: 451.15ms, tok/sec: 1162104.92
step: 4705 | loss: 3.454445 | lr: 5.3947e-04 | norm: 0.2832, dt: 450.67ms, tok/sec: 1163349.87
step: 4706 | loss: 3.419904 | lr: 5.3944e-04 | norm: 0.3323, dt: 450.79ms, tok/sec: 1163042.84
step: 4707 | loss: 3.405756 | lr: 5.3941e-04 | norm: 0.3163, dt: 451.12ms, tok/sec: 1162190.91
step: 4708 | loss: 3.373777 | lr: 5.3938e-04 | norm: 0.2709, dt: 450.58ms, tok/sec: 1163582.56
step: 4709 | loss: 3.261101 | lr: 5.3935e-04 | norm: 0.3294, dt: 450.69ms, tok/sec: 1163295.71
step: 4710 | loss: 3.296192 | lr: 5.3932e-04 | norm: 0.2924, dt: 451.11ms, tok/sec: 1162219.78
step: 4711 | loss: 3.348152 | lr: 5.3929e-04 | norm: 0.2810, dt: 450.45ms, tok/sec: 1163928.68
step: 4712 | loss: 3.387022 | lr: 5.3926e-04 | norm: 0.2659, dt: 451.52ms, tok/sec: 1161170.97
step: 4713 | loss: 3.300836 | lr: 5.3923e-04 | norm: 0.2559, dt: 450.28ms, tok/sec: 1164362.55
step: 4714 | loss: 3.392564 | lr: 5.3920e-04 | norm: 0.2505, dt: 450.74ms, tok/sec: 1163174.49
step: 4715 | loss: 3.385308 | lr: 5.3918e-04 | norm: 0.2819, dt: 451.66ms, tok/sec: 1160809.94
step: 4716 | loss: 3.308465 | lr: 5.3915e-04 | norm: 0.2905, dt: 451.00ms, tok/sec: 1162488.88
step: 4717 | loss: 3.284226 | lr: 5.3912e-04 | norm: 0.2646, dt: 451.55ms, tok/sec: 1161090.04
step: 4718 | loss: 3.376391 | lr: 5.3909e-04 | norm: 0.2441, dt: 451.56ms, tok/sec: 1161071.65
step: 4719 | loss: 3.307812 | lr: 5.3906e-04 | norm: 0.2461, dt: 451.50ms, tok/sec: 1161203.46
step: 4720 | loss: 3.387376 | lr: 5.3903e-04 | norm: 0.2622, dt: 450.66ms, tok/sec: 1163384.34
step: 4721 | loss: 3.365703 | lr: 5.3900e-04 | norm: 0.2444, dt: 450.65ms, tok/sec: 1163399.72
step: 4722 | loss: 3.381794 | lr: 5.3897e-04 | norm: 0.2717, dt: 450.98ms, tok/sec: 1162560.17
step: 4723 | loss: 3.323262 | lr: 5.3894e-04 | norm: 0.2207, dt: 450.88ms, tok/sec: 1162807.91
step: 4724 | loss: 3.355052 | lr: 5.3891e-04 | norm: 0.2496, dt: 1135.66ms, tok/sec: 461659.52
step: 4725 | loss: 3.359825 | lr: 5.3888e-04 | norm: 0.2472, dt: 451.64ms, tok/sec: 1160861.41
step: 4726 | loss: 3.351197 | lr: 5.3885e-04 | norm: 0.2672, dt: 451.30ms, tok/sec: 1161738.40
step: 4727 | loss: 3.378188 | lr: 5.3882e-04 | norm: 0.2499, dt: 451.08ms, tok/sec: 1162292.88
step: 4728 | loss: 3.359440 | lr: 5.3880e-04 | norm: 0.2403, dt: 450.66ms, tok/sec: 1163367.72
step: 4729 | loss: 3.342960 | lr: 5.3877e-04 | norm: 0.2284, dt: 450.39ms, tok/sec: 1164069.16
step: 4730 | loss: 3.379002 | lr: 5.3874e-04 | norm: 0.2520, dt: 450.78ms, tok/sec: 1163077.91
step: 4731 | loss: 3.371761 | lr: 5.3871e-04 | norm: 0.2647, dt: 451.13ms, tok/sec: 1162158.97
step: 4732 | loss: 3.385745 | lr: 5.3868e-04 | norm: 0.2912, dt: 450.21ms, tok/sec: 1164547.53
step: 4733 | loss: 3.242238 | lr: 5.3865e-04 | norm: 0.2889, dt: 450.04ms, tok/sec: 1164988.03
step: 4734 | loss: 3.229561 | lr: 5.3862e-04 | norm: 0.3321, dt: 450.52ms, tok/sec: 1163727.88
step: 4735 | loss: 3.269800 | lr: 5.3859e-04 | norm: 0.3207, dt: 451.15ms, tok/sec: 1162114.14
step: 4736 | loss: 3.271766 | lr: 5.3856e-04 | norm: 0.3317, dt: 451.19ms, tok/sec: 1161999.30
step: 4737 | loss: 3.280389 | lr: 5.3853e-04 | norm: 0.2740, dt: 451.34ms, tok/sec: 1161613.21
step: 4738 | loss: 3.241283 | lr: 5.3850e-04 | norm: 0.2555, dt: 451.13ms, tok/sec: 1162165.11
step: 4739 | loss: 3.227270 | lr: 5.3847e-04 | norm: 0.2856, dt: 451.96ms, tok/sec: 1160038.98
step: 4740 | loss: 3.220378 | lr: 5.3844e-04 | norm: 0.2728, dt: 451.73ms, tok/sec: 1160626.75
step: 4741 | loss: 3.236471 | lr: 5.3841e-04 | norm: 0.2658, dt: 451.12ms, tok/sec: 1162203.81
step: 4742 | loss: 3.223214 | lr: 5.3838e-04 | norm: 0.2705, dt: 450.55ms, tok/sec: 1163654.60
step: 4743 | loss: 3.195662 | lr: 5.3836e-04 | norm: 0.2575, dt: 451.39ms, tok/sec: 1161484.37
step: 4744 | loss: 3.204530 | lr: 5.3833e-04 | norm: 0.2491, dt: 450.49ms, tok/sec: 1163828.27
step: 4745 | loss: 3.367231 | lr: 5.3830e-04 | norm: 0.2716, dt: 451.49ms, tok/sec: 1161234.12
step: 4746 | loss: 3.366431 | lr: 5.3827e-04 | norm: 0.2672, dt: 450.03ms, tok/sec: 1164994.82
step: 4747 | loss: 3.416759 | lr: 5.3824e-04 | norm: 0.2502, dt: 451.52ms, tok/sec: 1161157.48
step: 4748 | loss: 3.365437 | lr: 5.3821e-04 | norm: 0.2766, dt: 451.29ms, tok/sec: 1161758.04
step: 4749 | loss: 3.401838 | lr: 5.3818e-04 | norm: 0.2750, dt: 1226.38ms, tok/sec: 427508.40
validation loss: 3.3893
HellaSwag accuracy: 2744/10042=0.2733
rank 0 sample 0: Hello, I'm a Manpreet, and I think I'll talk more about using all the vocabulary in some lesson in the next time."
He said
rank 0 sample 1: Hello, I'm a Manpreet, I feel like I'm a God my God is that I am. All my thoughts, I am happy, I
rank 0 sample 2: Hello, I'm a Manpreet, I'm only really concerned just about being a nice person. If that's not so, I'm not going to
rank 6 sample 0: Hello, I'm a Manpreet, or an OpenOffice application as you would expect. But what I was expecting for you was to have a simple,
rank 0 sample 3: Hello, I'm a Manpreet, and all the others are just like this one who is a part of me and whose own, I was a man
rank 6 sample 1: Hello, I'm a Manpreet, my mom has a baby. It's no easy task to get them, but it can be very rewarding and stressful
rank 6 sample 2: Hello, I'm a Manpreet, a Teacher of God.<|endoftext|>An overview of the history of art by David L. S. Taylor
From A
rank 4 sample 0: Hello, I'm a Manpreet, I'm on my way to the world I've never seen before. My son is my next realisation and I
rank 6 sample 3: Hello, I'm a Manpreet, and we can speak for a man, but it's not a joke. It's just your voice that needs to
rank 4 sample 1: Hello, I'm a Manpreet, here you can bring in a letter if you want to join me. Here I write "Hello, here is Your
rank 4 sample 2: Hello, I'm a Manpreet, so your answers really help you. But if you're not ready, you're back with a new story. What
rank 4 sample 3: Hello, I'm a Manpreet, and it'd me so much help here at MOMJ who has provided you with a quick read on the things
rank 5 sample 0: Hello, I'm a Manpreet, a B-Wiggling, a 'I want you to come out of my house'. I'm going to
rank 5 sample 1: Hello, I'm a Manpreet, "
So we put it up here." [I] is our "all." We are our "all"
rank 5 sample 2: Hello, I'm a Manpreet, I've lived a most miserable life; I'm a man, I'm going to write a book. I'm
rank 5 sample 3: Hello, I'm a Manpreet, if you know the name of a man you'll probably have a servant, son or sister, an uncle or uncle
rank 7 sample 0: Hello, I'm a Manpreet, and I'm a Woman and I'm getting married now. I'm here. Okay, I won't get married
rank 7 sample 1: Hello, I'm a Manpreet, thanks to Dr. Joko at Dr. Joko.
So, I see it in the light of my
rank 7 sample 2: Hello, I'm a Manpreet, M.I. :)
Here are my questions to help you out the next day:
- What does I
rank 2 sample 0: Hello, I'm a Manpreet, etc." I could not read it again, so I went back to my room.
I could not think of
rank 7 sample 3: Hello, I'm a Manpreet, although it's not very important. Because I'd like to come back as a Manpreet member. I'm
rank 2 sample 1: Hello, I'm a Manpreet, as I make it happen that I feel confident that as I have, if I can find it I will do it
rank 2 sample 2: Hello, I'm a Manpreet, I'm going to do my thing but now I'm going to make things like this one, we get this nice
rank 2 sample 3: Hello, I'm a Manpreet, so the thing is, you're the ones who would have, you're the ones who should call me a Man
rank 1 sample 0: Hello, I'm a Manpreet, just my Mother. Let me know please, it's an example of my parents, my grandparents were my mother,
rank 1 sample 1: Hello, I'm a Manpreet, but you can't just say, "Yeah, I see you don't want me.")
One day, I
rank 1 sample 2: Hello, I'm a Manpreet, a Styla, a Styla, a Santa Clarita and a Santa Clarita. The other people
rank 1 sample 3: Hello, I'm a Manpreet, and I'm here. (Thanks: It's nice too late...)
3. After hearing that, I have
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going with it, not really, so I don't pay with my salary to that very day,
rank 3 sample 1: Hello, I'm a Manpreet, and the rest of the crowd at the top of the house is the National Poetry Room.
I'm here
rank 3 sample 2: Hello, I'm a Manpreet, and you have to show me an example as this is an illustration. I am writing this, since it's about
rank 3 sample 3: Hello, I'm a Manpreet, and have a lot of wonderful, simple things in life for you to look at. If I hadn't, I
step: 4750 | loss: 3.338924 | lr: 5.3815e-04 | norm: 0.2540, dt: 12060.36ms, tok/sec: 43472.01
step: 4751 | loss: 3.438910 | lr: 5.3812e-04 | norm: 0.2675, dt: 448.45ms, tok/sec: 1169122.96
step: 4752 | loss: 3.313596 | lr: 5.3809e-04 | norm: 0.2693, dt: 448.17ms, tok/sec: 1169830.12
step: 4753 | loss: 3.390313 | lr: 5.3806e-04 | norm: 0.2450, dt: 448.41ms, tok/sec: 1169224.29
step: 4754 | loss: 3.374050 | lr: 5.3803e-04 | norm: 0.2531, dt: 449.34ms, tok/sec: 1166808.47
step: 4755 | loss: 3.309606 | lr: 5.3800e-04 | norm: 0.2568, dt: 449.61ms, tok/sec: 1166093.84
step: 4756 | loss: 3.279901 | lr: 5.3797e-04 | norm: 0.2443, dt: 449.78ms, tok/sec: 1165653.12
step: 4757 | loss: 3.281432 | lr: 5.3794e-04 | norm: 0.2673, dt: 449.30ms, tok/sec: 1166889.58
step: 4758 | loss: 3.359160 | lr: 5.3791e-04 | norm: 0.2916, dt: 450.02ms, tok/sec: 1165043.58
step: 4759 | loss: 3.335427 | lr: 5.3788e-04 | norm: 0.2783, dt: 449.63ms, tok/sec: 1166042.52
step: 4760 | loss: 3.360013 | lr: 5.3785e-04 | norm: 0.2485, dt: 450.87ms, tok/sec: 1162825.75
step: 4761 | loss: 3.327351 | lr: 5.3783e-04 | norm: 0.2951, dt: 450.74ms, tok/sec: 1163164.03
step: 4762 | loss: 3.306047 | lr: 5.3780e-04 | norm: 0.2970, dt: 450.49ms, tok/sec: 1163809.79
step: 4763 | loss: 3.337773 | lr: 5.3777e-04 | norm: 0.2606, dt: 449.61ms, tok/sec: 1166101.88
step: 4764 | loss: 3.313167 | lr: 5.3774e-04 | norm: 0.3040, dt: 451.40ms, tok/sec: 1161461.06
step: 4765 | loss: 3.352267 | lr: 5.3771e-04 | norm: 0.2780, dt: 454.37ms, tok/sec: 1153888.63
step: 4766 | loss: 3.375938 | lr: 5.3768e-04 | norm: 0.2605, dt: 450.05ms, tok/sec: 1164954.71
step: 4767 | loss: 3.432430 | lr: 5.3765e-04 | norm: 0.2952, dt: 452.83ms, tok/sec: 1157804.17
step: 4768 | loss: 3.314780 | lr: 5.3762e-04 | norm: 0.2845, dt: 451.41ms, tok/sec: 1161437.13
step: 4769 | loss: 3.388231 | lr: 5.3759e-04 | norm: 0.2808, dt: 450.62ms, tok/sec: 1163481.59
step: 4770 | loss: 3.336666 | lr: 5.3756e-04 | norm: 0.3470, dt: 450.45ms, tok/sec: 1163921.90
step: 4771 | loss: 3.388560 | lr: 5.3753e-04 | norm: 0.2937, dt: 453.81ms, tok/sec: 1155308.40
step: 4772 | loss: 3.366003 | lr: 5.3750e-04 | norm: 0.3278, dt: 460.60ms, tok/sec: 1138265.00
step: 4773 | loss: 3.356571 | lr: 5.3747e-04 | norm: 0.3185, dt: 451.36ms, tok/sec: 1161561.67
step: 4774 | loss: 3.355695 | lr: 5.3744e-04 | norm: 0.2845, dt: 450.95ms, tok/sec: 1162641.92
step: 4775 | loss: 3.316401 | lr: 5.3741e-04 | norm: 0.3067, dt: 451.32ms, tok/sec: 1161687.46
step: 4776 | loss: 3.417933 | lr: 5.3738e-04 | norm: 0.3090, dt: 451.50ms, tok/sec: 1161224.93
step: 4777 | loss: 3.368248 | lr: 5.3735e-04 | norm: 0.3163, dt: 450.62ms, tok/sec: 1163484.05
step: 4778 | loss: 3.337556 | lr: 5.3732e-04 | norm: 0.2989, dt: 450.63ms, tok/sec: 1163460.05
step: 4779 | loss: 3.376707 | lr: 5.3729e-04 | norm: 0.2936, dt: 450.17ms, tok/sec: 1164654.85
step: 4780 | loss: 3.210573 | lr: 5.3726e-04 | norm: 0.2693, dt: 450.98ms, tok/sec: 1162551.57
step: 4781 | loss: 3.317713 | lr: 5.3723e-04 | norm: 0.3053, dt: 450.76ms, tok/sec: 1163111.74
step: 4782 | loss: 3.191285 | lr: 5.3720e-04 | norm: 0.2730, dt: 450.56ms, tok/sec: 1163647.21
step: 4783 | loss: 3.227350 | lr: 5.3717e-04 | norm: 0.2767, dt: 451.38ms, tok/sec: 1161515.66
step: 4784 | loss: 3.183002 | lr: 5.3715e-04 | norm: 0.2568, dt: 452.54ms, tok/sec: 1158545.30
step: 4785 | loss: 3.201034 | lr: 5.3712e-04 | norm: 0.2736, dt: 452.04ms, tok/sec: 1159827.29
step: 4786 | loss: 3.279749 | lr: 5.3709e-04 | norm: 0.2363, dt: 449.78ms, tok/sec: 1165653.73
step: 4787 | loss: 3.259351 | lr: 5.3706e-04 | norm: 0.2526, dt: 451.16ms, tok/sec: 1162086.50
step: 4788 | loss: 3.248143 | lr: 5.3703e-04 | norm: 0.2326, dt: 451.27ms, tok/sec: 1161813.28
step: 4789 | loss: 3.308646 | lr: 5.3700e-04 | norm: 0.2574, dt: 450.46ms, tok/sec: 1163888.02
step: 4790 | loss: 3.187442 | lr: 5.3697e-04 | norm: 0.2433, dt: 450.35ms, tok/sec: 1164169.61
step: 4791 | loss: 3.268066 | lr: 5.3694e-04 | norm: 0.2494, dt: 452.09ms, tok/sec: 1159702.51
step: 4792 | loss: 3.364844 | lr: 5.3691e-04 | norm: 0.2413, dt: 451.19ms, tok/sec: 1162016.50
step: 4793 | loss: 3.402687 | lr: 5.3688e-04 | norm: 0.2700, dt: 450.84ms, tok/sec: 1162921.68
step: 4794 | loss: 3.473366 | lr: 5.3685e-04 | norm: 0.2797, dt: 451.40ms, tok/sec: 1161462.28
step: 4795 | loss: 3.352613 | lr: 5.3682e-04 | norm: 0.3061, dt: 450.98ms, tok/sec: 1162542.35
step: 4796 | loss: 3.469898 | lr: 5.3679e-04 | norm: 0.3409, dt: 451.33ms, tok/sec: 1161638.98
step: 4797 | loss: 3.314881 | lr: 5.3676e-04 | norm: 0.2825, dt: 450.24ms, tok/sec: 1164451.33
step: 4798 | loss: 3.374741 | lr: 5.3673e-04 | norm: 0.2805, dt: 451.06ms, tok/sec: 1162351.24
step: 4799 | loss: 3.348624 | lr: 5.3670e-04 | norm: 0.2677, dt: 450.89ms, tok/sec: 1162774.10
step: 4800 | loss: 3.362962 | lr: 5.3667e-04 | norm: 0.2516, dt: 451.71ms, tok/sec: 1160681.88
step: 4801 | loss: 3.338008 | lr: 5.3664e-04 | norm: 0.2544, dt: 450.32ms, tok/sec: 1164246.04
step: 4802 | loss: 3.341936 | lr: 5.3661e-04 | norm: 0.2814, dt: 450.80ms, tok/sec: 1163028.08
step: 4803 | loss: 3.311853 | lr: 5.3658e-04 | norm: 0.2648, dt: 449.86ms, tok/sec: 1165444.31
step: 4804 | loss: 3.363185 | lr: 5.3655e-04 | norm: 0.2420, dt: 449.99ms, tok/sec: 1165102.22
step: 4805 | loss: 3.350086 | lr: 5.3652e-04 | norm: 0.2661, dt: 450.93ms, tok/sec: 1162668.97
step: 4806 | loss: 3.324838 | lr: 5.3649e-04 | norm: 0.2508, dt: 451.01ms, tok/sec: 1162472.29
step: 4807 | loss: 3.367894 | lr: 5.3646e-04 | norm: 0.2783, dt: 450.39ms, tok/sec: 1164067.31
step: 4808 | loss: 3.250289 | lr: 5.3643e-04 | norm: 0.3146, dt: 450.24ms, tok/sec: 1164460.58
step: 4809 | loss: 3.318509 | lr: 5.3640e-04 | norm: 0.2692, dt: 450.37ms, tok/sec: 1164138.79
step: 4810 | loss: 3.296007 | lr: 5.3637e-04 | norm: 0.2684, dt: 452.17ms, tok/sec: 1159490.93
step: 4811 | loss: 3.305175 | lr: 5.3634e-04 | norm: 0.2839, dt: 450.24ms, tok/sec: 1164475.38
step: 4812 | loss: 3.338215 | lr: 5.3631e-04 | norm: 0.2426, dt: 451.10ms, tok/sec: 1162241.89
step: 4813 | loss: 3.379196 | lr: 5.3628e-04 | norm: 0.2561, dt: 450.33ms, tok/sec: 1164219.53
step: 4814 | loss: 3.443555 | lr: 5.3625e-04 | norm: 0.2972, dt: 450.94ms, tok/sec: 1162651.76
step: 4815 | loss: 3.323998 | lr: 5.3622e-04 | norm: 0.2651, dt: 451.43ms, tok/sec: 1161404.62
step: 4816 | loss: 3.382028 | lr: 5.3619e-04 | norm: 0.2374, dt: 451.27ms, tok/sec: 1161812.06
step: 4817 | loss: 3.401991 | lr: 5.3616e-04 | norm: 0.2650, dt: 450.96ms, tok/sec: 1162609.34
step: 4818 | loss: 3.425395 | lr: 5.3613e-04 | norm: 0.2605, dt: 450.37ms, tok/sec: 1164127.08
step: 4819 | loss: 3.363789 | lr: 5.3610e-04 | norm: 0.2630, dt: 452.41ms, tok/sec: 1158865.84
step: 4820 | loss: 3.339119 | lr: 5.3607e-04 | norm: 0.2403, dt: 450.63ms, tok/sec: 1163457.58
step: 4821 | loss: 3.371850 | lr: 5.3604e-04 | norm: 0.2447, dt: 450.81ms, tok/sec: 1162991.18
step: 4822 | loss: 3.369580 | lr: 5.3601e-04 | norm: 0.2544, dt: 450.98ms, tok/sec: 1162561.40
step: 4823 | loss: 3.318114 | lr: 5.3598e-04 | norm: 0.2683, dt: 450.42ms, tok/sec: 1163990.90
step: 4824 | loss: 3.374019 | lr: 5.3595e-04 | norm: 0.2573, dt: 450.62ms, tok/sec: 1163470.51
step: 4825 | loss: 3.358004 | lr: 5.3592e-04 | norm: 0.2457, dt: 450.25ms, tok/sec: 1164426.67
step: 4826 | loss: 3.292264 | lr: 5.3590e-04 | norm: 0.2582, dt: 450.87ms, tok/sec: 1162841.12
step: 4827 | loss: 3.205104 | lr: 5.3587e-04 | norm: 0.2494, dt: 450.32ms, tok/sec: 1164257.13
step: 4828 | loss: 3.264691 | lr: 5.3584e-04 | norm: 0.2880, dt: 451.34ms, tok/sec: 1161634.69
step: 4829 | loss: 3.196480 | lr: 5.3581e-04 | norm: 0.2843, dt: 450.47ms, tok/sec: 1163880.01
step: 4830 | loss: 3.256658 | lr: 5.3578e-04 | norm: 0.3055, dt: 450.78ms, tok/sec: 1163056.38
step: 4831 | loss: 3.244700 | lr: 5.3575e-04 | norm: 0.2796, dt: 450.80ms, tok/sec: 1163018.85
step: 4832 | loss: 3.151235 | lr: 5.3572e-04 | norm: 0.2928, dt: 451.22ms, tok/sec: 1161934.83
step: 4833 | loss: 3.210475 | lr: 5.3569e-04 | norm: 0.2946, dt: 451.69ms, tok/sec: 1160721.71
step: 4834 | loss: 3.217704 | lr: 5.3566e-04 | norm: 0.2680, dt: 450.84ms, tok/sec: 1162921.68
step: 4835 | loss: 3.225457 | lr: 5.3563e-04 | norm: 0.2583, dt: 451.40ms, tok/sec: 1161473.33
step: 4836 | loss: 3.185998 | lr: 5.3560e-04 | norm: 0.2718, dt: 451.08ms, tok/sec: 1162303.94
step: 4837 | loss: 3.174595 | lr: 5.3557e-04 | norm: 0.2906, dt: 451.57ms, tok/sec: 1161025.06
step: 4838 | loss: 3.256099 | lr: 5.3554e-04 | norm: 0.2576, dt: 451.12ms, tok/sec: 1162203.19
step: 4839 | loss: 3.392783 | lr: 5.3551e-04 | norm: 0.2695, dt: 450.01ms, tok/sec: 1165067.65
step: 4840 | loss: 3.355240 | lr: 5.3548e-04 | norm: 0.2809, dt: 450.84ms, tok/sec: 1162909.99
step: 4841 | loss: 3.362593 | lr: 5.3545e-04 | norm: 0.2671, dt: 450.79ms, tok/sec: 1163034.23
step: 4842 | loss: 3.381067 | lr: 5.3542e-04 | norm: 0.3038, dt: 450.27ms, tok/sec: 1164396.46
step: 4843 | loss: 3.397558 | lr: 5.3539e-04 | norm: 0.2841, dt: 451.01ms, tok/sec: 1162487.04
step: 4844 | loss: 3.309230 | lr: 5.3536e-04 | norm: 0.2666, dt: 451.02ms, tok/sec: 1162456.31
step: 4845 | loss: 3.338665 | lr: 5.3533e-04 | norm: 0.2722, dt: 451.04ms, tok/sec: 1162403.47
step: 4846 | loss: 3.399155 | lr: 5.3530e-04 | norm: 0.2710, dt: 450.56ms, tok/sec: 1163628.73
step: 4847 | loss: 3.393529 | lr: 5.3527e-04 | norm: 0.2702, dt: 451.22ms, tok/sec: 1161937.29
step: 4848 | loss: 3.328519 | lr: 5.3524e-04 | norm: 0.2534, dt: 451.09ms, tok/sec: 1162262.78
step: 4849 | loss: 3.286840 | lr: 5.3521e-04 | norm: 0.2702, dt: 450.29ms, tok/sec: 1164322.47
step: 4850 | loss: 3.381470 | lr: 5.3518e-04 | norm: 0.2921, dt: 450.94ms, tok/sec: 1162649.30
step: 4851 | loss: 3.368151 | lr: 5.3515e-04 | norm: 0.3278, dt: 451.22ms, tok/sec: 1161944.04
step: 4852 | loss: 3.372582 | lr: 5.3512e-04 | norm: 0.2789, dt: 450.72ms, tok/sec: 1163230.49
step: 4853 | loss: 3.400849 | lr: 5.3509e-04 | norm: 0.2898, dt: 451.14ms, tok/sec: 1162148.53
step: 4854 | loss: 3.359956 | lr: 5.3506e-04 | norm: 0.2975, dt: 451.40ms, tok/sec: 1161477.62
step: 4855 | loss: 3.421484 | lr: 5.3503e-04 | norm: 0.3761, dt: 450.74ms, tok/sec: 1163183.72
step: 4856 | loss: 3.365181 | lr: 5.3500e-04 | norm: 0.2528, dt: 451.45ms, tok/sec: 1161338.99
step: 4857 | loss: 3.318774 | lr: 5.3497e-04 | norm: 0.2544, dt: 451.09ms, tok/sec: 1162265.23
step: 4858 | loss: 3.265320 | lr: 5.3494e-04 | norm: 0.2361, dt: 449.83ms, tok/sec: 1165520.90
step: 4859 | loss: 3.349211 | lr: 5.3491e-04 | norm: 0.2546, dt: 451.00ms, tok/sec: 1162493.18
step: 4860 | loss: 3.376813 | lr: 5.3488e-04 | norm: 0.2763, dt: 450.61ms, tok/sec: 1163498.83
step: 4861 | loss: 3.413070 | lr: 5.3485e-04 | norm: 0.2654, dt: 450.87ms, tok/sec: 1162832.51
step: 4862 | loss: 3.410486 | lr: 5.3482e-04 | norm: 0.2682, dt: 449.83ms, tok/sec: 1165525.85
step: 4863 | loss: 3.383959 | lr: 5.3479e-04 | norm: 0.2623, dt: 451.12ms, tok/sec: 1162201.96
step: 4864 | loss: 3.352351 | lr: 5.3475e-04 | norm: 0.2524, dt: 451.04ms, tok/sec: 1162408.38
step: 4865 | loss: 3.362842 | lr: 5.3472e-04 | norm: 0.2634, dt: 450.37ms, tok/sec: 1164125.85
step: 4866 | loss: 3.365792 | lr: 5.3469e-04 | norm: 0.2700, dt: 450.35ms, tok/sec: 1164180.70
step: 4867 | loss: 3.335140 | lr: 5.3466e-04 | norm: 0.2645, dt: 452.85ms, tok/sec: 1157750.53
step: 4868 | loss: 3.346024 | lr: 5.3463e-04 | norm: 0.2815, dt: 449.99ms, tok/sec: 1165105.31
step: 4869 | loss: 3.355114 | lr: 5.3460e-04 | norm: 0.2649, dt: 450.51ms, tok/sec: 1163778.38
step: 4870 | loss: 3.316154 | lr: 5.3457e-04 | norm: 0.2649, dt: 450.22ms, tok/sec: 1164516.08
step: 4871 | loss: 3.286045 | lr: 5.3454e-04 | norm: 0.2602, dt: 450.76ms, tok/sec: 1163122.20
step: 4872 | loss: 3.396245 | lr: 5.3451e-04 | norm: 0.2754, dt: 450.61ms, tok/sec: 1163495.13
step: 4873 | loss: 3.151715 | lr: 5.3448e-04 | norm: 0.2841, dt: 450.40ms, tok/sec: 1164045.13
step: 4874 | loss: 3.399896 | lr: 5.3445e-04 | norm: 0.3216, dt: 453.69ms, tok/sec: 1155607.72
step: 4875 | loss: 3.213963 | lr: 5.3442e-04 | norm: 0.2995, dt: 450.79ms, tok/sec: 1163044.69
step: 4876 | loss: 3.207619 | lr: 5.3439e-04 | norm: 0.3196, dt: 450.89ms, tok/sec: 1162776.56
step: 4877 | loss: 3.233898 | lr: 5.3436e-04 | norm: 0.3038, dt: 451.07ms, tok/sec: 1162311.92
step: 4878 | loss: 3.179511 | lr: 5.3433e-04 | norm: 0.2620, dt: 450.82ms, tok/sec: 1162967.19
step: 4879 | loss: 3.257557 | lr: 5.3430e-04 | norm: 0.2839, dt: 452.67ms, tok/sec: 1158221.90
step: 4880 | loss: 3.233600 | lr: 5.3427e-04 | norm: 0.2698, dt: 450.80ms, tok/sec: 1163009.01
step: 4881 | loss: 3.181657 | lr: 5.3424e-04 | norm: 0.2438, dt: 450.31ms, tok/sec: 1164270.08
step: 4882 | loss: 3.349378 | lr: 5.3421e-04 | norm: 0.2755, dt: 451.13ms, tok/sec: 1162174.94
step: 4883 | loss: 3.195855 | lr: 5.3418e-04 | norm: 0.2745, dt: 450.24ms, tok/sec: 1164464.28
step: 4884 | loss: 3.201323 | lr: 5.3415e-04 | norm: 0.2745, dt: 450.67ms, tok/sec: 1163355.41
step: 4885 | loss: 3.398693 | lr: 5.3412e-04 | norm: 0.4132, dt: 450.15ms, tok/sec: 1164687.54
step: 4886 | loss: 3.375412 | lr: 5.3409e-04 | norm: 0.3041, dt: 451.54ms, tok/sec: 1161123.14
step: 4887 | loss: 3.385312 | lr: 5.3406e-04 | norm: 0.2812, dt: 451.78ms, tok/sec: 1160484.04
step: 4888 | loss: 3.369839 | lr: 5.3403e-04 | norm: 0.3051, dt: 450.31ms, tok/sec: 1164283.02
step: 4889 | loss: 3.375731 | lr: 5.3400e-04 | norm: 0.2976, dt: 451.17ms, tok/sec: 1162069.30
step: 4890 | loss: 3.417436 | lr: 5.3397e-04 | norm: 0.2861, dt: 450.61ms, tok/sec: 1163500.06
step: 4891 | loss: 3.407972 | lr: 5.3394e-04 | norm: 0.3039, dt: 451.80ms, tok/sec: 1160438.72
step: 4892 | loss: 3.355278 | lr: 5.3391e-04 | norm: 0.3045, dt: 451.28ms, tok/sec: 1161768.48
step: 4893 | loss: 3.401725 | lr: 5.3388e-04 | norm: 0.2748, dt: 453.97ms, tok/sec: 1154892.78
step: 4894 | loss: 3.392502 | lr: 5.3385e-04 | norm: 0.3058, dt: 451.61ms, tok/sec: 1160922.08
step: 4895 | loss: 3.313367 | lr: 5.3382e-04 | norm: 0.2811, dt: 450.64ms, tok/sec: 1163417.57
step: 4896 | loss: 3.337387 | lr: 5.3379e-04 | norm: 0.2771, dt: 450.76ms, tok/sec: 1163125.89
step: 4897 | loss: 3.329334 | lr: 5.3376e-04 | norm: 0.2926, dt: 450.18ms, tok/sec: 1164617.22
step: 4898 | loss: 3.341575 | lr: 5.3373e-04 | norm: 0.2814, dt: 451.09ms, tok/sec: 1162276.91
step: 4899 | loss: 3.337024 | lr: 5.3370e-04 | norm: 0.3238, dt: 450.73ms, tok/sec: 1163195.41
step: 4900 | loss: 3.380534 | lr: 5.3367e-04 | norm: 0.3128, dt: 449.28ms, tok/sec: 1166939.12
step: 4901 | loss: 3.348793 | lr: 5.3364e-04 | norm: 0.3029, dt: 451.22ms, tok/sec: 1161926.24
step: 4902 | loss: 3.306008 | lr: 5.3361e-04 | norm: 0.2780, dt: 450.88ms, tok/sec: 1162814.06
step: 4903 | loss: 3.408366 | lr: 5.3358e-04 | norm: 0.3037, dt: 450.95ms, tok/sec: 1162619.79
step: 4904 | loss: 3.400251 | lr: 5.3355e-04 | norm: 0.3121, dt: 450.63ms, tok/sec: 1163443.43
step: 4905 | loss: 3.350794 | lr: 5.3352e-04 | norm: 0.3014, dt: 451.11ms, tok/sec: 1162207.49
step: 4906 | loss: 3.392255 | lr: 5.3348e-04 | norm: 0.2536, dt: 451.70ms, tok/sec: 1160700.26
step: 4907 | loss: 3.381674 | lr: 5.3345e-04 | norm: 0.2808, dt: 450.61ms, tok/sec: 1163519.14
step: 4908 | loss: 3.414054 | lr: 5.3342e-04 | norm: 0.2610, dt: 451.02ms, tok/sec: 1162437.88
step: 4909 | loss: 3.364025 | lr: 5.3339e-04 | norm: 0.2774, dt: 449.96ms, tok/sec: 1165188.65
step: 4910 | loss: 3.334656 | lr: 5.3336e-04 | norm: 0.2707, dt: 450.53ms, tok/sec: 1163723.57
step: 4911 | loss: 3.357711 | lr: 5.3333e-04 | norm: 0.2830, dt: 450.77ms, tok/sec: 1163098.21
step: 4912 | loss: 3.312254 | lr: 5.3330e-04 | norm: 0.2646, dt: 449.91ms, tok/sec: 1165313.38
step: 4913 | loss: 3.399728 | lr: 5.3327e-04 | norm: 0.2789, dt: 1147.39ms, tok/sec: 456938.66
step: 4914 | loss: 3.365515 | lr: 5.3324e-04 | norm: 0.2934, dt: 449.80ms, tok/sec: 1165615.43
step: 4915 | loss: 3.348968 | lr: 5.3321e-04 | norm: 0.2565, dt: 450.38ms, tok/sec: 1164107.98
step: 4916 | loss: 3.297595 | lr: 5.3318e-04 | norm: 0.2491, dt: 451.44ms, tok/sec: 1161358.01
step: 4917 | loss: 3.390391 | lr: 5.3315e-04 | norm: 0.2620, dt: 450.37ms, tok/sec: 1164133.25
step: 4918 | loss: 3.315605 | lr: 5.3312e-04 | norm: 0.2704, dt: 450.29ms, tok/sec: 1164323.09
step: 4919 | loss: 3.376077 | lr: 5.3309e-04 | norm: 0.2648, dt: 451.88ms, tok/sec: 1160228.10
step: 4920 | loss: 3.265739 | lr: 5.3306e-04 | norm: 0.2799, dt: 449.94ms, tok/sec: 1165230.02
step: 4921 | loss: 3.251565 | lr: 5.3303e-04 | norm: 0.2577, dt: 449.46ms, tok/sec: 1166472.39
step: 4922 | loss: 3.242820 | lr: 5.3300e-04 | norm: 0.2877, dt: 450.83ms, tok/sec: 1162928.44
step: 4923 | loss: 3.284303 | lr: 5.3297e-04 | norm: 0.2969, dt: 450.06ms, tok/sec: 1164930.02
step: 4924 | loss: 3.254443 | lr: 5.3294e-04 | norm: 0.2569, dt: 450.12ms, tok/sec: 1164778.23
step: 4925 | loss: 3.233550 | lr: 5.3291e-04 | norm: 0.2529, dt: 450.27ms, tok/sec: 1164395.22
step: 4926 | loss: 3.178842 | lr: 5.3288e-04 | norm: 0.2648, dt: 449.90ms, tok/sec: 1165344.26
step: 4927 | loss: 3.229880 | lr: 5.3285e-04 | norm: 0.2718, dt: 451.27ms, tok/sec: 1161811.44
step: 4928 | loss: 3.200427 | lr: 5.3282e-04 | norm: 0.2595, dt: 450.97ms, tok/sec: 1162583.53
step: 4929 | loss: 3.257272 | lr: 5.3278e-04 | norm: 0.2811, dt: 450.32ms, tok/sec: 1164261.45
step: 4930 | loss: 3.185648 | lr: 5.3275e-04 | norm: 0.2730, dt: 450.70ms, tok/sec: 1163268.64
step: 4931 | loss: 3.226514 | lr: 5.3272e-04 | norm: 0.2642, dt: 451.84ms, tok/sec: 1160334.63
step: 4932 | loss: 3.392582 | lr: 5.3269e-04 | norm: 0.2984, dt: 451.37ms, tok/sec: 1161538.97
step: 4933 | loss: 3.337695 | lr: 5.3266e-04 | norm: 0.3026, dt: 450.35ms, tok/sec: 1164170.22
step: 4934 | loss: 3.371250 | lr: 5.3263e-04 | norm: 0.2955, dt: 451.73ms, tok/sec: 1160622.46
step: 4935 | loss: 3.392648 | lr: 5.3260e-04 | norm: 0.2911, dt: 450.34ms, tok/sec: 1164204.74
step: 4936 | loss: 3.321330 | lr: 5.3257e-04 | norm: 0.2465, dt: 450.66ms, tok/sec: 1163372.64
step: 4937 | loss: 3.412473 | lr: 5.3254e-04 | norm: 0.2710, dt: 450.45ms, tok/sec: 1163932.37
step: 4938 | loss: 3.431543 | lr: 5.3251e-04 | norm: 0.3051, dt: 451.48ms, tok/sec: 1161254.36
step: 4939 | loss: 3.326162 | lr: 5.3248e-04 | norm: 0.3070, dt: 1229.95ms, tok/sec: 426269.24
step: 4940 | loss: 3.341042 | lr: 5.3245e-04 | norm: 0.2715, dt: 452.38ms, tok/sec: 1158946.46
step: 4941 | loss: 3.386220 | lr: 5.3242e-04 | norm: 0.3131, dt: 450.60ms, tok/sec: 1163545.62
step: 4942 | loss: 3.307226 | lr: 5.3239e-04 | norm: 0.3150, dt: 449.92ms, tok/sec: 1165303.50
step: 4943 | loss: 3.407981 | lr: 5.3236e-04 | norm: 0.3138, dt: 451.18ms, tok/sec: 1162031.85
step: 4944 | loss: 3.346992 | lr: 5.3233e-04 | norm: 0.3322, dt: 451.02ms, tok/sec: 1162461.84
step: 4945 | loss: 3.344680 | lr: 5.3230e-04 | norm: 0.3075, dt: 450.22ms, tok/sec: 1164517.31
step: 4946 | loss: 3.367472 | lr: 5.3227e-04 | norm: 0.2801, dt: 449.68ms, tok/sec: 1165915.16
step: 4947 | loss: 3.335214 | lr: 5.3223e-04 | norm: 0.2839, dt: 450.24ms, tok/sec: 1164463.05
step: 4948 | loss: 3.343384 | lr: 5.3220e-04 | norm: 0.2833, dt: 451.63ms, tok/sec: 1160886.54
step: 4949 | loss: 3.335132 | lr: 5.3217e-04 | norm: 0.2629, dt: 451.14ms, tok/sec: 1162143.00
step: 4950 | loss: 3.351696 | lr: 5.3214e-04 | norm: 0.2515, dt: 450.17ms, tok/sec: 1164648.68
step: 4951 | loss: 3.421101 | lr: 5.3211e-04 | norm: 0.2758, dt: 450.21ms, tok/sec: 1164550.00
step: 4952 | loss: 3.315331 | lr: 5.3208e-04 | norm: 0.2737, dt: 450.77ms, tok/sec: 1163106.20
step: 4953 | loss: 3.431520 | lr: 5.3205e-04 | norm: 0.2614, dt: 450.18ms, tok/sec: 1164611.06
step: 4954 | loss: 3.446798 | lr: 5.3202e-04 | norm: 0.2427, dt: 450.19ms, tok/sec: 1164582.07
step: 4955 | loss: 3.336287 | lr: 5.3199e-04 | norm: 0.2666, dt: 450.35ms, tok/sec: 1164177.00
step: 4956 | loss: 3.322745 | lr: 5.3196e-04 | norm: 0.2608, dt: 451.52ms, tok/sec: 1161172.19
step: 4957 | loss: 3.370085 | lr: 5.3193e-04 | norm: 0.2748, dt: 450.27ms, tok/sec: 1164395.22
step: 4958 | loss: 3.383174 | lr: 5.3190e-04 | norm: 0.2743, dt: 450.29ms, tok/sec: 1164345.90
step: 4959 | loss: 3.366214 | lr: 5.3187e-04 | norm: 0.2624, dt: 450.69ms, tok/sec: 1163309.25
step: 4960 | loss: 3.349872 | lr: 5.3184e-04 | norm: 0.2533, dt: 450.94ms, tok/sec: 1162651.76
step: 4961 | loss: 3.292830 | lr: 5.3181e-04 | norm: 0.2426, dt: 450.87ms, tok/sec: 1162828.20
step: 4962 | loss: 3.374649 | lr: 5.3177e-04 | norm: 0.2361, dt: 451.12ms, tok/sec: 1162181.69
step: 4963 | loss: 3.342608 | lr: 5.3174e-04 | norm: 0.2428, dt: 450.49ms, tok/sec: 1163808.56
step: 4964 | loss: 3.392802 | lr: 5.3171e-04 | norm: 0.2382, dt: 450.23ms, tok/sec: 1164498.20
step: 4965 | loss: 3.326619 | lr: 5.3168e-04 | norm: 0.2605, dt: 450.97ms, tok/sec: 1162579.84
step: 4966 | loss: 3.222096 | lr: 5.3165e-04 | norm: 0.2548, dt: 450.76ms, tok/sec: 1163107.43
step: 4967 | loss: 3.187553 | lr: 5.3162e-04 | norm: 0.2721, dt: 451.93ms, tok/sec: 1160102.02
step: 4968 | loss: 3.214248 | lr: 5.3159e-04 | norm: 0.2746, dt: 449.32ms, tok/sec: 1166859.86
step: 4969 | loss: 3.254416 | lr: 5.3156e-04 | norm: 0.2708, dt: 451.30ms, tok/sec: 1161740.86
step: 4970 | loss: 3.192532 | lr: 5.3153e-04 | norm: 0.3034, dt: 450.68ms, tok/sec: 1163314.79
step: 4971 | loss: 3.153435 | lr: 5.3150e-04 | norm: 0.2949, dt: 450.98ms, tok/sec: 1162549.11
step: 4972 | loss: 3.258007 | lr: 5.3147e-04 | norm: 0.3037, dt: 451.29ms, tok/sec: 1161748.84
step: 4973 | loss: 3.199833 | lr: 5.3144e-04 | norm: 0.3525, dt: 450.59ms, tok/sec: 1163562.24
step: 4974 | loss: 3.259788 | lr: 5.3141e-04 | norm: 0.2989, dt: 451.27ms, tok/sec: 1161797.94
step: 4975 | loss: 3.135572 | lr: 5.3138e-04 | norm: 0.2667, dt: 451.30ms, tok/sec: 1161727.97
step: 4976 | loss: 3.227151 | lr: 5.3134e-04 | norm: 0.2603, dt: 450.47ms, tok/sec: 1163881.24
step: 4977 | loss: 3.278779 | lr: 5.3131e-04 | norm: 0.2913, dt: 451.02ms, tok/sec: 1162437.26
step: 4978 | loss: 3.396759 | lr: 5.3128e-04 | norm: 0.2625, dt: 451.72ms, tok/sec: 1160642.68
step: 4979 | loss: 3.413281 | lr: 5.3125e-04 | norm: 0.2890, dt: 450.09ms, tok/sec: 1164859.67
step: 4980 | loss: 3.334194 | lr: 5.3122e-04 | norm: 0.2884, dt: 450.52ms, tok/sec: 1163750.66
step: 4981 | loss: 3.357352 | lr: 5.3119e-04 | norm: 0.2578, dt: 450.13ms, tok/sec: 1164756.64
step: 4982 | loss: 3.450308 | lr: 5.3116e-04 | norm: 0.2689, dt: 451.63ms, tok/sec: 1160874.89
step: 4983 | loss: 3.377475 | lr: 5.3113e-04 | norm: 0.3208, dt: 451.32ms, tok/sec: 1161677.65
step: 4984 | loss: 3.379990 | lr: 5.3110e-04 | norm: 0.3090, dt: 449.89ms, tok/sec: 1165376.99
step: 4985 | loss: 3.382246 | lr: 5.3107e-04 | norm: 0.2757, dt: 450.47ms, tok/sec: 1163861.53
step: 4986 | loss: 3.312169 | lr: 5.3104e-04 | norm: 0.2896, dt: 450.87ms, tok/sec: 1162846.65
step: 4987 | loss: 3.254163 | lr: 5.3101e-04 | norm: 0.2705, dt: 451.04ms, tok/sec: 1162391.18
step: 4988 | loss: 3.343851 | lr: 5.3097e-04 | norm: 0.2572, dt: 451.61ms, tok/sec: 1160933.73
step: 4989 | loss: 3.331480 | lr: 5.3094e-04 | norm: 0.2592, dt: 451.18ms, tok/sec: 1162034.30
step: 4990 | loss: 3.307742 | lr: 5.3091e-04 | norm: 0.2712, dt: 450.55ms, tok/sec: 1163662.60
step: 4991 | loss: 3.315286 | lr: 5.3088e-04 | norm: 0.2392, dt: 450.44ms, tok/sec: 1163942.85
step: 4992 | loss: 3.353068 | lr: 5.3085e-04 | norm: 0.2673, dt: 453.04ms, tok/sec: 1157277.11
step: 4993 | loss: 3.363940 | lr: 5.3082e-04 | norm: 0.2478, dt: 451.71ms, tok/sec: 1160681.27
step: 4994 | loss: 3.325217 | lr: 5.3079e-04 | norm: 0.2739, dt: 450.29ms, tok/sec: 1164331.72
step: 4995 | loss: 3.358217 | lr: 5.3076e-04 | norm: 0.3016, dt: 450.85ms, tok/sec: 1162886.62
step: 4996 | loss: 3.338652 | lr: 5.3073e-04 | norm: 0.2811, dt: 451.35ms, tok/sec: 1161598.49
step: 4997 | loss: 3.295882 | lr: 5.3070e-04 | norm: 0.2604, dt: 450.71ms, tok/sec: 1163241.56
step: 4998 | loss: 3.398284 | lr: 5.3067e-04 | norm: 0.2606, dt: 451.39ms, tok/sec: 1161487.44
step: 4999 | loss: 3.380265 | lr: 5.3063e-04 | norm: 0.2668, dt: 451.64ms, tok/sec: 1160849.77
validation loss: 3.3758
HellaSwag accuracy: 2785/10042=0.2773
rank 6 sample 0: Hello, I'm a Manpreet, it's a way we interact with the people around us and our minds, as well as with our physical environment.
rank 6 sample 1: Hello, I'm a Manpreet, my favorite of all people, of course who do the most.
If you're in a wheelchair and want to
rank 6 sample 2: Hello, I'm a Manpreet, a woman who has done all I can to please the Lord of the Rings in the story.
For many stories
rank 6 sample 3: Hello, I'm a Manpreet, and I'll leave him to get back.
I'm a Nanny! I've given him $50;
rank 2 sample 0: Hello, I'm a Manpreet, am the first guy to meet you today, I'm an Indian.
He had a hard time with your family
rank 2 sample 1: Hello, I'm a Manpreet, we're moving forward. I'm standing nearby, listening to a man-of-the-street.
So
rank 2 sample 2: Hello, I'm a Manpreet, I'm the person who has just taken me to the front seat, it's all a bit like "I'm
rank 2 sample 3: Hello, I'm a Manpreet, a nice guest.<|endoftext|>Today, the government of Bangladesh has issued a new legal requirement on legal education.
In
rank 4 sample 0: Hello, I'm a Manpreet, I'm also a Friend. I am always a Helping someone. When I saw her with my back legs and
rank 4 sample 1: Hello, I'm a Manpreet, now that I thought that I thought I had done this the next time I'd found you a little something interesting this
rank 4 sample 2: Hello, I'm a Manpreet, but remember I made this with a lot of extra help. You know, I make my feet so my feet and
rank 4 sample 3: Hello, I'm a Manpreet, and it reads
Hello, I'm a Red Pen. Here is an image of the book called
The text
rank 0 sample 0: Hello, I'm a Manpreet, and I just want you to take out each. I have to show you some of you have got the basics.
rank 0 sample 1: Hello, I'm a Manpreet, I do not know the name of Manpreet." When we see God as the Father of all we are,
rank 0 sample 2: Hello, I'm a Manpreet, I'm a Smart Company Man, I'm a Smart Company Smart Group Smart Group Smart Group Smart Group Smart Group Smart
rank 0 sample 3: Hello, I'm a Manpreet, my Mom, and I'm the person to walk my dog. I was the "cow" to make my home
rank 7 sample 0: Hello, I'm a Manpreet, and I will be there as a couple moments later.
For the beginning of writing my essays I would write a
rank 7 sample 1: Hello, I'm a Manpreet, don't forget to add an email note (in Arabic on the right side of the page, above you, and
rank 7 sample 2: Hello, I'm a Manpreet, not a man) and I mean no more. Maybe I can make things better. I can make things better or
rank 7 sample 3: Hello, I'm a Manpreet, Mandy, and I think it goes pretty deep into the very basic.
- If you see any of these
rank 5 sample 0: Hello, I'm a Manpreet, a small child. The other day, my mother asked my family questions:
- What is your home?

rank 5 sample 1: Hello, I'm a Manpreet, there is a little red circle at the top that looks like I would be very sad if you got out at night
rank 5 sample 2: Hello, I'm a Manpreet, I am really excited or excited. If you look at the picture from the moment, think about the way you are
rank 5 sample 3: Hello, I'm a Manpreet, here are a few links with some nice links for other users. Here are some things that you also should know:
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going to show you, how I'll talk to you first one time with this "chimpy
rank 3 sample 1: Hello, I'm a Manpreet, and my wife is a Marit. I love this girl.
I have a son, and I am also
rank 3 sample 2: Hello, I'm a Manpreet, and you have two women. This is going to be the day when you get in touch with what is going on
rank 3 sample 3: Hello, I'm a Manpreet, and want to talk about it again. I have no problem.
(I agree with the folks...) I have
rank 1 sample 0: Hello, I'm a Manpreet, just kidding me. Let me explain (a) the word here.
Let's take a picture of my dad
rank 1 sample 1: Hello, I'm a Manpreet, so you know why I'm writing that because I know when I don't understand, like, where do I start
rank 1 sample 2: Hello, I'm a Manpreet, a Friend of the World. I'm a Manpreet I've met people who are really important, and have
rank 1 sample 3: Hello, I'm a Manpreet, and I'm my little granddaughter. The "I" signs in the center:
What my little granddaughter is trying
step: 5000 | loss: 3.401085 | lr: 5.3060e-04 | norm: 0.2610, dt: 13111.29ms, tok/sec: 39987.52
step: 5001 | loss: 3.334237 | lr: 5.3057e-04 | norm: 0.2531, dt: 448.53ms, tok/sec: 1168914.15
step: 5002 | loss: 3.327708 | lr: 5.3054e-04 | norm: 0.2468, dt: 452.18ms, tok/sec: 1159464.03
step: 5003 | loss: 3.363178 | lr: 5.3051e-04 | norm: 0.3268, dt: 449.97ms, tok/sec: 1165154.08
step: 5004 | loss: 3.315080 | lr: 5.3048e-04 | norm: 0.3100, dt: 449.53ms, tok/sec: 1166296.07
step: 5005 | loss: 3.376943 | lr: 5.3045e-04 | norm: 0.2901, dt: 451.04ms, tok/sec: 1162407.77
step: 5006 | loss: 3.266878 | lr: 5.3042e-04 | norm: 0.2651, dt: 450.22ms, tok/sec: 1164511.76
step: 5007 | loss: 3.317916 | lr: 5.3039e-04 | norm: 0.2757, dt: 450.19ms, tok/sec: 1164593.79
step: 5008 | loss: 3.316616 | lr: 5.3036e-04 | norm: 0.2845, dt: 451.36ms, tok/sec: 1161562.29
step: 5009 | loss: 3.383582 | lr: 5.3033e-04 | norm: 0.2619, dt: 450.71ms, tok/sec: 1163257.56
step: 5010 | loss: 3.354852 | lr: 5.3029e-04 | norm: 0.2551, dt: 451.16ms, tok/sec: 1162079.74
step: 5011 | loss: 3.340352 | lr: 5.3026e-04 | norm: 0.2749, dt: 450.77ms, tok/sec: 1163095.13
step: 5012 | loss: 3.281188 | lr: 5.3023e-04 | norm: 0.2893, dt: 449.70ms, tok/sec: 1165858.29
step: 5013 | loss: 3.176012 | lr: 5.3020e-04 | norm: 0.2690, dt: 451.25ms, tok/sec: 1161856.25
step: 5014 | loss: 3.150098 | lr: 5.3017e-04 | norm: 0.2918, dt: 451.71ms, tok/sec: 1160662.89
step: 5015 | loss: 3.234979 | lr: 5.3014e-04 | norm: 0.3465, dt: 450.12ms, tok/sec: 1164780.70
step: 5016 | loss: 3.156179 | lr: 5.3011e-04 | norm: 0.2938, dt: 450.47ms, tok/sec: 1163859.07
step: 5017 | loss: 3.189070 | lr: 5.3008e-04 | norm: 0.2903, dt: 450.37ms, tok/sec: 1164137.56
step: 5018 | loss: 3.174704 | lr: 5.3005e-04 | norm: 0.2639, dt: 452.46ms, tok/sec: 1158751.04
step: 5019 | loss: 3.211491 | lr: 5.3002e-04 | norm: 0.2658, dt: 450.93ms, tok/sec: 1162679.42
step: 5020 | loss: 3.209285 | lr: 5.2998e-04 | norm: 0.2702, dt: 451.29ms, tok/sec: 1161749.45
step: 5021 | loss: 3.182538 | lr: 5.2995e-04 | norm: 0.2856, dt: 451.04ms, tok/sec: 1162402.24
step: 5022 | loss: 3.166547 | lr: 5.2992e-04 | norm: 0.2472, dt: 450.96ms, tok/sec: 1162601.35
step: 5023 | loss: 3.192820 | lr: 5.2989e-04 | norm: 0.2447, dt: 451.26ms, tok/sec: 1161836.61
step: 5024 | loss: 3.351800 | lr: 5.2986e-04 | norm: 0.2386, dt: 451.84ms, tok/sec: 1160345.04
step: 5025 | loss: 3.402353 | lr: 5.2983e-04 | norm: 0.2610, dt: 450.73ms, tok/sec: 1163190.49
step: 5026 | loss: 3.413785 | lr: 5.2980e-04 | norm: 0.2582, dt: 451.84ms, tok/sec: 1160349.32
step: 5027 | loss: 3.399245 | lr: 5.2977e-04 | norm: 0.2527, dt: 452.08ms, tok/sec: 1159722.08
step: 5028 | loss: 3.396433 | lr: 5.2974e-04 | norm: 0.2448, dt: 451.37ms, tok/sec: 1161556.76
step: 5029 | loss: 3.392281 | lr: 5.2970e-04 | norm: 0.2676, dt: 451.15ms, tok/sec: 1162103.69
step: 5030 | loss: 3.325837 | lr: 5.2967e-04 | norm: 0.2465, dt: 450.26ms, tok/sec: 1164400.77
step: 5031 | loss: 3.337132 | lr: 5.2964e-04 | norm: 0.2535, dt: 450.56ms, tok/sec: 1163639.20
step: 5032 | loss: 3.344317 | lr: 5.2961e-04 | norm: 0.2496, dt: 451.12ms, tok/sec: 1162203.19
step: 5033 | loss: 3.354019 | lr: 5.2958e-04 | norm: 0.2666, dt: 451.51ms, tok/sec: 1161182.62
step: 5034 | loss: 3.303416 | lr: 5.2955e-04 | norm: 0.2395, dt: 451.31ms, tok/sec: 1161698.51
step: 5035 | loss: 3.287187 | lr: 5.2952e-04 | norm: 0.2601, dt: 450.75ms, tok/sec: 1163141.89
step: 5036 | loss: 3.308082 | lr: 5.2949e-04 | norm: 0.2686, dt: 450.99ms, tok/sec: 1162531.29
step: 5037 | loss: 3.280486 | lr: 5.2946e-04 | norm: 0.2686, dt: 451.37ms, tok/sec: 1161543.27
step: 5038 | loss: 3.294740 | lr: 5.2942e-04 | norm: 0.2661, dt: 450.70ms, tok/sec: 1163269.87
step: 5039 | loss: 3.309062 | lr: 5.2939e-04 | norm: 0.2855, dt: 452.45ms, tok/sec: 1158780.96
step: 5040 | loss: 3.326967 | lr: 5.2936e-04 | norm: 0.2961, dt: 452.52ms, tok/sec: 1158595.36
step: 5041 | loss: 3.377445 | lr: 5.2933e-04 | norm: 0.3195, dt: 450.42ms, tok/sec: 1164008.16
step: 5042 | loss: 3.323047 | lr: 5.2930e-04 | norm: 0.2900, dt: 451.23ms, tok/sec: 1161904.75
step: 5043 | loss: 3.336267 | lr: 5.2927e-04 | norm: 0.2953, dt: 452.36ms, tok/sec: 1159011.82
step: 5044 | loss: 3.330547 | lr: 5.2924e-04 | norm: 0.3367, dt: 450.70ms, tok/sec: 1163271.71
step: 5045 | loss: 3.467733 | lr: 5.2921e-04 | norm: 0.3103, dt: 450.12ms, tok/sec: 1164773.29
step: 5046 | loss: 3.400754 | lr: 5.2917e-04 | norm: 0.3075, dt: 450.79ms, tok/sec: 1163047.15
step: 5047 | loss: 3.346490 | lr: 5.2914e-04 | norm: 0.2885, dt: 450.68ms, tok/sec: 1163336.33
step: 5048 | loss: 3.328756 | lr: 5.2911e-04 | norm: 0.2907, dt: 451.15ms, tok/sec: 1162113.52
step: 5049 | loss: 3.328114 | lr: 5.2908e-04 | norm: 0.2460, dt: 450.75ms, tok/sec: 1163141.27
step: 5050 | loss: 3.340080 | lr: 5.2905e-04 | norm: 0.2704, dt: 451.62ms, tok/sec: 1160912.28
step: 5051 | loss: 3.369233 | lr: 5.2902e-04 | norm: 0.2659, dt: 450.72ms, tok/sec: 1163225.56
step: 5052 | loss: 3.326200 | lr: 5.2899e-04 | norm: 0.2552, dt: 450.22ms, tok/sec: 1164520.40
step: 5053 | loss: 3.367879 | lr: 5.2896e-04 | norm: 0.2638, dt: 452.72ms, tok/sec: 1158088.92
step: 5054 | loss: 3.399514 | lr: 5.2893e-04 | norm: 0.2420, dt: 450.35ms, tok/sec: 1164167.76
step: 5055 | loss: 3.359003 | lr: 5.2889e-04 | norm: 0.2947, dt: 451.12ms, tok/sec: 1162197.66
step: 5056 | loss: 3.354017 | lr: 5.2886e-04 | norm: 0.2472, dt: 450.96ms, tok/sec: 1162593.98
step: 5057 | loss: 3.299407 | lr: 5.2883e-04 | norm: 0.3013, dt: 452.04ms, tok/sec: 1159823.00
step: 5058 | loss: 3.376764 | lr: 5.2880e-04 | norm: 0.2569, dt: 450.18ms, tok/sec: 1164620.93
step: 5059 | loss: 3.217420 | lr: 5.2877e-04 | norm: 0.2940, dt: 450.54ms, tok/sec: 1163700.78
step: 5060 | loss: 3.124826 | lr: 5.2874e-04 | norm: 0.3502, dt: 450.53ms, tok/sec: 1163707.56
step: 5061 | loss: 3.183780 | lr: 5.2871e-04 | norm: 0.3393, dt: 450.87ms, tok/sec: 1162848.50
step: 5062 | loss: 3.218807 | lr: 5.2868e-04 | norm: 0.3244, dt: 453.00ms, tok/sec: 1157361.17
step: 5063 | loss: 3.212620 | lr: 5.2864e-04 | norm: 0.3300, dt: 450.39ms, tok/sec: 1164082.10
step: 5064 | loss: 3.229141 | lr: 5.2861e-04 | norm: 0.2832, dt: 450.69ms, tok/sec: 1163292.64
step: 5065 | loss: 3.179899 | lr: 5.2858e-04 | norm: 0.2783, dt: 450.44ms, tok/sec: 1163945.31
step: 5066 | loss: 3.172311 | lr: 5.2855e-04 | norm: 0.2604, dt: 450.66ms, tok/sec: 1163386.80
step: 5067 | loss: 3.204646 | lr: 5.2852e-04 | norm: 0.2569, dt: 451.52ms, tok/sec: 1161151.96
step: 5068 | loss: 3.201993 | lr: 5.2849e-04 | norm: 0.2592, dt: 451.40ms, tok/sec: 1161473.33
step: 5069 | loss: 3.202165 | lr: 5.2846e-04 | norm: 0.2575, dt: 451.35ms, tok/sec: 1161599.10
step: 5070 | loss: 3.318339 | lr: 5.2842e-04 | norm: 0.2419, dt: 451.61ms, tok/sec: 1160934.34
step: 5071 | loss: 3.343564 | lr: 5.2839e-04 | norm: 0.2570, dt: 451.25ms, tok/sec: 1161861.16
step: 5072 | loss: 3.358897 | lr: 5.2836e-04 | norm: 0.2553, dt: 450.79ms, tok/sec: 1163039.15
step: 5073 | loss: 3.364331 | lr: 5.2833e-04 | norm: 0.2826, dt: 450.75ms, tok/sec: 1163141.27
step: 5074 | loss: 3.377054 | lr: 5.2830e-04 | norm: 0.2787, dt: 451.09ms, tok/sec: 1162278.75
step: 5075 | loss: 3.381500 | lr: 5.2827e-04 | norm: 0.2728, dt: 452.90ms, tok/sec: 1157612.79
step: 5076 | loss: 3.398209 | lr: 5.2824e-04 | norm: 0.2763, dt: 450.62ms, tok/sec: 1163484.05
step: 5077 | loss: 3.375703 | lr: 5.2821e-04 | norm: 0.2838, dt: 450.40ms, tok/sec: 1164045.13
step: 5078 | loss: 3.393348 | lr: 5.2817e-04 | norm: 0.2869, dt: 451.98ms, tok/sec: 1159992.48
step: 5079 | loss: 3.322505 | lr: 5.2814e-04 | norm: 0.2788, dt: 450.62ms, tok/sec: 1163480.36
step: 5080 | loss: 3.365627 | lr: 5.2811e-04 | norm: 0.2776, dt: 449.86ms, tok/sec: 1165448.63
step: 5081 | loss: 3.352437 | lr: 5.2808e-04 | norm: 0.2958, dt: 449.93ms, tok/sec: 1165255.33
step: 5082 | loss: 3.323109 | lr: 5.2805e-04 | norm: 0.2785, dt: 451.46ms, tok/sec: 1161323.66
step: 5083 | loss: 3.363293 | lr: 5.2802e-04 | norm: 0.2800, dt: 450.94ms, tok/sec: 1162667.12
step: 5084 | loss: 3.339532 | lr: 5.2799e-04 | norm: 0.2647, dt: 449.59ms, tok/sec: 1166144.54
step: 5085 | loss: 3.294032 | lr: 5.2795e-04 | norm: 0.2719, dt: 451.33ms, tok/sec: 1161656.17
step: 5086 | loss: 3.322309 | lr: 5.2792e-04 | norm: 0.2700, dt: 450.33ms, tok/sec: 1164228.78
step: 5087 | loss: 3.324228 | lr: 5.2789e-04 | norm: 0.3030, dt: 450.45ms, tok/sec: 1163931.76
step: 5088 | loss: 3.367735 | lr: 5.2786e-04 | norm: 0.3265, dt: 450.45ms, tok/sec: 1163923.13
step: 5089 | loss: 3.309884 | lr: 5.2783e-04 | norm: 0.3301, dt: 451.80ms, tok/sec: 1160441.17
step: 5090 | loss: 3.363000 | lr: 5.2780e-04 | norm: 0.3074, dt: 450.93ms, tok/sec: 1162692.33
step: 5091 | loss: 3.389374 | lr: 5.2777e-04 | norm: 0.3110, dt: 450.82ms, tok/sec: 1162974.57
step: 5092 | loss: 3.455484 | lr: 5.2773e-04 | norm: 0.3459, dt: 450.98ms, tok/sec: 1162546.65
step: 5093 | loss: 3.437492 | lr: 5.2770e-04 | norm: 0.3190, dt: 451.43ms, tok/sec: 1161384.38
step: 5094 | loss: 3.350065 | lr: 5.2767e-04 | norm: 0.2743, dt: 450.27ms, tok/sec: 1164374.88
step: 5095 | loss: 3.428435 | lr: 5.2764e-04 | norm: 0.3205, dt: 450.95ms, tok/sec: 1162624.10
step: 5096 | loss: 3.336550 | lr: 5.2761e-04 | norm: 0.3077, dt: 450.57ms, tok/sec: 1163615.80
step: 5097 | loss: 3.358899 | lr: 5.2758e-04 | norm: 0.2647, dt: 450.28ms, tok/sec: 1164347.75
step: 5098 | loss: 3.323199 | lr: 5.2754e-04 | norm: 0.2587, dt: 449.70ms, tok/sec: 1165865.71
step: 5099 | loss: 3.382118 | lr: 5.2751e-04 | norm: 0.2519, dt: 451.25ms, tok/sec: 1161868.53
step: 5100 | loss: 3.389303 | lr: 5.2748e-04 | norm: 0.2499, dt: 451.79ms, tok/sec: 1160480.36
step: 5101 | loss: 3.317631 | lr: 5.2745e-04 | norm: 0.2569, dt: 451.19ms, tok/sec: 1162011.58
step: 5102 | loss: 3.348185 | lr: 5.2742e-04 | norm: 0.2628, dt: 1154.72ms, tok/sec: 454039.24
step: 5103 | loss: 3.322260 | lr: 5.2739e-04 | norm: 0.2455, dt: 451.21ms, tok/sec: 1161950.80
step: 5104 | loss: 3.340887 | lr: 5.2736e-04 | norm: 0.2427, dt: 452.49ms, tok/sec: 1158669.22
step: 5105 | loss: 3.283710 | lr: 5.2732e-04 | norm: 0.2682, dt: 451.22ms, tok/sec: 1161940.97
step: 5106 | loss: 3.252192 | lr: 5.2729e-04 | norm: 0.2523, dt: 449.71ms, tok/sec: 1165846.55
step: 5107 | loss: 3.209324 | lr: 5.2726e-04 | norm: 0.2702, dt: 450.99ms, tok/sec: 1162518.99
step: 5108 | loss: 3.161712 | lr: 5.2723e-04 | norm: 0.2638, dt: 451.01ms, tok/sec: 1162471.06
step: 5109 | loss: 3.130633 | lr: 5.2720e-04 | norm: 0.2697, dt: 451.28ms, tok/sec: 1161767.86
step: 5110 | loss: 3.168318 | lr: 5.2717e-04 | norm: 0.2577, dt: 449.88ms, tok/sec: 1165383.78
step: 5111 | loss: 3.148049 | lr: 5.2714e-04 | norm: 0.2357, dt: 451.30ms, tok/sec: 1161735.95
step: 5112 | loss: 3.132148 | lr: 5.2710e-04 | norm: 0.2679, dt: 451.92ms, tok/sec: 1160137.51
step: 5113 | loss: 3.183954 | lr: 5.2707e-04 | norm: 0.2469, dt: 451.32ms, tok/sec: 1161667.21
step: 5114 | loss: 3.185497 | lr: 5.2704e-04 | norm: 0.2703, dt: 451.23ms, tok/sec: 1161912.12
step: 5115 | loss: 3.127880 | lr: 5.2701e-04 | norm: 0.2902, dt: 450.93ms, tok/sec: 1162682.49
step: 5116 | loss: 3.189256 | lr: 5.2698e-04 | norm: 0.2789, dt: 451.09ms, tok/sec: 1162278.13
step: 5117 | loss: 3.356457 | lr: 5.2695e-04 | norm: 0.2894, dt: 451.21ms, tok/sec: 1161959.39
step: 5118 | loss: 3.361002 | lr: 5.2691e-04 | norm: 0.2934, dt: 451.23ms, tok/sec: 1161912.73
step: 5119 | loss: 3.378563 | lr: 5.2688e-04 | norm: 0.2655, dt: 450.48ms, tok/sec: 1163854.76
step: 5120 | loss: 3.358060 | lr: 5.2685e-04 | norm: 0.3063, dt: 450.82ms, tok/sec: 1162968.42
step: 5121 | loss: 3.328424 | lr: 5.2682e-04 | norm: 0.2797, dt: 451.07ms, tok/sec: 1162329.12
step: 5122 | loss: 3.355894 | lr: 5.2679e-04 | norm: 0.2683, dt: 452.20ms, tok/sec: 1159423.08
step: 5123 | loss: 3.319566 | lr: 5.2676e-04 | norm: 0.2611, dt: 451.03ms, tok/sec: 1162420.67
step: 5124 | loss: 3.276186 | lr: 5.2672e-04 | norm: 0.2661, dt: 450.61ms, tok/sec: 1163500.06
step: 5125 | loss: 3.301545 | lr: 5.2669e-04 | norm: 0.2809, dt: 451.05ms, tok/sec: 1162377.66
step: 5126 | loss: 3.312008 | lr: 5.2666e-04 | norm: 0.2475, dt: 451.04ms, tok/sec: 1162399.17
step: 5127 | loss: 3.247681 | lr: 5.2663e-04 | norm: 0.2670, dt: 458.68ms, tok/sec: 1143024.28
step: 5128 | loss: 3.317564 | lr: 5.2660e-04 | norm: 0.2520, dt: 450.96ms, tok/sec: 1162614.88
step: 5129 | loss: 3.332891 | lr: 5.2657e-04 | norm: 0.3019, dt: 1228.41ms, tok/sec: 426802.46
step: 5130 | loss: 3.363081 | lr: 5.2653e-04 | norm: 0.2747, dt: 453.44ms, tok/sec: 1156235.99
step: 5131 | loss: 3.296459 | lr: 5.2650e-04 | norm: 0.2583, dt: 450.91ms, tok/sec: 1162729.22
step: 5132 | loss: 3.294258 | lr: 5.2647e-04 | norm: 0.2522, dt: 451.27ms, tok/sec: 1161801.62
step: 5133 | loss: 3.266912 | lr: 5.2644e-04 | norm: 0.2516, dt: 451.80ms, tok/sec: 1160453.42
step: 5134 | loss: 3.311273 | lr: 5.2641e-04 | norm: 0.2497, dt: 450.61ms, tok/sec: 1163497.60
step: 5135 | loss: 3.371062 | lr: 5.2638e-04 | norm: 0.2857, dt: 450.62ms, tok/sec: 1163468.66
step: 5136 | loss: 3.435303 | lr: 5.2634e-04 | norm: 0.2568, dt: 449.74ms, tok/sec: 1165750.13
step: 5137 | loss: 3.385277 | lr: 5.2631e-04 | norm: 0.2366, dt: 449.59ms, tok/sec: 1166155.06
step: 5138 | loss: 3.397988 | lr: 5.2628e-04 | norm: 0.2684, dt: 451.36ms, tok/sec: 1161562.90
step: 5139 | loss: 3.435699 | lr: 5.2625e-04 | norm: 0.2576, dt: 451.44ms, tok/sec: 1161358.62
step: 5140 | loss: 3.340165 | lr: 5.2622e-04 | norm: 0.2739, dt: 450.63ms, tok/sec: 1163446.50
step: 5141 | loss: 3.358114 | lr: 5.2619e-04 | norm: 0.2351, dt: 451.11ms, tok/sec: 1162208.72
step: 5142 | loss: 3.322179 | lr: 5.2615e-04 | norm: 0.2563, dt: 451.36ms, tok/sec: 1161570.26
step: 5143 | loss: 3.318787 | lr: 5.2612e-04 | norm: 0.2695, dt: 450.50ms, tok/sec: 1163803.02
step: 5144 | loss: 3.321272 | lr: 5.2609e-04 | norm: 0.2477, dt: 451.15ms, tok/sec: 1162107.99
step: 5145 | loss: 3.321285 | lr: 5.2606e-04 | norm: 0.2706, dt: 450.96ms, tok/sec: 1162615.49
step: 5146 | loss: 3.307249 | lr: 5.2603e-04 | norm: 0.2725, dt: 451.00ms, tok/sec: 1162509.16
step: 5147 | loss: 3.343301 | lr: 5.2599e-04 | norm: 0.2722, dt: 450.52ms, tok/sec: 1163740.19
step: 5148 | loss: 3.356038 | lr: 5.2596e-04 | norm: 0.2503, dt: 450.23ms, tok/sec: 1164487.71
step: 5149 | loss: 3.328714 | lr: 5.2593e-04 | norm: 0.2737, dt: 450.17ms, tok/sec: 1164634.49
step: 5150 | loss: 3.307648 | lr: 5.2590e-04 | norm: 0.2843, dt: 450.91ms, tok/sec: 1162741.51
step: 5151 | loss: 3.223272 | lr: 5.2587e-04 | norm: 0.2739, dt: 450.60ms, tok/sec: 1163540.07
step: 5152 | loss: 3.186643 | lr: 5.2584e-04 | norm: 0.2944, dt: 452.30ms, tok/sec: 1159165.78
step: 5153 | loss: 3.210621 | lr: 5.2580e-04 | norm: 0.2755, dt: 451.71ms, tok/sec: 1160673.92
step: 5154 | loss: 3.290100 | lr: 5.2577e-04 | norm: 0.2864, dt: 451.50ms, tok/sec: 1161213.89
step: 5155 | loss: 3.117310 | lr: 5.2574e-04 | norm: 0.3160, dt: 452.41ms, tok/sec: 1158865.23
step: 5156 | loss: 3.186776 | lr: 5.2571e-04 | norm: 0.2864, dt: 451.53ms, tok/sec: 1161137.86
step: 5157 | loss: 3.197452 | lr: 5.2568e-04 | norm: 0.2433, dt: 452.03ms, tok/sec: 1159839.52
step: 5158 | loss: 3.116206 | lr: 5.2564e-04 | norm: 0.2657, dt: 451.24ms, tok/sec: 1161890.63
step: 5159 | loss: 3.217752 | lr: 5.2561e-04 | norm: 0.2826, dt: 451.57ms, tok/sec: 1161040.38
step: 5160 | loss: 3.227997 | lr: 5.2558e-04 | norm: 0.2485, dt: 450.84ms, tok/sec: 1162924.75
step: 5161 | loss: 3.205605 | lr: 5.2555e-04 | norm: 0.2516, dt: 453.59ms, tok/sec: 1155871.34
step: 5162 | loss: 3.280996 | lr: 5.2552e-04 | norm: 0.2453, dt: 451.86ms, tok/sec: 1160289.94
step: 5163 | loss: 3.319395 | lr: 5.2549e-04 | norm: 0.2663, dt: 451.43ms, tok/sec: 1161401.56
step: 5164 | loss: 3.376490 | lr: 5.2545e-04 | norm: 0.2934, dt: 452.87ms, tok/sec: 1157695.68
step: 5165 | loss: 3.311905 | lr: 5.2542e-04 | norm: 0.3218, dt: 452.04ms, tok/sec: 1159822.39
step: 5166 | loss: 3.309469 | lr: 5.2539e-04 | norm: 0.3341, dt: 450.98ms, tok/sec: 1162542.96
step: 5167 | loss: 3.357188 | lr: 5.2536e-04 | norm: 0.2941, dt: 451.33ms, tok/sec: 1161646.35
step: 5168 | loss: 3.345876 | lr: 5.2533e-04 | norm: 0.2469, dt: 453.01ms, tok/sec: 1157336.19
step: 5169 | loss: 3.320444 | lr: 5.2529e-04 | norm: 0.2926, dt: 449.89ms, tok/sec: 1165359.69
step: 5170 | loss: 3.348174 | lr: 5.2526e-04 | norm: 0.2441, dt: 451.55ms, tok/sec: 1161075.94
step: 5171 | loss: 3.382638 | lr: 5.2523e-04 | norm: 0.2545, dt: 451.73ms, tok/sec: 1160630.43
step: 5172 | loss: 3.436260 | lr: 5.2520e-04 | norm: 0.3918, dt: 449.53ms, tok/sec: 1166292.98
step: 5173 | loss: 3.263898 | lr: 5.2517e-04 | norm: 0.3092, dt: 450.96ms, tok/sec: 1162598.89
step: 5174 | loss: 3.364241 | lr: 5.2513e-04 | norm: 0.2900, dt: 451.41ms, tok/sec: 1161453.08
step: 5175 | loss: 3.324357 | lr: 5.2510e-04 | norm: 0.2763, dt: 450.81ms, tok/sec: 1162988.10
step: 5176 | loss: 3.315996 | lr: 5.2507e-04 | norm: 0.2562, dt: 450.82ms, tok/sec: 1162970.88
step: 5177 | loss: 3.279160 | lr: 5.2504e-04 | norm: 0.2511, dt: 450.13ms, tok/sec: 1164756.02
step: 5178 | loss: 3.266330 | lr: 5.2501e-04 | norm: 0.2751, dt: 451.80ms, tok/sec: 1160449.75
step: 5179 | loss: 3.278306 | lr: 5.2497e-04 | norm: 0.2569, dt: 449.56ms, tok/sec: 1166219.38
step: 5180 | loss: 3.314039 | lr: 5.2494e-04 | norm: 0.2427, dt: 450.44ms, tok/sec: 1163937.92
step: 5181 | loss: 3.339447 | lr: 5.2491e-04 | norm: 0.2619, dt: 449.47ms, tok/sec: 1166455.07
step: 5182 | loss: 3.379377 | lr: 5.2488e-04 | norm: 0.2593, dt: 450.45ms, tok/sec: 1163928.06
step: 5183 | loss: 3.384409 | lr: 5.2485e-04 | norm: 0.2730, dt: 450.36ms, tok/sec: 1164146.19
step: 5184 | loss: 3.416047 | lr: 5.2481e-04 | norm: 0.3033, dt: 450.45ms, tok/sec: 1163920.05
step: 5185 | loss: 3.399545 | lr: 5.2478e-04 | norm: 0.3296, dt: 449.80ms, tok/sec: 1165609.87
step: 5186 | loss: 3.354671 | lr: 5.2475e-04 | norm: 0.2947, dt: 450.53ms, tok/sec: 1163716.79
step: 5187 | loss: 3.348399 | lr: 5.2472e-04 | norm: 0.2722, dt: 450.95ms, tok/sec: 1162625.94
step: 5188 | loss: 3.325993 | lr: 5.2469e-04 | norm: 0.3158, dt: 452.02ms, tok/sec: 1159878.06
step: 5189 | loss: 3.331781 | lr: 5.2465e-04 | norm: 0.2862, dt: 450.48ms, tok/sec: 1163846.75
step: 5190 | loss: 3.320098 | lr: 5.2462e-04 | norm: 0.2894, dt: 449.81ms, tok/sec: 1165575.27
step: 5191 | loss: 3.338348 | lr: 5.2459e-04 | norm: 0.2813, dt: 455.37ms, tok/sec: 1151354.23
step: 5192 | loss: 3.359045 | lr: 5.2456e-04 | norm: 0.2851, dt: 450.97ms, tok/sec: 1162589.06
step: 5193 | loss: 3.309580 | lr: 5.2453e-04 | norm: 0.2541, dt: 450.58ms, tok/sec: 1163593.02
step: 5194 | loss: 3.355217 | lr: 5.2449e-04 | norm: 0.2832, dt: 450.00ms, tok/sec: 1165089.88
step: 5195 | loss: 3.321287 | lr: 5.2446e-04 | norm: 0.2818, dt: 451.33ms, tok/sec: 1161639.60
step: 5196 | loss: 3.336387 | lr: 5.2443e-04 | norm: 0.2506, dt: 451.32ms, tok/sec: 1161683.78
step: 5197 | loss: 3.297978 | lr: 5.2440e-04 | norm: 0.2932, dt: 451.37ms, tok/sec: 1161537.13
step: 5198 | loss: 3.220300 | lr: 5.2437e-04 | norm: 0.2722, dt: 450.84ms, tok/sec: 1162901.38
step: 5199 | loss: 3.135442 | lr: 5.2433e-04 | norm: 0.2680, dt: 451.33ms, tok/sec: 1161638.98
step: 5200 | loss: 3.155999 | lr: 5.2430e-04 | norm: 0.2829, dt: 451.34ms, tok/sec: 1161627.33
step: 5201 | loss: 3.194685 | lr: 5.2427e-04 | norm: 0.2619, dt: 451.73ms, tok/sec: 1160620.62
step: 5202 | loss: 3.200514 | lr: 5.2424e-04 | norm: 0.2625, dt: 450.36ms, tok/sec: 1164144.96
step: 5203 | loss: 3.206503 | lr: 5.2421e-04 | norm: 0.2481, dt: 451.04ms, tok/sec: 1162408.38
step: 5204 | loss: 3.165153 | lr: 5.2417e-04 | norm: 0.2540, dt: 451.02ms, tok/sec: 1162441.56
step: 5205 | loss: 3.200037 | lr: 5.2414e-04 | norm: 0.2477, dt: 450.78ms, tok/sec: 1163060.68
step: 5206 | loss: 3.186458 | lr: 5.2411e-04 | norm: 0.2417, dt: 451.18ms, tok/sec: 1162030.00
step: 5207 | loss: 3.186172 | lr: 5.2408e-04 | norm: 0.2417, dt: 450.22ms, tok/sec: 1164521.63
step: 5208 | loss: 3.191688 | lr: 5.2405e-04 | norm: 0.2672, dt: 450.96ms, tok/sec: 1162601.97
step: 5209 | loss: 3.310457 | lr: 5.2401e-04 | norm: 0.3164, dt: 451.29ms, tok/sec: 1161752.52
step: 5210 | loss: 3.316626 | lr: 5.2398e-04 | norm: 0.2781, dt: 451.64ms, tok/sec: 1160861.41
step: 5211 | loss: 3.330100 | lr: 5.2395e-04 | norm: 0.2646, dt: 450.96ms, tok/sec: 1162608.73
step: 5212 | loss: 3.333272 | lr: 5.2392e-04 | norm: 0.2676, dt: 451.04ms, tok/sec: 1162391.79
step: 5213 | loss: 3.338434 | lr: 5.2388e-04 | norm: 0.3037, dt: 451.43ms, tok/sec: 1161401.56
step: 5214 | loss: 3.307700 | lr: 5.2385e-04 | norm: 0.2758, dt: 450.85ms, tok/sec: 1162881.09
step: 5215 | loss: 3.378757 | lr: 5.2382e-04 | norm: 0.2631, dt: 450.17ms, tok/sec: 1164646.83
step: 5216 | loss: 3.310989 | lr: 5.2379e-04 | norm: 0.2692, dt: 449.91ms, tok/sec: 1165313.38
step: 5217 | loss: 3.311159 | lr: 5.2376e-04 | norm: 0.2791, dt: 450.31ms, tok/sec: 1164291.65
step: 5218 | loss: 3.270447 | lr: 5.2372e-04 | norm: 0.3098, dt: 450.24ms, tok/sec: 1164476.00
step: 5219 | loss: 3.296231 | lr: 5.2369e-04 | norm: 0.2710, dt: 450.84ms, tok/sec: 1162918.60
step: 5220 | loss: 3.339510 | lr: 5.2366e-04 | norm: 0.2611, dt: 449.90ms, tok/sec: 1165344.87
step: 5221 | loss: 3.285377 | lr: 5.2363e-04 | norm: 0.2751, dt: 451.15ms, tok/sec: 1162101.85
step: 5222 | loss: 3.301030 | lr: 5.2360e-04 | norm: 0.2735, dt: 451.13ms, tok/sec: 1162156.51
step: 5223 | loss: 3.306024 | lr: 5.2356e-04 | norm: 0.2756, dt: 450.59ms, tok/sec: 1163549.92
step: 5224 | loss: 3.259905 | lr: 5.2353e-04 | norm: 0.2710, dt: 450.58ms, tok/sec: 1163587.48
step: 5225 | loss: 3.331607 | lr: 5.2350e-04 | norm: 0.2675, dt: 450.84ms, tok/sec: 1162917.99
step: 5226 | loss: 3.283478 | lr: 5.2347e-04 | norm: 0.2822, dt: 450.21ms, tok/sec: 1164550.00
step: 5227 | loss: 3.283704 | lr: 5.2343e-04 | norm: 0.2638, dt: 449.97ms, tok/sec: 1165159.63
step: 5228 | loss: 3.346655 | lr: 5.2340e-04 | norm: 0.2747, dt: 452.13ms, tok/sec: 1159597.93
step: 5229 | loss: 3.355757 | lr: 5.2337e-04 | norm: 0.2645, dt: 451.12ms, tok/sec: 1162198.28
step: 5230 | loss: 3.378242 | lr: 5.2334e-04 | norm: 0.2552, dt: 451.31ms, tok/sec: 1161699.12
step: 5231 | loss: 3.356813 | lr: 5.2331e-04 | norm: 0.2621, dt: 451.60ms, tok/sec: 1160950.89
step: 5232 | loss: 3.349081 | lr: 5.2327e-04 | norm: 0.2962, dt: 451.59ms, tok/sec: 1160982.76
step: 5233 | loss: 3.334882 | lr: 5.2324e-04 | norm: 0.2704, dt: 449.87ms, tok/sec: 1165431.34
step: 5234 | loss: 3.292203 | lr: 5.2321e-04 | norm: 0.2823, dt: 450.55ms, tok/sec: 1163658.29
step: 5235 | loss: 3.344007 | lr: 5.2318e-04 | norm: 0.2570, dt: 452.22ms, tok/sec: 1159368.06
step: 5236 | loss: 3.363853 | lr: 5.2314e-04 | norm: 0.2690, dt: 451.05ms, tok/sec: 1162375.20
step: 5237 | loss: 3.340904 | lr: 5.2311e-04 | norm: 0.2693, dt: 450.38ms, tok/sec: 1164098.74
step: 5238 | loss: 3.295710 | lr: 5.2308e-04 | norm: 0.2307, dt: 450.95ms, tok/sec: 1162620.41
step: 5239 | loss: 3.270707 | lr: 5.2305e-04 | norm: 0.2382, dt: 450.95ms, tok/sec: 1162641.31
step: 5240 | loss: 3.306136 | lr: 5.2301e-04 | norm: 0.2207, dt: 451.46ms, tok/sec: 1161316.91
step: 5241 | loss: 3.320976 | lr: 5.2298e-04 | norm: 0.2319, dt: 450.96ms, tok/sec: 1162598.28
step: 5242 | loss: 3.327716 | lr: 5.2295e-04 | norm: 0.2568, dt: 451.41ms, tok/sec: 1161450.63
step: 5243 | loss: 3.305258 | lr: 5.2292e-04 | norm: 0.2546, dt: 450.52ms, tok/sec: 1163750.66
step: 5244 | loss: 3.180113 | lr: 5.2289e-04 | norm: 0.2569, dt: 451.28ms, tok/sec: 1161770.93
step: 5245 | loss: 3.195120 | lr: 5.2285e-04 | norm: 0.2986, dt: 450.55ms, tok/sec: 1163674.30
step: 5246 | loss: 3.228491 | lr: 5.2282e-04 | norm: 0.3091, dt: 451.83ms, tok/sec: 1160368.92
step: 5247 | loss: 3.216827 | lr: 5.2279e-04 | norm: 0.2476, dt: 450.30ms, tok/sec: 1164300.90
step: 5248 | loss: 3.195257 | lr: 5.2276e-04 | norm: 0.3070, dt: 450.64ms, tok/sec: 1163421.27
step: 5249 | loss: 3.179054 | lr: 5.2272e-04 | norm: 0.2863, dt: 451.52ms, tok/sec: 1161170.97
validation loss: 3.3645
HellaSwag accuracy: 2785/10042=0.2773
rank 4 sample 0: Hello, I'm a Manpreet, so I just want to write a message back. So I'm going to take a bit of stuff with a bit
rank 4 sample 1: Hello, I'm a Manpreet, where the name appears before the "man" appears on the screen. To me, I am the same man first
rank 4 sample 2: Hello, I'm a Manpreet, I'm a Shaker, I'm a Quaker, I'm a Quaker, I'm a Quaker
rank 0 sample 0: Hello, I'm a Manpreet, and I love to be your Godly! (See picture) What? What? (Hang-Dong
rank 4 sample 3: Hello, I'm a Manpreet, and it brings on some very interesting patterns to my point of attachment, when you put your own identity into the first
rank 0 sample 1: Hello, I'm a Manpreet, so don't worry, I can put my own knowledge from my perspective on my subject and let's face it.
rank 0 sample 2: Hello, I'm a Manpreet, so I always feel like they're not the people.
Now you'll see I've got a conversation with you
rank 0 sample 3: Hello, I'm a Manpreet, and all of a sudden I get up to meet someone who's coming and he walks out in front of me,
rank 6 sample 0: Hello, I'm a Manpreet, with one of my children - what does it mean? What's what? Who's the right person?
I
rank 6 sample 1: Hello, I'm a Manpreet, and I can't even tell you that your message doesn't have to look like a regular email. I know that
rank 6 sample 2: Hello, I'm a Manpreet, but when I say this it makes me feel better.
So you know who I'm talking about, the I
rank 6 sample 3: Hello, I'm a Manpreet, and so we have our own "Wired" and "Wired" version. We send the contents, some
rank 2 sample 0: Hello, I'm a Manpreet, now it is just a "I'm" and that makes it a huge problem.
I'm not in the
rank 2 sample 1: Hello, I'm a Manpreet, or I might get married, but don't worry. I've done many things, and I'll just do it
rank 2 sample 2: Hello, I'm a Manpreet, so I can get the job I am looking for.
How To Change the Job Job?
What is the
rank 2 sample 3: Hello, I'm a Manpreet, but this comes with some bad consequences. As I stated before, the first big surprise came and most people got stuck
rank 7 sample 0: Hello, I'm a Manpreet, and I know that. The other day everyone had to explain to me they love singing a song while they are in
rank 7 sample 1: Hello, I'm a Manpreet, friend of you.
Your new word about the concept does not go away. My kids will be so impressed that
rank 7 sample 2: Hello, I'm a Manpreet, and I've taken a lot of that to make things easier for me.
I've been working on a small
rank 5 sample 0: Hello, I'm a Manpreet, but that is what I'm doing with a new version.
"Hi, I'm a Manpreet,
rank 7 sample 3: Hello, I'm a Manpreet, isn't it? I was told for several reasons that I wanted a meeting with my boyfriend, it was a way
rank 5 sample 1: Hello, I'm a Manpreet, no longer a man anymore."
This is actually a bit of a misfortunes moment. If an article
rank 5 sample 2: Hello, I'm a Manpreet, so you may be tempted to switch in the first two places. That's all. This is a great way to
rank 5 sample 3: Hello, I'm a Manpreet, one of the kids in class, and it's all the same thing. He said he's using a lot more
rank 1 sample 0: Hello, I'm a Manpreet, and when you're finished with the definition you are now an Example.
And you are now just a Example.
rank 1 sample 1: Hello, I'm a Manpreet, a man. There's a lot of "manpont" who are not-automated, but they
rank 1 sample 2: Hello, I'm a Manpreet, but maybe it's a little bit more complicated.
Now when we get to the end of our sentence we are
rank 1 sample 3: Hello, I'm a Manpreet, and I'm very excited to teach and meet the challenges!
To prepare I'll put 'My Name' to
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going to tell you to say 'Well, thanks' And not, to take a deep breath and
rank 3 sample 1: Hello, I'm a Manpreet, and that means I'm welcome from my family.
When I was a kid, I was the only one,
rank 3 sample 2: Hello, I'm a Manpreet, and you've got "beware" when you actually need to use a shortcut called "sod".
That
rank 3 sample 3: Hello, I'm a Manpreet, and are happy for you.”<|endoftext|>This post may be the second in a series on the difference between �
step: 5250 | loss: 3.120388 | lr: 5.2269e-04 | norm: 0.2935, dt: 12085.88ms, tok/sec: 43380.22
step: 5251 | loss: 3.176702 | lr: 5.2266e-04 | norm: 0.2820, dt: 451.14ms, tok/sec: 1162131.95
step: 5252 | loss: 3.156165 | lr: 5.2263e-04 | norm: 0.2954, dt: 450.82ms, tok/sec: 1162953.66
step: 5253 | loss: 3.171610 | lr: 5.2259e-04 | norm: 0.3003, dt: 449.54ms, tok/sec: 1166273.19
step: 5254 | loss: 3.204082 | lr: 5.2256e-04 | norm: 0.2836, dt: 450.19ms, tok/sec: 1164597.49
step: 5255 | loss: 3.294014 | lr: 5.2253e-04 | norm: 0.2724, dt: 448.82ms, tok/sec: 1168139.84
step: 5256 | loss: 3.369822 | lr: 5.2250e-04 | norm: 0.3059, dt: 449.87ms, tok/sec: 1165412.81
step: 5257 | loss: 3.336638 | lr: 5.2246e-04 | norm: 0.2736, dt: 450.19ms, tok/sec: 1164588.85
step: 5258 | loss: 3.335583 | lr: 5.2243e-04 | norm: 0.2619, dt: 450.15ms, tok/sec: 1164694.33
step: 5259 | loss: 3.323785 | lr: 5.2240e-04 | norm: 0.2499, dt: 451.22ms, tok/sec: 1161933.61
step: 5260 | loss: 3.368413 | lr: 5.2237e-04 | norm: 0.2582, dt: 449.94ms, tok/sec: 1165244.22
step: 5261 | loss: 3.453441 | lr: 5.2233e-04 | norm: 0.2795, dt: 450.45ms, tok/sec: 1163912.66
step: 5262 | loss: 3.321923 | lr: 5.2230e-04 | norm: 0.3062, dt: 449.94ms, tok/sec: 1165249.16

step: 8433 | loss: 3.201859 | lr: 3.9680e-04 | norm: 0.2767, dt: 451.12ms, tok/sec: 1162195.21
step: 8434 | loss: 3.279241 | lr: 3.9676e-04 | norm: 0.2663, dt: 452.10ms, tok/sec: 1159684.77
step: 8435 | loss: 3.228068 | lr: 3.9671e-04 | norm: 0.2618, dt: 450.34ms, tok/sec: 1164193.03
step: 8436 | loss: 3.275089 | lr: 3.9667e-04 | norm: 0.2676, dt: 451.33ms, tok/sec: 1161661.08
step: 8437 | loss: 3.266828 | lr: 3.9663e-04 | norm: 0.2576, dt: 451.35ms, tok/sec: 1161591.12
step: 8438 | loss: 3.289873 | lr: 3.9658e-04 | norm: 0.2897, dt: 452.28ms, tok/sec: 1159210.38
step: 8439 | loss: 3.293252 | lr: 3.9654e-04 | norm: 0.2821, dt: 451.83ms, tok/sec: 1160354.83
step: 8440 | loss: 3.251642 | lr: 3.9649e-04 | norm: 0.2617, dt: 450.54ms, tok/sec: 1163689.08
step: 8441 | loss: 3.275834 | lr: 3.9645e-04 | norm: 0.2570, dt: 451.93ms, tok/sec: 1160100.18
step: 8442 | loss: 3.205586 | lr: 3.9640e-04 | norm: 0.2719, dt: 450.17ms, tok/sec: 1164653.00
step: 8443 | loss: 3.247163 | lr: 3.9636e-04 | norm: 0.2521, dt: 451.89ms, tok/sec: 1160201.78
step: 8444 | loss: 3.213861 | lr: 3.9631e-04 | norm: 0.2698, dt: 451.59ms, tok/sec: 1160982.76
step: 8445 | loss: 3.325989 | lr: 3.9627e-04 | norm: 0.2735, dt: 450.98ms, tok/sec: 1162555.26
step: 8446 | loss: 3.260731 | lr: 3.9622e-04 | norm: 0.2817, dt: 450.57ms, tok/sec: 1163611.49
step: 8447 | loss: 3.238617 | lr: 3.9618e-04 | norm: 0.2495, dt: 450.16ms, tok/sec: 1164673.97
step: 8448 | loss: 3.273791 | lr: 3.9613e-04 | norm: 0.2900, dt: 451.28ms, tok/sec: 1161785.05
step: 8449 | loss: 3.235065 | lr: 3.9609e-04 | norm: 0.2444, dt: 451.72ms, tok/sec: 1160653.70
step: 8450 | loss: 3.218044 | lr: 3.9604e-04 | norm: 0.2835, dt: 450.41ms, tok/sec: 1164021.71
step: 8451 | loss: 3.227588 | lr: 3.9600e-04 | norm: 0.2831, dt: 451.11ms, tok/sec: 1162228.38
step: 8452 | loss: 3.216017 | lr: 3.9595e-04 | norm: 0.2904, dt: 450.59ms, tok/sec: 1163562.85
step: 8453 | loss: 3.239535 | lr: 3.9591e-04 | norm: 0.2638, dt: 450.48ms, tok/sec: 1163843.05
step: 8454 | loss: 3.217777 | lr: 3.9586e-04 | norm: 0.2821, dt: 450.66ms, tok/sec: 1163388.03
step: 8455 | loss: 3.259630 | lr: 3.9582e-04 | norm: 0.3037, dt: 451.61ms, tok/sec: 1160923.31
step: 8456 | loss: 3.272379 | lr: 3.9577e-04 | norm: 0.2656, dt: 450.91ms, tok/sec: 1162724.30
step: 8457 | loss: 3.291811 | lr: 3.9573e-04 | norm: 0.3101, dt: 450.30ms, tok/sec: 1164305.21
step: 8458 | loss: 3.218074 | lr: 3.9568e-04 | norm: 0.2861, dt: 451.30ms, tok/sec: 1161731.65
step: 8459 | loss: 3.282145 | lr: 3.9564e-04 | norm: 0.2634, dt: 450.76ms, tok/sec: 1163110.51
step: 8460 | loss: 3.184262 | lr: 3.9559e-04 | norm: 0.2674, dt: 451.39ms, tok/sec: 1161493.57
step: 8461 | loss: 3.217484 | lr: 3.9555e-04 | norm: 0.2770, dt: 451.16ms, tok/sec: 1162090.80
step: 8462 | loss: 3.241682 | lr: 3.9551e-04 | norm: 0.2833, dt: 450.37ms, tok/sec: 1164115.99
step: 8463 | loss: 3.259733 | lr: 3.9546e-04 | norm: 0.2806, dt: 451.58ms, tok/sec: 1160997.47
step: 8464 | loss: 3.263586 | lr: 3.9542e-04 | norm: 0.2740, dt: 451.51ms, tok/sec: 1161182.00
step: 8465 | loss: 3.326689 | lr: 3.9537e-04 | norm: 0.2886, dt: 450.86ms, tok/sec: 1162865.71
step: 8466 | loss: 3.244941 | lr: 3.9533e-04 | norm: 0.2913, dt: 451.46ms, tok/sec: 1161329.18
step: 8467 | loss: 3.264619 | lr: 3.9528e-04 | norm: 0.2744, dt: 451.28ms, tok/sec: 1161778.30
step: 8468 | loss: 3.238823 | lr: 3.9524e-04 | norm: 0.2686, dt: 450.34ms, tok/sec: 1164209.67
step: 8469 | loss: 3.244750 | lr: 3.9519e-04 | norm: 0.2767, dt: 450.92ms, tok/sec: 1162705.85
step: 8470 | loss: 3.211701 | lr: 3.9515e-04 | norm: 0.2514, dt: 451.02ms, tok/sec: 1162443.41
step: 8471 | loss: 3.244448 | lr: 3.9510e-04 | norm: 0.2685, dt: 451.37ms, tok/sec: 1161537.13
step: 8472 | loss: 3.182325 | lr: 3.9506e-04 | norm: 0.2729, dt: 451.39ms, tok/sec: 1161492.96
step: 8473 | loss: 3.261385 | lr: 3.9501e-04 | norm: 0.2565, dt: 451.11ms, tok/sec: 1162217.32
step: 8474 | loss: 3.268182 | lr: 3.9497e-04 | norm: 0.2671, dt: 450.52ms, tok/sec: 1163738.35
step: 8475 | loss: 3.299242 | lr: 3.9492e-04 | norm: 0.2855, dt: 450.87ms, tok/sec: 1162834.97
step: 8476 | loss: 3.233128 | lr: 3.9488e-04 | norm: 0.2389, dt: 452.07ms, tok/sec: 1159753.27
step: 8477 | loss: 3.261413 | lr: 3.9483e-04 | norm: 0.2829, dt: 451.44ms, tok/sec: 1161376.41
step: 8478 | loss: 3.236197 | lr: 3.9479e-04 | norm: 0.2562, dt: 451.05ms, tok/sec: 1162367.83
step: 8479 | loss: 3.350455 | lr: 3.9474e-04 | norm: 0.2679, dt: 451.07ms, tok/sec: 1162332.81
step: 8480 | loss: 3.208404 | lr: 3.9470e-04 | norm: 0.2369, dt: 451.91ms, tok/sec: 1160161.38
step: 8481 | loss: 3.267747 | lr: 3.9465e-04 | norm: 0.2846, dt: 450.84ms, tok/sec: 1162920.45
step: 8482 | loss: 3.277567 | lr: 3.9461e-04 | norm: 0.2415, dt: 451.49ms, tok/sec: 1161242.10
step: 8483 | loss: 3.234832 | lr: 3.9456e-04 | norm: 0.2708, dt: 451.54ms, tok/sec: 1161101.69
step: 8484 | loss: 3.239760 | lr: 3.9452e-04 | norm: 0.2609, dt: 450.74ms, tok/sec: 1163162.19
step: 8485 | loss: 3.297135 | lr: 3.9447e-04 | norm: 0.2601, dt: 451.29ms, tok/sec: 1161759.88
step: 8486 | loss: 3.329550 | lr: 3.9443e-04 | norm: 0.2749, dt: 451.01ms, tok/sec: 1162476.59
step: 8487 | loss: 3.211937 | lr: 3.9438e-04 | norm: 0.2850, dt: 449.75ms, tok/sec: 1165736.54
step: 8488 | loss: 3.261666 | lr: 3.9434e-04 | norm: 0.3036, dt: 451.10ms, tok/sec: 1162244.35
step: 8489 | loss: 3.249729 | lr: 3.9429e-04 | norm: 0.2824, dt: 451.10ms, tok/sec: 1162246.80
step: 8490 | loss: 3.213669 | lr: 3.9425e-04 | norm: 0.2585, dt: 450.56ms, tok/sec: 1163645.98
step: 8491 | loss: 3.193428 | lr: 3.9420e-04 | norm: 0.2658, dt: 451.26ms, tok/sec: 1161831.70
step: 8492 | loss: 3.232738 | lr: 3.9416e-04 | norm: 0.2648, dt: 451.39ms, tok/sec: 1161497.25
step: 8493 | loss: 3.225211 | lr: 3.9411e-04 | norm: 0.2649, dt: 451.86ms, tok/sec: 1160280.75
step: 8494 | loss: 3.225136 | lr: 3.9407e-04 | norm: 0.2707, dt: 450.92ms, tok/sec: 1162712.00
step: 8495 | loss: 3.232132 | lr: 3.9402e-04 | norm: 0.2728, dt: 451.36ms, tok/sec: 1161573.94
step: 8496 | loss: 3.214340 | lr: 3.9398e-04 | norm: 0.2677, dt: 451.23ms, tok/sec: 1161896.77
step: 8497 | loss: 3.233507 | lr: 3.9394e-04 | norm: 0.2936, dt: 451.40ms, tok/sec: 1161467.19
step: 8498 | loss: 3.284208 | lr: 3.9389e-04 | norm: 0.2737, dt: 451.43ms, tok/sec: 1161402.78
step: 8499 | loss: 3.230179 | lr: 3.9385e-04 | norm: 0.4648, dt: 451.10ms, tok/sec: 1162232.68
validation loss: 3.2470
HellaSwag accuracy: 2846/10042=0.2834
rank 6 sample 0: Hello, I'm a Manpreet, manpreet, a servant of God. I love my Lord.
So, I had to bring the man
rank 6 sample 1: Hello, I'm a Manpreet, you're welcome to go and visit the "Sustainable Development" page here. I've got all of your business
rank 0 sample 0: Hello, I'm a Manpreet, I'm a ManPreet! That's where I want to focus if it's the difference between my English and
rank 6 sample 2: Hello, I'm a Manpreet, a manpreet. You have your own, but I am in love with you. I'm having my kids
rank 0 sample 1: Hello, I'm a Manpreet, and now I'm a Manpreet. So just do it with my pen and then when you use your pen
rank 6 sample 3: Hello, I'm a Manpreet, I'm a Manpreet, I'm a Manpreet, I'm a Manplein, we're
rank 0 sample 2: Hello, I'm a Manpreet, and I see that
You're having trouble loading your document! (If you do not see the error message in
rank 0 sample 3: Hello, I'm a Manpreet, you look like this:
You should also put the two boxes before the letter itself (with a marker) to
rank 2 sample 0: Hello, I'm a Manpreet, let me see some of the other's words, so you're welcome. I don't know what I think of
rank 2 sample 1: Hello, I'm a Manpreet, just like myself."
The man he'll be as soon and then will be a man who's good for you
rank 2 sample 2: Hello, I'm a Manpreet, and I want to know how much land I have to occupy to cover myself. So I got off my back p
rank 2 sample 3: Hello, I'm a Manpreet, a guy, a woman. I'm proud of having my new wife, Mariah. A nice man, arank 7 sample 0: Hello, I'm a Manpreet, I'm a Manpreet, I'm a Manpreet, there's none I can manage, and I

rank 7 sample 1: Hello, I'm a Manpreet, for people. I'm just the old man. I will be the new man on my first day so I'll
rank 7 sample 2: Hello, I'm a Manpreet, So I can look up the person at your door.
- What should I do?
- Who should call
rank 7 sample 3: Hello, I'm a Manpreet, The Manpreet's first letter must be his name. I am really sorry to see you don't know all
rank 4 sample 0: Hello, I'm a Manpreet, and I just want to have a little, so I'm not going to put out your house any more for that
rank 4 sample 1: Hello, I'm a Manpreet, if it's hard to understand me you will see my explanation.
My wife loves dogs, but my wife makes
rank 4 sample 2: Hello, I'm a Manpreet, but at the height of my feet is my mouth. I know the language but then I understand. As I say
rank 4 sample 3: Hello, I'm a Manpreet, I've left over $5. Thanks so much. I will go home. I'm not very good at using
rank 5 sample 0: Hello, I'm a Manpreet, a Manpie, a Manpie, a Manpie. I know a Manpie, a Manpie, a
rank 5 sample 1: Hello, I'm a Manpreet, someone like me! We were so proud of everyone's success that I was afraid for the rest of my life.
rank 5 sample 2: Hello, I'm a Manpreet, and I can do quite a bit using the app.
"Hi, how much fun is this?"
"
rank 5 sample 3: Hello, I'm a Manpreet, as long as my life depends on it, so the next time you're sitting outside and I start to get angry
rank 1 sample 0: Hello, I'm a Manpreet, how did you do that?
At this point in time someone asked me:
Well, my name is David
rank 1 sample 1: Hello, I'm a Manpreet, so you're very much like I am
I'm getting dressed every morning at Ira Caulle, and
rank 1 sample 2: Hello, I'm a Manpreet, you didn't even know what I was talking about. You got a new point of view and that's the key
rank 1 sample 3: Hello, I'm a Manpreet, I'm a woman. (Bosain)
Dentals - She is usually just a woman. (
rank 3 sample 0: Hello, I'm a Manpreet, I'm a man-of-the-art-world-and-has-disliked-a-
rank 3 sample 1: Hello, I'm a Manpreet, I can use the command wcip -d and i'm a Manpreet, I can use the file
rank 3 sample 2: Hello, I'm a Manpreet, I am just a Manpreet. Let me back up here.
In my previous post, we discussed different
rank 3 sample 3: Hello, I'm a Manpreet, I just want to work out that.
"So now, if we're actually looking for a line of the
step: 8500 | loss: 3.242841 | lr: 3.9380e-04 | norm: 0.2799, dt: 12093.89ms, tok/sec: 43351.49
step: 8501 | loss: 3.246747 | lr: 3.9376e-04 | norm: 0.2685, dt: 451.57ms, tok/sec: 1161027.51
step: 8502 | loss: 3.240337 | lr: 3.9371e-04 | norm: 0.3118, dt: 448.92ms, tok/sec: 1167899.75
step: 8503 | loss: 3.184870 | lr: 3.9367e-04 | norm: 0.2973, dt: 449.54ms, tok/sec: 1166273.19
step: 8504 | loss: 3.254505 | lr: 3.9362e-04 | norm: 0.2951, dt: 1162.90ms, tok/sec: 450846.33
step: 8505 | loss: 3.231427 | lr: 3.9358e-04 | norm: 0.3235, dt: 449.75ms, tok/sec: 1165743.34
step: 8506 | loss: 3.210197 | lr: 3.9353e-04 | norm: 0.2606, dt: 448.92ms, tok/sec: 1167898.51
step: 8507 | loss: 3.235110 | lr: 3.9349e-04 | norm: 0.2857, dt: 450.28ms, tok/sec: 1164372.41
step: 8508 | loss: 3.240169 | lr: 3.9344e-04 | norm: 0.2739, dt: 449.89ms, tok/sec: 1165373.90
step: 8509 | loss: 3.273638 | lr: 3.9340e-04 | norm: 0.2711, dt: 450.88ms, tok/sec: 1162814.06
step: 8510 | loss: 3.271682 | lr: 3.9335e-04 | norm: 0.2871, dt: 449.99ms, tok/sec: 1165111.48
step: 8511 | loss: 3.250747 | lr: 3.9331e-04 | norm: 0.2574, dt: 450.53ms, tok/sec: 1163719.26
step: 8512 | loss: 3.275612 | lr: 3.9326e-04 | norm: 0.2640, dt: 450.90ms, tok/sec: 1162762.41
step: 8513 | loss: 3.217258 | lr: 3.9322e-04 | norm: 0.2530, dt: 450.37ms, tok/sec: 1164116.61
step: 8514 | loss: 3.226804 | lr: 3.9317e-04 | norm: 0.2780, dt: 450.93ms, tok/sec: 1162674.50
step: 8515 | loss: 3.256463 | lr: 3.9313e-04 | norm: 0.2786, dt: 450.35ms, tok/sec: 1164191.80
step: 8516 | loss: 3.228555 | lr: 3.9308e-04 | norm: 0.2512, dt: 453.80ms, tok/sec: 1155336.32
step: 8517 | loss: 3.238569 | lr: 3.9304e-04 | norm: 0.2633, dt: 450.79ms, tok/sec: 1163050.84
step: 8518 | loss: 3.231252 | lr: 3.9299e-04 | norm: 0.2200, dt: 452.08ms, tok/sec: 1159716.57
step: 8519 | loss: 3.240780 | lr: 3.9295e-04 | norm: 0.2584, dt: 451.36ms, tok/sec: 1161569.03
step: 8520 | loss: 3.248882 | lr: 3.9290e-04 | norm: 0.2356, dt: 451.12ms, tok/sec: 1162195.21
step: 8521 | loss: 3.226259 | lr: 3.9286e-04 | norm: 0.2729, dt: 451.42ms, tok/sec: 1161426.09
step: 8522 | loss: 3.335836 | lr: 3.9281e-04 | norm: 0.2567, dt: 453.96ms, tok/sec: 1154912.19
step: 8523 | loss: 3.216717 | lr: 3.9277e-04 | norm: 0.2706, dt: 451.56ms, tok/sec: 1161048.96
step: 8524 | loss: 3.229718 | lr: 3.9272e-04 | norm: 0.2534, dt: 451.65ms, tok/sec: 1160826.48
step: 8525 | loss: 3.237183 | lr: 3.9268e-04 | norm: 0.2368, dt: 452.92ms, tok/sec: 1157581.11
step: 8526 | loss: 3.217899 | lr: 3.9263e-04 | norm: 0.2819, dt: 451.51ms, tok/sec: 1161176.49
step: 8527 | loss: 3.228030 | lr: 3.9259e-04 | norm: 0.2376, dt: 451.01ms, tok/sec: 1162464.30
step: 8528 | loss: 3.219785 | lr: 3.9254e-04 | norm: 0.2706, dt: 452.14ms, tok/sec: 1159561.86
step: 8529 | loss: 3.233596 | lr: 3.9250e-04 | norm: 0.2510, dt: 451.69ms, tok/sec: 1160718.64
step: 8530 | loss: 3.229779 | lr: 3.9245e-04 | norm: 0.2441, dt: 451.48ms, tok/sec: 1161273.37
step: 8531 | loss: 3.248019 | lr: 3.9241e-04 | norm: 0.2602, dt: 451.19ms, tok/sec: 1162014.65
step: 8532 | loss: 3.269286 | lr: 3.9236e-04 | norm: 0.2301, dt: 451.50ms, tok/sec: 1161208.98
step: 8533 | loss: 3.263373 | lr: 3.9232e-04 | norm: 0.2738, dt: 450.76ms, tok/sec: 1163119.74
step: 8534 | loss: 3.351304 | lr: 3.9227e-04 | norm: 0.2585, dt: 451.49ms, tok/sec: 1161229.83
step: 8535 | loss: 3.238914 | lr: 3.9223e-04 | norm: 0.2985, dt: 451.02ms, tok/sec: 1162454.47
step: 8536 | loss: 3.220099 | lr: 3.9218e-04 | norm: 0.2679, dt: 451.12ms, tok/sec: 1162189.68
step: 8537 | loss: 3.270290 | lr: 3.9214e-04 | norm: 0.2491, dt: 450.64ms, tok/sec: 1163423.11
step: 8538 | loss: 3.224561 | lr: 3.9209e-04 | norm: 0.2803, dt: 450.86ms, tok/sec: 1162849.73
step: 8539 | loss: 3.310791 | lr: 3.9205e-04 | norm: 0.2812, dt: 451.27ms, tok/sec: 1161801.62
step: 8540 | loss: 3.222906 | lr: 3.9200e-04 | norm: 0.2999, dt: 450.26ms, tok/sec: 1164414.34
step: 8541 | loss: 3.242038 | lr: 3.9196e-04 | norm: 0.2769, dt: 451.22ms, tok/sec: 1161936.06
step: 8542 | loss: 3.262701 | lr: 3.9191e-04 | norm: 0.2910, dt: 450.97ms, tok/sec: 1162573.69
step: 8543 | loss: 3.253072 | lr: 3.9187e-04 | norm: 0.2819, dt: 449.23ms, tok/sec: 1167076.61
step: 8544 | loss: 3.275116 | lr: 3.9182e-04 | norm: 0.2614, dt: 451.18ms, tok/sec: 1162043.51
step: 8545 | loss: 3.245196 | lr: 3.9178e-04 | norm: 0.2548, dt: 450.73ms, tok/sec: 1163187.41
step: 8546 | loss: 3.279641 | lr: 3.9173e-04 | norm: 0.2415, dt: 449.78ms, tok/sec: 1165664.24
step: 8547 | loss: 3.213194 | lr: 3.9169e-04 | norm: 0.2308, dt: 449.95ms, tok/sec: 1165202.85
step: 8548 | loss: 3.266589 | lr: 3.9164e-04 | norm: 0.2489, dt: 449.72ms, tok/sec: 1165818.12
step: 8549 | loss: 3.242359 | lr: 3.9160e-04 | norm: 0.2374, dt: 1232.66ms, tok/sec: 425331.14
step: 8550 | loss: 3.243265 | lr: 3.9155e-04 | norm: 0.2658, dt: 449.39ms, tok/sec: 1166659.29
step: 8551 | loss: 3.196452 | lr: 3.9151e-04 | norm: 0.2884, dt: 450.64ms, tok/sec: 1163428.65
step: 8552 | loss: 3.209755 | lr: 3.9146e-04 | norm: 0.2666, dt: 450.62ms, tok/sec: 1163469.28
step: 8553 | loss: 3.232643 | lr: 3.9142e-04 | norm: 0.2813, dt: 450.39ms, tok/sec: 1164073.47
step: 8554 | loss: 3.220164 | lr: 3.9137e-04 | norm: 0.2425, dt: 451.18ms, tok/sec: 1162033.69
step: 8555 | loss: 3.269528 | lr: 3.9133e-04 | norm: 0.2777, dt: 450.45ms, tok/sec: 1163921.29
step: 8556 | loss: 3.205560 | lr: 3.9128e-04 | norm: 0.2604, dt: 451.93ms, tok/sec: 1160098.96
step: 8557 | loss: 3.220885 | lr: 3.9124e-04 | norm: 0.2685, dt: 449.56ms, tok/sec: 1166226.18
step: 8558 | loss: 3.179784 | lr: 3.9119e-04 | norm: 0.2929, dt: 450.93ms, tok/sec: 1162691.10
step: 8559 | loss: 3.237593 | lr: 3.9115e-04 | norm: 0.2547, dt: 450.92ms, tok/sec: 1162706.47
step: 8560 | loss: 3.235710 | lr: 3.9110e-04 | norm: 0.2979, dt: 450.97ms, tok/sec: 1162588.45
step: 8561 | loss: 3.288045 | lr: 3.9106e-04 | norm: 0.2630, dt: 450.73ms, tok/sec: 1163198.49
step: 8562 | loss: 3.218519 | lr: 3.9101e-04 | norm: 0.2594, dt: 450.70ms, tok/sec: 1163283.41
step: 8563 | loss: 3.218603 | lr: 3.9097e-04 | norm: 0.2578, dt: 455.57ms, tok/sec: 1150844.47
step: 8564 | loss: 3.315291 | lr: 3.9092e-04 | norm: 0.2655, dt: 451.64ms, tok/sec: 1160864.48
step: 8565 | loss: 3.247527 | lr: 3.9088e-04 | norm: 0.2649, dt: 451.91ms, tok/sec: 1160160.77
step: 8566 | loss: 3.241014 | lr: 3.9083e-04 | norm: 0.2522, dt: 450.82ms, tok/sec: 1162973.34
step: 8567 | loss: 3.215117 | lr: 3.9079e-04 | norm: 0.2569, dt: 451.05ms, tok/sec: 1162383.81
step: 8568 | loss: 3.279632 | lr: 3.9074e-04 | norm: 0.2657, dt: 450.45ms, tok/sec: 1163918.21
step: 8569 | loss: 3.266238 | lr: 3.9070e-04 | norm: 0.2893, dt: 453.15ms, tok/sec: 1156995.81
step: 8570 | loss: 3.442670 | lr: 3.9065e-04 | norm: 0.3907, dt: 450.85ms, tok/sec: 1162893.39
step: 8571 | loss: 3.298136 | lr: 3.9061e-04 | norm: 0.4236, dt: 449.92ms, tok/sec: 1165278.80
step: 8572 | loss: 3.238278 | lr: 3.9056e-04 | norm: 0.3112, dt: 450.56ms, tok/sec: 1163640.43
step: 8573 | loss: 3.420499 | lr: 3.9052e-04 | norm: 0.3587, dt: 451.55ms, tok/sec: 1161092.49
step: 8574 | loss: 3.291940 | lr: 3.9047e-04 | norm: 0.3238, dt: 451.38ms, tok/sec: 1161515.66
step: 8575 | loss: 3.239765 | lr: 3.9043e-04 | norm: 0.3054, dt: 451.22ms, tok/sec: 1161922.55
step: 8576 | loss: 3.258543 | lr: 3.9038e-04 | norm: 0.2618, dt: 451.94ms, tok/sec: 1160094.06
step: 8577 | loss: 3.263817 | lr: 3.9034e-04 | norm: 0.2814, dt: 451.71ms, tok/sec: 1160661.67
step: 8578 | loss: 3.260568 | lr: 3.9029e-04 | norm: 0.2876, dt: 450.02ms, tok/sec: 1165044.82
step: 8579 | loss: 3.262125 | lr: 3.9025e-04 | norm: 0.2699, dt: 451.06ms, tok/sec: 1162334.65
step: 8580 | loss: 3.240497 | lr: 3.9020e-04 | norm: 0.2741, dt: 451.84ms, tok/sec: 1160340.14
step: 8581 | loss: 3.186180 | lr: 3.9016e-04 | norm: 0.2615, dt: 450.60ms, tok/sec: 1163530.22
step: 8582 | loss: 3.221230 | lr: 3.9011e-04 | norm: 0.2964, dt: 451.20ms, tok/sec: 1161982.11
step: 8583 | loss: 3.243564 | lr: 3.9007e-04 | norm: 0.2616, dt: 451.00ms, tok/sec: 1162512.23
step: 8584 | loss: 3.209070 | lr: 3.9002e-04 | norm: 0.2652, dt: 450.47ms, tok/sec: 1163864.00
step: 8585 | loss: 3.253104 | lr: 3.8998e-04 | norm: 0.2672, dt: 450.35ms, tok/sec: 1164181.32
step: 8586 | loss: 3.260134 | lr: 3.8993e-04 | norm: 0.2816, dt: 454.75ms, tok/sec: 1152906.17
step: 8587 | loss: 3.227415 | lr: 3.8989e-04 | norm: 0.2832, dt: 450.76ms, tok/sec: 1163129.58
step: 8588 | loss: 3.235262 | lr: 3.8984e-04 | norm: 0.2477, dt: 450.99ms, tok/sec: 1162523.30
step: 8589 | loss: 3.336888 | lr: 3.8980e-04 | norm: 0.2559, dt: 451.29ms, tok/sec: 1161757.43
step: 8590 | loss: 3.223748 | lr: 3.8975e-04 | norm: 0.2654, dt: 451.01ms, tok/sec: 1162471.06
step: 8591 | loss: 3.209121 | lr: 3.8971e-04 | norm: 0.2456, dt: 452.31ms, tok/sec: 1159137.06
step: 8592 | loss: 3.151298 | lr: 3.8966e-04 | norm: 0.2648, dt: 450.41ms, tok/sec: 1164029.72
step: 8593 | loss: 3.199698 | lr: 3.8962e-04 | norm: 0.2482, dt: 451.36ms, tok/sec: 1161568.42
step: 8594 | loss: 3.265931 | lr: 3.8957e-04 | norm: 0.2548, dt: 449.88ms, tok/sec: 1165399.84
step: 8595 | loss: 3.200604 | lr: 3.8953e-04 | norm: 0.2686, dt: 450.88ms, tok/sec: 1162805.45
step: 8596 | loss: 3.225221 | lr: 3.8948e-04 | norm: 0.2472, dt: 451.59ms, tok/sec: 1160984.60
step: 8597 | loss: 3.290429 | lr: 3.8944e-04 | norm: 0.2526, dt: 451.28ms, tok/sec: 1161772.77
step: 8598 | loss: 3.240311 | lr: 3.8939e-04 | norm: 0.2543, dt: 450.33ms, tok/sec: 1164242.95
step: 8599 | loss: 3.226780 | lr: 3.8935e-04 | norm: 0.2584, dt: 450.96ms, tok/sec: 1162609.96
step: 8600 | loss: 3.208780 | lr: 3.8930e-04 | norm: 0.2540, dt: 451.92ms, tok/sec: 1160143.02
step: 8601 | loss: 3.251027 | lr: 3.8926e-04 | norm: 0.2460, dt: 451.72ms, tok/sec: 1160639.00
step: 8602 | loss: 3.209652 | lr: 3.8921e-04 | norm: 0.3040, dt: 450.77ms, tok/sec: 1163099.44
step: 8603 | loss: 3.291379 | lr: 3.8917e-04 | norm: 0.2913, dt: 451.27ms, tok/sec: 1161818.19
step: 8604 | loss: 3.266775 | lr: 3.8912e-04 | norm: 0.2500, dt: 451.40ms, tok/sec: 1161472.10
step: 8605 | loss: 3.212944 | lr: 3.8908e-04 | norm: 0.2656, dt: 450.37ms, tok/sec: 1164130.78
step: 8606 | loss: 3.307996 | lr: 3.8903e-04 | norm: 0.2506, dt: 450.97ms, tok/sec: 1162589.67
step: 8607 | loss: 3.248805 | lr: 3.8899e-04 | norm: 0.2553, dt: 450.76ms, tok/sec: 1163114.82
step: 8608 | loss: 3.285802 | lr: 3.8894e-04 | norm: 0.2656, dt: 451.34ms, tok/sec: 1161631.62
step: 8609 | loss: 3.245400 | lr: 3.8890e-04 | norm: 0.2551, dt: 451.57ms, tok/sec: 1161041.00
step: 8610 | loss: 3.261836 | lr: 3.8885e-04 | norm: 0.2553, dt: 451.85ms, tok/sec: 1160323.61
step: 8611 | loss: 3.285321 | lr: 3.8881e-04 | norm: 0.2640, dt: 451.54ms, tok/sec: 1161104.75
step: 8612 | loss: 3.161711 | lr: 3.8876e-04 | norm: 0.2650, dt: 451.49ms, tok/sec: 1161229.22
step: 8613 | loss: 3.253466 | lr: 3.8872e-04 | norm: 0.2656, dt: 451.74ms, tok/sec: 1160605.92
step: 8614 | loss: 3.279845 | lr: 3.8867e-04 | norm: 0.3085, dt: 450.60ms, tok/sec: 1163543.15
step: 8615 | loss: 3.279044 | lr: 3.8863e-04 | norm: 0.2785, dt: 451.33ms, tok/sec: 1161658.01
step: 8616 | loss: 3.237533 | lr: 3.8858e-04 | norm: 0.2577, dt: 450.89ms, tok/sec: 1162788.24
step: 8617 | loss: 3.256217 | lr: 3.8854e-04 | norm: 0.2623, dt: 450.76ms, tok/sec: 1163111.74
step: 8618 | loss: 3.305578 | lr: 3.8849e-04 | norm: 0.2855, dt: 451.09ms, tok/sec: 1162276.91
step: 8619 | loss: 3.217654 | lr: 3.8844e-04 | norm: 0.2351, dt: 451.09ms, tok/sec: 1162259.09
step: 8620 | loss: 3.246253 | lr: 3.8840e-04 | norm: 0.2614, dt: 450.34ms, tok/sec: 1164195.49
step: 8621 | loss: 3.216058 | lr: 3.8835e-04 | norm: 0.2473, dt: 452.47ms, tok/sec: 1158721.12
step: 8622 | loss: 3.203252 | lr: 3.8831e-04 | norm: 0.2464, dt: 451.07ms, tok/sec: 1162320.52
step: 8623 | loss: 3.285233 | lr: 3.8826e-04 | norm: 0.2789, dt: 450.76ms, tok/sec: 1163117.28
step: 8624 | loss: 3.166811 | lr: 3.8822e-04 | norm: 0.2841, dt: 451.47ms, tok/sec: 1161292.38
step: 8625 | loss: 3.250874 | lr: 3.8817e-04 | norm: 0.2763, dt: 451.29ms, tok/sec: 1161755.59
step: 8626 | loss: 3.264753 | lr: 3.8813e-04 | norm: 0.2774, dt: 450.22ms, tok/sec: 1164525.95
step: 8627 | loss: 3.215276 | lr: 3.8808e-04 | norm: 0.2807, dt: 450.67ms, tok/sec: 1163364.64
step: 8628 | loss: 3.254631 | lr: 3.8804e-04 | norm: 0.2534, dt: 450.56ms, tok/sec: 1163645.98
step: 8629 | loss: 3.240828 | lr: 3.8799e-04 | norm: 0.3077, dt: 450.80ms, tok/sec: 1163023.77
step: 8630 | loss: 3.213684 | lr: 3.8795e-04 | norm: 0.2659, dt: 451.21ms, tok/sec: 1161967.37
step: 8631 | loss: 3.264406 | lr: 3.8790e-04 | norm: 0.2759, dt: 449.88ms, tok/sec: 1165391.19
step: 8632 | loss: 3.201102 | lr: 3.8786e-04 | norm: 0.2675, dt: 450.64ms, tok/sec: 1163442.19
step: 8633 | loss: 3.202795 | lr: 3.8781e-04 | norm: 0.2743, dt: 450.38ms, tok/sec: 1164098.12
step: 8634 | loss: 3.265105 | lr: 3.8777e-04 | norm: 0.2785, dt: 451.33ms, tok/sec: 1161656.17
step: 8635 | loss: 3.219387 | lr: 3.8772e-04 | norm: 0.2691, dt: 451.11ms, tok/sec: 1162206.26
step: 8636 | loss: 3.272919 | lr: 3.8768e-04 | norm: 0.2921, dt: 451.00ms, tok/sec: 1162499.33
step: 8637 | loss: 3.213429 | lr: 3.8763e-04 | norm: 0.2761, dt: 450.86ms, tok/sec: 1162868.17
step: 8638 | loss: 3.214970 | lr: 3.8759e-04 | norm: 0.2698, dt: 450.53ms, tok/sec: 1163713.71
step: 8639 | loss: 3.255611 | lr: 3.8754e-04 | norm: 0.3053, dt: 450.48ms, tok/sec: 1163843.67
step: 8640 | loss: 3.255708 | lr: 3.8750e-04 | norm: 0.3195, dt: 450.57ms, tok/sec: 1163608.42
step: 8641 | loss: 3.223835 | lr: 3.8745e-04 | norm: 0.3060, dt: 450.36ms, tok/sec: 1164156.05
step: 8642 | loss: 3.253960 | lr: 3.8741e-04 | norm: 0.2732, dt: 450.61ms, tok/sec: 1163496.98
step: 8643 | loss: 3.291912 | lr: 3.8736e-04 | norm: 0.2922, dt: 451.25ms, tok/sec: 1161855.03
step: 8644 | loss: 3.394810 | lr: 3.8732e-04 | norm: 0.3109, dt: 450.66ms, tok/sec: 1163382.49
step: 8645 | loss: 3.273625 | lr: 3.8727e-04 | norm: 0.3015, dt: 450.55ms, tok/sec: 1163672.45
step: 8646 | loss: 3.264238 | lr: 3.8723e-04 | norm: 0.2878, dt: 450.18ms, tok/sec: 1164611.67
step: 8647 | loss: 3.217059 | lr: 3.8718e-04 | norm: 0.2943, dt: 451.31ms, tok/sec: 1161708.94
step: 8648 | loss: 3.293614 | lr: 3.8714e-04 | norm: 0.2698, dt: 450.40ms, tok/sec: 1164059.30
step: 8649 | loss: 3.262921 | lr: 3.8709e-04 | norm: 0.2850, dt: 450.79ms, tok/sec: 1163049.61
step: 8650 | loss: 3.256090 | lr: 3.8705e-04 | norm: 0.2598, dt: 450.88ms, tok/sec: 1162808.53
step: 8651 | loss: 3.223039 | lr: 3.8700e-04 | norm: 0.2799, dt: 449.75ms, tok/sec: 1165729.74
step: 8652 | loss: 3.230596 | lr: 3.8696e-04 | norm: 0.2391, dt: 451.11ms, tok/sec: 1162209.33
step: 8653 | loss: 3.338821 | lr: 3.8691e-04 | norm: 0.2695, dt: 450.91ms, tok/sec: 1162730.44
step: 8654 | loss: 3.237925 | lr: 3.8687e-04 | norm: 0.2518, dt: 451.02ms, tok/sec: 1162461.84
step: 8655 | loss: 3.202134 | lr: 3.8682e-04 | norm: 0.2646, dt: 450.02ms, tok/sec: 1165021.36
step: 8656 | loss: 3.261108 | lr: 3.8677e-04 | norm: 0.2699, dt: 451.15ms, tok/sec: 1162114.14
step: 8657 | loss: 3.303990 | lr: 3.8673e-04 | norm: 0.2896, dt: 450.81ms, tok/sec: 1162989.94
step: 8658 | loss: 3.251547 | lr: 3.8668e-04 | norm: 0.3150, dt: 450.00ms, tok/sec: 1165073.83
step: 8659 | loss: 3.205207 | lr: 3.8664e-04 | norm: 0.2625, dt: 450.24ms, tok/sec: 1164459.96
step: 8660 | loss: 3.202590 | lr: 3.8659e-04 | norm: 0.2779, dt: 452.09ms, tok/sec: 1159690.28
step: 8661 | loss: 3.293647 | lr: 3.8655e-04 | norm: 0.2781, dt: 450.83ms, tok/sec: 1162935.82
step: 8662 | loss: 3.252309 | lr: 3.8650e-04 | norm: 0.2673, dt: 450.75ms, tok/sec: 1163133.89
step: 8663 | loss: 3.237198 | lr: 3.8646e-04 | norm: 0.2791, dt: 450.82ms, tok/sec: 1162961.04
step: 8664 | loss: 3.206323 | lr: 3.8641e-04 | norm: 0.2710, dt: 450.25ms, tok/sec: 1164427.90
step: 8665 | loss: 3.264886 | lr: 3.8637e-04 | norm: 0.2671, dt: 452.23ms, tok/sec: 1159327.72
step: 8666 | loss: 3.228956 | lr: 3.8632e-04 | norm: 0.2458, dt: 451.31ms, tok/sec: 1161694.21
step: 8667 | loss: 3.203391 | lr: 3.8628e-04 | norm: 0.2429, dt: 451.06ms, tok/sec: 1162338.34
step: 8668 | loss: 3.178850 | lr: 3.8623e-04 | norm: 0.2521, dt: 451.32ms, tok/sec: 1161670.28
step: 8669 | loss: 3.192872 | lr: 3.8619e-04 | norm: 0.2458, dt: 451.15ms, tok/sec: 1162104.92
step: 8670 | loss: 3.222939 | lr: 3.8614e-04 | norm: 0.2503, dt: 450.67ms, tok/sec: 1163364.03
step: 8671 | loss: 3.270551 | lr: 3.8610e-04 | norm: 0.2567, dt: 451.02ms, tok/sec: 1162455.08
step: 8672 | loss: 3.286547 | lr: 3.8605e-04 | norm: 0.2577, dt: 450.41ms, tok/sec: 1164014.93
step: 8673 | loss: 3.235714 | lr: 3.8601e-04 | norm: 0.2650, dt: 450.29ms, tok/sec: 1164330.49
step: 8674 | loss: 3.239003 | lr: 3.8596e-04 | norm: 0.2830, dt: 451.05ms, tok/sec: 1162369.67
step: 8675 | loss: 3.186341 | lr: 3.8592e-04 | norm: 0.2520, dt: 450.81ms, tok/sec: 1163002.25
step: 8676 | loss: 3.261902 | lr: 3.8587e-04 | norm: 0.2784, dt: 451.13ms, tok/sec: 1162157.13
step: 8677 | loss: 3.237889 | lr: 3.8583e-04 | norm: 0.2728, dt: 451.04ms, tok/sec: 1162394.25
step: 8678 | loss: 3.206975 | lr: 3.8578e-04 | norm: 0.2797, dt: 451.04ms, tok/sec: 1162394.87
step: 8679 | loss: 3.243396 | lr: 3.8574e-04 | norm: 0.2461, dt: 450.71ms, tok/sec: 1163239.10
step: 8680 | loss: 3.226820 | lr: 3.8569e-04 | norm: 0.2686, dt: 450.16ms, tok/sec: 1164662.87
step: 8681 | loss: 3.224318 | lr: 3.8564e-04 | norm: 0.2776, dt: 451.18ms, tok/sec: 1162042.29
step: 8682 | loss: 3.213121 | lr: 3.8560e-04 | norm: 0.2825, dt: 451.14ms, tok/sec: 1162147.30
step: 8683 | loss: 3.336734 | lr: 3.8555e-04 | norm: 0.2792, dt: 450.26ms, tok/sec: 1164420.50
step: 8684 | loss: 3.253500 | lr: 3.8551e-04 | norm: 0.2936, dt: 451.27ms, tok/sec: 1161798.55
step: 8685 | loss: 3.230098 | lr: 3.8546e-04 | norm: 0.2743, dt: 450.95ms, tok/sec: 1162619.79
step: 8686 | loss: 3.241854 | lr: 3.8542e-04 | norm: 0.2861, dt: 450.91ms, tok/sec: 1162726.76
step: 8687 | loss: 3.285663 | lr: 3.8537e-04 | norm: 0.2850, dt: 450.32ms, tok/sec: 1164244.19
step: 8688 | loss: 3.251919 | lr: 3.8533e-04 | norm: 0.2610, dt: 450.77ms, tok/sec: 1163103.13
step: 8689 | loss: 3.293083 | lr: 3.8528e-04 | norm: 0.2783, dt: 451.00ms, tok/sec: 1162490.73
step: 8690 | loss: 3.218304 | lr: 3.8524e-04 | norm: 0.2643, dt: 450.67ms, tok/sec: 1163347.41
step: 8691 | loss: 3.239210 | lr: 3.8519e-04 | norm: 0.2601, dt: 450.93ms, tok/sec: 1162678.19
step: 8692 | loss: 3.188650 | lr: 3.8515e-04 | norm: 0.2605, dt: 450.02ms, tok/sec: 1165022.60
step: 8693 | loss: 3.236962 | lr: 3.8510e-04 | norm: 0.2611, dt: 1145.04ms, tok/sec: 457876.58
step: 8694 | loss: 3.191517 | lr: 3.8506e-04 | norm: 0.2509, dt: 450.85ms, tok/sec: 1162897.69
step: 8695 | loss: 3.214507 | lr: 3.8501e-04 | norm: 0.2708, dt: 450.87ms, tok/sec: 1162838.66
step: 8696 | loss: 3.233028 | lr: 3.8497e-04 | norm: 0.2658, dt: 450.09ms, tok/sec: 1164854.74
step: 8697 | loss: 3.209912 | lr: 3.8492e-04 | norm: 0.2773, dt: 450.92ms, tok/sec: 1162694.79
step: 8698 | loss: 3.158335 | lr: 3.8488e-04 | norm: 0.2782, dt: 449.68ms, tok/sec: 1165913.31
step: 8699 | loss: 3.225292 | lr: 3.8483e-04 | norm: 0.2406, dt: 450.37ms, tok/sec: 1164138.18
step: 8700 | loss: 3.174356 | lr: 3.8479e-04 | norm: 0.2942, dt: 450.57ms, tok/sec: 1163609.03
step: 8701 | loss: 3.249443 | lr: 3.8474e-04 | norm: 0.2561, dt: 449.57ms, tok/sec: 1166194.02
step: 8702 | loss: 3.190915 | lr: 3.8470e-04 | norm: 0.2766, dt: 450.78ms, tok/sec: 1163076.06
step: 8703 | loss: 3.187447 | lr: 3.8465e-04 | norm: 0.2509, dt: 450.63ms, tok/sec: 1163449.58
step: 8704 | loss: 3.259653 | lr: 3.8460e-04 | norm: 0.2531, dt: 451.16ms, tok/sec: 1162080.36
step: 8705 | loss: 3.197569 | lr: 3.8456e-04 | norm: 0.2659, dt: 451.22ms, tok/sec: 1161940.36
step: 8706 | loss: 3.330254 | lr: 3.8451e-04 | norm: 0.2931, dt: 450.89ms, tok/sec: 1162785.78
step: 8707 | loss: 3.168704 | lr: 3.8447e-04 | norm: 0.2727, dt: 451.71ms, tok/sec: 1160673.92
step: 8708 | loss: 3.218796 | lr: 3.8442e-04 | norm: 0.2816, dt: 450.49ms, tok/sec: 1163821.49
step: 8709 | loss: 3.210881 | lr: 3.8438e-04 | norm: 0.2989, dt: 451.53ms, tok/sec: 1161137.25
step: 8710 | loss: 3.229616 | lr: 3.8433e-04 | norm: 0.2652, dt: 450.72ms, tok/sec: 1163234.79
step: 8711 | loss: 3.226663 | lr: 3.8429e-04 | norm: 0.2824, dt: 453.49ms, tok/sec: 1156115.63
step: 8712 | loss: 3.202875 | lr: 3.8424e-04 | norm: 0.2553, dt: 450.55ms, tok/sec: 1163663.22
step: 8713 | loss: 3.213152 | lr: 3.8420e-04 | norm: 0.2730, dt: 450.81ms, tok/sec: 1163002.25
step: 8714 | loss: 3.237220 | lr: 3.8415e-04 | norm: 0.3015, dt: 451.48ms, tok/sec: 1161262.33
step: 8715 | loss: 3.202343 | lr: 3.8411e-04 | norm: 0.2588, dt: 451.44ms, tok/sec: 1161365.98
step: 8716 | loss: 3.142394 | lr: 3.8406e-04 | norm: 0.2568, dt: 449.86ms, tok/sec: 1165436.90
step: 8717 | loss: 3.215809 | lr: 3.8402e-04 | norm: 0.3018, dt: 450.70ms, tok/sec: 1163280.94
step: 8718 | loss: 3.201227 | lr: 3.8397e-04 | norm: 0.2443, dt: 451.43ms, tok/sec: 1161385.61
step: 8719 | loss: 3.227496 | lr: 3.8393e-04 | norm: 0.3467, dt: 450.85ms, tok/sec: 1162876.78
step: 8720 | loss: 3.240506 | lr: 3.8388e-04 | norm: 0.2675, dt: 450.45ms, tok/sec: 1163924.98
step: 8721 | loss: 3.242198 | lr: 3.8384e-04 | norm: 0.2876, dt: 450.45ms, tok/sec: 1163924.37
step: 8722 | loss: 3.256397 | lr: 3.8379e-04 | norm: 0.2483, dt: 451.22ms, tok/sec: 1161935.45
step: 8723 | loss: 3.219147 | lr: 3.8374e-04 | norm: 0.2574, dt: 451.06ms, tok/sec: 1162340.18
step: 8724 | loss: 3.243357 | lr: 3.8370e-04 | norm: 0.3217, dt: 451.45ms, tok/sec: 1161354.33
step: 8725 | loss: 3.242404 | lr: 3.8365e-04 | norm: 0.2640, dt: 451.16ms, tok/sec: 1162079.13
step: 8726 | loss: 3.258113 | lr: 3.8361e-04 | norm: 0.2949, dt: 451.11ms, tok/sec: 1162206.26
step: 8727 | loss: 3.222437 | lr: 3.8356e-04 | norm: 0.2819, dt: 450.31ms, tok/sec: 1164294.12
step: 8728 | loss: 3.291398 | lr: 3.8352e-04 | norm: 0.2899, dt: 450.97ms, tok/sec: 1162578.61
step: 8729 | loss: 3.278634 | lr: 3.8347e-04 | norm: 0.2839, dt: 451.05ms, tok/sec: 1162373.36
step: 8730 | loss: 3.243299 | lr: 3.8343e-04 | norm: 0.2506, dt: 450.77ms, tok/sec: 1163090.21
step: 8731 | loss: 3.267661 | lr: 3.8338e-04 | norm: 0.2676, dt: 450.79ms, tok/sec: 1163048.38
step: 8732 | loss: 3.194545 | lr: 3.8334e-04 | norm: 0.2616, dt: 451.90ms, tok/sec: 1160197.50
step: 8733 | loss: 3.301628 | lr: 3.8329e-04 | norm: 0.2599, dt: 450.90ms, tok/sec: 1162748.89
step: 8734 | loss: 3.222600 | lr: 3.8325e-04 | norm: 0.2813, dt: 450.48ms, tok/sec: 1163844.28
step: 8735 | loss: 3.188479 | lr: 3.8320e-04 | norm: 0.2602, dt: 451.27ms, tok/sec: 1161800.39
step: 8736 | loss: 3.271277 | lr: 3.8316e-04 | norm: 0.2770, dt: 450.87ms, tok/sec: 1162831.89
step: 8737 | loss: 3.233555 | lr: 3.8311e-04 | norm: 0.2521, dt: 450.54ms, tok/sec: 1163695.85
step: 8738 | loss: 3.260796 | lr: 3.8307e-04 | norm: 0.2799, dt: 450.60ms, tok/sec: 1163539.46
step: 8739 | loss: 3.192938 | lr: 3.8302e-04 | norm: 0.2766, dt: 1219.42ms, tok/sec: 429947.33
step: 8740 | loss: 3.190831 | lr: 3.8297e-04 | norm: 0.2751, dt: 450.18ms, tok/sec: 1164607.97
step: 8741 | loss: 3.246047 | lr: 3.8293e-04 | norm: 0.2644, dt: 450.39ms, tok/sec: 1164079.02
step: 8742 | loss: 3.169534 | lr: 3.8288e-04 | norm: 0.2628, dt: 451.23ms, tok/sec: 1161901.68
step: 8743 | loss: 3.208130 | lr: 3.8284e-04 | norm: 0.2555, dt: 450.81ms, tok/sec: 1163003.48
step: 8744 | loss: 3.249615 | lr: 3.8279e-04 | norm: 0.2647, dt: 450.57ms, tok/sec: 1163613.96
step: 8745 | loss: 3.374349 | lr: 3.8275e-04 | norm: 0.2984, dt: 451.36ms, tok/sec: 1161572.72
step: 8746 | loss: 3.206534 | lr: 3.8270e-04 | norm: 0.2560, dt: 450.95ms, tok/sec: 1162633.32
step: 8747 | loss: 3.247938 | lr: 3.8266e-04 | norm: 0.2738, dt: 452.19ms, tok/sec: 1159454.25
step: 8748 | loss: 3.181644 | lr: 3.8261e-04 | norm: 0.2505, dt: 450.62ms, tok/sec: 1163477.90
step: 8749 | loss: 3.340635 | lr: 3.8257e-04 | norm: 0.2596, dt: 451.18ms, tok/sec: 1162041.06
validation loss: 3.2367
HellaSwag accuracy: 2877/10042=0.2865
rank 0 sample 0: Hello, I'm a Manpreet, and I love the idea of having people out the door for me or doing some other type of project that we can
rank 0 sample 1: Hello, I'm a Manpreet, because a man is a man, man is not actually a person. Manpreet, the word itself has a
rank 0 sample 2: Hello, I'm a Manpreet, I'm being a Man-Panchor of a Man and it's a different thing to me. I am
rank 0 sample 3: Hello, I'm a Manpreet,
Now, that's not a true Mancatcher.
There are two of them—the ones that you
rank 6 sample 0: Hello, I'm a Manpreet, thank you for the invitation in my mind, and my husband's mom. So now I have a better understanding of
rank 6 sample 1: Hello, I'm a Manpreet,
so what's in there?
Oh, it's a pretty small thing,
and I don't know
rank 6 sample 2: Hello, I'm a Manpreet, so what is the function to find the minimum, maximum, or equal of values of the variables?
You could
rank 6 sample 3: Hello, I'm a Manpreet, and this is the second time, and it's the second time.
SOUSE 4.10) We
rank 5 sample 0: Hello, I'm a Manpreet, so that is my business. I've no business interest, so now you can see that I am a Manp
rank 5 sample 1: Hello, I'm a Manpreet, is what I need to call to help.”
(It is actually in this article that we take the
rank 7 sample 0: Hello, I'm a Manpreet, and I'm a Cee.
Tapping your child's voice or any facial expression would still not work.
rank 5 sample 2: Hello, I'm a Manpreet, in the language that does the best jobs. My name is Dr. Sill. Dr. Sill is a
rank 5 sample 3: Hello, I'm a Manpreet, i'm a Manpreet. I am a Manpreet, i'm a Manpreet, i am
rank 7 sample 1: Hello, I'm a Manpreet, have a man, you'll love those of us if we're able to do things like make a machine and get
rank 7 sample 2: Hello, I'm a Manpreet, who's been reading the book the old time. When I started my training, I had to tell my parents the
rank 7 sample 3: Hello, I'm a Manpreet, kind of
someone that loves to travel. This is a gentleman. There's a guy who looks like a

rank 2 sample 0: Hello, I'm a Manpreet, someone I know but I'm a Manpie, someone that is a Manpreet, someone that is, someone
rank 2 sample 1: Hello, I'm a Manpreet, please. No problem, I'm getting pretty good at what you mean with this.
So now I can get
rank 2 sample 2: Hello, I'm a Manpreet, I'm not a Manpreet in America, but I'm just living in North America, which means I'm
rank 4 sample 0: Hello, I'm a Manpreet, I'm A Manpreet, and a Manpreet, I'm A Manpreet, and a Man
rank 2 sample 3: Hello, I'm a Manpreet, so if I'm going to be a child, why are some of my names, or parents, and a new
rank 4 sample 1: Hello, I'm a Manpreet, isn't it my business to write for us, they had to send that. We were going to make sure them
rank 4 sample 2: Hello, I'm a Manpreet, but then I start with it. You'll be able to write about yourself and to understand its purpose if you do
rank 4 sample 3: Hello, I'm a Manpreet, and my cat came home and I don't remember its exact owner, if you could tell it by the name of
rank 1 sample 0: Hello, I'm a Manpreet, don't you know the answer? Try the answer, in particular, by answering the first question below.
Which
rank 1 sample 1: Hello, I'm a Manpreet, you know that!
I was really wondering if you understand the answer. How can somebody see you, and if
rank 1 sample 2: Hello, I'm a Manpreet, so will my girlfriend be a Manpreet?
A simple man is a man. If she was not her
rank 1 sample 3: Hello, I'm a Manpreet, and I'm also an Excel employee. Of course, Microsoft calls employees "K" that I have to work with
rank 3 sample 0: Hello, I'm a Manpreet, and I'm not in love. We love the idea of my being self-confident. No disrespect for anyone
rank 3 sample 1: Hello, I'm a Manpreet, and that is the reason I can't be on the go. I'm a Manpreet, and that doesn
rank 3 sample 2: Hello, I'm a Manpreet, and you've got to decide you're wrong. Why, we're not gonna try to tell you what to put
rank 3 sample 3: Hello, I'm a Manpreet, and
I can't talk a bit. I'll try
I can't walk with my husband anymore, but
step: 8750 | loss: 3.189708 | lr: 3.8252e-04 | norm: 0.2689, dt: 12076.42ms, tok/sec: 43414.19
step: 8751 | loss: 3.230070 | lr: 3.8248e-04 | norm: 0.2950, dt: 448.99ms, tok/sec: 1167704.40
step: 8752 | loss: 3.241900 | lr: 3.8243e-04 | norm: 0.2679, dt: 448.86ms, tok/sec: 1168047.39
step: 8753 | loss: 3.235174 | lr: 3.8239e-04 | norm: 0.2668, dt: 449.43ms, tok/sec: 1166572.02
step: 8754 | loss: 3.289185 | lr: 3.8234e-04 | norm: 0.2991, dt: 449.60ms, tok/sec: 1166113.01
step: 8755 | loss: 3.242895 | lr: 3.8229e-04 | norm: 0.2940, dt: 452.66ms, tok/sec: 1158248.74
step: 8756 | loss: 3.289425 | lr: 3.8225e-04 | norm: 0.2561, dt: 449.72ms, tok/sec: 1165808.23
step: 8757 | loss: 3.185577 | lr: 3.8220e-04 | norm: 0.2741, dt: 449.73ms, tok/sec: 1165794.01
step: 8758 | loss: 3.216083 | lr: 3.8216e-04 | norm: 0.2773, dt: 449.49ms, tok/sec: 1166411.76
step: 8759 | loss: 3.235374 | lr: 3.8211e-04 | norm: 0.2778, dt: 451.97ms, tok/sec: 1160018.18
step: 8760 | loss: 3.214211 | lr: 3.8207e-04 | norm: 0.2509, dt: 450.11ms, tok/sec: 1164797.35
step: 8761 | loss: 3.237611 | lr: 3.8202e-04 | norm: 0.2557, dt: 450.05ms, tok/sec: 1164965.20
step: 8762 | loss: 3.239343 | lr: 3.8198e-04 | norm: 0.2589, dt: 451.58ms, tok/sec: 1161005.44
step: 8763 | loss: 3.206664 | lr: 3.8193e-04 | norm: 0.2386, dt: 450.63ms, tok/sec: 1163463.74
step: 8764 | loss: 3.238694 | lr: 3.8189e-04 | norm: 0.2471, dt: 451.25ms, tok/sec: 1161858.71
step: 8765 | loss: 3.356534 | lr: 3.8184e-04 | norm: 0.2725, dt: 450.95ms, tok/sec: 1162630.24
step: 8766 | loss: 3.246681 | lr: 3.8180e-04 | norm: 0.2770, dt: 451.11ms, tok/sec: 1162208.11
step: 8767 | loss: 3.203843 | lr: 3.8175e-04 | norm: 0.2736, dt: 450.32ms, tok/sec: 1164259.60
step: 8768 | loss: 3.237276 | lr: 3.8171e-04 | norm: 0.2833, dt: 451.44ms, tok/sec: 1161374.57
step: 8769 | loss: 3.172419 | lr: 3.8166e-04 | norm: 0.2547, dt: 451.35ms, tok/sec: 1161599.71
step: 8770 | loss: 3.238939 | lr: 3.8161e-04 | norm: 0.2553, dt: 450.98ms, tok/sec: 1162551.57
step: 8771 | loss: 3.285278 | lr: 3.8157e-04 | norm: 0.2696, dt: 451.17ms, tok/sec: 1162072.38
step: 8772 | loss: 3.191332 | lr: 3.8152e-04 | norm: 0.2610, dt: 450.54ms, tok/sec: 1163686.00
step: 8773 | loss: 3.164672 | lr: 3.8148e-04 | norm: 0.2650, dt: 452.00ms, tok/sec: 1159940.47
step: 8774 | loss: 3.252026 | lr: 3.8143e-04 | norm: 0.2670, dt: 451.42ms, tok/sec: 1161410.14
step: 8775 | loss: 3.231269 | lr: 3.8139e-04 | norm: 0.2675, dt: 451.04ms, tok/sec: 1162403.47
step: 8776 | loss: 3.202817 | lr: 3.8134e-04 | norm: 0.2406, dt: 452.25ms, tok/sec: 1159282.49
step: 8777 | loss: 3.212204 | lr: 3.8130e-04 | norm: 0.2408, dt: 451.08ms, tok/sec: 1162297.79
step: 8778 | loss: 3.245109 | lr: 3.8125e-04 | norm: 0.2626, dt: 452.04ms, tok/sec: 1159826.06
step: 8779 | loss: 3.252898 | lr: 3.8121e-04 | norm: 0.2591, dt: 451.52ms, tok/sec: 1161149.51
step: 8780 | loss: 3.190777 | lr: 3.8116e-04 | norm: 0.2548, dt: 451.05ms, tok/sec: 1162368.44
step: 8781 | loss: 3.276953 | lr: 3.8112e-04 | norm: 0.2490, dt: 452.21ms, tok/sec: 1159382.12
step: 8782 | loss: 3.243796 | lr: 3.8107e-04 | norm: 0.2730, dt: 450.97ms, tok/sec: 1162574.92
step: 8783 | loss: 3.281527 | lr: 3.8102e-04 | norm: 0.2433, dt: 451.08ms, tok/sec: 1162283.05
step: 8784 | loss: 3.233463 | lr: 3.8098e-04 | norm: 0.2543, dt: 451.47ms, tok/sec: 1161298.52
step: 8785 | loss: 3.205176 | lr: 3.8093e-04 | norm: 0.2547, dt: 450.73ms, tok/sec: 1163198.49
step: 8786 | loss: 3.222744 | lr: 3.8089e-04 | norm: 0.2305, dt: 450.77ms, tok/sec: 1163085.90
step: 8787 | loss: 3.232852 | lr: 3.8084e-04 | norm: 0.2569, dt: 450.93ms, tok/sec: 1162689.26
step: 8788 | loss: 3.221525 | lr: 3.8080e-04 | norm: 0.2466, dt: 451.22ms, tok/sec: 1161930.54
step: 8789 | loss: 3.283118 | lr: 3.8075e-04 | norm: 0.2511, dt: 451.76ms, tok/sec: 1160534.26
step: 8790 | loss: 3.254621 | lr: 3.8071e-04 | norm: 0.2454, dt: 451.72ms, tok/sec: 1160650.64
step: 8791 | loss: 3.327331 | lr: 3.8066e-04 | norm: 0.2559, dt: 451.30ms, tok/sec: 1161733.49
step: 8792 | loss: 3.240375 | lr: 3.8062e-04 | norm: 0.2632, dt: 451.86ms, tok/sec: 1160288.71
step: 8793 | loss: 3.303230 | lr: 3.8057e-04 | norm: 0.2626, dt: 450.46ms, tok/sec: 1163899.11
step: 8794 | loss: 3.269231 | lr: 3.8053e-04 | norm: 0.3023, dt: 451.07ms, tok/sec: 1162314.38
step: 8795 | loss: 3.278860 | lr: 3.8048e-04 | norm: 0.2888, dt: 451.04ms, tok/sec: 1162410.84
step: 8796 | loss: 3.248422 | lr: 3.8044e-04 | norm: 0.2531, dt: 451.48ms, tok/sec: 1161277.66
step: 8797 | loss: 3.248905 | lr: 3.8039e-04 | norm: 0.2749, dt: 450.94ms, tok/sec: 1162651.76
step: 8798 | loss: 3.195727 | lr: 3.8034e-04 | norm: 0.2730, dt: 450.93ms, tok/sec: 1162685.57
step: 8799 | loss: 3.180441 | lr: 3.8030e-04 | norm: 0.2529, dt: 451.46ms, tok/sec: 1161327.34
step: 8800 | loss: 3.284547 | lr: 3.8025e-04 | norm: 0.2613, dt: 451.90ms, tok/sec: 1160196.89
step: 8801 | loss: 3.281328 | lr: 3.8021e-04 | norm: 0.2536, dt: 451.85ms, tok/sec: 1160319.93
step: 8802 | loss: 3.292006 | lr: 3.8016e-04 | norm: 0.2637, dt: 450.89ms, tok/sec: 1162793.16
step: 8803 | loss: 3.220824 | lr: 3.8012e-04 | norm: 0.2590, dt: 450.89ms, tok/sec: 1162783.93
step: 8804 | loss: 3.139545 | lr: 3.8007e-04 | norm: 0.2586, dt: 450.32ms, tok/sec: 1164255.28
step: 8805 | loss: 3.227276 | lr: 3.8003e-04 | norm: 0.2754, dt: 451.13ms, tok/sec: 1162178.01
step: 8806 | loss: 3.236187 | lr: 3.7998e-04 | norm: 0.2415, dt: 450.26ms, tok/sec: 1164419.89
step: 8807 | loss: 3.214011 | lr: 3.7994e-04 | norm: 0.2433, dt: 450.52ms, tok/sec: 1163727.88
step: 8808 | loss: 3.189460 | lr: 3.7989e-04 | norm: 0.2669, dt: 451.83ms, tok/sec: 1160373.20
step: 8809 | loss: 3.206300 | lr: 3.7984e-04 | norm: 0.2592, dt: 451.11ms, tok/sec: 1162216.09
step: 8810 | loss: 3.275310 | lr: 3.7980e-04 | norm: 0.2651, dt: 451.71ms, tok/sec: 1160676.98
step: 8811 | loss: 3.165251 | lr: 3.7975e-04 | norm: 0.2546, dt: 450.70ms, tok/sec: 1163285.87
step: 8812 | loss: 3.235471 | lr: 3.7971e-04 | norm: 0.2390, dt: 451.09ms, tok/sec: 1162260.32
step: 8813 | loss: 3.166848 | lr: 3.7966e-04 | norm: 0.2778, dt: 451.40ms, tok/sec: 1161482.53
step: 8814 | loss: 3.258673 | lr: 3.7962e-04 | norm: 0.2581, dt: 450.84ms, tok/sec: 1162902.61
step: 8815 | loss: 3.314542 | lr: 3.7957e-04 | norm: 0.2902, dt: 450.86ms, tok/sec: 1162867.56
step: 8816 | loss: 3.223434 | lr: 3.7953e-04 | norm: 0.2997, dt: 450.50ms, tok/sec: 1163783.92
step: 8817 | loss: 3.274910 | lr: 3.7948e-04 | norm: 0.2676, dt: 451.55ms, tok/sec: 1161098.01
step: 8818 | loss: 3.206717 | lr: 3.7944e-04 | norm: 0.2845, dt: 451.02ms, tok/sec: 1162445.25
step: 8819 | loss: 3.299897 | lr: 3.7939e-04 | norm: 0.2679, dt: 450.25ms, tok/sec: 1164445.78
step: 8820 | loss: 3.216912 | lr: 3.7935e-04 | norm: 0.2879, dt: 453.37ms, tok/sec: 1156422.05
step: 8821 | loss: 3.189166 | lr: 3.7930e-04 | norm: 0.2709, dt: 450.98ms, tok/sec: 1162546.04
step: 8822 | loss: 3.244942 | lr: 3.7925e-04 | norm: 0.2616, dt: 451.30ms, tok/sec: 1161729.20
step: 8823 | loss: 3.218620 | lr: 3.7921e-04 | norm: 0.2837, dt: 451.95ms, tok/sec: 1160050.61
step: 8824 | loss: 3.290344 | lr: 3.7916e-04 | norm: 0.3269, dt: 451.85ms, tok/sec: 1160310.14
step: 8825 | loss: 3.230663 | lr: 3.7912e-04 | norm: 0.2932, dt: 451.39ms, tok/sec: 1161498.48
step: 8826 | loss: 3.232603 | lr: 3.7907e-04 | norm: 0.2921, dt: 451.75ms, tok/sec: 1160575.91
step: 8827 | loss: 3.297695 | lr: 3.7903e-04 | norm: 0.2940, dt: 452.15ms, tok/sec: 1159533.73
step: 8828 | loss: 3.239537 | lr: 3.7898e-04 | norm: 0.2797, dt: 451.39ms, tok/sec: 1161504.61
step: 8829 | loss: 3.205261 | lr: 3.7894e-04 | norm: 0.2895, dt: 452.49ms, tok/sec: 1158664.95
step: 8830 | loss: 3.250465 | lr: 3.7889e-04 | norm: 0.3030, dt: 451.83ms, tok/sec: 1160360.34
step: 8831 | loss: 3.241313 | lr: 3.7885e-04 | norm: 0.2991, dt: 451.73ms, tok/sec: 1160620.01
step: 8832 | loss: 3.220057 | lr: 3.7880e-04 | norm: 0.2626, dt: 451.22ms, tok/sec: 1161940.36
step: 8833 | loss: 3.188164 | lr: 3.7875e-04 | norm: 0.2717, dt: 451.81ms, tok/sec: 1160420.96
step: 8834 | loss: 3.252404 | lr: 3.7871e-04 | norm: 0.2575, dt: 454.37ms, tok/sec: 1153870.47
step: 8835 | loss: 3.253802 | lr: 3.7866e-04 | norm: 0.2664, dt: 450.44ms, tok/sec: 1163947.16
step: 8836 | loss: 3.212119 | lr: 3.7862e-04 | norm: 0.2628, dt: 451.50ms, tok/sec: 1161202.85
step: 8837 | loss: 3.203567 | lr: 3.7857e-04 | norm: 0.3150, dt: 452.38ms, tok/sec: 1158966.62
step: 8838 | loss: 3.207030 | lr: 3.7853e-04 | norm: 0.2883, dt: 451.30ms, tok/sec: 1161739.02
step: 8839 | loss: 3.207168 | lr: 3.7848e-04 | norm: 0.2830, dt: 451.44ms, tok/sec: 1161362.30
step: 8840 | loss: 3.184393 | lr: 3.7844e-04 | norm: 0.2615, dt: 451.28ms, tok/sec: 1161772.16
step: 8841 | loss: 3.217795 | lr: 3.7839e-04 | norm: 0.2628, dt: 451.64ms, tok/sec: 1160862.64
step: 8842 | loss: 3.208393 | lr: 3.7835e-04 | norm: 0.2871, dt: 450.90ms, tok/sec: 1162771.02
step: 8843 | loss: 3.241607 | lr: 3.7830e-04 | norm: 0.2423, dt: 450.72ms, tok/sec: 1163216.33
step: 8844 | loss: 3.218078 | lr: 3.7825e-04 | norm: 0.2631, dt: 451.01ms, tok/sec: 1162471.67
step: 8845 | loss: 3.175538 | lr: 3.7821e-04 | norm: 0.2679, dt: 450.54ms, tok/sec: 1163686.00
step: 8846 | loss: 3.185933 | lr: 3.7816e-04 | norm: 0.2571, dt: 451.68ms, tok/sec: 1160752.34
step: 8847 | loss: 3.192392 | lr: 3.7812e-04 | norm: 0.2916, dt: 451.57ms, tok/sec: 1161027.51
step: 8848 | loss: 3.282549 | lr: 3.7807e-04 | norm: 0.2457, dt: 451.26ms, tok/sec: 1161823.11
step: 8849 | loss: 3.244203 | lr: 3.7803e-04 | norm: 0.2786, dt: 451.32ms, tok/sec: 1161669.67
step: 8850 | loss: 3.219912 | lr: 3.7798e-04 | norm: 0.2518, dt: 451.26ms, tok/sec: 1161836.61
step: 8851 | loss: 3.195960 | lr: 3.7794e-04 | norm: 0.2563, dt: 451.39ms, tok/sec: 1161485.60
step: 8852 | loss: 3.241116 | lr: 3.7789e-04 | norm: 0.2678, dt: 450.70ms, tok/sec: 1163266.18
step: 8853 | loss: 3.252869 | lr: 3.7785e-04 | norm: 0.2600, dt: 451.03ms, tok/sec: 1162427.43
step: 8854 | loss: 3.150725 | lr: 3.7780e-04 | norm: 0.2798, dt: 450.93ms, tok/sec: 1162686.80
step: 8855 | loss: 3.210504 | lr: 3.7775e-04 | norm: 0.2740, dt: 450.46ms, tok/sec: 1163906.50
step: 8856 | loss: 3.310213 | lr: 3.7771e-04 | norm: 0.2809, dt: 451.20ms, tok/sec: 1161987.02
step: 8857 | loss: 3.308464 | lr: 3.7766e-04 | norm: 0.2665, dt: 451.32ms, tok/sec: 1161664.76
step: 8858 | loss: 3.284646 | lr: 3.7762e-04 | norm: 0.2436, dt: 451.95ms, tok/sec: 1160057.95
step: 8859 | loss: 3.337061 | lr: 3.7757e-04 | norm: 0.3381, dt: 451.84ms, tok/sec: 1160337.69
step: 8860 | loss: 3.255071 | lr: 3.7753e-04 | norm: 0.3017, dt: 451.21ms, tok/sec: 1161949.57
step: 8861 | loss: 3.266868 | lr: 3.7748e-04 | norm: 0.2983, dt: 450.49ms, tok/sec: 1163811.02
step: 8862 | loss: 3.229405 | lr: 3.7744e-04 | norm: 0.2754, dt: 450.48ms, tok/sec: 1163838.12
step: 8863 | loss: 3.209891 | lr: 3.7739e-04 | norm: 0.3141, dt: 451.97ms, tok/sec: 1160003.49
step: 8864 | loss: 3.173736 | lr: 3.7735e-04 | norm: 0.2819, dt: 450.74ms, tok/sec: 1163179.42
step: 8865 | loss: 3.305142 | lr: 3.7730e-04 | norm: 0.3150, dt: 451.19ms, tok/sec: 1162020.18
step: 8866 | loss: 3.290193 | lr: 3.7725e-04 | norm: 0.3020, dt: 452.03ms, tok/sec: 1159854.81
step: 8867 | loss: 3.288481 | lr: 3.7721e-04 | norm: 0.2519, dt: 451.10ms, tok/sec: 1162244.35
step: 8868 | loss: 3.226750 | lr: 3.7716e-04 | norm: 0.2957, dt: 451.11ms, tok/sec: 1162219.16
step: 8869 | loss: 3.251448 | lr: 3.7712e-04 | norm: 0.2562, dt: 451.26ms, tok/sec: 1161823.11
step: 8870 | loss: 3.309800 | lr: 3.7707e-04 | norm: 0.3147, dt: 451.94ms, tok/sec: 1160079.98
step: 8871 | loss: 3.289364 | lr: 3.7703e-04 | norm: 0.2380, dt: 450.49ms, tok/sec: 1163812.25
step: 8872 | loss: 3.273491 | lr: 3.7698e-04 | norm: 0.2980, dt: 451.31ms, tok/sec: 1161689.92
step: 8873 | loss: 3.252531 | lr: 3.7694e-04 | norm: 0.2653, dt: 450.76ms, tok/sec: 1163126.51
step: 8874 | loss: 3.208770 | lr: 3.7689e-04 | norm: 0.2531, dt: 451.11ms, tok/sec: 1162223.46
step: 8875 | loss: 3.255642 | lr: 3.7684e-04 | norm: 0.2650, dt: 450.67ms, tok/sec: 1163348.02
step: 8876 | loss: 3.234530 | lr: 3.7680e-04 | norm: 0.2607, dt: 451.66ms, tok/sec: 1160802.58
step: 8877 | loss: 3.218413 | lr: 3.7675e-04 | norm: 0.2492, dt: 451.45ms, tok/sec: 1161338.99
step: 8878 | loss: 3.256799 | lr: 3.7671e-04 | norm: 0.2478, dt: 451.13ms, tok/sec: 1162177.40
step: 8879 | loss: 3.226276 | lr: 3.7666e-04 | norm: 0.2596, dt: 451.20ms, tok/sec: 1161990.71
step: 8880 | loss: 3.203515 | lr: 3.7662e-04 | norm: 0.2589, dt: 450.58ms, tok/sec: 1163591.79
step: 8881 | loss: 3.216888 | lr: 3.7657e-04 | norm: 0.2617, dt: 451.15ms, tok/sec: 1162114.75
step: 8882 | loss: 3.269760 | lr: 3.7653e-04 | norm: 0.2591, dt: 1158.15ms, tok/sec: 452693.75
step: 8883 | loss: 3.225312 | lr: 3.7648e-04 | norm: 0.2534, dt: 450.54ms, tok/sec: 1163676.15
step: 8884 | loss: 3.205455 | lr: 3.7643e-04 | norm: 0.2586, dt: 450.37ms, tok/sec: 1164122.77
step: 8885 | loss: 3.228542 | lr: 3.7639e-04 | norm: 0.2700, dt: 449.91ms, tok/sec: 1165307.20
step: 8886 | loss: 3.228417 | lr: 3.7634e-04 | norm: 0.2475, dt: 450.70ms, tok/sec: 1163287.10
step: 8887 | loss: 3.223800 | lr: 3.7630e-04 | norm: 0.2729, dt: 449.86ms, tok/sec: 1165441.22
step: 8888 | loss: 3.182205 | lr: 3.7625e-04 | norm: 0.2632, dt: 450.07ms, tok/sec: 1164905.34
step: 8889 | loss: 3.295586 | lr: 3.7621e-04 | norm: 0.2549, dt: 449.87ms, tok/sec: 1165423.31
step: 8890 | loss: 3.271655 | lr: 3.7616e-04 | norm: 0.3287, dt: 450.80ms, tok/sec: 1163013.93
step: 8891 | loss: 3.243633 | lr: 3.7612e-04 | norm: 0.3326, dt: 450.90ms, tok/sec: 1162771.02
step: 8892 | loss: 3.216526 | lr: 3.7607e-04 | norm: 0.2687, dt: 450.07ms, tok/sec: 1164899.17
step: 8893 | loss: 3.185440 | lr: 3.7603e-04 | norm: 0.2771, dt: 450.37ms, tok/sec: 1164115.37
step: 8894 | loss: 3.302893 | lr: 3.7598e-04 | norm: 0.2542, dt: 450.84ms, tok/sec: 1162908.76
step: 8895 | loss: 3.256490 | lr: 3.7593e-04 | norm: 0.2824, dt: 450.46ms, tok/sec: 1163902.19
step: 8896 | loss: 3.196965 | lr: 3.7589e-04 | norm: 0.2757, dt: 451.68ms, tok/sec: 1160751.73
step: 8897 | loss: 3.307716 | lr: 3.7584e-04 | norm: 0.2960, dt: 450.88ms, tok/sec: 1162806.68
step: 8898 | loss: 3.258475 | lr: 3.7580e-04 | norm: 0.2871, dt: 450.74ms, tok/sec: 1163183.11
step: 8899 | loss: 3.239793 | lr: 3.7575e-04 | norm: 0.2758, dt: 450.56ms, tok/sec: 1163645.98
step: 8900 | loss: 3.222380 | lr: 3.7571e-04 | norm: 0.2698, dt: 453.52ms, tok/sec: 1156038.44
step: 8901 | loss: 3.249114 | lr: 3.7566e-04 | norm: 0.2590, dt: 451.04ms, tok/sec: 1162400.40
step: 8902 | loss: 3.245448 | lr: 3.7562e-04 | norm: 0.2526, dt: 451.09ms, tok/sec: 1162258.48
step: 8903 | loss: 3.192391 | lr: 3.7557e-04 | norm: 0.2665, dt: 451.04ms, tok/sec: 1162388.72
step: 8904 | loss: 3.249408 | lr: 3.7552e-04 | norm: 0.2603, dt: 451.59ms, tok/sec: 1160977.25
step: 8905 | loss: 3.290939 | lr: 3.7548e-04 | norm: 0.2411, dt: 450.99ms, tok/sec: 1162529.44
step: 8906 | loss: 3.254462 | lr: 3.7543e-04 | norm: 0.2570, dt: 450.89ms, tok/sec: 1162793.16
step: 8907 | loss: 3.177239 | lr: 3.7539e-04 | norm: 0.2354, dt: 452.90ms, tok/sec: 1157628.64
step: 8908 | loss: 3.197472 | lr: 3.7534e-04 | norm: 0.2653, dt: 450.99ms, tok/sec: 1162535.59
step: 8909 | loss: 3.202950 | lr: 3.7530e-04 | norm: 0.2356, dt: 451.37ms, tok/sec: 1161551.85
step: 8910 | loss: 3.214309 | lr: 3.7525e-04 | norm: 0.2547, dt: 450.54ms, tok/sec: 1163683.54
step: 8911 | loss: 3.202208 | lr: 3.7521e-04 | norm: 0.2634, dt: 450.31ms, tok/sec: 1164281.79
step: 8912 | loss: 3.178618 | lr: 3.7516e-04 | norm: 0.2624, dt: 451.83ms, tok/sec: 1160378.10
step: 8913 | loss: 3.169927 | lr: 3.7511e-04 | norm: 0.2752, dt: 450.30ms, tok/sec: 1164296.58
step: 8914 | loss: 3.260418 | lr: 3.7507e-04 | norm: 0.2629, dt: 450.85ms, tok/sec: 1162879.86
step: 8915 | loss: 3.234390 | lr: 3.7502e-04 | norm: 0.2920, dt: 451.34ms, tok/sec: 1161613.21
step: 8916 | loss: 3.235494 | lr: 3.7498e-04 | norm: 0.3061, dt: 451.21ms, tok/sec: 1161958.78
step: 8917 | loss: 3.217855 | lr: 3.7493e-04 | norm: 0.3248, dt: 450.37ms, tok/sec: 1164138.79
step: 8918 | loss: 3.198741 | lr: 3.7489e-04 | norm: 0.2790, dt: 451.29ms, tok/sec: 1161759.88
step: 8919 | loss: 3.265487 | lr: 3.7484e-04 | norm: 0.2946, dt: 449.89ms, tok/sec: 1165370.19
step: 8920 | loss: 3.228678 | lr: 3.7480e-04 | norm: 0.2707, dt: 451.51ms, tok/sec: 1161178.94
step: 8921 | loss: 3.220351 | lr: 3.7475e-04 | norm: 0.3307, dt: 450.58ms, tok/sec: 1163596.10
step: 8922 | loss: 3.259011 | lr: 3.7470e-04 | norm: 0.2612, dt: 450.87ms, tok/sec: 1162826.98
step: 8923 | loss: 3.186810 | lr: 3.7466e-04 | norm: 0.3130, dt: 450.56ms, tok/sec: 1163645.98
step: 8924 | loss: 3.208819 | lr: 3.7461e-04 | norm: 0.2843, dt: 450.42ms, tok/sec: 1164003.23
step: 8925 | loss: 3.242511 | lr: 3.7457e-04 | norm: 0.2849, dt: 451.16ms, tok/sec: 1162076.67
step: 8926 | loss: 3.280949 | lr: 3.7452e-04 | norm: 0.2854, dt: 451.21ms, tok/sec: 1161967.37
step: 8927 | loss: 3.213859 | lr: 3.7448e-04 | norm: 0.2486, dt: 450.37ms, tok/sec: 1164121.54
step: 8928 | loss: 3.256024 | lr: 3.7443e-04 | norm: 0.2808, dt: 450.87ms, tok/sec: 1162844.19
step: 8929 | loss: 3.226219 | lr: 3.7439e-04 | norm: 0.2504, dt: 1223.63ms, tok/sec: 428471.08
step: 8930 | loss: 3.224874 | lr: 3.7434e-04 | norm: 0.2711, dt: 451.85ms, tok/sec: 1160310.14
step: 8931 | loss: 3.422095 | lr: 3.7429e-04 | norm: 0.3079, dt: 449.73ms, tok/sec: 1165776.71
step: 8932 | loss: 3.280531 | lr: 3.7425e-04 | norm: 0.3194, dt: 450.97ms, tok/sec: 1162577.38
step: 8933 | loss: 3.319848 | lr: 3.7420e-04 | norm: 0.3409, dt: 449.94ms, tok/sec: 1165231.87
step: 8934 | loss: 3.204583 | lr: 3.7416e-04 | norm: 0.3118, dt: 451.00ms, tok/sec: 1162498.10
step: 8935 | loss: 3.247127 | lr: 3.7411e-04 | norm: 0.3219, dt: 450.61ms, tok/sec: 1163512.99
step: 8936 | loss: 3.305256 | lr: 3.7407e-04 | norm: 0.2899, dt: 450.60ms, tok/sec: 1163532.69
step: 8937 | loss: 3.200590 | lr: 3.7402e-04 | norm: 0.3301, dt: 450.17ms, tok/sec: 1164640.66
step: 8938 | loss: 3.295095 | lr: 3.7398e-04 | norm: 0.2868, dt: 450.52ms, tok/sec: 1163748.20
step: 8939 | loss: 3.255528 | lr: 3.7393e-04 | norm: 0.3049, dt: 451.58ms, tok/sec: 1161010.96
step: 8940 | loss: 3.267894 | lr: 3.7388e-04 | norm: 0.2896, dt: 450.06ms, tok/sec: 1164934.96
step: 8941 | loss: 3.293043 | lr: 3.7384e-04 | norm: 0.2890, dt: 451.21ms, tok/sec: 1161964.92
step: 8942 | loss: 3.229566 | lr: 3.7379e-04 | norm: 0.2662, dt: 450.60ms, tok/sec: 1163527.15
step: 8943 | loss: 3.221112 | lr: 3.7375e-04 | norm: 0.2692, dt: 451.34ms, tok/sec: 1161624.87
step: 8944 | loss: 3.235203 | lr: 3.7370e-04 | norm: 0.2787, dt: 449.94ms, tok/sec: 1165232.49
step: 8945 | loss: 3.225688 | lr: 3.7366e-04 | norm: 0.2381, dt: 450.57ms, tok/sec: 1163618.88
step: 8946 | loss: 3.194263 | lr: 3.7361e-04 | norm: 0.2546, dt: 451.00ms, tok/sec: 1162493.18
step: 8947 | loss: 3.237384 | lr: 3.7356e-04 | norm: 0.2530, dt: 451.22ms, tok/sec: 1161932.99
step: 8948 | loss: 3.216899 | lr: 3.7352e-04 | norm: 0.2638, dt: 450.72ms, tok/sec: 1163234.79
step: 8949 | loss: 3.260627 | lr: 3.7347e-04 | norm: 0.2723, dt: 451.16ms, tok/sec: 1162100.01
step: 8950 | loss: 3.262054 | lr: 3.7343e-04 | norm: 0.2777, dt: 451.46ms, tok/sec: 1161329.18
step: 8951 | loss: 3.263444 | lr: 3.7338e-04 | norm: 0.3159, dt: 451.11ms, tok/sec: 1162217.32
step: 8952 | loss: 3.210849 | lr: 3.7334e-04 | norm: 0.2845, dt: 450.93ms, tok/sec: 1162678.19
step: 8953 | loss: 3.252955 | lr: 3.7329e-04 | norm: 0.2790, dt: 450.36ms, tok/sec: 1164157.90
step: 8954 | loss: 3.206857 | lr: 3.7325e-04 | norm: 0.2661, dt: 451.49ms, tok/sec: 1161239.03
step: 8955 | loss: 3.187123 | lr: 3.7320e-04 | norm: 0.3182, dt: 450.42ms, tok/sec: 1163993.98
step: 8956 | loss: 3.225784 | lr: 3.7315e-04 | norm: 0.2824, dt: 451.11ms, tok/sec: 1162228.99
step: 8957 | loss: 3.248034 | lr: 3.7311e-04 | norm: 0.2849, dt: 452.44ms, tok/sec: 1158800.50
step: 8958 | loss: 3.194605 | lr: 3.7306e-04 | norm: 0.2881, dt: 450.17ms, tok/sec: 1164636.96
step: 8959 | loss: 3.238136 | lr: 3.7302e-04 | norm: 0.2690, dt: 450.39ms, tok/sec: 1164080.87
step: 8960 | loss: 3.228431 | lr: 3.7297e-04 | norm: 0.2749, dt: 450.49ms, tok/sec: 1163812.25
step: 8961 | loss: 3.259583 | lr: 3.7293e-04 | norm: 0.2818, dt: 450.67ms, tok/sec: 1163360.33
step: 8962 | loss: 3.277812 | lr: 3.7288e-04 | norm: 0.2685, dt: 451.11ms, tok/sec: 1162217.32
step: 8963 | loss: 3.214870 | lr: 3.7284e-04 | norm: 0.2785, dt: 451.80ms, tok/sec: 1160443.01
step: 8964 | loss: 3.167421 | lr: 3.7279e-04 | norm: 0.2635, dt: 451.08ms, tok/sec: 1162282.43
step: 8965 | loss: 3.220007 | lr: 3.7274e-04 | norm: 0.2574, dt: 451.59ms, tok/sec: 1160980.31
step: 8966 | loss: 3.254993 | lr: 3.7270e-04 | norm: 0.2622, dt: 450.97ms, tok/sec: 1162588.45
step: 8967 | loss: 3.237881 | lr: 3.7265e-04 | norm: 0.2581, dt: 450.61ms, tok/sec: 1163510.52
step: 8968 | loss: 3.233737 | lr: 3.7261e-04 | norm: 0.2569, dt: 450.40ms, tok/sec: 1164041.43
step: 8969 | loss: 3.205200 | lr: 3.7256e-04 | norm: 0.2421, dt: 451.50ms, tok/sec: 1161225.54
step: 8970 | loss: 3.273996 | lr: 3.7252e-04 | norm: 0.2657, dt: 450.20ms, tok/sec: 1164574.67
step: 8971 | loss: 3.211482 | lr: 3.7247e-04 | norm: 0.2556, dt: 450.68ms, tok/sec: 1163317.25
step: 8972 | loss: 3.272645 | lr: 3.7242e-04 | norm: 0.2643, dt: 450.52ms, tok/sec: 1163734.04
step: 8973 | loss: 3.223113 | lr: 3.7238e-04 | norm: 0.2629, dt: 450.98ms, tok/sec: 1162544.81
step: 8974 | loss: 3.202480 | lr: 3.7233e-04 | norm: 0.2639, dt: 450.51ms, tok/sec: 1163760.52
step: 8975 | loss: 3.211678 | lr: 3.7229e-04 | norm: 0.2743, dt: 451.26ms, tok/sec: 1161831.09
step: 8976 | loss: 3.237099 | lr: 3.7224e-04 | norm: 0.2427, dt: 451.35ms, tok/sec: 1161596.03
step: 8977 | loss: 3.250752 | lr: 3.7220e-04 | norm: 0.2629, dt: 450.25ms, tok/sec: 1164435.92
step: 8978 | loss: 3.242690 | lr: 3.7215e-04 | norm: 0.2589, dt: 450.51ms, tok/sec: 1163772.84
step: 8979 | loss: 3.205771 | lr: 3.7210e-04 | norm: 0.2350, dt: 449.89ms, tok/sec: 1165367.11
step: 8980 | loss: 3.235316 | lr: 3.7206e-04 | norm: 0.2602, dt: 451.89ms, tok/sec: 1160212.19
step: 8981 | loss: 3.184633 | lr: 3.7201e-04 | norm: 0.2537, dt: 450.31ms, tok/sec: 1164289.19
step: 8982 | loss: 3.164230 | lr: 3.7197e-04 | norm: 0.2554, dt: 450.94ms, tok/sec: 1162644.99
step: 8983 | loss: 3.203969 | lr: 3.7192e-04 | norm: 0.2412, dt: 451.14ms, tok/sec: 1162151.60
step: 8984 | loss: 3.217323 | lr: 3.7188e-04 | norm: 0.2683, dt: 451.51ms, tok/sec: 1161199.79
step: 8985 | loss: 3.185820 | lr: 3.7183e-04 | norm: 0.2664, dt: 450.36ms, tok/sec: 1164146.19
step: 8986 | loss: 3.236530 | lr: 3.7179e-04 | norm: 0.2487, dt: 451.07ms, tok/sec: 1162332.81
step: 8987 | loss: 3.262058 | lr: 3.7174e-04 | norm: 0.2780, dt: 451.78ms, tok/sec: 1160499.35
step: 8988 | loss: 3.220277 | lr: 3.7169e-04 | norm: 0.2615, dt: 451.04ms, tok/sec: 1162392.41
step: 8989 | loss: 3.239080 | lr: 3.7165e-04 | norm: 0.2629, dt: 450.49ms, tok/sec: 1163829.50
step: 8990 | loss: 3.285493 | lr: 3.7160e-04 | norm: 0.2649, dt: 450.85ms, tok/sec: 1162889.08
step: 8991 | loss: 3.195304 | lr: 3.7156e-04 | norm: 0.2943, dt: 451.06ms, tok/sec: 1162350.01
step: 8992 | loss: 3.235536 | lr: 3.7151e-04 | norm: 0.3184, dt: 450.39ms, tok/sec: 1164070.39
step: 8993 | loss: 3.244640 | lr: 3.7147e-04 | norm: 0.2748, dt: 450.21ms, tok/sec: 1164538.28
step: 8994 | loss: 3.295045 | lr: 3.7142e-04 | norm: 0.3277, dt: 451.21ms, tok/sec: 1161956.94
step: 8995 | loss: 3.250158 | lr: 3.7137e-04 | norm: 0.2937, dt: 449.82ms, tok/sec: 1165560.44
step: 8996 | loss: 3.209939 | lr: 3.7133e-04 | norm: 0.2868, dt: 452.76ms, tok/sec: 1157971.84
step: 8997 | loss: 3.248318 | lr: 3.7128e-04 | norm: 0.2847, dt: 450.50ms, tok/sec: 1163795.62
step: 8998 | loss: 3.239480 | lr: 3.7124e-04 | norm: 0.2896, dt: 450.86ms, tok/sec: 1162871.25
step: 8999 | loss: 3.301046 | lr: 3.7119e-04 | norm: 0.2816, dt: 451.78ms, tok/sec: 1160499.35
validation loss: 3.2311
HellaSwag accuracy: 2902/10042=0.2890
rank 2 sample 0: Hello, I'm a Manpreet, Pempham, and a Master of Science in English and I have been to Canada for over a while.
rank 2 sample 1: Hello, I'm a Manpreet, who's called me. I'm called this. Thanks for that. That's a great thing to show.

rank 2 sample 2: Hello, I'm a Manpreet, I'm not a Manpreet here, I'm a Manpreet in that case, just an Manp
rank 2 sample 3: Hello, I'm a Manpreet, a M'S, and a Khat. Let's make a little list of some possible scenarios that would make
rank 0 sample 0: Hello, I'm a Manpreet, and I just want you to do all the shopping.
The way I can do all this and more, I
rank 6 sample 0: Hello, I'm a Manpreet, thank you for listening to and discussing all of the questions that have come to mind.
How do you use therank 0 sample 1: Hello, I'm a Manpreet, I live in a place where people really do things
right. That's a great way to talk about what's

rank 5 sample 0: Hello, I'm a Manpreet, but that doesn't mean I'm just writing to somebody else. My mom told me that I could write to people
rank 0 sample 2: Hello, I'm a Manpreet, I'm trying to look good.
I mean, I know the words, I'm not a Manpreet
rank 6 sample 1: Hello, I'm a Manpreet, so I want to give a brief overview a little, so I'm actually going to go to a few places on
rank 4 sample 0: Hello, I'm a Manpreet, I'm your Manpreet.
What are your thoughts on this project?
We like to think of the
rank 0 sample 3: Hello, I'm a Manpreet, so a manpreet is just my manhood. I'm from Australia and New Zealand.
P.Srank 5 sample 1: Hello, I'm a Manpreet, sorry my name is Gollac. I think I've lost interest. Thanks in advance for your information<|endoftext|>An

rank 6 sample 2: Hello, I'm a Manpreet, and I have been trying to figure out how to get the right number. What I didn't know is where do
rank 4 sample 1: Hello, I'm a Manpreet, Mama, we'll get off of that later! How do you write to an online account? (email@
rank 5 sample 2: Hello, I'm a Manpreet, my wife.
One of our wives, in my house, who was going out by the back door, was
rank 6 sample 3: Hello, I'm a Manpreet, and I am using this as my starting point, and I am about to get a little niggling at about
rank 4 sample 2: Hello, I'm a Manpreet, a Godwrought person."
And the person who, you know, is standing next to a mother, anrank 5 sample 3: Hello, I'm a Manpreet, is this the case? You know, it's an important word, and what that is, 'I'm not

rank 4 sample 3: Hello, I'm a Manpreet, and the old,
new, etc. name would it follow?’
So I think the old man
rank 7 sample 0: Hello, I'm a Manpreet, and I'm a Mother
I'm Going to New School. I do not read. I cannot go
to
rank 7 sample 1: Hello, I'm a Manpreet, here is how to write the message into your email. As soon as I write I am ready to enter it in
rank 7 sample 2: Hello, I'm a Manpreet, manpreet; am I gonna put it in case of any emergency
- I am going to give it an
rank 7 sample 3: Hello, I'm a Manpreet, really. I'm the most awesome that I have ever seen, I want to show you how awesome you are.
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going back over the post I wrote on Monday.
From Friday, February 28, 2004:

rank 3 sample 1: Hello, I'm a Manpreet, and my mother is a pretty woman.
My Dad's name was Marley.
My Mom's name:
rank 3 sample 2: Hello, I'm a Manpreet, and you've put lots of new stuff in there.
Now, you are an old man: I've tried
rank 3 sample 3: Hello, I'm a Manpreet, and for the rest of the things, I have no desire.
My wife just got tired of what she is
rank 1 sample 0: Hello, I'm a Manpreet, Androgyne, Androchrome and it's going by the way.
Santin, what's
rank 1 sample 1: Hello, I'm a Manpreet, you know. Now, I am a Manpreet a manper. Now, your new Name, I'm
rank 1 sample 2: Hello, I'm a Manpreet, so far as I'm concerned. I'm not a new employee at this company. I have always been in touch
rank 1 sample 3: Hello, I'm a Manpreet, and I'm also an African American. Welcome to the home page of the Indian Country Heritage in America.
We
step: 9000 | loss: 3.262787 | lr: 3.7115e-04 | norm: 0.3086, dt: 12093.13ms, tok/sec: 43354.21
step: 9001 | loss: 3.273562 | lr: 3.7110e-04 | norm: 0.2680, dt: 449.28ms, tok/sec: 1166959.56
step: 9002 | loss: 3.210220 | lr: 3.7105e-04 | norm: 0.2701, dt: 449.14ms, tok/sec: 1167318.84
step: 9003 | loss: 3.230048 | lr: 3.7101e-04 | norm: 0.2595, dt: 449.62ms, tok/sec: 1166070.34
step: 9004 | loss: 3.204009 | lr: 3.7096e-04 | norm: 0.2632, dt: 449.82ms, tok/sec: 1165562.29
step: 9005 | loss: 3.254689 | lr: 3.7092e-04 | norm: 0.2723, dt: 450.92ms, tok/sec: 1162704.62
step: 9006 | loss: 3.260834 | lr: 3.7087e-04 | norm: 0.2743, dt: 453.28ms, tok/sec: 1156643.45
step: 9007 | loss: 3.228540 | lr: 3.7083e-04 | norm: 0.2599, dt: 450.82ms, tok/sec: 1162960.42
step: 9008 | loss: 3.223403 | lr: 3.7078e-04 | norm: 0.2613, dt: 450.24ms, tok/sec: 1164468.60
step: 9009 | loss: 3.195011 | lr: 3.7074e-04 | norm: 0.2659, dt: 451.66ms, tok/sec: 1160792.78
step: 9010 | loss: 3.229276 | lr: 3.7069e-04 | norm: 0.2667, dt: 451.45ms, tok/sec: 1161335.93
step: 9011 | loss: 3.166279 | lr: 3.7064e-04 | norm: 0.2701, dt: 451.47ms, tok/sec: 1161289.32
step: 9012 | loss: 3.209296 | lr: 3.7060e-04 | norm: 0.2548, dt: 450.39ms, tok/sec: 1164075.32
step: 9013 | loss: 3.235485 | lr: 3.7055e-04 | norm: 0.2625, dt: 450.93ms, tok/sec: 1162679.42
step: 9014 | loss: 3.204618 | lr: 3.7051e-04 | norm: 0.2683, dt: 450.71ms, tok/sec: 1163244.64
step: 9015 | loss: 3.179345 | lr: 3.7046e-04 | norm: 0.2514, dt: 451.68ms, tok/sec: 1160744.99
step: 9016 | loss: 3.198311 | lr: 3.7042e-04 | norm: 0.2525, dt: 451.37ms, tok/sec: 1161541.42
step: 9017 | loss: 3.238556 | lr: 3.7037e-04 | norm: 0.2669, dt: 451.76ms, tok/sec: 1160533.04
step: 9018 | loss: 3.247012 | lr: 3.7032e-04 | norm: 0.2551, dt: 451.39ms, tok/sec: 1161484.98
step: 9019 | loss: 3.182030 | lr: 3.7028e-04 | norm: 0.2528, dt: 451.79ms, tok/sec: 1160479.75
step: 9020 | loss: 3.203029 | lr: 3.7023e-04 | norm: 0.2507, dt: 451.01ms, tok/sec: 1162479.66
step: 9021 | loss: 3.221167 | lr: 3.7019e-04 | norm: 0.2611, dt: 451.41ms, tok/sec: 1161439.59
step: 9022 | loss: 3.212641 | lr: 3.7014e-04 | norm: 0.2622, dt: 450.92ms, tok/sec: 1162698.48
step: 9023 | loss: 3.211839 | lr: 3.7010e-04 | norm: 0.2711, dt: 452.06ms, tok/sec: 1159785.08
step: 9024 | loss: 3.212708 | lr: 3.7005e-04 | norm: 0.2705, dt: 450.30ms, tok/sec: 1164304.60
step: 9025 | loss: 3.228863 | lr: 3.7000e-04 | norm: 0.2683, dt: 450.94ms, tok/sec: 1162662.21
step: 9026 | loss: 3.186686 | lr: 3.6996e-04 | norm: 0.2610, dt: 451.57ms, tok/sec: 1161038.54
step: 9027 | loss: 3.270640 | lr: 3.6991e-04 | norm: 0.2849, dt: 451.14ms, tok/sec: 1162147.91
step: 9028 | loss: 3.249076 | lr: 3.6987e-04 | norm: 0.2713, dt: 450.34ms, tok/sec: 1164199.81
step: 9029 | loss: 3.233513 | lr: 3.6982e-04 | norm: 0.2892, dt: 451.23ms, tok/sec: 1161904.75
step: 9030 | loss: 3.205168 | lr: 3.6978e-04 | norm: 0.2522, dt: 451.90ms, tok/sec: 1160182.81
step: 9031 | loss: 3.285429 | lr: 3.6973e-04 | norm: 0.2777, dt: 450.37ms, tok/sec: 1164122.15
step: 9032 | loss: 3.347717 | lr: 3.6968e-04 | norm: 0.2967, dt: 450.37ms, tok/sec: 1164136.33
step: 9033 | loss: 3.238626 | lr: 3.6964e-04 | norm: 0.3056, dt: 451.16ms, tok/sec: 1162077.90
step: 9034 | loss: 3.239030 | lr: 3.6959e-04 | norm: 0.2718, dt: 450.79ms, tok/sec: 1163039.15
step: 9035 | loss: 3.249259 | lr: 3.6955e-04 | norm: 0.3071, dt: 450.81ms, tok/sec: 1162983.79
step: 9036 | loss: 3.244800 | lr: 3.6950e-04 | norm: 0.2649, dt: 450.86ms, tok/sec: 1162873.71
step: 9037 | loss: 3.294864 | lr: 3.6946e-04 | norm: 0.3197, dt: 451.31ms, tok/sec: 1161702.81
step: 9038 | loss: 3.274286 | lr: 3.6941e-04 | norm: 0.3000, dt: 451.00ms, tok/sec: 1162501.79
step: 9039 | loss: 3.201612 | lr: 3.6936e-04 | norm: 0.2741, dt: 451.83ms, tok/sec: 1160364.02
step: 9040 | loss: 3.266109 | lr: 3.6932e-04 | norm: 0.2822, dt: 450.58ms, tok/sec: 1163591.79
step: 9041 | loss: 3.200326 | lr: 3.6927e-04 | norm: 0.2762, dt: 451.43ms, tok/sec: 1161388.06
step: 9042 | loss: 3.205000 | lr: 3.6923e-04 | norm: 0.2507, dt: 451.36ms, tok/sec: 1161582.53
step: 9043 | loss: 3.226210 | lr: 3.6918e-04 | norm: 0.2731, dt: 450.94ms, tok/sec: 1162667.12
step: 9044 | loss: 3.211209 | lr: 3.6914e-04 | norm: 0.2536, dt: 450.29ms, tok/sec: 1164345.90
step: 9045 | loss: 3.304516 | lr: 3.6909e-04 | norm: 0.2825, dt: 450.75ms, tok/sec: 1163142.50
step: 9046 | loss: 3.217404 | lr: 3.6904e-04 | norm: 0.2463, dt: 451.65ms, tok/sec: 1160832.61
step: 9047 | loss: 3.266945 | lr: 3.6900e-04 | norm: 0.2731, dt: 450.60ms, tok/sec: 1163526.53
step: 9048 | loss: 3.218364 | lr: 3.6895e-04 | norm: 0.2461, dt: 451.01ms, tok/sec: 1162479.66
step: 9049 | loss: 3.195570 | lr: 3.6891e-04 | norm: 0.2561, dt: 450.81ms, tok/sec: 1162986.87
step: 9050 | loss: 3.232722 | lr: 3.6886e-04 | norm: 0.2430, dt: 450.65ms, tok/sec: 1163391.72
step: 9051 | loss: 3.349277 | lr: 3.6882e-04 | norm: 0.3205, dt: 451.02ms, tok/sec: 1162439.72
step: 9052 | loss: 3.330030 | lr: 3.6877e-04 | norm: 0.3093, dt: 451.59ms, tok/sec: 1160984.60
step: 9053 | loss: 3.235585 | lr: 3.6872e-04 | norm: 0.2861, dt: 451.19ms, tok/sec: 1162019.57
step: 9054 | loss: 3.182495 | lr: 3.6868e-04 | norm: 0.2701, dt: 450.01ms, tok/sec: 1165048.52
step: 9055 | loss: 3.221242 | lr: 3.6863e-04 | norm: 0.2640, dt: 451.23ms, tok/sec: 1161899.84
step: 9056 | loss: 3.216155 | lr: 3.6859e-04 | norm: 0.2584, dt: 451.32ms, tok/sec: 1161676.42
step: 9057 | loss: 3.264071 | lr: 3.6854e-04 | norm: 0.2623, dt: 450.78ms, tok/sec: 1163056.38
step: 9058 | loss: 3.200255 | lr: 3.6850e-04 | norm: 0.2679, dt: 451.21ms, tok/sec: 1161971.67
step: 9059 | loss: 3.256078 | lr: 3.6845e-04 | norm: 0.2608, dt: 450.34ms, tok/sec: 1164194.26
step: 9060 | loss: 3.276416 | lr: 3.6840e-04 | norm: 0.2621, dt: 451.63ms, tok/sec: 1160888.38
step: 9061 | loss: 3.219674 | lr: 3.6836e-04 | norm: 0.2735, dt: 450.36ms, tok/sec: 1164148.04
step: 9062 | loss: 3.208368 | lr: 3.6831e-04 | norm: 0.2728, dt: 450.86ms, tok/sec: 1162871.25
step: 9063 | loss: 3.279299 | lr: 3.6827e-04 | norm: 0.2767, dt: 451.54ms, tok/sec: 1161101.69
step: 9064 | loss: 3.274623 | lr: 3.6822e-04 | norm: 0.2841, dt: 451.58ms, tok/sec: 1161020.77
step: 9065 | loss: 3.250154 | lr: 3.6818e-04 | norm: 0.2947, dt: 450.78ms, tok/sec: 1163066.22
step: 9066 | loss: 3.265361 | lr: 3.6813e-04 | norm: 0.2485, dt: 450.76ms, tok/sec: 1163113.59
step: 9067 | loss: 3.231520 | lr: 3.6808e-04 | norm: 0.2655, dt: 451.18ms, tok/sec: 1162041.67
step: 9068 | loss: 3.235364 | lr: 3.6804e-04 | norm: 0.2744, dt: 450.37ms, tok/sec: 1164140.03
step: 9069 | loss: 3.254610 | lr: 3.6799e-04 | norm: 0.2434, dt: 450.90ms, tok/sec: 1162767.95
step: 9070 | loss: 3.257637 | lr: 3.6795e-04 | norm: 0.2846, dt: 450.98ms, tok/sec: 1162564.47
step: 9071 | loss: 3.217358 | lr: 3.6790e-04 | norm: 0.2546, dt: 1160.55ms, tok/sec: 451759.10
step: 9072 | loss: 3.353757 | lr: 3.6786e-04 | norm: 0.2974, dt: 448.87ms, tok/sec: 1168005.83
step: 9073 | loss: 3.216100 | lr: 3.6781e-04 | norm: 0.3215, dt: 450.78ms, tok/sec: 1163058.84
step: 9074 | loss: 3.253471 | lr: 3.6776e-04 | norm: 0.2896, dt: 451.39ms, tok/sec: 1161491.73
step: 9075 | loss: 3.283775 | lr: 3.6772e-04 | norm: 0.2645, dt: 451.42ms, tok/sec: 1161413.82
step: 9076 | loss: 3.320042 | lr: 3.6767e-04 | norm: 0.3648, dt: 450.42ms, tok/sec: 1164002.61
step: 9077 | loss: 3.310624 | lr: 3.6763e-04 | norm: 0.3734, dt: 451.91ms, tok/sec: 1160172.40
step: 9078 | loss: 3.220518 | lr: 3.6758e-04 | norm: 0.3024, dt: 452.97ms, tok/sec: 1157440.97
step: 9079 | loss: 3.229968 | lr: 3.6754e-04 | norm: 0.2940, dt: 450.08ms, tok/sec: 1164864.61
step: 9080 | loss: 3.259396 | lr: 3.6749e-04 | norm: 0.2874, dt: 450.98ms, tok/sec: 1162564.47
step: 9081 | loss: 3.195398 | lr: 3.6744e-04 | norm: 0.2729, dt: 452.73ms, tok/sec: 1158057.21
step: 9082 | loss: 3.248670 | lr: 3.6740e-04 | norm: 0.3215, dt: 451.47ms, tok/sec: 1161287.48
step: 9083 | loss: 3.222243 | lr: 3.6735e-04 | norm: 0.2830, dt: 450.91ms, tok/sec: 1162736.59
step: 9084 | loss: 3.201110 | lr: 3.6731e-04 | norm: 0.2712, dt: 451.41ms, tok/sec: 1161451.24
step: 9085 | loss: 3.190732 | lr: 3.6726e-04 | norm: 0.2665, dt: 451.46ms, tok/sec: 1161317.53
step: 9086 | loss: 3.162183 | lr: 3.6721e-04 | norm: 0.2935, dt: 450.02ms, tok/sec: 1165039.26
step: 9087 | loss: 3.190403 | lr: 3.6717e-04 | norm: 0.2813, dt: 450.55ms, tok/sec: 1163665.06
step: 9088 | loss: 3.195976 | lr: 3.6712e-04 | norm: 0.2993, dt: 452.26ms, tok/sec: 1159265.38
step: 9089 | loss: 3.203240 | lr: 3.6708e-04 | norm: 0.2576, dt: 451.22ms, tok/sec: 1161942.20
step: 9090 | loss: 3.238345 | lr: 3.6703e-04 | norm: 0.2541, dt: 451.73ms, tok/sec: 1160631.04
step: 9091 | loss: 3.224045 | lr: 3.6699e-04 | norm: 0.2565, dt: 453.15ms, tok/sec: 1156987.29
step: 9092 | loss: 3.177402 | lr: 3.6694e-04 | norm: 0.2649, dt: 450.23ms, tok/sec: 1164479.70
step: 9093 | loss: 3.212659 | lr: 3.6689e-04 | norm: 0.2530, dt: 451.10ms, tok/sec: 1162232.06
step: 9094 | loss: 3.196244 | lr: 3.6685e-04 | norm: 0.2567, dt: 450.78ms, tok/sec: 1163065.60
step: 9095 | loss: 3.200559 | lr: 3.6680e-04 | norm: 0.2770, dt: 452.01ms, tok/sec: 1159903.76
step: 9096 | loss: 3.210812 | lr: 3.6676e-04 | norm: 0.2543, dt: 451.34ms, tok/sec: 1161618.74
step: 9097 | loss: 3.237064 | lr: 3.6671e-04 | norm: 0.2804, dt: 451.64ms, tok/sec: 1160846.09
step: 9098 | loss: 3.264440 | lr: 3.6667e-04 | norm: 0.2848, dt: 451.98ms, tok/sec: 1159988.19
step: 9099 | loss: 3.332042 | lr: 3.6662e-04 | norm: 0.2905, dt: 451.00ms, tok/sec: 1162511.62
step: 9100 | loss: 3.231437 | lr: 3.6657e-04 | norm: 0.3088, dt: 450.88ms, tok/sec: 1162811.60
step: 9101 | loss: 3.219568 | lr: 3.6653e-04 | norm: 0.2705, dt: 451.80ms, tok/sec: 1160430.76
step: 9102 | loss: 3.209802 | lr: 3.6648e-04 | norm: 0.2668, dt: 451.04ms, tok/sec: 1162394.25
step: 9103 | loss: 3.248962 | lr: 3.6644e-04 | norm: 0.2800, dt: 451.58ms, tok/sec: 1161003.60
step: 9104 | loss: 3.238338 | lr: 3.6639e-04 | norm: 0.2775, dt: 451.24ms, tok/sec: 1161885.72
step: 9105 | loss: 3.210143 | lr: 3.6635e-04 | norm: 0.3705, dt: 451.23ms, tok/sec: 1161907.82
step: 9106 | loss: 3.303349 | lr: 3.6630e-04 | norm: 0.2843, dt: 451.30ms, tok/sec: 1161734.11
step: 9107 | loss: 3.266277 | lr: 3.6625e-04 | norm: 0.2938, dt: 451.13ms, tok/sec: 1162157.13
step: 9108 | loss: 3.191179 | lr: 3.6621e-04 | norm: 0.2716, dt: 450.79ms, tok/sec: 1163049.61
step: 9109 | loss: 3.201559 | lr: 3.6616e-04 | norm: 0.2875, dt: 451.77ms, tok/sec: 1160531.81
step: 9110 | loss: 3.197909 | lr: 3.6612e-04 | norm: 0.2781, dt: 451.47ms, tok/sec: 1161281.96
step: 9111 | loss: 3.219937 | lr: 3.6607e-04 | norm: 0.2751, dt: 450.32ms, tok/sec: 1164250.97
step: 9112 | loss: 3.229069 | lr: 3.6602e-04 | norm: 0.2543, dt: 451.32ms, tok/sec: 1161680.10
step: 9113 | loss: 3.236308 | lr: 3.6598e-04 | norm: 0.2714, dt: 451.91ms, tok/sec: 1160155.88
step: 9114 | loss: 3.275522 | lr: 3.6593e-04 | norm: 0.2743, dt: 450.85ms, tok/sec: 1162880.47
step: 9115 | loss: 3.207247 | lr: 3.6589e-04 | norm: 0.2695, dt: 450.60ms, tok/sec: 1163541.31
step: 9116 | loss: 3.236550 | lr: 3.6584e-04 | norm: 0.3023, dt: 451.35ms, tok/sec: 1161605.24
step: 9117 | loss: 3.223287 | lr: 3.6580e-04 | norm: 0.2464, dt: 451.81ms, tok/sec: 1160419.74
step: 9118 | loss: 3.241132 | lr: 3.6575e-04 | norm: 0.2760, dt: 450.68ms, tok/sec: 1163315.41
step: 9119 | loss: 3.256364 | lr: 3.6570e-04 | norm: 0.2626, dt: 1230.05ms, tok/sec: 426233.05
step: 9120 | loss: 3.273932 | lr: 3.6566e-04 | norm: 0.2701, dt: 449.04ms, tok/sec: 1167563.66
step: 9121 | loss: 3.227876 | lr: 3.6561e-04 | norm: 0.2542, dt: 450.75ms, tok/sec: 1163138.19
step: 9122 | loss: 3.155927 | lr: 3.6557e-04 | norm: 0.2643, dt: 450.71ms, tok/sec: 1163238.48
step: 9123 | loss: 3.314802 | lr: 3.6552e-04 | norm: 0.2670, dt: 449.40ms, tok/sec: 1166650.00
step: 9124 | loss: 3.241084 | lr: 3.6547e-04 | norm: 0.2579, dt: 450.05ms, tok/sec: 1164965.20
step: 9125 | loss: 3.244672 | lr: 3.6543e-04 | norm: 0.2474, dt: 450.57ms, tok/sec: 1163613.96
step: 9126 | loss: 3.221380 | lr: 3.6538e-04 | norm: 0.2635, dt: 450.00ms, tok/sec: 1165073.21
step: 9127 | loss: 3.195731 | lr: 3.6534e-04 | norm: 0.2562, dt: 450.04ms, tok/sec: 1164976.92
step: 9128 | loss: 3.211938 | lr: 3.6529e-04 | norm: 0.2809, dt: 450.43ms, tok/sec: 1163973.65
step: 9129 | loss: 3.220259 | lr: 3.6525e-04 | norm: 0.2374, dt: 450.90ms, tok/sec: 1162760.57
step: 9130 | loss: 3.257896 | lr: 3.6520e-04 | norm: 0.2629, dt: 450.54ms, tok/sec: 1163692.78
step: 9131 | loss: 3.165335 | lr: 3.6515e-04 | norm: 0.2354, dt: 450.60ms, tok/sec: 1163520.37
step: 9132 | loss: 3.194426 | lr: 3.6511e-04 | norm: 0.2656, dt: 451.23ms, tok/sec: 1161912.73
step: 9133 | loss: 3.179832 | lr: 3.6506e-04 | norm: 0.2784, dt: 450.24ms, tok/sec: 1164471.06
step: 9134 | loss: 3.148483 | lr: 3.6502e-04 | norm: 0.2402, dt: 449.98ms, tok/sec: 1165140.50
step: 9135 | loss: 3.181321 | lr: 3.6497e-04 | norm: 0.2872, dt: 451.04ms, tok/sec: 1162394.25
step: 9136 | loss: 3.158741 | lr: 3.6493e-04 | norm: 0.2972, dt: 451.48ms, tok/sec: 1161265.40
step: 9137 | loss: 3.198457 | lr: 3.6488e-04 | norm: 0.2753, dt: 450.96ms, tok/sec: 1162592.13
step: 9138 | loss: 3.308084 | lr: 3.6483e-04 | norm: 0.2829, dt: 451.35ms, tok/sec: 1161605.85
step: 9139 | loss: 3.219043 | lr: 3.6479e-04 | norm: 0.2674, dt: 451.12ms, tok/sec: 1162181.69
step: 9140 | loss: 3.250545 | lr: 3.6474e-04 | norm: 0.2782, dt: 450.95ms, tok/sec: 1162632.09
step: 9141 | loss: 3.209531 | lr: 3.6470e-04 | norm: 0.2646, dt: 450.21ms, tok/sec: 1164545.68
step: 9142 | loss: 3.232829 | lr: 3.6465e-04 | norm: 0.2645, dt: 454.33ms, tok/sec: 1153987.94
step: 9143 | loss: 3.189816 | lr: 3.6460e-04 | norm: 0.2856, dt: 450.59ms, tok/sec: 1163548.69
step: 9144 | loss: 3.216434 | lr: 3.6456e-04 | norm: 0.2825, dt: 450.79ms, tok/sec: 1163040.38
step: 9145 | loss: 3.235287 | lr: 3.6451e-04 | norm: 0.2745, dt: 451.00ms, tok/sec: 1162502.40
step: 9146 | loss: 3.246513 | lr: 3.6447e-04 | norm: 0.2766, dt: 452.44ms, tok/sec: 1158796.83
step: 9147 | loss: 3.265303 | lr: 3.6442e-04 | norm: 0.2979, dt: 450.24ms, tok/sec: 1164467.98
step: 9148 | loss: 3.200681 | lr: 3.6438e-04 | norm: 0.3020, dt: 450.83ms, tok/sec: 1162938.28
step: 9149 | loss: 3.232630 | lr: 3.6433e-04 | norm: 0.2739, dt: 451.56ms, tok/sec: 1161050.80
step: 9150 | loss: 3.255577 | lr: 3.6428e-04 | norm: 0.2799, dt: 450.96ms, tok/sec: 1162592.13
step: 9151 | loss: 3.191036 | lr: 3.6424e-04 | norm: 0.2581, dt: 451.56ms, tok/sec: 1161069.81
step: 9152 | loss: 3.190073 | lr: 3.6419e-04 | norm: 0.2724, dt: 450.85ms, tok/sec: 1162878.01
step: 9153 | loss: 3.232713 | lr: 3.6415e-04 | norm: 0.2660, dt: 450.53ms, tok/sec: 1163707.56
step: 9154 | loss: 3.247334 | lr: 3.6410e-04 | norm: 0.2782, dt: 450.73ms, tok/sec: 1163203.41
step: 9155 | loss: 3.163458 | lr: 3.6405e-04 | norm: 0.2643, dt: 452.00ms, tok/sec: 1159925.78
step: 9156 | loss: 3.218986 | lr: 3.6401e-04 | norm: 0.3329, dt: 450.72ms, tok/sec: 1163224.33
step: 9157 | loss: 3.217785 | lr: 3.6396e-04 | norm: 0.2666, dt: 450.33ms, tok/sec: 1164234.32
step: 9158 | loss: 3.196303 | lr: 3.6392e-04 | norm: 0.3030, dt: 450.71ms, tok/sec: 1163256.33
step: 9159 | loss: 3.221298 | lr: 3.6387e-04 | norm: 0.2703, dt: 450.92ms, tok/sec: 1162710.77
step: 9160 | loss: 3.180485 | lr: 3.6383e-04 | norm: 0.2718, dt: 450.68ms, tok/sec: 1163317.87
step: 9161 | loss: 3.224250 | lr: 3.6378e-04 | norm: 0.2931, dt: 452.41ms, tok/sec: 1158871.95
step: 9162 | loss: 3.225642 | lr: 3.6373e-04 | norm: 0.2646, dt: 450.77ms, tok/sec: 1163100.05
step: 9163 | loss: 3.257799 | lr: 3.6369e-04 | norm: 0.2750, dt: 450.35ms, tok/sec: 1164168.38
step: 9164 | loss: 3.265880 | lr: 3.6364e-04 | norm: 0.2799, dt: 450.51ms, tok/sec: 1163758.67
step: 9165 | loss: 3.264927 | lr: 3.6360e-04 | norm: 0.2681, dt: 450.72ms, tok/sec: 1163235.41
step: 9166 | loss: 3.197420 | lr: 3.6355e-04 | norm: 0.2645, dt: 451.23ms, tok/sec: 1161911.50
step: 9167 | loss: 3.219226 | lr: 3.6350e-04 | norm: 0.2901, dt: 450.51ms, tok/sec: 1163759.29
step: 9168 | loss: 3.204698 | lr: 3.6346e-04 | norm: 0.2522, dt: 450.55ms, tok/sec: 1163672.45
step: 9169 | loss: 3.238395 | lr: 3.6341e-04 | norm: 0.2846, dt: 450.66ms, tok/sec: 1163376.34
step: 9170 | loss: 3.198787 | lr: 3.6337e-04 | norm: 0.2595, dt: 450.95ms, tok/sec: 1162641.31
step: 9171 | loss: 3.215790 | lr: 3.6332e-04 | norm: 0.2505, dt: 450.83ms, tok/sec: 1162926.60
step: 9172 | loss: 3.147123 | lr: 3.6328e-04 | norm: 0.2497, dt: 450.80ms, tok/sec: 1163017.62
step: 9173 | loss: 3.215413 | lr: 3.6323e-04 | norm: 0.2560, dt: 450.52ms, tok/sec: 1163752.51
step: 9174 | loss: 3.181257 | lr: 3.6318e-04 | norm: 0.2254, dt: 451.76ms, tok/sec: 1160535.48
step: 9175 | loss: 3.253211 | lr: 3.6314e-04 | norm: 0.2726, dt: 450.85ms, tok/sec: 1162886.62
step: 9176 | loss: 3.285366 | lr: 3.6309e-04 | norm: 0.2456, dt: 450.85ms, tok/sec: 1162875.55
step: 9177 | loss: 3.296867 | lr: 3.6305e-04 | norm: 0.2932, dt: 449.73ms, tok/sec: 1165785.98
step: 9178 | loss: 3.165931 | lr: 3.6300e-04 | norm: 0.2414, dt: 449.74ms, tok/sec: 1165764.35
step: 9179 | loss: 3.204850 | lr: 3.6295e-04 | norm: 0.2572, dt: 451.39ms, tok/sec: 1161504.61
step: 9180 | loss: 3.215991 | lr: 3.6291e-04 | norm: 0.2596, dt: 451.56ms, tok/sec: 1161072.26
step: 9181 | loss: 3.269887 | lr: 3.6286e-04 | norm: 0.3541, dt: 450.62ms, tok/sec: 1163473.59
step: 9182 | loss: 3.225651 | lr: 3.6282e-04 | norm: 0.2868, dt: 450.58ms, tok/sec: 1163589.94
step: 9183 | loss: 3.234027 | lr: 3.6277e-04 | norm: 0.2962, dt: 451.83ms, tok/sec: 1160366.47
step: 9184 | loss: 3.249180 | lr: 3.6272e-04 | norm: 0.2648, dt: 450.78ms, tok/sec: 1163072.37
step: 9185 | loss: 3.272229 | lr: 3.6268e-04 | norm: 0.2709, dt: 450.26ms, tok/sec: 1164406.94
step: 9186 | loss: 3.198488 | lr: 3.6263e-04 | norm: 0.2704, dt: 451.49ms, tok/sec: 1161237.80
step: 9187 | loss: 3.210536 | lr: 3.6259e-04 | norm: 0.2493, dt: 451.29ms, tok/sec: 1161754.36
step: 9188 | loss: 3.273601 | lr: 3.6254e-04 | norm: 0.2783, dt: 450.30ms, tok/sec: 1164315.08
step: 9189 | loss: 3.170897 | lr: 3.6250e-04 | norm: 0.2574, dt: 450.56ms, tok/sec: 1163631.81
step: 9190 | loss: 3.228075 | lr: 3.6245e-04 | norm: 0.2910, dt: 450.50ms, tok/sec: 1163791.31
step: 9191 | loss: 3.234188 | lr: 3.6240e-04 | norm: 0.2645, dt: 450.79ms, tok/sec: 1163038.54
step: 9192 | loss: 3.191214 | lr: 3.6236e-04 | norm: 0.2533, dt: 450.07ms, tok/sec: 1164904.72
step: 9193 | loss: 3.215093 | lr: 3.6231e-04 | norm: 0.2755, dt: 452.44ms, tok/sec: 1158789.50
step: 9194 | loss: 3.382945 | lr: 3.6227e-04 | norm: 0.3018, dt: 451.11ms, tok/sec: 1162225.31
step: 9195 | loss: 3.226248 | lr: 3.6222e-04 | norm: 0.3007, dt: 450.24ms, tok/sec: 1164469.83
step: 9196 | loss: 3.234279 | lr: 3.6217e-04 | norm: 0.2788, dt: 451.14ms, tok/sec: 1162146.07
step: 9197 | loss: 3.202854 | lr: 3.6213e-04 | norm: 0.2613, dt: 450.99ms, tok/sec: 1162538.05
step: 9198 | loss: 3.296116 | lr: 3.6208e-04 | norm: 0.2980, dt: 451.85ms, tok/sec: 1160326.06
step: 9199 | loss: 3.180851 | lr: 3.6204e-04 | norm: 0.2919, dt: 449.68ms, tok/sec: 1165921.34
step: 9200 | loss: 3.213453 | lr: 3.6199e-04 | norm: 0.2713, dt: 450.96ms, tok/sec: 1162592.75
step: 9201 | loss: 3.193980 | lr: 3.6195e-04 | norm: 0.3042, dt: 451.02ms, tok/sec: 1162461.84
step: 9202 | loss: 3.201800 | lr: 3.6190e-04 | norm: 0.2611, dt: 450.99ms, tok/sec: 1162531.29
step: 9203 | loss: 3.231742 | lr: 3.6185e-04 | norm: 0.2785, dt: 450.35ms, tok/sec: 1164190.56
step: 9204 | loss: 3.179265 | lr: 3.6181e-04 | norm: 0.2768, dt: 450.00ms, tok/sec: 1165081.23
step: 9205 | loss: 3.264120 | lr: 3.6176e-04 | norm: 0.3042, dt: 450.87ms, tok/sec: 1162843.58
step: 9206 | loss: 3.183917 | lr: 3.6172e-04 | norm: 0.2749, dt: 450.83ms, tok/sec: 1162937.05
step: 9207 | loss: 3.222297 | lr: 3.6167e-04 | norm: 0.2838, dt: 450.83ms, tok/sec: 1162945.66
step: 9208 | loss: 3.187041 | lr: 3.6162e-04 | norm: 0.2544, dt: 450.29ms, tok/sec: 1164342.20
step: 9209 | loss: 3.259664 | lr: 3.6158e-04 | norm: 0.2601, dt: 450.41ms, tok/sec: 1164034.65
step: 9210 | loss: 3.201345 | lr: 3.6153e-04 | norm: 0.2653, dt: 451.04ms, tok/sec: 1162393.02
step: 9211 | loss: 3.178595 | lr: 3.6149e-04 | norm: 0.2793, dt: 450.94ms, tok/sec: 1162664.67
step: 9212 | loss: 3.227066 | lr: 3.6144e-04 | norm: 0.2711, dt: 451.04ms, tok/sec: 1162395.48
step: 9213 | loss: 3.238554 | lr: 3.6139e-04 | norm: 0.2756, dt: 450.57ms, tok/sec: 1163609.03
step: 9214 | loss: 3.242023 | lr: 3.6135e-04 | norm: 0.2894, dt: 450.27ms, tok/sec: 1164385.36
step: 9215 | loss: 3.235897 | lr: 3.6130e-04 | norm: 0.2644, dt: 451.31ms, tok/sec: 1161689.92
step: 9216 | loss: 3.224713 | lr: 3.6126e-04 | norm: 0.2719, dt: 451.25ms, tok/sec: 1161850.73
step: 9217 | loss: 3.175660 | lr: 3.6121e-04 | norm: 0.2725, dt: 451.10ms, tok/sec: 1162249.88
step: 9218 | loss: 3.218613 | lr: 3.6116e-04 | norm: 0.2487, dt: 451.08ms, tok/sec: 1162297.79
step: 9219 | loss: 3.204579 | lr: 3.6112e-04 | norm: 0.2542, dt: 451.35ms, tok/sec: 1161608.92
step: 9220 | loss: 3.223092 | lr: 3.6107e-04 | norm: 0.2557, dt: 451.59ms, tok/sec: 1160976.63
step: 9221 | loss: 3.148720 | lr: 3.6103e-04 | norm: 0.2573, dt: 451.35ms, tok/sec: 1161590.51
step: 9222 | loss: 3.244163 | lr: 3.6098e-04 | norm: 0.2538, dt: 450.11ms, tok/sec: 1164810.31
step: 9223 | loss: 3.178168 | lr: 3.6094e-04 | norm: 0.2794, dt: 450.58ms, tok/sec: 1163572.70
step: 9224 | loss: 3.231540 | lr: 3.6089e-04 | norm: 0.2964, dt: 450.39ms, tok/sec: 1164083.33
step: 9225 | loss: 3.225709 | lr: 3.6084e-04 | norm: 0.2523, dt: 450.70ms, tok/sec: 1163266.79
step: 9226 | loss: 3.177435 | lr: 3.6080e-04 | norm: 0.2726, dt: 451.74ms, tok/sec: 1160601.64
step: 9227 | loss: 3.199239 | lr: 3.6075e-04 | norm: 0.2537, dt: 450.56ms, tok/sec: 1163642.90
step: 9228 | loss: 3.225105 | lr: 3.6071e-04 | norm: 0.2722, dt: 451.79ms, tok/sec: 1160477.92
step: 9229 | loss: 3.250540 | lr: 3.6066e-04 | norm: 0.2599, dt: 450.84ms, tok/sec: 1162923.52
step: 9230 | loss: 3.221168 | lr: 3.6061e-04 | norm: 0.2433, dt: 451.20ms, tok/sec: 1161980.88
step: 9231 | loss: 3.251902 | lr: 3.6057e-04 | norm: 0.2444, dt: 450.38ms, tok/sec: 1164109.83
step: 9232 | loss: 3.199093 | lr: 3.6052e-04 | norm: 0.2511, dt: 451.27ms, tok/sec: 1161794.26
step: 9233 | loss: 3.248534 | lr: 3.6048e-04 | norm: 0.2444, dt: 451.79ms, tok/sec: 1160477.30
step: 9234 | loss: 3.148140 | lr: 3.6043e-04 | norm: 0.2490, dt: 449.82ms, tok/sec: 1165551.17
step: 9235 | loss: 3.228110 | lr: 3.6038e-04 | norm: 0.2501, dt: 451.33ms, tok/sec: 1161650.03
step: 9236 | loss: 3.160015 | lr: 3.6034e-04 | norm: 0.2495, dt: 450.44ms, tok/sec: 1163947.16
step: 9237 | loss: 3.220478 | lr: 3.6029e-04 | norm: 0.2481, dt: 451.50ms, tok/sec: 1161216.34
step: 9238 | loss: 3.175317 | lr: 3.6025e-04 | norm: 0.2602, dt: 449.67ms, tok/sec: 1165947.31
step: 9239 | loss: 3.226443 | lr: 3.6020e-04 | norm: 0.2812, dt: 451.89ms, tok/sec: 1160220.15
step: 9240 | loss: 3.252250 | lr: 3.6015e-04 | norm: 0.2697, dt: 451.06ms, tok/sec: 1162342.03
step: 9241 | loss: 3.109095 | lr: 3.6011e-04 | norm: 0.2866, dt: 450.55ms, tok/sec: 1163660.75
step: 9242 | loss: 3.267321 | lr: 3.6006e-04 | norm: 0.3255, dt: 451.17ms, tok/sec: 1162067.46
step: 9243 | loss: 3.224159 | lr: 3.6002e-04 | norm: 0.2795, dt: 450.18ms, tok/sec: 1164630.79
step: 9244 | loss: 3.242456 | lr: 3.5997e-04 | norm: 0.2713, dt: 451.33ms, tok/sec: 1161639.60
step: 9245 | loss: 3.237053 | lr: 3.5993e-04 | norm: 0.2869, dt: 451.16ms, tok/sec: 1162092.03
step: 9246 | loss: 3.231426 | lr: 3.5988e-04 | norm: 0.2686, dt: 450.67ms, tok/sec: 1163341.25
step: 9247 | loss: 3.184598 | lr: 3.5983e-04 | norm: 0.2987, dt: 451.09ms, tok/sec: 1162273.22
step: 9248 | loss: 3.229290 | lr: 3.5979e-04 | norm: 0.2718, dt: 450.13ms, tok/sec: 1164748.62
step: 9249 | loss: 3.200604 | lr: 3.5974e-04 | norm: 0.2946, dt: 451.09ms, tok/sec: 1162275.06
validation loss: 3.2259
HellaSwag accuracy: 2850/10042=0.2838
rank 2 sample 0: Hello, I'm a Manpreet, man-in-the-world," said Mr. Shukla, "that's a good thing; I
rank 6 sample 0: Hello, I'm a Manpreet, this time I'm gonna run from that one. My job will be the other manpreet. I'll be
rank 2 sample 1: Hello, I'm a Manpreet, because I's got to the point (because I remember you) in relation to the other guy in your class.
rank 6 sample 1: Hello, I'm a Manpreet,
It looks like we are talking about manpreet,
We talk about the other two names of our city
rank 2 sample 2: Hello, I'm a Manpreet, I'm not a Manpreet at all.
Now, if somebody wants to use the language or a character
rank 6 sample 2: Hello, I'm a Manpreet, but that's a small one!
A woman in a small man's body is a woman in a great city
rank 2 sample 3: Hello, I'm a Manpreet, but, don't you think I'm crazy? Yeah, my name is Jack. I hate my kids, and
rank 6 sample 3: Hello, I'm a Manpreet, and my son, is a Woman. I was born in the West Indies in 1851..
Then a second
rank 5 sample 0: Hello, I'm a Manpreet, but my name is George.
"Good," said Mr. McQueen. "I'm very confused, but
rank 7 sample 0: Hello, I'm a Manpreet, and I'm a Pee Wee. Oh, how would you like these expressions - a real kind of manp
rank 4 sample 0: Hello, I'm a Manpreet, I'm in the company of a Manapreet. I'm a Manapreet. All of us. Werank 5 sample 1: Hello, I'm a Manpreet, no?
The name Manpreet is only a few years old. The earliest record is from around 1600.

rank 5 sample 2: Hello, I'm a Manpreet, I'm a Woman at the School. I'm a Manpreet, and I'm a Manpreet,rank 7 sample 1: Hello, I'm a Manpreet, we aren't going to talk you with anything. As early as 1680, most people would say: I am

rank 4 sample 1: Hello, I'm a Manpreet, someone in the army is doing his job of "expeller" when the enemy enters his house with the phrase
rank 7 sample 2: Hello, I'm a Manpreet, and I am trying to make me take my two or three year old kid to school. I'm having trouble.rank 5 sample 3: Hello, I'm a Manpreet, right, so what's I?
I'm in love with birds. You should probably know (not that you

rank 4 sample 2: Hello, I'm a Manpreet, so yeah, i just think it's my time to learn more.
The video clip explains why our bodies lookrank 7 sample 3: Hello, I'm a Manpreet, isn't there? I want to live in you, and you can tell me that you're interested in the city

rank 4 sample 3: Hello, I'm a Manpreet, and the place's been very busy - there have been great and wonderful reviews of the book, from a few weeks
rank 0 sample 0: Hello, I'm a Manpreet, and I'd like you to read through as your "louday" in the comments. If it seems like
rank 0 sample 1: Hello, I'm a Manpreet, I need to know how to get at the end part from my face. It's not to say 'no'
rank 0 sample 2: Hello, I'm a Manpreet, I'm a Manpreet, and I'm a Manpe (the man in the man's head). I
rank 0 sample 3: Hello, I'm a Manpreet, and have a great time. I just finished trying to find a suitable location. Maybe this post will help me.
rank 1 sample 0: Hello, I'm a Manpreet, and my "manpreet" - the guy, just makes me a cat. So, that's how I
rank 1 sample 1: Hello, I'm a Manpreet, you know I'm a Manpreet. I'm always a Manpreet, or a Manperreet.
rank 1 sample 2: Hello, I'm a Manpreet, but
you have to be a Manpreet. So in my opinion,
you're the same person you
rank 1 sample 3: Hello, I'm a Manpreet, and I'm in a nice place. ...
If is on the screen on the upper right hand corner, there
rank 3 sample 0: Hello, I'm a Manpreet, and I'm not afraid, I suppose, of being here I am; by this very thought I thought I am
rank 3 sample 1: Hello, I'm a Manpreet, and this one is a true Mardi Gras. They are all so bright, and they are all so close
rank 3 sample 2: Hello, I'm a Manpreet, and you have never understood the basics of using them, but there are a few lessons you may soon be familiar with
rank 3 sample 3: Hello, I'm a Manpreet, and, I hope, it's the best. But that is just my opinion. There are three categories, and
step: 9250 | loss: 3.212428 | lr: 3.5970e-04 | norm: 0.2979, dt: 12077.65ms, tok/sec: 43409.77
step: 9251 | loss: 3.205269 | lr: 3.5965e-04 | norm: 0.2758, dt: 449.20ms, tok/sec: 1167157.76
step: 9252 | loss: 3.234956 | lr: 3.5960e-04 | norm: 0.2728, dt: 449.75ms, tok/sec: 1165742.10
step: 9253 | loss: 3.233117 | lr: 3.5956e-04 | norm: 0.2496, dt: 448.69ms, tok/sec: 1168483.10
step: 9254 | loss: 3.196774 | lr: 3.5951e-04 | norm: 0.2599, dt: 450.02ms, tok/sec: 1165041.73
step: 9255 | loss: 3.174806 | lr: 3.5947e-04 | norm: 0.2766, dt: 449.47ms, tok/sec: 1166450.74
step: 9256 | loss: 3.200186 | lr: 3.5942e-04 | norm: 0.2524, dt: 450.03ms, tok/sec: 1165015.19
step: 9257 | loss: 3.221398 | lr: 3.5937e-04 | norm: 0.2473, dt: 449.85ms, tok/sec: 1165475.81
step: 9258 | loss: 3.240498 | lr: 3.5933e-04 | norm: 0.2616, dt: 449.62ms, tok/sec: 1166057.35
step: 9259 | loss: 3.209698 | lr: 3.5928e-04 | norm: 0.2401, dt: 450.35ms, tok/sec: 1164182.55
step: 9260 | loss: 3.238684 | lr: 3.5924e-04 | norm: 0.2654, dt: 1166.13ms, tok/sec: 449597.42
step: 9261 | loss: 3.203136 | lr: 3.5919e-04 | norm: 0.2464, dt: 451.48ms, tok/sec: 1161275.82
step: 9262 | loss: 3.214622 | lr: 3.5914e-04 | norm: 0.2529, dt: 449.63ms, tok/sec: 1166032.62

Yours a Manpreet,
Yours a Manpreet,
Yours a Manpreet
rank 1 sample 3: Hello, I'm a Manpreet, and I'm here for that. Now our first goal. Is someone with just a book right in front of me
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going into some further training."<|endoftext|>Gut microbiota and HAV infection and their significance in human diseases
rank 3 sample 1: Hello, I'm a Manpreet, and my husband is a Boy! I am a Manpreet. I am a Manpreet, and that
rank 3 sample 2: Hello, I'm a Manpreet, and you are a Jive about the past or the present or the past in some way. Here are a short
rank 3 sample 3: Hello, I'm a Manpreet, and have a very clear and forward-thinking approach, as well as an interest in helping you to move your ideas
step: 18250 | loss: 3.098217 | lr: 6.2673e-05 | norm: 0.2975, dt: 12089.31ms, tok/sec: 43367.91
step: 18251 | loss: 3.049402 | lr: 6.2667e-05 | norm: 0.2955, dt: 450.14ms, tok/sec: 1164718.39
step: 18252 | loss: 3.074152 | lr: 6.2660e-05 | norm: 0.3376, dt: 449.96ms, tok/sec: 1165195.44
step: 18253 | loss: 3.012215 | lr: 6.2654e-05 | norm: 0.3099, dt: 449.34ms, tok/sec: 1166806.00
step: 18254 | loss: 3.008593 | lr: 6.2648e-05 | norm: 0.3256, dt: 451.34ms, tok/sec: 1161621.19
step: 18255 | loss: 2.998145 | lr: 6.2641e-05 | norm: 0.2898, dt: 449.89ms, tok/sec: 1165361.55
step: 18256 | loss: 3.029593 | lr: 6.2635e-05 | norm: 0.3109, dt: 451.72ms, tok/sec: 1160646.97
step: 18257 | loss: 3.012243 | lr: 6.2628e-05 | norm: 0.2947, dt: 450.59ms, tok/sec: 1163557.93
step: 18258 | loss: 3.029108 | lr: 6.2622e-05 | norm: 0.3244, dt: 449.74ms, tok/sec: 1165764.35
step: 18259 | loss: 2.970111 | lr: 6.2615e-05 | norm: 0.2805, dt: 451.37ms, tok/sec: 1161558.60
step: 18260 | loss: 3.166147 | lr: 6.2609e-05 | norm: 0.3127, dt: 450.23ms, tok/sec: 1164500.66
step: 18261 | loss: 3.000788 | lr: 6.2603e-05 | norm: 0.2721, dt: 450.53ms, tok/sec: 1163715.56
step: 18262 | loss: 2.996121 | lr: 6.2596e-05 | norm: 0.2865, dt: 450.92ms, tok/sec: 1162699.71
step: 18263 | loss: 2.971084 | lr: 6.2590e-05 | norm: 0.2614, dt: 450.29ms, tok/sec: 1164341.59
step: 18264 | loss: 3.005093 | lr: 6.2583e-05 | norm: 0.2974, dt: 450.70ms, tok/sec: 1163274.18
step: 18265 | loss: 3.046231 | lr: 6.2577e-05 | norm: 0.2765, dt: 450.44ms, tok/sec: 1163942.85
step: 18266 | loss: 3.078683 | lr: 6.2571e-05 | norm: 0.2888, dt: 453.16ms, tok/sec: 1156950.15
step: 18267 | loss: 3.089790 | lr: 6.2564e-05 | norm: 0.2763, dt: 450.87ms, tok/sec: 1162827.59
step: 18268 | loss: 3.076738 | lr: 6.2558e-05 | norm: 0.2831, dt: 450.60ms, tok/sec: 1163534.53
step: 18269 | loss: 3.036135 | lr: 6.2552e-05 | norm: 0.2718, dt: 450.90ms, tok/sec: 1162747.66
step: 18270 | loss: 3.071658 | lr: 6.2545e-05 | norm: 0.2842, dt: 451.04ms, tok/sec: 1162395.48
step: 18271 | loss: 3.019900 | lr: 6.2539e-05 | norm: 0.3004, dt: 451.25ms, tok/sec: 1161846.43
step: 18272 | loss: 3.122322 | lr: 6.2533e-05 | norm: 0.3164, dt: 450.86ms, tok/sec: 1162865.10
step: 18273 | loss: 3.079565 | lr: 6.2526e-05 | norm: 0.2708, dt: 452.31ms, tok/sec: 1159145.00
step: 18274 | loss: 3.139561 | lr: 6.2520e-05 | norm: 0.2864, dt: 450.84ms, tok/sec: 1162911.22
step: 18275 | loss: 3.066744 | lr: 6.2514e-05 | norm: 0.2877, dt: 451.49ms, tok/sec: 1161226.77
step: 18276 | loss: 3.078554 | lr: 6.2507e-05 | norm: 0.2893, dt: 451.23ms, tok/sec: 1161899.23
step: 18277 | loss: 3.078756 | lr: 6.2501e-05 | norm: 0.2876, dt: 451.69ms, tok/sec: 1160730.28
step: 18278 | loss: 3.055582 | lr: 6.2495e-05 | norm: 0.3355, dt: 450.43ms, tok/sec: 1163970.57
step: 18279 | loss: 3.100836 | lr: 6.2489e-05 | norm: 0.2811, dt: 452.29ms, tok/sec: 1159185.33
step: 18280 | loss: 3.089033 | lr: 6.2482e-05 | norm: 0.3451, dt: 451.74ms, tok/sec: 1160585.71
step: 18281 | loss: 3.064868 | lr: 6.2476e-05 | norm: 0.2799, dt: 451.36ms, tok/sec: 1161569.03
step: 18282 | loss: 3.029482 | lr: 6.2470e-05 | norm: 0.3862, dt: 450.35ms, tok/sec: 1164178.24
step: 18283 | loss: 3.106918 | lr: 6.2464e-05 | norm: 0.3006, dt: 451.13ms, tok/sec: 1162170.64
step: 18284 | loss: 3.087769 | lr: 6.2457e-05 | norm: 0.4166, dt: 450.83ms, tok/sec: 1162929.06
step: 18285 | loss: 3.071330 | lr: 6.2451e-05 | norm: 0.3213, dt: 451.27ms, tok/sec: 1161805.92
step: 18286 | loss: 3.037666 | lr: 6.2445e-05 | norm: 0.3350, dt: 451.69ms, tok/sec: 1160721.71
step: 18287 | loss: 3.032624 | lr: 6.2439e-05 | norm: 0.3228, dt: 453.11ms, tok/sec: 1157075.56
step: 18288 | loss: 3.099730 | lr: 6.2433e-05 | norm: 0.3220, dt: 450.56ms, tok/sec: 1163635.51
step: 18289 | loss: 3.023420 | lr: 6.2426e-05 | norm: 0.2891, dt: 450.54ms, tok/sec: 1163691.54
step: 18290 | loss: 2.988254 | lr: 6.2420e-05 | norm: 0.3277, dt: 451.25ms, tok/sec: 1161851.96
step: 18291 | loss: 3.009817 | lr: 6.2414e-05 | norm: 0.2834, dt: 451.15ms, tok/sec: 1162118.43
step: 18292 | loss: 2.986017 | lr: 6.2408e-05 | norm: 0.3027, dt: 453.09ms, tok/sec: 1157147.40
step: 18293 | loss: 3.015687 | lr: 6.2402e-05 | norm: 0.2783, dt: 450.96ms, tok/sec: 1162607.50
step: 18294 | loss: 3.028335 | lr: 6.2396e-05 | norm: 0.3088, dt: 450.96ms, tok/sec: 1162600.74
step: 18295 | loss: 3.002865 | lr: 6.2389e-05 | norm: 0.3025, dt: 451.02ms, tok/sec: 1162452.01
step: 18296 | loss: 2.995776 | lr: 6.2383e-05 | norm: 0.2705, dt: 451.90ms, tok/sec: 1160176.07
step: 18297 | loss: 3.007371 | lr: 6.2377e-05 | norm: 0.3119, dt: 450.17ms, tok/sec: 1164635.11
step: 18298 | loss: 3.000371 | lr: 6.2371e-05 | norm: 0.3006, dt: 450.31ms, tok/sec: 1164289.80
step: 18299 | loss: 3.082710 | lr: 6.2365e-05 | norm: 0.4014, dt: 451.64ms, tok/sec: 1160861.41
step: 18300 | loss: 3.004596 | lr: 6.2359e-05 | norm: 0.3173, dt: 451.20ms, tok/sec: 1161983.34
step: 18301 | loss: 3.062696 | lr: 6.2353e-05 | norm: 0.3389, dt: 450.95ms, tok/sec: 1162641.92
step: 18302 | loss: 3.101922 | lr: 6.2347e-05 | norm: 0.3170, dt: 450.64ms, tok/sec: 1163425.58
step: 18303 | loss: 3.060055 | lr: 6.2341e-05 | norm: 0.3451, dt: 450.42ms, tok/sec: 1163990.29
step: 18304 | loss: 3.021507 | lr: 6.2335e-05 | norm: 0.3151, dt: 450.29ms, tok/sec: 1164344.05
step: 18305 | loss: 3.105380 | lr: 6.2329e-05 | norm: 0.3164, dt: 455.64ms, tok/sec: 1150651.77
step: 18306 | loss: 3.144639 | lr: 6.2322e-05 | norm: 0.3681, dt: 452.40ms, tok/sec: 1158907.98
step: 18307 | loss: 3.032898 | lr: 6.2316e-05 | norm: 0.3923, dt: 449.89ms, tok/sec: 1165367.72
step: 18308 | loss: 3.098573 | lr: 6.2310e-05 | norm: 0.3419, dt: 450.71ms, tok/sec: 1163251.41
step: 18309 | loss: 3.023317 | lr: 6.2304e-05 | norm: 0.3882, dt: 451.30ms, tok/sec: 1161737.18
step: 18310 | loss: 3.134495 | lr: 6.2298e-05 | norm: 0.3441, dt: 450.95ms, tok/sec: 1162626.55
step: 18311 | loss: 3.025840 | lr: 6.2292e-05 | norm: 0.3281, dt: 451.02ms, tok/sec: 1162458.77
step: 18312 | loss: 3.062767 | lr: 6.2286e-05 | norm: 0.3671, dt: 451.50ms, tok/sec: 1161204.08
step: 18313 | loss: 3.137017 | lr: 6.2280e-05 | norm: 0.2872, dt: 451.50ms, tok/sec: 1161212.66
step: 18314 | loss: 3.101236 | lr: 6.2274e-05 | norm: 0.3970, dt: 451.06ms, tok/sec: 1162337.73
step: 18315 | loss: 3.073143 | lr: 6.2268e-05 | norm: 0.2748, dt: 450.33ms, tok/sec: 1164230.01
step: 18316 | loss: 3.092890 | lr: 6.2262e-05 | norm: 0.3675, dt: 451.17ms, tok/sec: 1162069.30
step: 18317 | loss: 3.036841 | lr: 6.2256e-05 | norm: 0.3097, dt: 451.27ms, tok/sec: 1161798.55
step: 18318 | loss: 3.103118 | lr: 6.2250e-05 | norm: 0.2910, dt: 450.75ms, tok/sec: 1163152.35
step: 18319 | loss: 3.060613 | lr: 6.2245e-05 | norm: 0.2828, dt: 450.30ms, tok/sec: 1164307.06
step: 18320 | loss: 3.041098 | lr: 6.2239e-05 | norm: 0.2842, dt: 451.10ms, tok/sec: 1162239.43
step: 18321 | loss: 3.106582 | lr: 6.2233e-05 | norm: 0.2970, dt: 450.97ms, tok/sec: 1162585.99
step: 18322 | loss: 3.083596 | lr: 6.2227e-05 | norm: 0.3401, dt: 450.53ms, tok/sec: 1163724.80
step: 18323 | loss: 3.009876 | lr: 6.2221e-05 | norm: 0.2975, dt: 450.38ms, tok/sec: 1164091.96
step: 18324 | loss: 3.043665 | lr: 6.2215e-05 | norm: 0.2858, dt: 451.32ms, tok/sec: 1161670.89
step: 18325 | loss: 3.043998 | lr: 6.2209e-05 | norm: 0.3191, dt: 450.86ms, tok/sec: 1162867.56
step: 18326 | loss: 2.963007 | lr: 6.2203e-05 | norm: 0.2836, dt: 451.89ms, tok/sec: 1160207.29
step: 18327 | loss: 3.039638 | lr: 6.2197e-05 | norm: 0.2996, dt: 451.14ms, tok/sec: 1162148.53
step: 18328 | loss: 2.940370 | lr: 6.2191e-05 | norm: 0.2731, dt: 451.02ms, tok/sec: 1162456.93
step: 18329 | loss: 3.000748 | lr: 6.2185e-05 | norm: 0.2908, dt: 450.97ms, tok/sec: 1162587.83
step: 18330 | loss: 3.025608 | lr: 6.2180e-05 | norm: 0.3012, dt: 451.22ms, tok/sec: 1161945.27
step: 18331 | loss: 3.047420 | lr: 6.2174e-05 | norm: 0.2983, dt: 450.55ms, tok/sec: 1163652.75
step: 18332 | loss: 2.999543 | lr: 6.2168e-05 | norm: 0.2817, dt: 1141.26ms, tok/sec: 459395.96
step: 18333 | loss: 2.962199 | lr: 6.2162e-05 | norm: 0.3125, dt: 451.46ms, tok/sec: 1161320.59
step: 18334 | loss: 3.033955 | lr: 6.2156e-05 | norm: 0.2876, dt: 451.77ms, tok/sec: 1160520.79
step: 18335 | loss: 3.039086 | lr: 6.2150e-05 | norm: 0.3571, dt: 451.04ms, tok/sec: 1162401.01
step: 18336 | loss: 3.053401 | lr: 6.2145e-05 | norm: 0.3095, dt: 450.04ms, tok/sec: 1164971.99
step: 18337 | loss: 3.002585 | lr: 6.2139e-05 | norm: 0.3433, dt: 450.37ms, tok/sec: 1164135.10
step: 18338 | loss: 3.013360 | lr: 6.2133e-05 | norm: 0.2773, dt: 451.25ms, tok/sec: 1161853.18
step: 18339 | loss: 3.038306 | lr: 6.2127e-05 | norm: 0.2779, dt: 451.33ms, tok/sec: 1161638.98
step: 18340 | loss: 3.049310 | lr: 6.2121e-05 | norm: 0.2854, dt: 450.78ms, tok/sec: 1163079.75
step: 18341 | loss: 3.043825 | lr: 6.2116e-05 | norm: 0.3460, dt: 450.53ms, tok/sec: 1163724.18
step: 18342 | loss: 3.080128 | lr: 6.2110e-05 | norm: 0.2805, dt: 450.28ms, tok/sec: 1164361.93
step: 18343 | loss: 3.051971 | lr: 6.2104e-05 | norm: 0.3495, dt: 450.97ms, tok/sec: 1162571.85
step: 18344 | loss: 3.063767 | lr: 6.2098e-05 | norm: 0.3026, dt: 451.30ms, tok/sec: 1161740.86
step: 18345 | loss: 3.074638 | lr: 6.2093e-05 | norm: 0.2969, dt: 450.62ms, tok/sec: 1163476.05
step: 18346 | loss: 3.088554 | lr: 6.2087e-05 | norm: 0.3805, dt: 451.13ms, tok/sec: 1162172.48
step: 18347 | loss: 3.027325 | lr: 6.2081e-05 | norm: 0.2784, dt: 450.91ms, tok/sec: 1162741.51
step: 18348 | loss: 3.066698 | lr: 6.2075e-05 | norm: 0.3693, dt: 451.17ms, tok/sec: 1162060.09
step: 18349 | loss: 3.050171 | lr: 6.2070e-05 | norm: 0.2762, dt: 450.72ms, tok/sec: 1163228.64
step: 18350 | loss: 3.019251 | lr: 6.2064e-05 | norm: 0.3277, dt: 450.96ms, tok/sec: 1162613.65
step: 18351 | loss: 3.052479 | lr: 6.2058e-05 | norm: 0.2912, dt: 451.39ms, tok/sec: 1161505.84
step: 18352 | loss: 3.063301 | lr: 6.2053e-05 | norm: 0.3031, dt: 451.00ms, tok/sec: 1162505.47
step: 18353 | loss: 3.074473 | lr: 6.2047e-05 | norm: 0.3126, dt: 451.75ms, tok/sec: 1160575.91
step: 18354 | loss: 3.059503 | lr: 6.2041e-05 | norm: 0.3013, dt: 450.76ms, tok/sec: 1163128.35
step: 18355 | loss: 3.101491 | lr: 6.2036e-05 | norm: 0.3206, dt: 451.36ms, tok/sec: 1161575.17
step: 18356 | loss: 3.053488 | lr: 6.2030e-05 | norm: 0.4268, dt: 450.79ms, tok/sec: 1163035.46
step: 18357 | loss: 3.073671 | lr: 6.2024e-05 | norm: 0.3191, dt: 450.92ms, tok/sec: 1162713.85
step: 18358 | loss: 3.024485 | lr: 6.2019e-05 | norm: 0.3203, dt: 450.10ms, tok/sec: 1164829.44
step: 18359 | loss: 3.004347 | lr: 6.2013e-05 | norm: 0.3545, dt: 451.07ms, tok/sec: 1162317.45
step: 18360 | loss: 3.050236 | lr: 6.2007e-05 | norm: 0.2674, dt: 451.16ms, tok/sec: 1162092.03
step: 18361 | loss: 3.007772 | lr: 6.2002e-05 | norm: 0.3950, dt: 450.83ms, tok/sec: 1162935.21
step: 18362 | loss: 3.009387 | lr: 6.1996e-05 | norm: 0.3375, dt: 450.69ms, tok/sec: 1163300.02
step: 18363 | loss: 2.999071 | lr: 6.1991e-05 | norm: 0.4138, dt: 451.17ms, tok/sec: 1162068.08
step: 18364 | loss: 3.021551 | lr: 6.1985e-05 | norm: 0.3853, dt: 451.12ms, tok/sec: 1162184.77
step: 18365 | loss: 3.024960 | lr: 6.1979e-05 | norm: 0.2796, dt: 452.02ms, tok/sec: 1159875.00
step: 18366 | loss: 2.981745 | lr: 6.1974e-05 | norm: 0.4259, dt: 451.25ms, tok/sec: 1161849.50
step: 18367 | loss: 3.007528 | lr: 6.1968e-05 | norm: 0.3051, dt: 450.55ms, tok/sec: 1163654.60
step: 18368 | loss: 2.967895 | lr: 6.1963e-05 | norm: 0.3116, dt: 452.99ms, tok/sec: 1157404.42
step: 18369 | loss: 3.020238 | lr: 6.1957e-05 | norm: 0.4369, dt: 451.02ms, tok/sec: 1162455.70
step: 18370 | loss: 3.074764 | lr: 6.1952e-05 | norm: 0.2889, dt: 450.31ms, tok/sec: 1164277.47
step: 18371 | loss: 3.079775 | lr: 6.1946e-05 | norm: 0.3932, dt: 450.78ms, tok/sec: 1163072.37
step: 18372 | loss: 3.071345 | lr: 6.1940e-05 | norm: 0.3334, dt: 454.37ms, tok/sec: 1153876.52
step: 18373 | loss: 3.096908 | lr: 6.1935e-05 | norm: 0.3631, dt: 459.41ms, tok/sec: 1141214.46
step: 18374 | loss: 3.032270 | lr: 6.1929e-05 | norm: 0.3684, dt: 451.36ms, tok/sec: 1161567.19
step: 18375 | loss: 3.093177 | lr: 6.1924e-05 | norm: 0.3033, dt: 450.47ms, tok/sec: 1163872.62
step: 18376 | loss: 3.116108 | lr: 6.1918e-05 | norm: 0.3717, dt: 451.70ms, tok/sec: 1160691.07
step: 18377 | loss: 3.089566 | lr: 6.1913e-05 | norm: 0.3136, dt: 449.88ms, tok/sec: 1165403.54
step: 18378 | loss: 3.036109 | lr: 6.1907e-05 | norm: 0.2878, dt: 451.48ms, tok/sec: 1161259.88
step: 18379 | loss: 3.046419 | lr: 6.1902e-05 | norm: 0.3084, dt: 451.51ms, tok/sec: 1161178.33
step: 18380 | loss: 3.090503 | lr: 6.1896e-05 | norm: 0.2877, dt: 451.23ms, tok/sec: 1161913.35
step: 18381 | loss: 3.026292 | lr: 6.1891e-05 | norm: 0.2905, dt: 450.43ms, tok/sec: 1163984.13
step: 18382 | loss: 3.083498 | lr: 6.1886e-05 | norm: 0.3241, dt: 451.68ms, tok/sec: 1160745.60
step: 18383 | loss: 3.046761 | lr: 6.1880e-05 | norm: 0.2782, dt: 450.86ms, tok/sec: 1162852.19
step: 18384 | loss: 3.110071 | lr: 6.1875e-05 | norm: 0.3234, dt: 451.02ms, tok/sec: 1162461.23
step: 18385 | loss: 3.052328 | lr: 6.1869e-05 | norm: 0.3194, dt: 454.45ms, tok/sec: 1153669.49
step: 18386 | loss: 3.048518 | lr: 6.1864e-05 | norm: 0.2698, dt: 452.05ms, tok/sec: 1159805.88
step: 18387 | loss: 3.097546 | lr: 6.1858e-05 | norm: 0.3534, dt: 450.16ms, tok/sec: 1164680.14
step: 18388 | loss: 3.029732 | lr: 6.1853e-05 | norm: 0.2864, dt: 451.39ms, tok/sec: 1161494.18
step: 18389 | loss: 3.017217 | lr: 6.1848e-05 | norm: 0.3513, dt: 451.11ms, tok/sec: 1162216.09
step: 18390 | loss: 3.105237 | lr: 6.1842e-05 | norm: 0.2858, dt: 451.19ms, tok/sec: 1162015.27
step: 18391 | loss: 3.047398 | lr: 6.1837e-05 | norm: 0.3100, dt: 450.62ms, tok/sec: 1163471.13
step: 18392 | loss: 3.065779 | lr: 6.1831e-05 | norm: 0.4181, dt: 451.75ms, tok/sec: 1160574.07
step: 18393 | loss: 3.012621 | lr: 6.1826e-05 | norm: 0.3118, dt: 451.84ms, tok/sec: 1160341.36
step: 18394 | loss: 3.034497 | lr: 6.1821e-05 | norm: 0.3918, dt: 450.90ms, tok/sec: 1162759.96
step: 18395 | loss: 3.083596 | lr: 6.1815e-05 | norm: 0.3465, dt: 450.66ms, tok/sec: 1163378.80
step: 18396 | loss: 3.037799 | lr: 6.1810e-05 | norm: 0.3140, dt: 451.58ms, tok/sec: 1161004.83
step: 18397 | loss: 3.071001 | lr: 6.1805e-05 | norm: 0.4064, dt: 451.70ms, tok/sec: 1160708.84
step: 18398 | loss: 2.987783 | lr: 6.1799e-05 | norm: 0.3205, dt: 450.72ms, tok/sec: 1163219.41
step: 18399 | loss: 3.009423 | lr: 6.1794e-05 | norm: 0.3618, dt: 450.82ms, tok/sec: 1162969.65
step: 18400 | loss: 2.951811 | lr: 6.1789e-05 | norm: 0.3466, dt: 451.03ms, tok/sec: 1162434.80
step: 18401 | loss: 3.039810 | lr: 6.1783e-05 | norm: 0.2782, dt: 451.27ms, tok/sec: 1161793.03
step: 18402 | loss: 2.987410 | lr: 6.1778e-05 | norm: 0.3573, dt: 450.64ms, tok/sec: 1163419.42
step: 18403 | loss: 2.993822 | lr: 6.1773e-05 | norm: 0.3616, dt: 451.23ms, tok/sec: 1161896.77
step: 18404 | loss: 2.973686 | lr: 6.1768e-05 | norm: 0.2916, dt: 450.32ms, tok/sec: 1164255.90
step: 18405 | loss: 2.987612 | lr: 6.1762e-05 | norm: 0.3255, dt: 450.81ms, tok/sec: 1162980.72
step: 18406 | loss: 3.077298 | lr: 6.1757e-05 | norm: 0.3441, dt: 452.44ms, tok/sec: 1158813.93
step: 18407 | loss: 3.113096 | lr: 6.1752e-05 | norm: 0.4595, dt: 451.09ms, tok/sec: 1162266.46
step: 18408 | loss: 3.030231 | lr: 6.1746e-05 | norm: 0.3208, dt: 450.49ms, tok/sec: 1163819.65
step: 18409 | loss: 3.049878 | lr: 6.1741e-05 | norm: 0.3185, dt: 452.79ms, tok/sec: 1157909.64
step: 18410 | loss: 3.112497 | lr: 6.1736e-05 | norm: 0.3509, dt: 451.31ms, tok/sec: 1161707.10
step: 18411 | loss: 2.997562 | lr: 6.1731e-05 | norm: 0.3070, dt: 450.18ms, tok/sec: 1164609.21
step: 18412 | loss: 2.970378 | lr: 6.1726e-05 | norm: 0.2983, dt: 451.35ms, tok/sec: 1161601.55
step: 18413 | loss: 3.039157 | lr: 6.1720e-05 | norm: 0.2982, dt: 450.34ms, tok/sec: 1164196.73
step: 18414 | loss: 3.030859 | lr: 6.1715e-05 | norm: 0.3364, dt: 451.34ms, tok/sec: 1161615.05
step: 18415 | loss: 3.193314 | lr: 6.1710e-05 | norm: 0.4091, dt: 450.57ms, tok/sec: 1163609.03
step: 18416 | loss: 3.055297 | lr: 6.1705e-05 | norm: 0.3458, dt: 451.03ms, tok/sec: 1162418.83
step: 18417 | loss: 3.057795 | lr: 6.1700e-05 | norm: 0.3224, dt: 451.61ms, tok/sec: 1160934.95
step: 18418 | loss: 3.044344 | lr: 6.1694e-05 | norm: 0.3141, dt: 451.54ms, tok/sec: 1161098.62
step: 18419 | loss: 3.045013 | lr: 6.1689e-05 | norm: 0.3009, dt: 450.84ms, tok/sec: 1162917.99
step: 18420 | loss: 3.055968 | lr: 6.1684e-05 | norm: 0.2878, dt: 451.22ms, tok/sec: 1161939.13
step: 18421 | loss: 3.006749 | lr: 6.1679e-05 | norm: 0.3275, dt: 450.94ms, tok/sec: 1162646.84
step: 18422 | loss: 3.070499 | lr: 6.1674e-05 | norm: 0.3104, dt: 450.86ms, tok/sec: 1162864.48
step: 18423 | loss: 3.019894 | lr: 6.1669e-05 | norm: 0.3043, dt: 450.77ms, tok/sec: 1163082.83
step: 18424 | loss: 3.078233 | lr: 6.1664e-05 | norm: 0.2864, dt: 450.54ms, tok/sec: 1163695.85
step: 18425 | loss: 3.098602 | lr: 6.1658e-05 | norm: 0.3029, dt: 451.52ms, tok/sec: 1161151.96
step: 18426 | loss: 3.020059 | lr: 6.1653e-05 | norm: 0.3018, dt: 451.41ms, tok/sec: 1161442.65
step: 18427 | loss: 3.092587 | lr: 6.1648e-05 | norm: 0.3006, dt: 450.89ms, tok/sec: 1162795.62
step: 18428 | loss: 3.083488 | lr: 6.1643e-05 | norm: 0.3059, dt: 450.69ms, tok/sec: 1163308.64
step: 18429 | loss: 3.040133 | lr: 6.1638e-05 | norm: 0.2633, dt: 1229.79ms, tok/sec: 426324.69
step: 18430 | loss: 3.124747 | lr: 6.1633e-05 | norm: 0.3129, dt: 452.52ms, tok/sec: 1158587.42
step: 18431 | loss: 3.005457 | lr: 6.1628e-05 | norm: 0.2990, dt: 450.99ms, tok/sec: 1162514.69
step: 18432 | loss: 3.053072 | lr: 6.1623e-05 | norm: 0.2712, dt: 449.97ms, tok/sec: 1165157.17
step: 18433 | loss: 3.043115 | lr: 6.1618e-05 | norm: 0.2998, dt: 451.03ms, tok/sec: 1162426.20
step: 18434 | loss: 3.038436 | lr: 6.1613e-05 | norm: 0.2856, dt: 450.58ms, tok/sec: 1163573.94
step: 18435 | loss: 3.018604 | lr: 6.1608e-05 | norm: 0.2980, dt: 451.19ms, tok/sec: 1162007.28
step: 18436 | loss: 2.994628 | lr: 6.1603e-05 | norm: 0.2860, dt: 450.55ms, tok/sec: 1163669.38
step: 18437 | loss: 3.009231 | lr: 6.1598e-05 | norm: 0.2744, dt: 450.74ms, tok/sec: 1163165.27
step: 18438 | loss: 2.975153 | lr: 6.1593e-05 | norm: 0.3121, dt: 451.43ms, tok/sec: 1161391.13
step: 18439 | loss: 3.029245 | lr: 6.1588e-05 | norm: 0.2736, dt: 449.63ms, tok/sec: 1166041.28
step: 18440 | loss: 2.997211 | lr: 6.1583e-05 | norm: 0.3005, dt: 452.86ms, tok/sec: 1157715.79
step: 18441 | loss: 3.039412 | lr: 6.1578e-05 | norm: 0.3418, dt: 450.24ms, tok/sec: 1164454.41
step: 18442 | loss: 3.036022 | lr: 6.1573e-05 | norm: 0.3374, dt: 451.02ms, tok/sec: 1162458.16
step: 18443 | loss: 3.055441 | lr: 6.1568e-05 | norm: 0.2706, dt: 450.80ms, tok/sec: 1163014.55
step: 18444 | loss: 3.069524 | lr: 6.1563e-05 | norm: 0.3505, dt: 451.32ms, tok/sec: 1161673.96
step: 18445 | loss: 3.012088 | lr: 6.1558e-05 | norm: 0.2940, dt: 451.13ms, tok/sec: 1162167.57
step: 18446 | loss: 3.046990 | lr: 6.1553e-05 | norm: 0.3856, dt: 450.35ms, tok/sec: 1164168.38
step: 18447 | loss: 2.982135 | lr: 6.1548e-05 | norm: 0.2863, dt: 451.10ms, tok/sec: 1162240.05
step: 18448 | loss: 3.060933 | lr: 6.1543e-05 | norm: 0.3868, dt: 450.74ms, tok/sec: 1163178.19
step: 18449 | loss: 3.017799 | lr: 6.1538e-05 | norm: 0.3434, dt: 451.00ms, tok/sec: 1162510.39
step: 18450 | loss: 3.049906 | lr: 6.1533e-05 | norm: 0.3661, dt: 450.42ms, tok/sec: 1163992.14
step: 18451 | loss: 3.123344 | lr: 6.1528e-05 | norm: 0.3072, dt: 450.49ms, tok/sec: 1163809.18
step: 18452 | loss: 3.056692 | lr: 6.1523e-05 | norm: 0.3611, dt: 451.48ms, tok/sec: 1161258.65
step: 18453 | loss: 3.091049 | lr: 6.1518e-05 | norm: 0.2926, dt: 450.35ms, tok/sec: 1164180.70
step: 18454 | loss: 3.077637 | lr: 6.1513e-05 | norm: 0.3608, dt: 451.53ms, tok/sec: 1161139.70
step: 18455 | loss: 3.064038 | lr: 6.1509e-05 | norm: 0.2745, dt: 450.76ms, tok/sec: 1163117.89
step: 18456 | loss: 3.063845 | lr: 6.1504e-05 | norm: 0.3522, dt: 450.90ms, tok/sec: 1162755.04
step: 18457 | loss: 3.068645 | lr: 6.1499e-05 | norm: 0.3533, dt: 449.90ms, tok/sec: 1165347.34
step: 18458 | loss: 3.025639 | lr: 6.1494e-05 | norm: 0.3484, dt: 450.48ms, tok/sec: 1163847.98
step: 18459 | loss: 3.060870 | lr: 6.1489e-05 | norm: 0.4062, dt: 451.07ms, tok/sec: 1162332.20
step: 18460 | loss: 3.120768 | lr: 6.1484e-05 | norm: 0.3564, dt: 449.90ms, tok/sec: 1165347.34
step: 18461 | loss: 3.029624 | lr: 6.1479e-05 | norm: 0.3376, dt: 450.80ms, tok/sec: 1163018.24
step: 18462 | loss: 3.093569 | lr: 6.1475e-05 | norm: 0.2964, dt: 450.34ms, tok/sec: 1164204.74
step: 18463 | loss: 3.077870 | lr: 6.1470e-05 | norm: 0.3356, dt: 451.23ms, tok/sec: 1161900.45
step: 18464 | loss: 3.113632 | lr: 6.1465e-05 | norm: 0.3038, dt: 449.58ms, tok/sec: 1166181.65
step: 18465 | loss: 3.048220 | lr: 6.1460e-05 | norm: 0.3083, dt: 451.15ms, tok/sec: 1162122.12
step: 18466 | loss: 3.029255 | lr: 6.1455e-05 | norm: 0.2770, dt: 450.77ms, tok/sec: 1163093.90
step: 18467 | loss: 3.012491 | lr: 6.1451e-05 | norm: 0.3013, dt: 450.01ms, tok/sec: 1165063.95
step: 18468 | loss: 3.022378 | lr: 6.1446e-05 | norm: 0.3017, dt: 450.89ms, tok/sec: 1162796.23
step: 18469 | loss: 3.020229 | lr: 6.1441e-05 | norm: 0.2920, dt: 450.72ms, tok/sec: 1163217.56
step: 18470 | loss: 3.024303 | lr: 6.1436e-05 | norm: 0.2767, dt: 451.27ms, tok/sec: 1161807.76
step: 18471 | loss: 3.038185 | lr: 6.1432e-05 | norm: 0.2912, dt: 451.59ms, tok/sec: 1160988.28
step: 18472 | loss: 3.019753 | lr: 6.1427e-05 | norm: 0.2747, dt: 450.13ms, tok/sec: 1164735.04
step: 18473 | loss: 2.999901 | lr: 6.1422e-05 | norm: 0.3347, dt: 450.79ms, tok/sec: 1163042.23
step: 18474 | loss: 2.998991 | lr: 6.1417e-05 | norm: 0.2616, dt: 450.48ms, tok/sec: 1163841.82
step: 18475 | loss: 2.956808 | lr: 6.1413e-05 | norm: 0.2917, dt: 450.25ms, tok/sec: 1164424.82
step: 18476 | loss: 3.029901 | lr: 6.1408e-05 | norm: 0.2821, dt: 449.79ms, tok/sec: 1165632.73
step: 18477 | loss: 2.998812 | lr: 6.1403e-05 | norm: 0.2826, dt: 451.28ms, tok/sec: 1161775.23
step: 18478 | loss: 3.050153 | lr: 6.1398e-05 | norm: 0.3416, dt: 450.58ms, tok/sec: 1163583.17
step: 18479 | loss: 3.089998 | lr: 6.1394e-05 | norm: 0.3017, dt: 450.54ms, tok/sec: 1163692.78
step: 18480 | loss: 3.070073 | lr: 6.1389e-05 | norm: 0.3121, dt: 449.84ms, tok/sec: 1165498.67
step: 18481 | loss: 3.054475 | lr: 6.1384e-05 | norm: 0.3054, dt: 451.26ms, tok/sec: 1161828.63
step: 18482 | loss: 3.038332 | lr: 6.1380e-05 | norm: 0.2885, dt: 450.58ms, tok/sec: 1163575.78
step: 18483 | loss: 3.001132 | lr: 6.1375e-05 | norm: 0.2828, dt: 451.11ms, tok/sec: 1162218.55
step: 18484 | loss: 3.058071 | lr: 6.1370e-05 | norm: 0.2928, dt: 451.79ms, tok/sec: 1160458.32
step: 18485 | loss: 3.141232 | lr: 6.1366e-05 | norm: 0.3361, dt: 450.45ms, tok/sec: 1163923.75
step: 18486 | loss: 3.059494 | lr: 6.1361e-05 | norm: 0.3260, dt: 454.07ms, tok/sec: 1154652.04
step: 18487 | loss: 3.083995 | lr: 6.1356e-05 | norm: 0.3216, dt: 450.69ms, tok/sec: 1163301.25
step: 18488 | loss: 3.088921 | lr: 6.1352e-05 | norm: 0.3510, dt: 450.53ms, tok/sec: 1163720.49
step: 18489 | loss: 3.104851 | lr: 6.1347e-05 | norm: 0.3173, dt: 449.62ms, tok/sec: 1166068.48
step: 18490 | loss: 3.061135 | lr: 6.1343e-05 | norm: 0.3413, dt: 450.85ms, tok/sec: 1162886.01
step: 18491 | loss: 3.070268 | lr: 6.1338e-05 | norm: 0.3039, dt: 451.18ms, tok/sec: 1162033.07
step: 18492 | loss: 3.060261 | lr: 6.1333e-05 | norm: 0.3520, dt: 450.60ms, tok/sec: 1163531.46
step: 18493 | loss: 3.079232 | lr: 6.1329e-05 | norm: 0.2984, dt: 452.89ms, tok/sec: 1157659.11
step: 18494 | loss: 3.007133 | lr: 6.1324e-05 | norm: 0.3192, dt: 450.63ms, tok/sec: 1163466.82
step: 18495 | loss: 3.084582 | lr: 6.1320e-05 | norm: 0.2978, dt: 450.50ms, tok/sec: 1163801.78
step: 18496 | loss: 3.024194 | lr: 6.1315e-05 | norm: 0.2799, dt: 451.00ms, tok/sec: 1162501.17
step: 18497 | loss: 3.041943 | lr: 6.1311e-05 | norm: 0.2922, dt: 451.07ms, tok/sec: 1162324.21
step: 18498 | loss: 3.037817 | lr: 6.1306e-05 | norm: 0.2673, dt: 451.06ms, tok/sec: 1162340.80
step: 18499 | loss: 3.045189 | lr: 6.1302e-05 | norm: 0.3026, dt: 450.28ms, tok/sec: 1164357.61
validation loss: 3.0868
HellaSwag accuracy: 3010/10042=0.2997
rank 5 sample 0: Hello, I'm a Manpreet, a very smart guy who's got a problem. We've got our eyes on it. We've got the mouth
rank 5 sample 1: Hello, I'm a Manpreet, don't feel guilty and just give me a couple of questions about those. Yes, you can tell them something about
rank 5 sample 2: Hello, I'm a Manpreet, is it that you hear? Are you a man?
Well, let me know right now. I'm a
rank 5 sample 3: Hello, I'm a Manpreet, just like a fish. (And I mean, if the fish were a really interesting animal.) At first, it
rank 0 sample 0: Hello, I'm a Manpreet, and I know that you will get something right the first time, that's it. But wait... what happens when
rank 0 sample 1: Hello, I'm a Manpreet, an Indonesian or a Dutchman, have you seen many interesting and wonderful articles in this area? I love reading about
rank 0 sample 2: Hello, I'm a Manpreet, I'm a Shillist, and I'm a Shalt-Loving Shillist. I'm not
rank 0 sample 3: Hello, I'm a Manpreet,
In the previous post, we mentioned how simple it is to add a few rules, which will simplify the whole
rank 4 sample 0: Hello, I'm a Manpreet, I'm my Son, I'm my Head. And I'm my Teacher. And we did it this time with
rank 4 sample 1: Hello, I'm a Manpreet, if that's confusing for you now, that's too soon. This page was last updated at 10:53 UTC
rank 2 sample 0: Hello, I'm a Manpreet, as it is known that, the woman who was married in that time was the mother of the child. That's
rank 7 sample 0: Hello, I'm a Manpreet, and I'm a woman - I'm now a man.
I feel like...
But because you are arank 4 sample 2: Hello, I'm a Manpreet, you use the form. This is where we all get together: the last time we've just talked about the other

rank 2 sample 1: Hello, I'm a Manpreet, it's true."
The idea from Michael's
"It's one of the most exciting
I've ever
rank 4 sample 3: Hello, I'm a Manpreet, and this idea makes you happy. Just kidding…we see why we just did it. We do this because we
rank 7 sample 1: Hello, I'm a Manpreet, one day while I'm making tea you just sit down from the floor and I give a few words:
I
rank 2 sample 2: Hello, I'm a Manpreet, I'm the King! I just know that I am a person. In a very positive light!" You know,
rank 7 sample 2: Hello, I'm a Manpreet, right? I had to go a few years ago and I had to wait for my son to come out. My
rank 2 sample 3: Hello, I'm a Manpreet, a student at the university, and I wanted to show you an example of a new technology of self-driving vehicles
rank 7 sample 3: Hello, I'm a Manpreet, let me get back to what you really want to do. I want to get rid of the idea that I'm
rank 6 sample 0: Hello, I'm a Manpreet, please send me a post I love
- The Great War, 1916–1918
- The Great Depression,
rank 6 sample 1: Hello, I'm a Manpreet,
you'll be very pleased,
in my humble opinion,
by the time the first man in a business
rank 6 sample 2: Hello, I'm a Manpreet, a woman who is passionate in music and who also loves to write in her daily life. I'm doing everything by
rank 6 sample 3: Hello, I'm a Manpreet, and it's kind of like "this is my name", and it's there. It works like "Hello,
rank 1 sample 0: Hello, I'm a Manpreet, we'll keep you posted on the 'What You Do':".
In these sentences the second is the one that
rank 1 sample 1: Hello, I'm a Manpreet, so this is a new lesson! :)
I want for a simple lesson: in Lesson 21 we are going
rank 1 sample 2: Hello, I'm a Manpreet,
Well, now I'm going to be a Manpreet
and I'm going to be a manping
rank 1 sample 3: Hello, I'm a Manpreet, and I'm in a little office right now. I feel like I'm very important enough to be able to communicate
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going into some house on the other side of the river" at that, it's something I have
rank 3 sample 1: Hello, I'm a Manpreet, and that is the first thing about my life. I have a passion for and love to be the person who takes
rank 3 sample 2: Hello, I'm a Manpreet, and you are a Wanna (Wanna I'm a Wanna) (You're a Wanna) that
rank 3 sample 3: Hello, I'm a Manpreet, and thanks for taking the time to make it! It really helps if we can teach students the things these people are
step: 18500 | loss: 3.051025 | lr: 6.1297e-05 | norm: 0.3488, dt: 12079.97ms, tok/sec: 43401.43
step: 18501 | loss: 3.028464 | lr: 6.1292e-05 | norm: 0.2707, dt: 449.00ms, tok/sec: 1167673.39
step: 18502 | loss: 2.990915 | lr: 6.1288e-05 | norm: 0.3029, dt: 449.80ms, tok/sec: 1165614.19
step: 18503 | loss: 3.027586 | lr: 6.1283e-05 | norm: 0.3233, dt: 450.78ms, tok/sec: 1163069.91
step: 18504 | loss: 3.030178 | lr: 6.1279e-05 | norm: 0.3369, dt: 449.27ms, tok/sec: 1166981.85
step: 18505 | loss: 3.009816 | lr: 6.1274e-05 | norm: 0.3213, dt: 451.64ms, tok/sec: 1160855.90
step: 18506 | loss: 3.073270 | lr: 6.1270e-05 | norm: 0.3145, dt: 449.10ms, tok/sec: 1167414.90
step: 18507 | loss: 2.998758 | lr: 6.1266e-05 | norm: 0.3336, dt: 450.11ms, tok/sec: 1164788.10
step: 18508 | loss: 2.961616 | lr: 6.1261e-05 | norm: 0.2645, dt: 450.14ms, tok/sec: 1164710.98
step: 18509 | loss: 2.989213 | lr: 6.1257e-05 | norm: 0.3197, dt: 450.75ms, tok/sec: 1163151.11
step: 18510 | loss: 2.971206 | lr: 6.1252e-05 | norm: 0.2882, dt: 451.68ms, tok/sec: 1160746.83
step: 18511 | loss: 2.927083 | lr: 6.1248e-05 | norm: 0.2943, dt: 449.97ms, tok/sec: 1165173.22
step: 18512 | loss: 2.978127 | lr: 6.1243e-05 | norm: 0.3037, dt: 450.15ms, tok/sec: 1164698.65
step: 18513 | loss: 3.040017 | lr: 6.1239e-05 | norm: 0.3010, dt: 450.65ms, tok/sec: 1163412.65
step: 18514 | loss: 3.081098 | lr: 6.1234e-05 | norm: 0.2875, dt: 451.58ms, tok/sec: 1160998.09
step: 18515 | loss: 3.075264 | lr: 6.1230e-05 | norm: 0.2877, dt: 450.09ms, tok/sec: 1164847.95
step: 18516 | loss: 3.069010 | lr: 6.1226e-05 | norm: 0.3109, dt: 449.26ms, tok/sec: 1167000.43
step: 18517 | loss: 3.025595 | lr: 6.1221e-05 | norm: 0.2918, dt: 450.27ms, tok/sec: 1164376.11
step: 18518 | loss: 3.040664 | lr: 6.1217e-05 | norm: 0.3537, dt: 450.82ms, tok/sec: 1162974.57
step: 18519 | loss: 3.073071 | lr: 6.1212e-05 | norm: 0.3341, dt: 450.80ms, tok/sec: 1163009.01
step: 18520 | loss: 3.071381 | lr: 6.1208e-05 | norm: 0.3390, dt: 449.95ms, tok/sec: 1165205.94
step: 18521 | loss: 3.134819 | lr: 6.1204e-05 | norm: 0.3473, dt: 876.23ms, tok/sec: 598344.37
step: 18522 | loss: 3.053627 | lr: 6.1199e-05 | norm: 0.3197, dt: 450.41ms, tok/sec: 1164019.86
step: 18523 | loss: 3.108854 | lr: 6.1195e-05 | norm: 0.3432, dt: 451.29ms, tok/sec: 1161750.68
step: 18524 | loss: 3.056458 | lr: 6.1191e-05 | norm: 0.3260, dt: 450.90ms, tok/sec: 1162748.27
step: 18525 | loss: 3.088832 | lr: 6.1186e-05 | norm: 0.2931, dt: 450.91ms, tok/sec: 1162730.44
step: 18526 | loss: 3.083879 | lr: 6.1182e-05 | norm: 0.3069, dt: 450.86ms, tok/sec: 1162870.02
step: 18527 | loss: 3.086762 | lr: 6.1178e-05 | norm: 0.3288, dt: 451.67ms, tok/sec: 1160768.88
step: 18528 | loss: 3.059949 | lr: 6.1173e-05 | norm: 0.3227, dt: 450.22ms, tok/sec: 1164510.53
step: 18529 | loss: 3.038168 | lr: 6.1169e-05 | norm: 0.2988, dt: 450.74ms, tok/sec: 1163174.49
step: 18530 | loss: 3.082448 | lr: 6.1165e-05 | norm: 0.3301, dt: 451.89ms, tok/sec: 1160211.58
step: 18531 | loss: 3.078516 | lr: 6.1161e-05 | norm: 0.2907, dt: 450.56ms, tok/sec: 1163634.89
step: 18532 | loss: 3.122072 | lr: 6.1156e-05 | norm: 0.3097, dt: 451.22ms, tok/sec: 1161923.17
step: 18533 | loss: 3.067181 | lr: 6.1152e-05 | norm: 0.3351, dt: 450.97ms, tok/sec: 1162573.08
step: 18534 | loss: 3.092943 | lr: 6.1148e-05 | norm: 0.2990, dt: 451.73ms, tok/sec: 1160612.66
step: 18535 | loss: 3.018532 | lr: 6.1144e-05 | norm: 0.3046, dt: 450.19ms, tok/sec: 1164601.80
step: 18536 | loss: 3.018486 | lr: 6.1139e-05 | norm: 0.2868, dt: 450.79ms, tok/sec: 1163044.69
step: 18537 | loss: 2.993598 | lr: 6.1135e-05 | norm: 0.2973, dt: 450.72ms, tok/sec: 1163227.41
step: 18538 | loss: 3.047558 | lr: 6.1131e-05 | norm: 0.2723, dt: 450.93ms, tok/sec: 1162683.72
step: 18539 | loss: 3.011407 | lr: 6.1127e-05 | norm: 0.2820, dt: 450.41ms, tok/sec: 1164015.55
step: 18540 | loss: 2.991518 | lr: 6.1122e-05 | norm: 0.3142, dt: 451.39ms, tok/sec: 1161492.96
step: 18541 | loss: 3.016063 | lr: 6.1118e-05 | norm: 0.2788, dt: 451.45ms, tok/sec: 1161352.49
step: 18542 | loss: 2.982379 | lr: 6.1114e-05 | norm: 0.3153, dt: 450.70ms, tok/sec: 1163264.94
step: 18543 | loss: 3.010399 | lr: 6.1110e-05 | norm: 0.3150, dt: 450.24ms, tok/sec: 1164456.26
step: 18544 | loss: 3.000050 | lr: 6.1106e-05 | norm: 0.2882, dt: 450.78ms, tok/sec: 1163074.22
step: 18545 | loss: 2.974356 | lr: 6.1101e-05 | norm: 0.3389, dt: 450.34ms, tok/sec: 1164202.89
step: 18546 | loss: 2.979078 | lr: 6.1097e-05 | norm: 0.3664, dt: 450.45ms, tok/sec: 1163912.05
step: 18547 | loss: 3.022107 | lr: 6.1093e-05 | norm: 0.3243, dt: 451.15ms, tok/sec: 1162108.61
step: 18548 | loss: 3.014476 | lr: 6.1089e-05 | norm: 0.3608, dt: 450.79ms, tok/sec: 1163040.38
step: 18549 | loss: 3.060755 | lr: 6.1085e-05 | norm: 0.3309, dt: 450.32ms, tok/sec: 1164264.53
step: 18550 | loss: 3.225433 | lr: 6.1081e-05 | norm: 0.3726, dt: 451.38ms, tok/sec: 1161513.82
step: 18551 | loss: 3.082994 | lr: 6.1077e-05 | norm: 0.3627, dt: 451.53ms, tok/sec: 1161142.15
step: 18552 | loss: 3.068160 | lr: 6.1072e-05 | norm: 0.3056, dt: 449.66ms, tok/sec: 1165956.58
step: 18553 | loss: 3.014995 | lr: 6.1068e-05 | norm: 0.3699, dt: 450.63ms, tok/sec: 1163453.89
step: 18554 | loss: 3.076221 | lr: 6.1064e-05 | norm: 0.3524, dt: 452.00ms, tok/sec: 1159938.63
step: 18555 | loss: 3.028222 | lr: 6.1060e-05 | norm: 0.3017, dt: 450.52ms, tok/sec: 1163732.19
step: 18556 | loss: 3.116050 | lr: 6.1056e-05 | norm: 0.2987, dt: 449.34ms, tok/sec: 1166795.47
step: 18557 | loss: 3.066242 | lr: 6.1052e-05 | norm: 0.3927, dt: 451.21ms, tok/sec: 1161960.01
step: 18558 | loss: 3.167620 | lr: 6.1048e-05 | norm: 0.3203, dt: 450.86ms, tok/sec: 1162871.25
step: 18559 | loss: 3.035182 | lr: 6.1044e-05 | norm: 0.3556, dt: 450.64ms, tok/sec: 1163423.73
step: 18560 | loss: 3.018327 | lr: 6.1040e-05 | norm: 0.3322, dt: 451.06ms, tok/sec: 1162346.33
step: 18561 | loss: 3.070670 | lr: 6.1036e-05 | norm: 0.3330, dt: 450.56ms, tok/sec: 1163641.05
step: 18562 | loss: 3.024490 | lr: 6.1032e-05 | norm: 0.3104, dt: 450.59ms, tok/sec: 1163564.09
step: 18563 | loss: 3.042652 | lr: 6.1028e-05 | norm: 0.3477, dt: 450.85ms, tok/sec: 1162888.47
step: 18564 | loss: 3.016025 | lr: 6.1024e-05 | norm: 0.3128, dt: 451.09ms, tok/sec: 1162267.08
step: 18565 | loss: 3.029476 | lr: 6.1020e-05 | norm: 0.3102, dt: 451.08ms, tok/sec: 1162305.78
step: 18566 | loss: 3.050953 | lr: 6.1016e-05 | norm: 0.2925, dt: 450.30ms, tok/sec: 1164308.91
step: 18567 | loss: 3.099798 | lr: 6.1012e-05 | norm: 0.4258, dt: 451.41ms, tok/sec: 1161454.31
step: 18568 | loss: 3.066854 | lr: 6.1008e-05 | norm: 0.2774, dt: 450.97ms, tok/sec: 1162578.61
step: 18569 | loss: 3.058649 | lr: 6.1004e-05 | norm: 0.3987, dt: 451.05ms, tok/sec: 1162372.75
step: 18570 | loss: 3.021756 | lr: 6.1000e-05 | norm: 0.2766, dt: 450.62ms, tok/sec: 1163490.82
step: 18571 | loss: 2.949576 | lr: 6.0996e-05 | norm: 0.4200, dt: 451.87ms, tok/sec: 1160270.34
step: 18572 | loss: 3.122075 | lr: 6.0992e-05 | norm: 0.3253, dt: 450.23ms, tok/sec: 1164493.26
step: 18573 | loss: 3.018959 | lr: 6.0988e-05 | norm: 0.3785, dt: 450.48ms, tok/sec: 1163849.83
step: 18574 | loss: 2.966328 | lr: 6.0984e-05 | norm: 0.2766, dt: 451.00ms, tok/sec: 1162512.23
step: 18575 | loss: 3.008092 | lr: 6.0980e-05 | norm: 0.3763, dt: 449.95ms, tok/sec: 1165225.08
step: 18576 | loss: 3.028940 | lr: 6.0976e-05 | norm: 0.2978, dt: 450.29ms, tok/sec: 1164326.79
step: 18577 | loss: 3.049873 | lr: 6.0972e-05 | norm: 0.3291, dt: 450.27ms, tok/sec: 1164379.81
step: 18578 | loss: 2.970738 | lr: 6.0968e-05 | norm: 0.2911, dt: 450.86ms, tok/sec: 1162854.65
step: 18579 | loss: 3.008914 | lr: 6.0964e-05 | norm: 0.3606, dt: 450.93ms, tok/sec: 1162686.80
step: 18580 | loss: 2.933978 | lr: 6.0960e-05 | norm: 0.2830, dt: 449.90ms, tok/sec: 1165334.99
step: 18581 | loss: 3.014258 | lr: 6.0956e-05 | norm: 0.3454, dt: 450.93ms, tok/sec: 1162672.04
step: 18582 | loss: 3.022392 | lr: 6.0953e-05 | norm: 0.3552, dt: 451.21ms, tok/sec: 1161955.71
step: 18583 | loss: 3.074026 | lr: 6.0949e-05 | norm: 0.3075, dt: 450.11ms, tok/sec: 1164789.33
step: 18584 | loss: 3.055514 | lr: 6.0945e-05 | norm: 0.3415, dt: 450.72ms, tok/sec: 1163223.72
step: 18585 | loss: 2.954047 | lr: 6.0941e-05 | norm: 0.3154, dt: 450.71ms, tok/sec: 1163248.95
step: 18586 | loss: 3.015828 | lr: 6.0937e-05 | norm: 0.3518, dt: 450.62ms, tok/sec: 1163477.90
step: 18587 | loss: 3.005633 | lr: 6.0933e-05 | norm: 0.3317, dt: 450.29ms, tok/sec: 1164342.82
step: 18588 | loss: 3.076650 | lr: 6.0929e-05 | norm: 0.3692, dt: 450.63ms, tok/sec: 1163453.27
step: 18589 | loss: 3.013493 | lr: 6.0926e-05 | norm: 0.2956, dt: 451.43ms, tok/sec: 1161382.54
step: 18590 | loss: 3.117935 | lr: 6.0922e-05 | norm: 0.3785, dt: 450.66ms, tok/sec: 1163389.26
step: 18591 | loss: 3.106584 | lr: 6.0918e-05 | norm: 0.3167, dt: 450.59ms, tok/sec: 1163569.01
step: 18592 | loss: 3.059295 | lr: 6.0914e-05 | norm: 0.3548, dt: 450.83ms, tok/sec: 1162949.35
step: 18593 | loss: 3.029909 | lr: 6.0910e-05 | norm: 0.2981, dt: 451.94ms, tok/sec: 1160079.98
step: 18594 | loss: 3.112381 | lr: 6.0907e-05 | norm: 0.3168, dt: 450.78ms, tok/sec: 1163063.14
step: 18595 | loss: 3.110876 | lr: 6.0903e-05 | norm: 0.3140, dt: 450.69ms, tok/sec: 1163298.79
step: 18596 | loss: 3.051414 | lr: 6.0899e-05 | norm: 0.4084, dt: 451.73ms, tok/sec: 1160633.49
step: 18597 | loss: 3.089972 | lr: 6.0895e-05 | norm: 0.3009, dt: 450.40ms, tok/sec: 1164061.15
step: 18598 | loss: 2.989635 | lr: 6.0892e-05 | norm: 0.3519, dt: 451.43ms, tok/sec: 1161401.56
step: 18599 | loss: 3.041526 | lr: 6.0888e-05 | norm: 0.2849, dt: 450.98ms, tok/sec: 1162540.51
step: 18600 | loss: 3.027113 | lr: 6.0884e-05 | norm: 0.3125, dt: 450.85ms, tok/sec: 1162895.23
step: 18601 | loss: 3.085654 | lr: 6.0880e-05 | norm: 0.3067, dt: 450.49ms, tok/sec: 1163824.57
step: 18602 | loss: 3.045722 | lr: 6.0877e-05 | norm: 0.3110, dt: 451.23ms, tok/sec: 1161900.45
step: 18603 | loss: 3.044659 | lr: 6.0873e-05 | norm: 0.2760, dt: 450.31ms, tok/sec: 1164278.09
step: 18604 | loss: 3.081950 | lr: 6.0869e-05 | norm: 0.3065, dt: 450.96ms, tok/sec: 1162609.34
step: 18605 | loss: 3.027394 | lr: 6.0865e-05 | norm: 0.3548, dt: 450.29ms, tok/sec: 1164321.86
step: 18606 | loss: 2.994997 | lr: 6.0862e-05 | norm: 0.2936, dt: 451.70ms, tok/sec: 1160710.07
step: 18607 | loss: 3.025612 | lr: 6.0858e-05 | norm: 0.2845, dt: 450.84ms, tok/sec: 1162911.22
step: 18608 | loss: 2.968102 | lr: 6.0854e-05 | norm: 0.3258, dt: 452.06ms, tok/sec: 1159765.50
step: 18609 | loss: 2.946490 | lr: 6.0851e-05 | norm: 0.2990, dt: 450.88ms, tok/sec: 1162810.99
step: 18610 | loss: 3.009204 | lr: 6.0847e-05 | norm: 0.3529, dt: 450.27ms, tok/sec: 1164379.81
step: 18611 | loss: 2.994100 | lr: 6.0843e-05 | norm: 0.2699, dt: 451.10ms, tok/sec: 1162254.18
step: 18612 | loss: 3.053175 | lr: 6.0840e-05 | norm: 0.3075, dt: 449.98ms, tok/sec: 1165143.58
step: 18613 | loss: 2.958158 | lr: 6.0836e-05 | norm: 0.3005, dt: 450.78ms, tok/sec: 1163064.37
step: 18614 | loss: 3.028479 | lr: 6.0833e-05 | norm: 0.2897, dt: 450.62ms, tok/sec: 1163470.51
step: 18615 | loss: 3.003637 | lr: 6.0829e-05 | norm: 0.3242, dt: 450.75ms, tok/sec: 1163149.88
step: 18616 | loss: 3.000738 | lr: 6.0825e-05 | norm: 0.2711, dt: 451.45ms, tok/sec: 1161345.13
step: 18617 | loss: 3.114330 | lr: 6.0822e-05 | norm: 0.3345, dt: 450.66ms, tok/sec: 1163383.11
step: 18618 | loss: 3.016255 | lr: 6.0818e-05 | norm: 0.2959, dt: 450.92ms, tok/sec: 1162694.79
step: 18619 | loss: 3.084138 | lr: 6.0814e-05 | norm: 0.3147, dt: 935.23ms, tok/sec: 560599.06
step: 18620 | loss: 3.060944 | lr: 6.0811e-05 | norm: 0.3298, dt: 449.94ms, tok/sec: 1165231.25
step: 18621 | loss: 3.074281 | lr: 6.0807e-05 | norm: 0.3618, dt: 449.96ms, tok/sec: 1165193.59
step: 18622 | loss: 3.066211 | lr: 6.0804e-05 | norm: 0.3364, dt: 450.57ms, tok/sec: 1163610.26
step: 18623 | loss: 3.091131 | lr: 6.0800e-05 | norm: 0.3513, dt: 1103.88ms, tok/sec: 474950.82
step: 18624 | loss: 3.037567 | lr: 6.0797e-05 | norm: 0.3662, dt: 450.78ms, tok/sec: 1163072.37
step: 18625 | loss: 3.087111 | lr: 6.0793e-05 | norm: 0.3190, dt: 451.97ms, tok/sec: 1160018.18
step: 18626 | loss: 3.099813 | lr: 6.0790e-05 | norm: 0.3627, dt: 450.55ms, tok/sec: 1163671.84
step: 18627 | loss: 3.033239 | lr: 6.0786e-05 | norm: 0.3991, dt: 451.48ms, tok/sec: 1161270.92
step: 18628 | loss: 3.027507 | lr: 6.0783e-05 | norm: 0.3131, dt: 450.15ms, tok/sec: 1164695.56
step: 18629 | loss: 3.044963 | lr: 6.0779e-05 | norm: 0.3553, dt: 451.10ms, tok/sec: 1162255.40
step: 18630 | loss: 3.050851 | lr: 6.0776e-05 | norm: 0.3410, dt: 450.32ms, tok/sec: 1164261.45
step: 18631 | loss: 3.100008 | lr: 6.0772e-05 | norm: 0.3920, dt: 450.59ms, tok/sec: 1163562.85
step: 18632 | loss: 3.107643 | lr: 6.0769e-05 | norm: 0.3855, dt: 451.07ms, tok/sec: 1162329.12
step: 18633 | loss: 3.091147 | lr: 6.0765e-05 | norm: 0.3621, dt: 451.65ms, tok/sec: 1160819.13
step: 18634 | loss: 3.052075 | lr: 6.0762e-05 | norm: 0.3373, dt: 449.92ms, tok/sec: 1165281.89
step: 18635 | loss: 3.113831 | lr: 6.0758e-05 | norm: 0.3623, dt: 451.37ms, tok/sec: 1161552.47
step: 18636 | loss: 3.137812 | lr: 6.0755e-05 | norm: 0.4134, dt: 450.21ms, tok/sec: 1164551.85
step: 18637 | loss: 3.091074 | lr: 6.0751e-05 | norm: 0.3830, dt: 451.40ms, tok/sec: 1161472.71
step: 18638 | loss: 3.111296 | lr: 6.0748e-05 | norm: 0.3341, dt: 450.96ms, tok/sec: 1162595.21
step: 18639 | loss: 3.135075 | lr: 6.0744e-05 | norm: 0.3303, dt: 452.25ms, tok/sec: 1159289.83
step: 18640 | loss: 3.023725 | lr: 6.0741e-05 | norm: 0.4169, dt: 450.17ms, tok/sec: 1164648.06
step: 18641 | loss: 3.006008 | lr: 6.0737e-05 | norm: 0.3356, dt: 451.59ms, tok/sec: 1160994.41
step: 18642 | loss: 3.011660 | lr: 6.0734e-05 | norm: 0.3736, dt: 451.67ms, tok/sec: 1160789.10
step: 18643 | loss: 2.984840 | lr: 6.0731e-05 | norm: 0.3353, dt: 451.32ms, tok/sec: 1161669.05
step: 18644 | loss: 3.006322 | lr: 6.0727e-05 | norm: 0.3352, dt: 451.33ms, tok/sec: 1161645.12
step: 18645 | loss: 3.043590 | lr: 6.0724e-05 | norm: 0.3196, dt: 451.53ms, tok/sec: 1161148.89
step: 18646 | loss: 3.017972 | lr: 6.0721e-05 | norm: 0.3600, dt: 451.68ms, tok/sec: 1160738.86
step: 18647 | loss: 2.991690 | lr: 6.0717e-05 | norm: 0.3043, dt: 450.79ms, tok/sec: 1163032.39
step: 18648 | loss: 2.984287 | lr: 6.0714e-05 | norm: 0.3145, dt: 450.70ms, tok/sec: 1163264.33
step: 18649 | loss: 2.974400 | lr: 6.0710e-05 | norm: 0.3365, dt: 450.55ms, tok/sec: 1163653.98
step: 18650 | loss: 3.026615 | lr: 6.0707e-05 | norm: 0.2977, dt: 452.54ms, tok/sec: 1158534.93
step: 18651 | loss: 3.044132 | lr: 6.0704e-05 | norm: 0.2856, dt: 451.32ms, tok/sec: 1161680.10
step: 18652 | loss: 3.111169 | lr: 6.0700e-05 | norm: 0.3443, dt: 450.82ms, tok/sec: 1162972.72
step: 18653 | loss: 3.119134 | lr: 6.0697e-05 | norm: 0.3439, dt: 450.66ms, tok/sec: 1163367.72
step: 18654 | loss: 3.062728 | lr: 6.0694e-05 | norm: 0.3037, dt: 452.53ms, tok/sec: 1158571.55
step: 18655 | loss: 3.062599 | lr: 6.0690e-05 | norm: 0.3999, dt: 450.28ms, tok/sec: 1164354.53
step: 18656 | loss: 3.080448 | lr: 6.0687e-05 | norm: 0.3271, dt: 454.33ms, tok/sec: 1153969.77
step: 18657 | loss: 3.131192 | lr: 6.0684e-05 | norm: 0.3616, dt: 451.34ms, tok/sec: 1161616.28
step: 18658 | loss: 3.062025 | lr: 6.0681e-05 | norm: 0.3043, dt: 450.25ms, tok/sec: 1164439.00
step: 18659 | loss: 3.047493 | lr: 6.0677e-05 | norm: 0.3505, dt: 452.89ms, tok/sec: 1157659.72
step: 18660 | loss: 3.063302 | lr: 6.0674e-05 | norm: 0.3297, dt: 450.95ms, tok/sec: 1162619.18
step: 18661 | loss: 3.009746 | lr: 6.0671e-05 | norm: 0.3580, dt: 450.26ms, tok/sec: 1164409.40
step: 18662 | loss: 3.126195 | lr: 6.0668e-05 | norm: 0.3346, dt: 450.21ms, tok/sec: 1164548.15
step: 18663 | loss: 3.051854 | lr: 6.0664e-05 | norm: 0.3191, dt: 450.18ms, tok/sec: 1164630.18
step: 18664 | loss: 3.017843 | lr: 6.0661e-05 | norm: 0.3266, dt: 451.87ms, tok/sec: 1160262.39
step: 18665 | loss: 2.991096 | lr: 6.0658e-05 | norm: 0.3311, dt: 450.83ms, tok/sec: 1162949.35
step: 18666 | loss: 3.039803 | lr: 6.0655e-05 | norm: 0.2800, dt: 450.96ms, tok/sec: 1162592.13
step: 18667 | loss: 2.995988 | lr: 6.0651e-05 | norm: 0.3476, dt: 450.33ms, tok/sec: 1164218.92
step: 18668 | loss: 2.995864 | lr: 6.0648e-05 | norm: 0.3086, dt: 450.87ms, tok/sec: 1162847.88
step: 18669 | loss: 3.047391 | lr: 6.0645e-05 | norm: 0.2856, dt: 450.99ms, tok/sec: 1162518.38
step: 18670 | loss: 3.107282 | lr: 6.0642e-05 | norm: 0.3705, dt: 451.14ms, tok/sec: 1162128.87
step: 18671 | loss: 3.103757 | lr: 6.0639e-05 | norm: 0.3160, dt: 450.38ms, tok/sec: 1164093.81
step: 18672 | loss: 3.078314 | lr: 6.0635e-05 | norm: 0.2945, dt: 451.21ms, tok/sec: 1161955.09
step: 18673 | loss: 3.065346 | lr: 6.0632e-05 | norm: 0.3116, dt: 450.73ms, tok/sec: 1163186.18
step: 18674 | loss: 3.062405 | lr: 6.0629e-05 | norm: 0.2887, dt: 450.48ms, tok/sec: 1163836.28
step: 18675 | loss: 3.124807 | lr: 6.0626e-05 | norm: 0.2803, dt: 451.44ms, tok/sec: 1161375.79
step: 18676 | loss: 3.019616 | lr: 6.0623e-05 | norm: 0.3106, dt: 451.16ms, tok/sec: 1162094.48
step: 18677 | loss: 2.978038 | lr: 6.0620e-05 | norm: 0.2864, dt: 451.29ms, tok/sec: 1161766.02
step: 18678 | loss: 3.029327 | lr: 6.0617e-05 | norm: 0.3146, dt: 451.28ms, tok/sec: 1161769.09
step: 18679 | loss: 3.000119 | lr: 6.0613e-05 | norm: 0.2789, dt: 450.70ms, tok/sec: 1163281.56
step: 18680 | loss: 3.006747 | lr: 6.0610e-05 | norm: 0.2773, dt: 450.66ms, tok/sec: 1163370.80
step: 18681 | loss: 2.996514 | lr: 6.0607e-05 | norm: 0.2708, dt: 451.03ms, tok/sec: 1162431.73
step: 18682 | loss: 2.987745 | lr: 6.0604e-05 | norm: 0.3147, dt: 451.14ms, tok/sec: 1162138.70
step: 18683 | loss: 3.021384 | lr: 6.0601e-05 | norm: 0.3108, dt: 451.28ms, tok/sec: 1161786.28
step: 18684 | loss: 3.049777 | lr: 6.0598e-05 | norm: 0.3132, dt: 450.67ms, tok/sec: 1163342.48
step: 18685 | loss: 3.006415 | lr: 6.0595e-05 | norm: 0.3724, dt: 450.48ms, tok/sec: 1163837.51
step: 18686 | loss: 3.015622 | lr: 6.0592e-05 | norm: 0.2681, dt: 451.44ms, tok/sec: 1161361.69
step: 18687 | loss: 2.999192 | lr: 6.0589e-05 | norm: 0.3374, dt: 450.96ms, tok/sec: 1162607.50
step: 18688 | loss: 3.118999 | lr: 6.0586e-05 | norm: 0.3288, dt: 451.05ms, tok/sec: 1162359.84
step: 18689 | loss: 3.133530 | lr: 6.0583e-05 | norm: 0.3039, dt: 452.24ms, tok/sec: 1159325.89
step: 18690 | loss: 3.060706 | lr: 6.0580e-05 | norm: 0.3182, dt: 451.63ms, tok/sec: 1160882.25
step: 18691 | loss: 3.146270 | lr: 6.0577e-05 | norm: 0.2956, dt: 450.74ms, tok/sec: 1163167.73
step: 18692 | loss: 3.045248 | lr: 6.0574e-05 | norm: 0.3460, dt: 450.64ms, tok/sec: 1163426.81
step: 18693 | loss: 3.090976 | lr: 6.0571e-05 | norm: 0.3020, dt: 451.68ms, tok/sec: 1160754.18
step: 18694 | loss: 3.083179 | lr: 6.0568e-05 | norm: 0.3575, dt: 450.68ms, tok/sec: 1163330.18
step: 18695 | loss: 3.068436 | lr: 6.0565e-05 | norm: 0.3514, dt: 451.00ms, tok/sec: 1162499.94
step: 18696 | loss: 3.018806 | lr: 6.0562e-05 | norm: 0.3157, dt: 451.76ms, tok/sec: 1160548.35
step: 18697 | loss: 3.122926 | lr: 6.0559e-05 | norm: 0.2782, dt: 451.16ms, tok/sec: 1162085.89
step: 18698 | loss: 3.090331 | lr: 6.0556e-05 | norm: 0.3824, dt: 450.37ms, tok/sec: 1164117.84
step: 18699 | loss: 3.063303 | lr: 6.0553e-05 | norm: 0.2939, dt: 451.26ms, tok/sec: 1161819.42
step: 18700 | loss: 3.010435 | lr: 6.0550e-05 | norm: 0.3703, dt: 451.38ms, tok/sec: 1161510.14
step: 18701 | loss: 3.044000 | lr: 6.0547e-05 | norm: 0.2863, dt: 450.93ms, tok/sec: 1162692.33
step: 18702 | loss: 3.032719 | lr: 6.0544e-05 | norm: 0.3654, dt: 451.04ms, tok/sec: 1162396.09
step: 18703 | loss: 3.030255 | lr: 6.0541e-05 | norm: 0.2994, dt: 451.10ms, tok/sec: 1162244.96
step: 18704 | loss: 3.013139 | lr: 6.0538e-05 | norm: 0.3140, dt: 450.76ms, tok/sec: 1163122.81
step: 18705 | loss: 3.041051 | lr: 6.0535e-05 | norm: 0.3497, dt: 450.74ms, tok/sec: 1163174.49
step: 18706 | loss: 3.056644 | lr: 6.0532e-05 | norm: 0.2920, dt: 453.24ms, tok/sec: 1156763.31
step: 18707 | loss: 3.009077 | lr: 6.0529e-05 | norm: 0.3692, dt: 450.05ms, tok/sec: 1164967.67
step: 18708 | loss: 3.056633 | lr: 6.0527e-05 | norm: 0.2781, dt: 450.28ms, tok/sec: 1164347.75
step: 18709 | loss: 3.022861 | lr: 6.0524e-05 | norm: 0.3325, dt: 451.76ms, tok/sec: 1160536.10
step: 18710 | loss: 3.043320 | lr: 6.0521e-05 | norm: 0.3599, dt: 450.44ms, tok/sec: 1163942.85
step: 18711 | loss: 3.046095 | lr: 6.0518e-05 | norm: 0.2971, dt: 450.16ms, tok/sec: 1164673.97
step: 18712 | loss: 2.998329 | lr: 6.0515e-05 | norm: 0.4367, dt: 451.17ms, tok/sec: 1162056.41
step: 18713 | loss: 2.954484 | lr: 6.0512e-05 | norm: 0.2844, dt: 451.39ms, tok/sec: 1161490.50
step: 18714 | loss: 3.033699 | lr: 6.0509e-05 | norm: 0.4192, dt: 450.53ms, tok/sec: 1163723.57
step: 18715 | loss: 2.972607 | lr: 6.0507e-05 | norm: 0.3168, dt: 450.39ms, tok/sec: 1164086.41
step: 18716 | loss: 3.016174 | lr: 6.0504e-05 | norm: 0.3510, dt: 451.26ms, tok/sec: 1161826.17
step: 18717 | loss: 3.055792 | lr: 6.0501e-05 | norm: 0.4066, dt: 451.34ms, tok/sec: 1161627.94
step: 18718 | loss: 3.020828 | lr: 6.0498e-05 | norm: 0.3180, dt: 450.65ms, tok/sec: 1163416.34
step: 18719 | loss: 3.046154 | lr: 6.0495e-05 | norm: 0.4407, dt: 451.33ms, tok/sec: 1161656.78
step: 18720 | loss: 3.013709 | lr: 6.0492e-05 | norm: 0.3421, dt: 450.69ms, tok/sec: 1163307.41
step: 18721 | loss: 2.978628 | lr: 6.0490e-05 | norm: 0.3556, dt: 1170.63ms, tok/sec: 447868.80
step: 18722 | loss: 3.063422 | lr: 6.0487e-05 | norm: 0.3308, dt: 449.78ms, tok/sec: 1165644.47
step: 18723 | loss: 3.163659 | lr: 6.0484e-05 | norm: 0.2907, dt: 449.93ms, tok/sec: 1165260.27
step: 18724 | loss: 3.135536 | lr: 6.0481e-05 | norm: 0.3295, dt: 449.70ms, tok/sec: 1165871.89
step: 18725 | loss: 3.200413 | lr: 6.0479e-05 | norm: 0.3095, dt: 450.78ms, tok/sec: 1163074.22
step: 18726 | loss: 3.165743 | lr: 6.0476e-05 | norm: 0.3301, dt: 450.81ms, tok/sec: 1162997.33
step: 18727 | loss: 3.132436 | lr: 6.0473e-05 | norm: 0.3388, dt: 449.49ms, tok/sec: 1166394.43
step: 18728 | loss: 3.225342 | lr: 6.0470e-05 | norm: 0.3716, dt: 450.51ms, tok/sec: 1163758.05
step: 18729 | loss: 3.112680 | lr: 6.0468e-05 | norm: 0.3368, dt: 451.35ms, tok/sec: 1161607.69
step: 18730 | loss: 3.149829 | lr: 6.0465e-05 | norm: 0.4231, dt: 450.63ms, tok/sec: 1163461.89
step: 18731 | loss: 3.112838 | lr: 6.0462e-05 | norm: 0.3408, dt: 450.53ms, tok/sec: 1163726.65
step: 18732 | loss: 3.078647 | lr: 6.0460e-05 | norm: 0.4506, dt: 451.30ms, tok/sec: 1161719.38
step: 18733 | loss: 3.057303 | lr: 6.0457e-05 | norm: 0.3090, dt: 450.50ms, tok/sec: 1163794.39
step: 18734 | loss: 3.060775 | lr: 6.0454e-05 | norm: 0.3750, dt: 450.94ms, tok/sec: 1162665.90
step: 18735 | loss: 3.003097 | lr: 6.0452e-05 | norm: 0.3780, dt: 450.75ms, tok/sec: 1163141.27
step: 18736 | loss: 3.079382 | lr: 6.0449e-05 | norm: 0.3446, dt: 451.56ms, tok/sec: 1161064.90
step: 18737 | loss: 3.088010 | lr: 6.0446e-05 | norm: 0.4938, dt: 451.02ms, tok/sec: 1162450.17
step: 18738 | loss: 3.017348 | lr: 6.0444e-05 | norm: 0.3274, dt: 450.68ms, tok/sec: 1163335.72
step: 18739 | loss: 3.091863 | lr: 6.0441e-05 | norm: 0.4254, dt: 450.75ms, tok/sec: 1163139.43
step: 18740 | loss: 3.036429 | lr: 6.0438e-05 | norm: 0.3501, dt: 450.93ms, tok/sec: 1162688.64
step: 18741 | loss: 2.983121 | lr: 6.0436e-05 | norm: 0.4404, dt: 450.65ms, tok/sec: 1163408.96
step: 18742 | loss: 3.015541 | lr: 6.0433e-05 | norm: 0.3386, dt: 449.92ms, tok/sec: 1165301.03
step: 18743 | loss: 3.052959 | lr: 6.0430e-05 | norm: 0.2960, dt: 450.53ms, tok/sec: 1163711.87
step: 18744 | loss: 3.174964 | lr: 6.0428e-05 | norm: 0.3416, dt: 450.00ms, tok/sec: 1165092.96
step: 18745 | loss: 2.989318 | lr: 6.0425e-05 | norm: 0.3359, dt: 450.92ms, tok/sec: 1162702.17
step: 18746 | loss: 3.101370 | lr: 6.0423e-05 | norm: 0.3602, dt: 451.88ms, tok/sec: 1160228.10
step: 18747 | loss: 3.063802 | lr: 6.0420e-05 | norm: 0.2919, dt: 450.35ms, tok/sec: 1164167.76
step: 18748 | loss: 3.014396 | lr: 6.0417e-05 | norm: 0.3342, dt: 450.04ms, tok/sec: 1164971.99
step: 18749 | loss: 3.014517 | lr: 6.0415e-05 | norm: 0.2860, dt: 450.96ms, tok/sec: 1162591.52
validation loss: 3.0836
HellaSwag accuracy: 2992/10042=0.2979
rank 5 sample 0: Hello, I'm a Manpreet, a very smart person and I'm very friendly with everybody, I am also a great teacher.
One thing I
rank 5 sample 1: Hello, I'm a Manpreet, have a look at another word and try to look at another word. I'll have another word, this one.
rank 5 sample 2: Hello, I'm a Manpreet, an English instructor. Your job is to teach the English language to kids in English class that is spoken by the students
rank 5 sample 3: Hello, I'm a Manpreet, don't be surprised if they don't say that they do. I'm A Liktuck. I am
rank 4 sample 0: Hello, I'm a Manpreet, I'm my Friend, I'm a Good Manpreet, I'm a Manpreet, I'm a
rank 0 sample 0: Hello, I'm a Manpreet, and I love to play a song like all good things, and a woman like me! There are many reasons why
rank 0 sample 1: Hello, I'm a Manpreet, thank you for your interest in the future of your post. I will let you know if the next posting of the
rank 4 sample 1: Hello, I'm a Manpreet, sorry!
Do you think he doesn't deserve that accolade? Let him come along.
If he wins
rank 7 sample 0: Hello, I'm a Manpreet, and I think I have good reason to trust him, but I'm confused. To him, your son's namerank 2 sample 0: Hello, I'm a Manpreet, as if you didn't know how to use it. So you're thinking about something that's been done on your
rank 6 sample 0: Hello, I'm a Manpreet, why can't I see his words! I'm trying to make sense out of that. When I'm trying to
rank 0 sample 2: Hello, I'm a Manpreet, I'm doing the whole day's chores. If I don't, I'll start running out of time.


rank 4 sample 2: Hello, I'm a Manpreet, so have a read out in the browser to the right of that particular page. As far as I can tell.
rank 2 sample 1: Hello, I'm a Manpreet, the man i know of. I guess as a guy, you could easily say "I'm the boss of therank 0 sample 3: Hello, I'm a Manpreet,
but I'm a Manpreet, of course.
If you have asked me what you thought about that

rank 7 sample 1: Hello, I'm a Manpreet, thanks for all the fun! Thanks."
"There are many words that are too old-fashioned to be used
rank 6 sample 1: Hello, I'm a Manpreet,
Now, I know you can use either the 'S' or the 'W' in your command
[
rank 4 sample 3: Hello, I'm a Manpreet, and this new document is in the course "Hello People" to get some practice.
After getting a few pages
rank 2 sample 2: Hello, I'm a Manpreet, I'm the Master.
Now in reality, the game is still an old one, and as an alternative,rank 7 sample 2: Hello, I'm a Manpreet, who is the manager of the Internet community in the local area of I.T.
I'm a man with

rank 6 sample 2: Hello, I'm a Manpreet, a woman who is working with me. I love the way you can be there when you need to give an update
rank 7 sample 3: Hello, I'm a Manpreet, for a short time now I've managed to complete my homework, and if I have a question at the moment,rank 2 sample 3: Hello, I'm a Manpreet, a friend, and you are a new baby. Let's look at the question in detail again at this point,

rank 6 sample 3: Hello, I'm a Manpreet, and it's on my computer at home. It's a little complicated, because it is being used by all different
rank 1 sample 0: Hello, I'm a Manpreet, please give me a try!
Hey guys, what will I do with this? Thank you in advance, I
rank 1 sample 1: Hello, I'm a Manpreet, but you don't have to have much respect for it if you'd like to understand him. Anyway, I'm
rank 1 sample 2: Hello, I'm a Manpreet,
When we go to the
menu, we're told you that
#include <input file/file>.
rank 1 sample 3: Hello, I'm a Manpreet, and I'm doing my PhD at the Hospital for Sick Children, located in Manchester, West Riding, England. In
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going into some world wide web, where I'll be called Geevah. Well, the
rank 3 sample 1: Hello, I'm a Manpreet, and my first job is to do my homework. I like to work with the kids, and I'll explain things
rank 3 sample 2: Hello, I'm a Manpreet, and you are like what you call a Human on the inside.
The Human looks like a young child. That
rank 3 sample 3: Hello, I'm a Manpreet, and have a good time! Thank you!
This video was taken on the third Tuesday in the 2nd grade
step: 18750 | loss: 3.060001 | lr: 6.0412e-05 | norm: 0.3074, dt: 12090.26ms, tok/sec: 43364.48
step: 18751 | loss: 3.052042 | lr: 6.0410e-05 | norm: 0.3039, dt: 449.12ms, tok/sec: 1167363.46
step: 18752 | loss: 3.011283 | lr: 6.0407e-05 | norm: 0.2821, dt: 449.92ms, tok/sec: 1165289.30
step: 18753 | loss: 3.020567 | lr: 6.0405e-05 | norm: 0.3307, dt: 451.39ms, tok/sec: 1161498.48
step: 18754 | loss: 3.042222 | lr: 6.0402e-05 | norm: 0.3438, dt: 450.36ms, tok/sec: 1164149.89
step: 18755 | loss: 3.113118 | lr: 6.0400e-05 | norm: 0.3207, dt: 448.66ms, tok/sec: 1168568.16
step: 18756 | loss: 3.090132 | lr: 6.0397e-05 | norm: 0.3216, dt: 450.39ms, tok/sec: 1164063.61
step: 18757 | loss: 3.010093 | lr: 6.0395e-05 | norm: 0.3218, dt: 451.86ms, tok/sec: 1160291.77
step: 18758 | loss: 3.035459 | lr: 6.0392e-05 | norm: 0.3126, dt: 450.62ms, tok/sec: 1163469.89
step: 18759 | loss: 2.983602 | lr: 6.0390e-05 | norm: 0.2958, dt: 449.09ms, tok/sec: 1167439.69
step: 18760 | loss: 2.973340 | lr: 6.0387e-05 | norm: 0.3214, dt: 450.57ms, tok/sec: 1163618.88
step: 18761 | loss: 3.008443 | lr: 6.0385e-05 | norm: 0.2975, dt: 451.14ms, tok/sec: 1162142.39
step: 18762 | loss: 3.033491 | lr: 6.0382e-05 | norm: 0.3107, dt: 450.07ms, tok/sec: 1164913.98
step: 18763 | loss: 3.068602 | lr: 6.0380e-05 | norm: 0.3339, dt: 450.63ms, tok/sec: 1163450.81
step: 18764 | loss: 3.059618 | lr: 6.0377e-05 | norm: 0.2917, dt: 450.68ms, tok/sec: 1163331.41
step: 18765 | loss: 3.056843 | lr: 6.0375e-05 | norm: 0.3312, dt: 451.20ms, tok/sec: 1161977.20
step: 18766 | loss: 2.997670 | lr: 6.0373e-05 | norm: 0.3099, dt: 450.07ms, tok/sec: 1164895.46
step: 18767 | loss: 3.026827 | lr: 6.0370e-05 | norm: 0.3406, dt: 451.10ms, tok/sec: 1162241.28
step: 18768 | loss: 3.004039 | lr: 6.0368e-05 | norm: 0.2612, dt: 451.81ms, tok/sec: 1160413.00
step: 18769 | loss: 3.094253 | lr: 6.0365e-05 | norm: 0.3677, dt: 450.37ms, tok/sec: 1164117.84
step: 18770 | loss: 3.164238 | lr: 6.0363e-05 | norm: 0.3755, dt: 450.43ms, tok/sec: 1163984.74
step: 18771 | loss: 3.112184 | lr: 6.0360e-05 | norm: 0.4392, dt: 450.26ms, tok/sec: 1164421.12
step: 18772 | loss: 3.093086 | lr: 6.0358e-05 | norm: 0.2979, dt: 450.71ms, tok/sec: 1163256.33
step: 18773 | loss: 3.109857 | lr: 6.0356e-05 | norm: 0.4351, dt: 451.32ms, tok/sec: 1161673.96
step: 18774 | loss: 3.101559 | lr: 6.0353e-05 | norm: 0.3089, dt: 451.77ms, tok/sec: 1160507.31
step: 18775 | loss: 3.105397 | lr: 6.0351e-05 | norm: 0.4044, dt: 451.34ms, tok/sec: 1161623.03
step: 18776 | loss: 3.052197 | lr: 6.0349e-05 | norm: 0.2759, dt: 451.74ms, tok/sec: 1160599.80
step: 18777 | loss: 3.095268 | lr: 6.0346e-05 | norm: 0.3812, dt: 450.37ms, tok/sec: 1164124.62
step: 18778 | loss: 3.096198 | lr: 6.0344e-05 | norm: 0.3310, dt: 451.02ms, tok/sec: 1162452.63
step: 18779 | loss: 3.083470 | lr: 6.0342e-05 | norm: 0.2866, dt: 450.76ms, tok/sec: 1163108.05
step: 18780 | loss: 3.052821 | lr: 6.0339e-05 | norm: 0.3709, dt: 450.62ms, tok/sec: 1163480.98
step: 18781 | loss: 3.038929 | lr: 6.0337e-05 | norm: 0.3348, dt: 451.09ms, tok/sec: 1162276.29
step: 18782 | loss: 3.061217 | lr: 6.0335e-05 | norm: 0.3052, dt: 450.66ms, tok/sec: 1163374.49
step: 18783 | loss: 2.982762 | lr: 6.0332e-05 | norm: 0.3348, dt: 450.87ms, tok/sec: 1162836.81
step: 18784 | loss: 3.053946 | lr: 6.0330e-05 | norm: 0.2723, dt: 451.09ms, tok/sec: 1162279.98
step: 18785 | loss: 3.084508 | lr: 6.0328e-05 | norm: 0.2668, dt: 450.75ms, tok/sec: 1163157.27
step: 18786 | loss: 3.054003 | lr: 6.0326e-05 | norm: 0.2733, dt: 451.02ms, tok/sec: 1162444.02
step: 18787 | loss: 3.121636 | lr: 6.0323e-05 | norm: 0.3148, dt: 451.82ms, tok/sec: 1160395.25
step: 18788 | loss: 3.049129 | lr: 6.0321e-05 | norm: 0.2702, dt: 450.46ms, tok/sec: 1163905.88
step: 18789 | loss: 3.142148 | lr: 6.0319e-05 | norm: 0.3151, dt: 449.89ms, tok/sec: 1165373.28
step: 18790 | loss: 3.102304 | lr: 6.0317e-05 | norm: 0.2985, dt: 453.10ms, tok/sec: 1157119.40
step: 18791 | loss: 3.061256 | lr: 6.0314e-05 | norm: 0.3352, dt: 450.58ms, tok/sec: 1163594.25
step: 18792 | loss: 3.070900 | lr: 6.0312e-05 | norm: 0.2838, dt: 450.24ms, tok/sec: 1164467.98
step: 18793 | loss: 3.075454 | lr: 6.0310e-05 | norm: 0.3245, dt: 450.37ms, tok/sec: 1164125.23
step: 18794 | loss: 3.028214 | lr: 6.0308e-05 | norm: 0.2775, dt: 450.88ms, tok/sec: 1162799.31
step: 18795 | loss: 3.083834 | lr: 6.0305e-05 | norm: 0.3063, dt: 450.25ms, tok/sec: 1164427.28
step: 18796 | loss: 3.078843 | lr: 6.0303e-05 | norm: 0.2815, dt: 452.46ms, tok/sec: 1158759.58
step: 18797 | loss: 3.068261 | lr: 6.0301e-05 | norm: 0.3110, dt: 451.11ms, tok/sec: 1162215.48
step: 18798 | loss: 3.049256 | lr: 6.0299e-05 | norm: 0.2684, dt: 450.98ms, tok/sec: 1162563.25
step: 18799 | loss: 3.013788 | lr: 6.0297e-05 | norm: 0.3350, dt: 450.17ms, tok/sec: 1164648.68
step: 18800 | loss: 3.039582 | lr: 6.0295e-05 | norm: 0.2928, dt: 450.39ms, tok/sec: 1164069.77
step: 18801 | loss: 3.057548 | lr: 6.0292e-05 | norm: 0.3183, dt: 451.18ms, tok/sec: 1162037.99
step: 18802 | loss: 3.053372 | lr: 6.0290e-05 | norm: 0.2736, dt: 450.52ms, tok/sec: 1163734.65
step: 18803 | loss: 2.955480 | lr: 6.0288e-05 | norm: 0.2682, dt: 449.16ms, tok/sec: 1167257.50
step: 18804 | loss: 3.053550 | lr: 6.0286e-05 | norm: 0.3507, dt: 450.81ms, tok/sec: 1162979.49
step: 18805 | loss: 2.975304 | lr: 6.0284e-05 | norm: 0.3176, dt: 450.21ms, tok/sec: 1164548.15
step: 18806 | loss: 2.998168 | lr: 6.0282e-05 | norm: 0.2646, dt: 454.23ms, tok/sec: 1154239.91
step: 18807 | loss: 2.938457 | lr: 6.0280e-05 | norm: 0.2748, dt: 450.92ms, tok/sec: 1162712.00
step: 18808 | loss: 2.963205 | lr: 6.0278e-05 | norm: 0.2642, dt: 450.66ms, tok/sec: 1163372.03
step: 18809 | loss: 3.015871 | lr: 6.0275e-05 | norm: 0.3082, dt: 451.01ms, tok/sec: 1162465.53
step: 18810 | loss: 3.007990 | lr: 6.0273e-05 | norm: 0.2685, dt: 450.47ms, tok/sec: 1163867.08
step: 18811 | loss: 3.026833 | lr: 6.0271e-05 | norm: 0.3074, dt: 449.89ms, tok/sec: 1165375.13
step: 18812 | loss: 3.016656 | lr: 6.0269e-05 | norm: 0.2969, dt: 1154.61ms, tok/sec: 454083.02
step: 18813 | loss: 3.026109 | lr: 6.0267e-05 | norm: 0.3083, dt: 450.00ms, tok/sec: 1165077.53
step: 18814 | loss: 3.027291 | lr: 6.0265e-05 | norm: 0.3295, dt: 453.34ms, tok/sec: 1156497.46
step: 18815 | loss: 3.042246 | lr: 6.0263e-05 | norm: 0.3195, dt: 450.48ms, tok/sec: 1163855.37
step: 18816 | loss: 3.099327 | lr: 6.0261e-05 | norm: 0.2947, dt: 450.41ms, tok/sec: 1164035.27
step: 18817 | loss: 3.161589 | lr: 6.0259e-05 | norm: 0.3148, dt: 450.83ms, tok/sec: 1162927.83
step: 18818 | loss: 3.120696 | lr: 6.0257e-05 | norm: 0.3292, dt: 451.10ms, tok/sec: 1162248.65
step: 18819 | loss: 3.088980 | lr: 6.0255e-05 | norm: 0.3075, dt: 450.21ms, tok/sec: 1164533.96
step: 18820 | loss: 3.048840 | lr: 6.0253e-05 | norm: 0.4366, dt: 450.47ms, tok/sec: 1163879.40
step: 18821 | loss: 3.107270 | lr: 6.0251e-05 | norm: 0.3885, dt: 453.08ms, tok/sec: 1157170.54
step: 18822 | loss: 3.055297 | lr: 6.0249e-05 | norm: 0.3897, dt: 450.20ms, tok/sec: 1164567.27
step: 18823 | loss: 3.150364 | lr: 6.0247e-05 | norm: 0.4635, dt: 450.56ms, tok/sec: 1163637.97
step: 18824 | loss: 3.098359 | lr: 6.0245e-05 | norm: 0.2986, dt: 450.74ms, tok/sec: 1163170.80
step: 18825 | loss: 3.105329 | lr: 6.0243e-05 | norm: 0.4321, dt: 450.46ms, tok/sec: 1163891.10
step: 18826 | loss: 3.102040 | lr: 6.0241e-05 | norm: 0.3321, dt: 450.76ms, tok/sec: 1163118.51
step: 18827 | loss: 3.033345 | lr: 6.0239e-05 | norm: 0.3449, dt: 451.18ms, tok/sec: 1162031.23
step: 18828 | loss: 3.093299 | lr: 6.0237e-05 | norm: 0.4274, dt: 451.34ms, tok/sec: 1161626.10
step: 18829 | loss: 3.094160 | lr: 6.0235e-05 | norm: 0.2945, dt: 451.08ms, tok/sec: 1162299.02
step: 18830 | loss: 3.083724 | lr: 6.0233e-05 | norm: 0.4095, dt: 450.24ms, tok/sec: 1164466.13
step: 18831 | loss: 3.053865 | lr: 6.0232e-05 | norm: 0.2976, dt: 451.10ms, tok/sec: 1162252.95
step: 18832 | loss: 3.080329 | lr: 6.0230e-05 | norm: 0.3416, dt: 451.37ms, tok/sec: 1161553.08
step: 18833 | loss: 3.075588 | lr: 6.0228e-05 | norm: 0.3449, dt: 450.76ms, tok/sec: 1163126.51
step: 18834 | loss: 3.042417 | lr: 6.0226e-05 | norm: 0.3014, dt: 450.98ms, tok/sec: 1162544.81
step: 18835 | loss: 3.013444 | lr: 6.0224e-05 | norm: 0.3352, dt: 451.40ms, tok/sec: 1161476.39
step: 18836 | loss: 3.117190 | lr: 6.0222e-05 | norm: 0.3576, dt: 450.97ms, tok/sec: 1162579.84
step: 18837 | loss: 3.029621 | lr: 6.0220e-05 | norm: 0.3768, dt: 450.51ms, tok/sec: 1163772.22
step: 18838 | loss: 3.076586 | lr: 6.0218e-05 | norm: 0.3081, dt: 451.75ms, tok/sec: 1160583.26
step: 18839 | loss: 3.075992 | lr: 6.0216e-05 | norm: 0.3314, dt: 450.90ms, tok/sec: 1162757.50
step: 18840 | loss: 3.076638 | lr: 6.0215e-05 | norm: 0.2998, dt: 450.90ms, tok/sec: 1162758.11
step: 18841 | loss: 3.060048 | lr: 6.0213e-05 | norm: 0.3943, dt: 450.92ms, tok/sec: 1162715.69
step: 18842 | loss: 3.035320 | lr: 6.0211e-05 | norm: 0.3327, dt: 450.47ms, tok/sec: 1163867.08
step: 18843 | loss: 3.061306 | lr: 6.0209e-05 | norm: 0.3378, dt: 451.04ms, tok/sec: 1162407.15
step: 18844 | loss: 3.053895 | lr: 6.0207e-05 | norm: 0.3202, dt: 450.42ms, tok/sec: 1163999.53
step: 18845 | loss: 3.062937 | lr: 6.0205e-05 | norm: 0.3447, dt: 451.66ms, tok/sec: 1160800.13
step: 18846 | loss: 3.068123 | lr: 6.0204e-05 | norm: 0.2977, dt: 449.75ms, tok/sec: 1165729.12
step: 18847 | loss: 3.061720 | lr: 6.0202e-05 | norm: 0.3042, dt: 450.42ms, tok/sec: 1164004.46
step: 18848 | loss: 3.044986 | lr: 6.0200e-05 | norm: 0.3169, dt: 450.25ms, tok/sec: 1164449.48
step: 18849 | loss: 3.019755 | lr: 6.0198e-05 | norm: 0.3131, dt: 451.88ms, tok/sec: 1160245.86
step: 18850 | loss: 3.006303 | lr: 6.0197e-05 | norm: 0.3448, dt: 450.77ms, tok/sec: 1163103.13
step: 18851 | loss: 2.964320 | lr: 6.0195e-05 | norm: 0.2841, dt: 451.31ms, tok/sec: 1161692.37
step: 18852 | loss: 2.998010 | lr: 6.0193e-05 | norm: 0.3449, dt: 450.58ms, tok/sec: 1163575.17
step: 18853 | loss: 2.928937 | lr: 6.0191e-05 | norm: 0.2768, dt: 451.25ms, tok/sec: 1161847.66
step: 18854 | loss: 3.008409 | lr: 6.0190e-05 | norm: 0.3078, dt: 451.01ms, tok/sec: 1162480.28
step: 18855 | loss: 3.001087 | lr: 6.0188e-05 | norm: 0.2950, dt: 451.39ms, tok/sec: 1161486.82
step: 18856 | loss: 3.065073 | lr: 6.0186e-05 | norm: 0.2961, dt: 450.21ms, tok/sec: 1164532.73
step: 18857 | loss: 3.092286 | lr: 6.0184e-05 | norm: 0.3364, dt: 450.14ms, tok/sec: 1164732.58
step: 18858 | loss: 3.011355 | lr: 6.0183e-05 | norm: 0.3166, dt: 450.46ms, tok/sec: 1163886.79
step: 18859 | loss: 3.004991 | lr: 6.0181e-05 | norm: 0.3067, dt: 450.83ms, tok/sec: 1162937.67
step: 18860 | loss: 3.031272 | lr: 6.0179e-05 | norm: 0.2792, dt: 450.42ms, tok/sec: 1164005.69
step: 18861 | loss: 3.069375 | lr: 6.0178e-05 | norm: 0.3290, dt: 449.96ms, tok/sec: 1165192.36
step: 18862 | loss: 3.147455 | lr: 6.0176e-05 | norm: 0.3121, dt: 450.16ms, tok/sec: 1164665.34
step: 18863 | loss: 3.131198 | lr: 6.0174e-05 | norm: 0.3864, dt: 450.89ms, tok/sec: 1162781.47
step: 18864 | loss: 3.134300 | lr: 6.0173e-05 | norm: 0.3554, dt: 449.80ms, tok/sec: 1165613.57
step: 18865 | loss: 3.092912 | lr: 6.0171e-05 | norm: 0.3343, dt: 450.24ms, tok/sec: 1164467.36
step: 18866 | loss: 3.135672 | lr: 6.0169e-05 | norm: 0.3709, dt: 450.67ms, tok/sec: 1163359.72
step: 18867 | loss: 3.057673 | lr: 6.0168e-05 | norm: 0.3184, dt: 451.06ms, tok/sec: 1162353.70
step: 18868 | loss: 3.087075 | lr: 6.0166e-05 | norm: 0.3290, dt: 450.21ms, tok/sec: 1164545.06
step: 18869 | loss: 3.154407 | lr: 6.0165e-05 | norm: 0.3375, dt: 450.88ms, tok/sec: 1162812.22
step: 18870 | loss: 3.185898 | lr: 6.0163e-05 | norm: 0.3040, dt: 450.20ms, tok/sec: 1164573.43
step: 18871 | loss: 3.147286 | lr: 6.0161e-05 | norm: 0.3383, dt: 450.75ms, tok/sec: 1163151.73
step: 18872 | loss: 3.162456 | lr: 6.0160e-05 | norm: 0.3221, dt: 450.93ms, tok/sec: 1162688.03
step: 18873 | loss: 3.042289 | lr: 6.0158e-05 | norm: 0.2960, dt: 450.97ms, tok/sec: 1162588.45
step: 18874 | loss: 3.088584 | lr: 6.0157e-05 | norm: 0.2990, dt: 452.11ms, tok/sec: 1159652.97
step: 18875 | loss: 3.074793 | lr: 6.0155e-05 | norm: 0.3088, dt: 450.57ms, tok/sec: 1163621.96
step: 18876 | loss: 3.020589 | lr: 6.0153e-05 | norm: 0.3614, dt: 451.00ms, tok/sec: 1162509.16
step: 18877 | loss: 3.103097 | lr: 6.0152e-05 | norm: 0.3789, dt: 451.08ms, tok/sec: 1162303.32
step: 18878 | loss: 3.004755 | lr: 6.0150e-05 | norm: 0.2896, dt: 450.98ms, tok/sec: 1162565.09
step: 18879 | loss: 3.122490 | lr: 6.0149e-05 | norm: 0.4003, dt: 450.32ms, tok/sec: 1164245.42
step: 18880 | loss: 3.174236 | lr: 6.0147e-05 | norm: 0.3197, dt: 450.70ms, tok/sec: 1163270.48
step: 18881 | loss: 3.067945 | lr: 6.0146e-05 | norm: 0.3678, dt: 450.96ms, tok/sec: 1162603.81
step: 18882 | loss: 3.018601 | lr: 6.0144e-05 | norm: 0.2930, dt: 451.78ms, tok/sec: 1160495.06
step: 18883 | loss: 3.099827 | lr: 6.0143e-05 | norm: 0.4440, dt: 451.52ms, tok/sec: 1161159.93
step: 18884 | loss: 3.120560 | lr: 6.0141e-05 | norm: 0.2876, dt: 451.34ms, tok/sec: 1161626.10
step: 18885 | loss: 3.056607 | lr: 6.0140e-05 | norm: 0.3868, dt: 451.33ms, tok/sec: 1161642.67
step: 18886 | loss: 3.009475 | lr: 6.0138e-05 | norm: 0.3128, dt: 451.03ms, tok/sec: 1162434.19
step: 18887 | loss: 3.057888 | lr: 6.0137e-05 | norm: 0.3485, dt: 450.41ms, tok/sec: 1164031.57
step: 18888 | loss: 3.083924 | lr: 6.0135e-05 | norm: 0.2829, dt: 450.01ms, tok/sec: 1165053.46
step: 18889 | loss: 3.040900 | lr: 6.0134e-05 | norm: 0.3222, dt: 450.77ms, tok/sec: 1163101.90
step: 18890 | loss: 3.072520 | lr: 6.0132e-05 | norm: 0.3055, dt: 451.21ms, tok/sec: 1161947.73
step: 18891 | loss: 3.023329 | lr: 6.0131e-05 | norm: 0.3025, dt: 450.54ms, tok/sec: 1163696.47
step: 18892 | loss: 3.042538 | lr: 6.0130e-05 | norm: 0.3215, dt: 451.06ms, tok/sec: 1162336.50
step: 18893 | loss: 3.028689 | lr: 6.0128e-05 | norm: 0.3245, dt: 452.02ms, tok/sec: 1159881.12
step: 18894 | loss: 2.964234 | lr: 6.0127e-05 | norm: 0.3143, dt: 450.57ms, tok/sec: 1163622.58
step: 18895 | loss: 3.017044 | lr: 6.0125e-05 | norm: 0.3019, dt: 450.80ms, tok/sec: 1163009.01
step: 18896 | loss: 2.975670 | lr: 6.0124e-05 | norm: 0.3148, dt: 450.36ms, tok/sec: 1164148.65
step: 18897 | loss: 2.994632 | lr: 6.0122e-05 | norm: 0.3036, dt: 451.29ms, tok/sec: 1161759.88
step: 18898 | loss: 2.978710 | lr: 6.0121e-05 | norm: 0.3421, dt: 449.62ms, tok/sec: 1166077.76
step: 18899 | loss: 3.066763 | lr: 6.0120e-05 | norm: 0.2753, dt: 450.74ms, tok/sec: 1163160.96
step: 18900 | loss: 3.071640 | lr: 6.0118e-05 | norm: 0.3676, dt: 451.89ms, tok/sec: 1160221.37
step: 18901 | loss: 3.004183 | lr: 6.0117e-05 | norm: 0.3042, dt: 450.64ms, tok/sec: 1163434.19
step: 18902 | loss: 2.945187 | lr: 6.0116e-05 | norm: 0.3527, dt: 450.26ms, tok/sec: 1164421.74
step: 18903 | loss: 3.025718 | lr: 6.0114e-05 | norm: 0.3539, dt: 451.25ms, tok/sec: 1161866.69
step: 18904 | loss: 3.030607 | lr: 6.0113e-05 | norm: 0.3888, dt: 450.75ms, tok/sec: 1163148.65
step: 18905 | loss: 3.008330 | lr: 6.0112e-05 | norm: 0.4666, dt: 451.26ms, tok/sec: 1161829.86
step: 18906 | loss: 3.061214 | lr: 6.0110e-05 | norm: 0.3610, dt: 450.86ms, tok/sec: 1162868.79
step: 18907 | loss: 3.066567 | lr: 6.0109e-05 | norm: 0.5293, dt: 450.98ms, tok/sec: 1162548.49
step: 18908 | loss: 3.076419 | lr: 6.0108e-05 | norm: 0.3735, dt: 450.41ms, tok/sec: 1164011.24
step: 18909 | loss: 3.119510 | lr: 6.0106e-05 | norm: 0.4536, dt: 450.14ms, tok/sec: 1164711.60
step: 18910 | loss: 3.072960 | lr: 6.0105e-05 | norm: 0.4610, dt: 451.23ms, tok/sec: 1161901.68
step: 18911 | loss: 3.091399 | lr: 6.0104e-05 | norm: 0.3701, dt: 1235.74ms, tok/sec: 424271.97
step: 18912 | loss: 3.070769 | lr: 6.0102e-05 | norm: 0.4422, dt: 451.12ms, tok/sec: 1162189.07
step: 18913 | loss: 3.151360 | lr: 6.0101e-05 | norm: 0.3325, dt: 450.42ms, tok/sec: 1163990.90
step: 18914 | loss: 3.095614 | lr: 6.0100e-05 | norm: 0.3963, dt: 449.93ms, tok/sec: 1165267.07
step: 18915 | loss: 3.104535 | lr: 6.0099e-05 | norm: 0.3470, dt: 452.03ms, tok/sec: 1159845.64
step: 18916 | loss: 3.184689 | lr: 6.0097e-05 | norm: 0.3794, dt: 452.34ms, tok/sec: 1159055.80
step: 18917 | loss: 3.044101 | lr: 6.0096e-05 | norm: 0.3434, dt: 450.91ms, tok/sec: 1162743.97
step: 18918 | loss: 3.083886 | lr: 6.0095e-05 | norm: 0.3235, dt: 454.57ms, tok/sec: 1153382.07
step: 18919 | loss: 3.082621 | lr: 6.0094e-05 | norm: 0.3624, dt: 451.79ms, tok/sec: 1160480.36
step: 18920 | loss: 3.027326 | lr: 6.0093e-05 | norm: 0.3095, dt: 450.97ms, tok/sec: 1162582.30
step: 18921 | loss: 3.073097 | lr: 6.0091e-05 | norm: 0.3188, dt: 450.46ms, tok/sec: 1163903.42
step: 18922 | loss: 3.098808 | lr: 6.0090e-05 | norm: 0.3136, dt: 450.02ms, tok/sec: 1165033.71
step: 18923 | loss: 3.066788 | lr: 6.0089e-05 | norm: 0.3885, dt: 451.84ms, tok/sec: 1160345.65
step: 18924 | loss: 3.008382 | lr: 6.0088e-05 | norm: 0.3011, dt: 451.02ms, tok/sec: 1162453.85
step: 18925 | loss: 3.038713 | lr: 6.0087e-05 | norm: 0.4385, dt: 451.02ms, tok/sec: 1162459.38
step: 18926 | loss: 3.080688 | lr: 6.0085e-05 | norm: 0.3135, dt: 451.51ms, tok/sec: 1161196.11
step: 18927 | loss: 3.094575 | lr: 6.0084e-05 | norm: 0.3373, dt: 452.87ms, tok/sec: 1157709.69
step: 18928 | loss: 3.117609 | lr: 6.0083e-05 | norm: 0.2998, dt: 450.89ms, tok/sec: 1162782.70
step: 18929 | loss: 3.047742 | lr: 6.0082e-05 | norm: 0.3252, dt: 450.95ms, tok/sec: 1162634.55
step: 18930 | loss: 3.012310 | lr: 6.0081e-05 | norm: 0.3338, dt: 451.21ms, tok/sec: 1161968.60
step: 18931 | loss: 3.112571 | lr: 6.0080e-05 | norm: 0.3517, dt: 451.12ms, tok/sec: 1162190.91
step: 18932 | loss: 3.063979 | lr: 6.0079e-05 | norm: 0.3521, dt: 451.81ms, tok/sec: 1160415.45
step: 18933 | loss: 3.050330 | lr: 6.0077e-05 | norm: 0.3004, dt: 453.22ms, tok/sec: 1156813.21
step: 18934 | loss: 3.120043 | lr: 6.0076e-05 | norm: 0.3329, dt: 451.51ms, tok/sec: 1161180.16
step: 18935 | loss: 3.006696 | lr: 6.0075e-05 | norm: 0.3341, dt: 450.53ms, tok/sec: 1163708.17
step: 18936 | loss: 3.079235 | lr: 6.0074e-05 | norm: 0.3722, dt: 449.99ms, tok/sec: 1165121.36
step: 18937 | loss: 2.993444 | lr: 6.0073e-05 | norm: 0.3422, dt: 452.44ms, tok/sec: 1158793.78
step: 18938 | loss: 3.056149 | lr: 6.0072e-05 | norm: 0.3493, dt: 451.32ms, tok/sec: 1161673.35
step: 18939 | loss: 3.012139 | lr: 6.0071e-05 | norm: 0.3301, dt: 451.42ms, tok/sec: 1161427.93
step: 18940 | loss: 3.040678 | lr: 6.0070e-05 | norm: 0.3105, dt: 452.95ms, tok/sec: 1157507.99
step: 18941 | loss: 3.041574 | lr: 6.0069e-05 | norm: 0.3406, dt: 451.23ms, tok/sec: 1161920.71
step: 18942 | loss: 2.984248 | lr: 6.0068e-05 | norm: 0.2919, dt: 450.77ms, tok/sec: 1163082.21
step: 18943 | loss: 3.047872 | lr: 6.0067e-05 | norm: 0.2885, dt: 453.12ms, tok/sec: 1157061.56
step: 18944 | loss: 2.966781 | lr: 6.0066e-05 | norm: 0.3042, dt: 450.99ms, tok/sec: 1162538.05
step: 18945 | loss: 2.996340 | lr: 6.0065e-05 | norm: 0.2864, dt: 451.07ms, tok/sec: 1162311.31
step: 18946 | loss: 3.068152 | lr: 6.0064e-05 | norm: 0.2916, dt: 451.74ms, tok/sec: 1160598.57
step: 18947 | loss: 3.014228 | lr: 6.0063e-05 | norm: 0.2608, dt: 450.29ms, tok/sec: 1164345.90
step: 18948 | loss: 3.017846 | lr: 6.0062e-05 | norm: 0.3735, dt: 450.87ms, tok/sec: 1162849.11
step: 18949 | loss: 3.016148 | lr: 6.0061e-05 | norm: 0.2979, dt: 450.36ms, tok/sec: 1164163.44
step: 18950 | loss: 3.010825 | lr: 6.0060e-05 | norm: 0.4020, dt: 450.68ms, tok/sec: 1163322.18
step: 18951 | loss: 2.988526 | lr: 6.0059e-05 | norm: 0.3083, dt: 451.44ms, tok/sec: 1161362.91
step: 18952 | loss: 3.000185 | lr: 6.0058e-05 | norm: 0.3709, dt: 450.71ms, tok/sec: 1163253.25
step: 18953 | loss: 3.078105 | lr: 6.0057e-05 | norm: 0.3848, dt: 450.09ms, tok/sec: 1164855.97
step: 18954 | loss: 3.141806 | lr: 6.0056e-05 | norm: 0.3126, dt: 451.50ms, tok/sec: 1161208.37
step: 18955 | loss: 3.092278 | lr: 6.0055e-05 | norm: 0.3617, dt: 451.22ms, tok/sec: 1161932.38
step: 18956 | loss: 3.111029 | lr: 6.0054e-05 | norm: 0.3322, dt: 449.61ms, tok/sec: 1166085.18
step: 18957 | loss: 3.113368 | lr: 6.0053e-05 | norm: 0.4425, dt: 450.82ms, tok/sec: 1162956.73
step: 18958 | loss: 3.124493 | lr: 6.0052e-05 | norm: 0.3537, dt: 450.46ms, tok/sec: 1163887.40
step: 18959 | loss: 3.045005 | lr: 6.0051e-05 | norm: 0.4241, dt: 451.20ms, tok/sec: 1161993.16
step: 18960 | loss: 3.116362 | lr: 6.0050e-05 | norm: 0.3167, dt: 450.06ms, tok/sec: 1164922.00
step: 18961 | loss: 3.088314 | lr: 6.0050e-05 | norm: 0.4264, dt: 450.51ms, tok/sec: 1163756.82
step: 18962 | loss: 3.160491 | lr: 6.0049e-05 | norm: 0.3788, dt: 453.47ms, tok/sec: 1156175.80
step: 18963 | loss: 3.064194 | lr: 6.0048e-05 | norm: 0.4529, dt: 451.31ms, tok/sec: 1161708.33
step: 18964 | loss: 3.123259 | lr: 6.0047e-05 | norm: 0.3197, dt: 450.26ms, tok/sec: 1164420.50
step: 18965 | loss: 3.067416 | lr: 6.0046e-05 | norm: 0.3749, dt: 450.51ms, tok/sec: 1163759.90
step: 18966 | loss: 3.056630 | lr: 6.0045e-05 | norm: 0.3074, dt: 450.28ms, tok/sec: 1164358.85
step: 18967 | loss: 3.036000 | lr: 6.0044e-05 | norm: 0.3216, dt: 451.19ms, tok/sec: 1162022.02
step: 18968 | loss: 3.061962 | lr: 6.0044e-05 | norm: 0.3020, dt: 451.54ms, tok/sec: 1161107.82
step: 18969 | loss: 3.047018 | lr: 6.0043e-05 | norm: 0.3033, dt: 450.29ms, tok/sec: 1164339.74
step: 18970 | loss: 2.985201 | lr: 6.0042e-05 | norm: 0.2871, dt: 450.81ms, tok/sec: 1162990.56
step: 18971 | loss: 3.006392 | lr: 6.0041e-05 | norm: 0.3060, dt: 451.19ms, tok/sec: 1162002.37
step: 18972 | loss: 3.068101 | lr: 6.0040e-05 | norm: 0.3113, dt: 450.70ms, tok/sec: 1163270.48
step: 18973 | loss: 3.128283 | lr: 6.0040e-05 | norm: 0.3579, dt: 450.33ms, tok/sec: 1164234.32
step: 18974 | loss: 3.056726 | lr: 6.0039e-05 | norm: 0.3593, dt: 450.39ms, tok/sec: 1164074.70
step: 18975 | loss: 3.136241 | lr: 6.0038e-05 | norm: 0.3005, dt: 450.36ms, tok/sec: 1164141.26
step: 18976 | loss: 3.037728 | lr: 6.0037e-05 | norm: 0.3453, dt: 450.32ms, tok/sec: 1164265.14
step: 18977 | loss: 3.045829 | lr: 6.0036e-05 | norm: 0.2732, dt: 451.46ms, tok/sec: 1161314.46
step: 18978 | loss: 3.049288 | lr: 6.0036e-05 | norm: 0.3003, dt: 449.54ms, tok/sec: 1166289.27
step: 18979 | loss: 3.048970 | lr: 6.0035e-05 | norm: 0.2840, dt: 450.11ms, tok/sec: 1164812.16
step: 18980 | loss: 3.055689 | lr: 6.0034e-05 | norm: 0.4159, dt: 449.68ms, tok/sec: 1165913.92
step: 18981 | loss: 3.060328 | lr: 6.0033e-05 | norm: 0.3110, dt: 450.45ms, tok/sec: 1163913.28
step: 18982 | loss: 3.029873 | lr: 6.0033e-05 | norm: 0.4227, dt: 451.42ms, tok/sec: 1161421.18
step: 18983 | loss: 3.023057 | lr: 6.0032e-05 | norm: 0.2954, dt: 451.26ms, tok/sec: 1161820.65
step: 18984 | loss: 3.083655 | lr: 6.0031e-05 | norm: 0.3558, dt: 451.41ms, tok/sec: 1161446.95
step: 18985 | loss: 3.036090 | lr: 6.0031e-05 | norm: 0.3257, dt: 451.28ms, tok/sec: 1161783.82
step: 18986 | loss: 3.062327 | lr: 6.0030e-05 | norm: 0.3094, dt: 449.75ms, tok/sec: 1165722.32
step: 18987 | loss: 3.058447 | lr: 6.0029e-05 | norm: 0.2961, dt: 450.41ms, tok/sec: 1164024.18
step: 18988 | loss: 2.986411 | lr: 6.0029e-05 | norm: 0.3340, dt: 450.34ms, tok/sec: 1164209.67
step: 18989 | loss: 2.984595 | lr: 6.0028e-05 | norm: 0.3097, dt: 451.32ms, tok/sec: 1161681.33
step: 18990 | loss: 3.037181 | lr: 6.0027e-05 | norm: 0.3458, dt: 450.53ms, tok/sec: 1163708.79
step: 18991 | loss: 2.984284 | lr: 6.0027e-05 | norm: 0.3058, dt: 451.32ms, tok/sec: 1161686.85
step: 18992 | loss: 2.959820 | lr: 6.0026e-05 | norm: 0.3418, dt: 451.64ms, tok/sec: 1160844.87
step: 18993 | loss: 3.007937 | lr: 6.0025e-05 | norm: 0.2915, dt: 450.26ms, tok/sec: 1164399.54
step: 18994 | loss: 3.082393 | lr: 6.0025e-05 | norm: 0.3380, dt: 450.92ms, tok/sec: 1162696.02
step: 18995 | loss: 3.014030 | lr: 6.0024e-05 | norm: 0.2953, dt: 451.52ms, tok/sec: 1161160.54
step: 18996 | loss: 3.024654 | lr: 6.0023e-05 | norm: 0.2953, dt: 451.71ms, tok/sec: 1160676.98
step: 18997 | loss: 3.001525 | lr: 6.0023e-05 | norm: 0.3247, dt: 451.75ms, tok/sec: 1160581.42
step: 18998 | loss: 3.020955 | lr: 6.0022e-05 | norm: 0.3311, dt: 450.78ms, tok/sec: 1163069.91
step: 18999 | loss: 3.050816 | lr: 6.0022e-05 | norm: 0.3185, dt: 451.55ms, tok/sec: 1161085.13
validation loss: 3.0805
HellaSwag accuracy: 3017/10042=0.3004
rank 0 sample 0: Hello, I'm a Manpreet, and I'll be the King and Queen of I love. I hope you will enjoy reading this story, or if
rank 7 sample 0: Hello, I'm a Manpreet, and I just want to tell you what each of you do. I tell you each of you on this day,
rank 0 sample 1: Hello, I'm a Manpreet, I think I'm a Manpreet, but yeah, that's your definition. And it's true, I
rank 2 sample 0: Hello, I'm a Manpreet, "Hey! The manpreet looks a bit like you're a bird, or a bird."
A birdrank 7 sample 1: Hello, I'm a Manpreet, (we have a friend and friend/professional) of Manpreet, a member of the International Day of Womenrank 0 sample 2: Hello, I'm a Manpreet, I'm looking at the page (with the head of the body, I'm looking at the page) and you


rank 0 sample 3: Hello, I'm a Manpreet, but why do you think it was an error - that's how it happened to your system? I understand that the
rank 7 sample 2: Hello, I'm a Manpreet, sorry, but let me know right away!
Do you know why people are so scared of the sea? They
rank 6 sample 0: Hello, I'm a Manpreet, why can't I say sorry to her?"
It was an annoying moment and I was happy to see that sherank 2 sample 1: Hello, I'm a Manpreet, my name is Nd, I think and I call it Manpreet. I'm a man.
I

rank 7 sample 3: Hello, I'm a Manpreet, no one ever said, just a fellow that means, "man, do you want to have any? I'm
rank 2 sample 2: Hello, I'm a Manpreet, I'm the Master, and my family is very close to me.<|endoftext|>This book describes how different languages and localrank 5 sample 0: Hello, I'm a Manpreet, you mean you're gonna be a mane. Yeah. I wanna have a good sense of your world and not
rank 6 sample 1: Hello, I'm a Manpreet, but I am a Teacher, too. A teacher who is a teacher for a college student.
The following is
rank 4 sample 0: Hello, I'm a Manpreet, I'm gonna be the best man in a class.
So, it's always great to see people in class

rank 2 sample 3: Hello, I'm a Manpreet, you see--
The manpreet you see on the bottom right of the post. Your image of your face
rank 6 sample 2: Hello, I'm a Manpreet, you should have been the one that is coming back.
"But, in the old days, no one evenrank 5 sample 1: Hello, I'm a Manpreet, don't you see. This is how the term "man" sounds." No more. I guess there, you
rank 4 sample 1: Hello, I'm a Manpreet, your Name is Gayhirnath (My name is Gayhirnath) and I live in


rank 6 sample 3: Hello, I'm a Manpreet, and it's something like that (except that you're a Manpreet and I have 'em's) who
rank 5 sample 2: Hello, I'm a Manpreet, You are very nice . . . [I just want to write down a story, what's the best way to
rank 4 sample 2: Hello, I'm a Manpreet, a Jitap,
I was the Director of the Indian Council of the Arts (ICACAF) -
rank 5 sample 3: Hello, I'm a Manpreet, thanks for pointing that out. I'm not going to say a lot about manpreet. Actually, it is
rank 4 sample 3: Hello, I'm a Manpreet, and this may surprise you. I grew up here in London having never wanted to go to college so I was forced
rank 1 sample 0: Hello, I'm a Manpreet, please write a comment, please don't write my opinion :)<|endoftext|>The United States is home to an estimated one-
rank 1 sample 1: Hello, I'm a Manpreet, so you're supposed to be on time to get back through the last five minutes. How would you like to know
rank 1 sample 2: Hello, I'm a Manpreet, but then what about you?
I'm a Manpreet and my girlfriend is a Manpreet, who
rank 1 sample 3: Hello, I'm a Manpreet, and I'm looking at myself and wondering that I'm taking notes that day before. One might have to ask what
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going back over to Ili. And when it's you just know when a problem with the machine
rank 3 sample 1: Hello, I'm a Manpreet, and that is what I used to do when I was a kid, and I was a professional soccer player. In
rank 3 sample 2: Hello, I'm a Manpreet, and you are a
What was the world like for me and my mother
in the 1930s and '90
rank 3 sample 3: Hello, I'm a Manpreet, and at the very least, if you want to get in touch with him, you might get in touch.

step: 19000 | loss: 3.071060 | lr: 6.0021e-05 | norm: 0.3740, dt: 12083.06ms, tok/sec: 43390.33
step: 19001 | loss: 3.128910 | lr: 6.0020e-05 | norm: 0.3093, dt: 1147.41ms, tok/sec: 456932.21
step: 19002 | loss: 3.110885 | lr: 6.0020e-05 | norm: 0.3113, dt: 448.19ms, tok/sec: 1169795.89
step: 19003 | loss: 3.089281 | lr: 6.0019e-05 | norm: 0.3064, dt: 448.80ms, tok/sec: 1168205.00
step: 19004 | loss: 3.107827 | lr: 6.0019e-05 | norm: 0.3479, dt: 449.50ms, tok/sec: 1166388.25
step: 19005 | loss: 3.149146 | lr: 6.0018e-05 | norm: 0.3245, dt: 452.45ms, tok/sec: 1158771.19
step: 19006 | loss: 3.058329 | lr: 6.0018e-05 | norm: 0.3072, dt: 449.42ms, tok/sec: 1166597.39
step: 19007 | loss: 3.074749 | lr: 6.0017e-05 | norm: 0.4096, dt: 450.68ms, tok/sec: 1163325.87
step: 19008 | loss: 3.126159 | lr: 6.0017e-05 | norm: 0.3235, dt: 451.27ms, tok/sec: 1161804.08
step: 19009 | loss: 3.069616 | lr: 6.0016e-05 | norm: 0.3141, dt: 451.43ms, tok/sec: 1161399.10
step: 19010 | loss: 3.041189 | lr: 6.0016e-05 | norm: 0.3079, dt: 450.02ms, tok/sec: 1165022.60
step: 19011 | loss: 3.134212 | lr: 6.0015e-05 | norm: 0.4039, dt: 450.89ms, tok/sec: 1162774.71
step: 19012 | loss: 2.977798 | lr: 6.0015e-05 | norm: 0.3539, dt: 450.95ms, tok/sec: 1162639.46
step: 19013 | loss: 3.019120 | lr: 6.0014e-05 | norm: 0.3542, dt: 450.60ms, tok/sec: 1163538.23
step: 19014 | loss: 3.095700 | lr: 6.0014e-05 | norm: 0.2982, dt: 450.41ms, tok/sec: 1164022.33
step: 19015 | loss: 3.042805 | lr: 6.0013e-05 | norm: 0.3203, dt: 450.73ms, tok/sec: 1163199.72
step: 19016 | loss: 3.054195 | lr: 6.0013e-05 | norm: 0.2979, dt: 452.80ms, tok/sec: 1157883.43
step: 19017 | loss: 3.063762 | lr: 6.0012e-05 | norm: 0.3414, dt: 450.49ms, tok/sec: 1163807.33
step: 19018 | loss: 3.084513 | lr: 6.0012e-05 | norm: 0.2832, dt: 455.07ms, tok/sec: 1152110.66
step: 19019 | loss: 3.095187 | lr: 6.0012e-05 | norm: 0.2871, dt: 452.40ms, tok/sec: 1158903.09
step: 19020 | loss: 3.024111 | lr: 6.0011e-05 | norm: 0.3081, dt: 450.69ms, tok/sec: 1163301.87
step: 19021 | loss: 3.033106 | lr: 6.0011e-05 | norm: 0.2896, dt: 451.34ms, tok/sec: 1161619.96
step: 19022 | loss: 3.120891 | lr: 6.0010e-05 | norm: 0.3093, dt: 451.47ms, tok/sec: 1161285.02
step: 19023 | loss: 3.059443 | lr: 6.0010e-05 | norm: 0.2912, dt: 450.70ms, tok/sec: 1163283.41
step: 19024 | loss: 3.085866 | lr: 6.0009e-05 | norm: 0.3167, dt: 451.37ms, tok/sec: 1161535.90
step: 19025 | loss: 3.056797 | lr: 6.0009e-05 | norm: 0.3173, dt: 452.15ms, tok/sec: 1159550.24
step: 19026 | loss: 3.030788 | lr: 6.0009e-05 | norm: 0.2809, dt: 452.28ms, tok/sec: 1159211.61
step: 19027 | loss: 2.985983 | lr: 6.0008e-05 | norm: 0.3024, dt: 450.71ms, tok/sec: 1163252.02
step: 19028 | loss: 3.021516 | lr: 6.0008e-05 | norm: 0.3392, dt: 450.23ms, tok/sec: 1164501.90
step: 19029 | loss: 3.028810 | lr: 6.0008e-05 | norm: 0.3045, dt: 451.41ms, tok/sec: 1161435.29
step: 19030 | loss: 3.061255 | lr: 6.0007e-05 | norm: 0.3858, dt: 451.24ms, tok/sec: 1161894.93
step: 19031 | loss: 2.965288 | lr: 6.0007e-05 | norm: 0.3196, dt: 452.06ms, tok/sec: 1159786.30
step: 19032 | loss: 3.055945 | lr: 6.0007e-05 | norm: 0.3708, dt: 452.20ms, tok/sec: 1159410.85
step: 19033 | loss: 3.062587 | lr: 6.0006e-05 | norm: 0.3076, dt: 450.58ms, tok/sec: 1163573.32
step: 19034 | loss: 2.969986 | lr: 6.0006e-05 | norm: 0.3337, dt: 450.87ms, tok/sec: 1162827.59
step: 19035 | loss: 2.998115 | lr: 6.0006e-05 | norm: 0.3351, dt: 451.24ms, tok/sec: 1161890.63
step: 19036 | loss: 2.997259 | lr: 6.0005e-05 | norm: 0.3161, dt: 452.45ms, tok/sec: 1158775.46
step: 19037 | loss: 3.052245 | lr: 6.0005e-05 | norm: 0.3135, dt: 450.37ms, tok/sec: 1164127.70
step: 19038 | loss: 3.039004 | lr: 6.0005e-05 | norm: 0.3636, dt: 453.47ms, tok/sec: 1156167.90
step: 19039 | loss: 3.042357 | lr: 6.0005e-05 | norm: 0.3662, dt: 451.28ms, tok/sec: 1161788.73
step: 19040 | loss: 3.013330 | lr: 6.0004e-05 | norm: 0.3445, dt: 450.58ms, tok/sec: 1163577.63
step: 19041 | loss: 3.034044 | lr: 6.0004e-05 | norm: 0.3049, dt: 451.35ms, tok/sec: 1161611.37
step: 19042 | loss: 3.009784 | lr: 6.0004e-05 | norm: 0.3026, dt: 451.36ms, tok/sec: 1161568.42
step: 19043 | loss: 3.027441 | lr: 6.0004e-05 | norm: 0.3167, dt: 450.92ms, tok/sec: 1162716.30
step: 19044 | loss: 3.013636 | lr: 6.0003e-05 | norm: 0.3059, dt: 449.65ms, tok/sec: 1165980.69
step: 19045 | loss: 3.079331 | lr: 6.0003e-05 | norm: 0.4338, dt: 450.21ms, tok/sec: 1164548.15
step: 19046 | loss: 3.075661 | lr: 6.0003e-05 | norm: 0.3511, dt: 451.28ms, tok/sec: 1161774.61
step: 19047 | loss: 3.107903 | lr: 6.0003e-05 | norm: 0.3561, dt: 451.16ms, tok/sec: 1162076.06
step: 19048 | loss: 3.124269 | lr: 6.0002e-05 | norm: 0.3642, dt: 450.57ms, tok/sec: 1163604.72
step: 19049 | loss: 3.112307 | lr: 6.0002e-05 | norm: 0.3976, dt: 451.17ms, tok/sec: 1162071.76
step: 19050 | loss: 3.093037 | lr: 6.0002e-05 | norm: 0.3201, dt: 451.25ms, tok/sec: 1161854.41
step: 19051 | loss: 3.123297 | lr: 6.0002e-05 | norm: 0.3424, dt: 450.91ms, tok/sec: 1162744.59
step: 19052 | loss: 3.156433 | lr: 6.0002e-05 | norm: 0.3390, dt: 451.32ms, tok/sec: 1161680.71
step: 19053 | loss: 3.139626 | lr: 6.0002e-05 | norm: 0.3297, dt: 451.54ms, tok/sec: 1161114.56
step: 19054 | loss: 3.117090 | lr: 6.0001e-05 | norm: 0.3455, dt: 454.22ms, tok/sec: 1154256.88
step: 19055 | loss: 3.119825 | lr: 6.0001e-05 | norm: 0.3365, dt: 451.64ms, tok/sec: 1160865.09
step: 19056 | loss: 3.109554 | lr: 6.0001e-05 | norm: 0.3577, dt: 450.99ms, tok/sec: 1162529.44
step: 19057 | loss: 3.101246 | lr: 6.0001e-05 | norm: 0.3796, dt: 450.90ms, tok/sec: 1162763.64
step: 19058 | loss: 3.085089 | lr: 6.0001e-05 | norm: 0.4787, dt: 450.44ms, tok/sec: 1163941.62
step: 19059 | loss: 3.033453 | lr: 6.0001e-05 | norm: 0.3679, dt: 451.36ms, tok/sec: 1161578.85
step: 19060 | loss: 3.094935 | lr: 6.0001e-05 | norm: 0.3783, dt: 451.32ms, tok/sec: 1161672.74
step: 19061 | loss: 3.026596 | lr: 6.0001e-05 | norm: 0.3296, dt: 451.74ms, tok/sec: 1160590.00
step: 19062 | loss: 3.076644 | lr: 6.0000e-05 | norm: 0.3610, dt: 455.71ms, tok/sec: 1150475.39
step: 19063 | loss: 3.099107 | lr: 6.0000e-05 | norm: 0.3003, dt: 451.43ms, tok/sec: 1161399.10
step: 19064 | loss: 3.112928 | lr: 6.0000e-05 | norm: 0.4256, dt: 451.46ms, tok/sec: 1161324.89
step: 19065 | loss: 3.045789 | lr: 6.0000e-05 | norm: 0.3192, dt: 451.50ms, tok/sec: 1161209.60
step: 19066 | loss: 3.022587 | lr: 6.0000e-05 | norm: 0.3940, dt: 450.85ms, tok/sec: 1162878.63
step: 19067 | loss: 3.029815 | lr: 6.0000e-05 | norm: 0.3506, dt: 451.16ms, tok/sec: 1162085.27
step: 19068 | loss: 3.123561 | lr: 6.0000e-05 | norm: 0.3710, dt: 450.70ms, tok/sec: 1163267.41
step: 19069 | loss: 3.055324 | lr: 6.0000e-05 | norm: 0.4074, dt: 450.83ms, tok/sec: 1162929.67
step: 19070 | loss: 3.059604 | lr: 6.0000e-05 | norm: 0.2838, dt: 450.96ms, tok/sec: 1162606.88
step: 19071 | loss: 3.078244 | lr: 6.0000e-05 | norm: 0.3745, dt: 451.18ms, tok/sec: 1162025.09
validation loss: 3.0793
HellaSwag accuracy: 3018/10042=0.3005
rank 2 sample 0: Hello, I'm a Manpreet, no need for much, thank you,
"You can also use "I'm a Manpreet..." to
rank 2 sample 1: Hello, I'm a Manpreet, thank you so much for the opportunity to participate in the 'Culti' programme. The first year I took
rank 2 sample 2: Hello, I'm a Manpreet, I'm the King! I just put you in my world of being there for your family and all of my friends
rank 2 sample 3: Hello, I'm a Manpreet, you see. I have been in this game for 30 years and I have tried and found people in my field.
rank 0 sample 0: Hello, I'm a Manpreet, and I don't have to deal with an "allergen,"”” or something, “
rank 0 sample 1: Hello, I'm a Manpreet, I live in a house that was rented out for people without any of the above. It's not exactly like the
rank 0 sample 2: Hello, I'm a Manpreet, I'm at work, I'm here, in the office or anywhere. I'm a man, I'm at
rank 0 sample 3: Hello, I'm a Manpreet,
Now I'm a Manpreet. Welcome to the Blog.
It came with our awesome book, "
rank 5 sample 0: Hello, I'm a Manpreet, you mean you're just a man! Okay. How about you say it? Well, I think you are actually
rank 5 sample 1: Hello, I'm a Manpreet, don't you see there a place you can actually play in your life. You might have to change or adapt your
rank 5 sample 2: Hello, I'm a Manpreet, just a small guy somewhere.<|endoftext|>“This is a little story of the man by the sea who is arank 7 sample 0: Hello, I'm a Manpreet, and I'm a Teacher for the entire Indian state of Goa. How are you?
Pupil:

rank 5 sample 3: Hello, I'm a Manpreet, an official English Language teacher that has been working for some time now in the country so it is only natural that you
rank 4 sample 0: Hello, I'm a Manpreet, I'm gonna be the new Manpreet, I'm gonna be the new Manpreet, I'm gonnarank 7 sample 1: Hello, I'm a Manpreet, where everybody has a job to do a bunch of things over and over again. How to get people to buy something

rank 7 sample 2: Hello, I'm a Manpreet, because I am now a Manpreet and I feel like I'm coming to the next level. I am an
rank 4 sample 1: Hello, I'm a Manpreet, Aisha, We're going, The Sun is on her way, but the Earth is not moving around, Not
rank 7 sample 3: Hello, I'm a Manpreet, with the first two words and the right combination - -y, and all the rest.
That's the end
rank 4 sample 2: Hello, I'm a Manpreet, a Lenny & Atee Lenny. I'm also a Lenny & Atee/Lenny
rank 4 sample 3: Hello, I'm a Manpreet, and the place's pretty well, all in, as are people's ears, and the stuff like it's kind
rank 6 sample 0: Hello, I'm a Manpreet, why can't I see who's there, and you know who's here?
I like to put a little
rank 6 sample 1: Hello, I'm a Manpreet,
But the question comes from you, don't worry,
I will be the best man on earth for you
rank 6 sample 2: Hello, I'm a Manpreet, you should have gone somewhere else and had a chance to see a beautiful, red, round, white fish on his
rank 6 sample 3: Hello, I'm a Manpreet, and it's me that's where the conversation begins.
I've been busy in this particular area. Now,
rank 1 sample 0: Hello, I'm a Manpreet, in this group, so you can refer me any number of times.
My name is Giambald.
rank 1 sample 1: Hello, I'm a Manpreet, so you're supposed to be an 'honest' Englishman.
We said today we had to do the
rank 1 sample 2: Hello, I'm a Manpreet,
who was just a little bit late to the party and that
I think I'm getting through to you a
rank 1 sample 3: Hello, I'm a Manpreet, and I'm in a mood to take advantage of the information we gain on various topics -- as long as it's
rank 3 sample 0: Hello, I'm a Manpreet, and I'm going into some world for our next trip. Our next friend says "hello, I'm going into
rank 3 sample 1: Hello, I'm a Manpreet, and that is what I wrote, so it's time to write it down.
I'm a manpreet
rank 3 sample 2: Hello, I'm a Manpreet, and you have some data in your head. When you look at the table below, you have got the data I
rank 3 sample 3: Hello, I'm a Manpreet, and have a nice day. And thank you. And it was really good. You just don't put the same
step: 19072 | loss: 3.045321 | lr: 6.0000e-05 | norm: 0.2866, dt: 12908.25ms, tok/sec: 40616.51
ubuntu@164-152-26-183:~/msgpt$ 


step: 38145 | loss: 2.997005 | lr: 6.0000e-05 | norm: 0.4810, dt: 18603.15ms, tok/sec: 28182.75