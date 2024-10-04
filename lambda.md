## Setup LambdaLabs

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