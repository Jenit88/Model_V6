# Training report - Model_v6_2_scratch_rtx

Generated 2026-09-03T06:13:10+00:00

## Where the run is

- Epochs completed: **49** of 120
- Learning rate: 2.091e-04
- Train loss: 0.1994  |  validation loss: nan
- Foreground mIoU: train 0.9002  |  validation nan

## Best checkpoint so far

- Selected on **mask_map50_95** = **0.8127** at epoch 50
- Instance precision 0.9048, recall 0.9678, F1 0.9353 at IoU 0.50
- Mask mAP50 0.9600, mAP50-95 0.8127
- TP 10,986 / FP 1,156 / FN 365

### Per class, at the best checkpoint

| class | TP | FP | FN | P | R | F1 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|---|
| Rectangle | 2,745 | 266 | 65 | 0.9117 | 0.9769 | 0.9431 | 0.9852 | 0.8381 |
| Rectangle_concave | 80 | 12 | 3 | 0.8696 | 0.9639 | 0.9143 | 0.9794 | 0.9085 |
| circle | 1,552 | 82 | 66 | 0.9498 | 0.9592 | 0.9545 | 0.9757 | 0.8229 |
| circle_full | 6,609 | 796 | 231 | 0.8925 | 0.9662 | 0.9279 | 0.8996 | 0.6814 |

### Evaluation history

| epoch | P | R | F1 | mAP50 | mAP50-95 | saved |
|---|---|---|---|---|---|---|
| 1 | 0.0184 | 0.0684 | 0.0290 | 0.0134 | 0.0043 | yes |
| 5 | 0.7198 | 0.8738 | 0.7894 | 0.8595 | 0.5751 | yes |
| 10 | 0.8438 | 0.9383 | 0.8885 | 0.9354 | 0.7284 | yes |
| 15 | 0.8779 | 0.9554 | 0.9150 | 0.9504 | 0.7657 | yes |
| 20 | 0.8820 | 0.9567 | 0.9178 | 0.9513 | 0.7814 | yes |
| 25 | 0.8870 | 0.9597 | 0.9219 | 0.9529 | 0.7892 | yes |
| 30 | 0.8938 | 0.9620 | 0.9267 | 0.9554 | 0.7966 | yes |
| 35 | 0.8996 | 0.9649 | 0.9311 | 0.9549 | 0.8003 | yes |
| 40 | 0.9002 | 0.9633 | 0.9307 | 0.9562 | 0.8052 | yes |
| 45 | 0.9027 | 0.9647 | 0.9327 | 0.9595 | 0.8095 | yes |
| 50 | 0.9048 | 0.9678 | 0.9353 | 0.9600 | 0.8127 | yes |

## What the numbers say

- Still improving: mask_map50_95 is rising about 0.0041 per evaluation over the last 5.
- False positives dominate: precision 0.905 versus recall 0.968. Raise CENTER_CONFIDENCE_THRESHOLD or MIN_COMPONENT_AREA_FRACTION; do not add capacity.

## Semantic IoU by class (last epoch)

| class | train IoU | val IoU |
|---|---|---|
| Rectangle | - | - |
| Rectangle_concave | - | - |
| circle | - | - |
| circle_full | - | - |
