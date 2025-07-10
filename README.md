# EHPE: A Segmented Architecture for Enhanced Hand Pose Estimation
![网络结构](https://raw.githubusercontent.com/SereinNout/EHPE/main/image/net.jpg)

## About
3D hand pose estimation has garnered great attention in recent years due to its critical applications in human-computer interaction, virtual reality, and related fields.
The accurate estimation of hand joints is essential for high-quality hand pose estimation.
However, existing methods neglect the importance of Distal Phalanx Tip (TIP) and Wrist in predicting hand joints overall and often fail to account for the phenomenon of error accumulation for distal joints in gesture estimation, which can cause certain joints to incur larger errors, resulting in misalignments and artifacts in the pose estimation and degrading the overall reconstruction quality.
To address this challenge, we propose a novel segmented architecture for enhanced hand pose estimation (EHPE).
We perform local extraction of TIP and wrist, thus alleviating the effect of error accumulation on TIP prediction and further reduce the predictive errors for all joints on this basis.
### 🔍 Our Main Contributions
- We systematically analyze the phenomenon of error accumulation for distal joints in gesture estimation, clarify the importance of the TIP joints under the guidance of structural priors, and design a novel segmented architecture for enhanced hand pose estimation.
- We design a dual-branch structure in which one branch involves a graphic attention layer of the dynamic topology structure to utilize the hand structure prior with the guidance of TIP and wrist, while the other branch estimates the hand poses from the visual features.
- Extensive experiments demonstrate the effectiveness of the proposed methods. Through comparison on public benchmarks, our EHPE surpasses all compared methods and achieves state-of-the-art performance.

Paper link: 

## Dataset 
https://lmb.informatik.uni-freiburg.de/projects/freihand/
