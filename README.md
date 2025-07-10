# EHPE: A Segmented Architecture for Enhanced Hand Pose Estimation
![image] (image/net.jpg)
# About
3D hand pose estimation has garnered great attention in recent years due to its critical applications in human-computer interaction, virtual reality, and related fields.
The accurate estimation of hand joints is essential for high-quality hand pose estimation.
However, existing methods neglect the importance of Distal Phalanx Tip (TIP) and Wrist in predicting hand joints overall and often fail to account for the phenomenon of error accumulation for distal joints in gesture estimation, which can cause certain joints to incur larger errors, resulting in misalignments and artifacts in the pose estimation and degrading the overall reconstruction quality.
To address this challenge, we propose a novel segmented architecture for enhanced hand pose estimation (EHPE).
We perform local extraction of TIP and wrist, thus alleviating the effect of error accumulation on TIP prediction and further reduce the predictive errors for all joints on this basis.
EHPE consists of two key stages: In the TIP and Wrist Joints Extraction stage (TW-stage), the positions of the TIP and wrist joints are estimated to provide an initial accurate joint configuration; In the Prior Guided Joints Estimation stage (PG-stage), a dual-branch interaction network is employed to refine the positions of the remaining joints. 
Extensive experiments on two widely used benchmarks demonstrate that EHPE achieves state-of-the-arts performance.
