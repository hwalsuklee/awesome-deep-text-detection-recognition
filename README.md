# awesome-deep-text-detection-recognition
A curated list of awesome deep learning based papers on text detection and recognition.

<p align='center'>
<img src = '/overall_pi_chart.png' height="300px">  
<img src = '/overall_histogram.png' height="450px">
</p>

## Text Detection
* Papers are sorted by published date.
* IC is shorts for ICDAR.
* Score is F1-score for localization task.
  * (L) stands for score in [leader-board](http://rrc.cvc.uab.es/).
  * If the reported score in leader-board is somewhat different from the paper, (L) is provided.
* `*CODE` means official code and `CODE(M)` means that traiend model is provided.  

*Conf.* | *Date* | *Title* | *IC13* | *IC15* | *Resources* |
:---: | :---: |:--- | :---: | :---: | :---: |
'14-ECCV	| 14/10/07	| [Robust Scene Text Detection with Convolution Neural Network Induced MSER Trees](	http://www.whuang.org/papers/whuang2014_eccv.pdf) |
15-CVPR	| 15/06/01	| [Symmetry-based text line detection in natural scenes](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Symmetry-Based_Text_Line_2015_CVPR_paper.pdf) | [0.8043](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=2197) | | [`PRJ`](http://mclab.eic.hust.edu.cn/~xbai/text/symmetry/SymmetryTextLineDetection.html) <br> [`CODE`](https://github.com/stupidZZ/Symmetry_Text_Line_Detection) |  
'16-TIP	| 15/10/12 | [Text-Attentional Convolutional Neural Networks for Scene Text Detection](https://arxiv.org/pdf/1510.03283.pdf) |	[0.8165](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=7187)	| 
'15-ICCV	| 15/12/13	| [Text Flow : A Unified Text Detection System in Natural Scene Images](https://pdfs.semanticscholar.org/11a0/8ced22775a217ba78c566528ed44ea98e3e3.pdf)	|0.8025 | 
'16-arXiv	| 16/03/31	| [Accurate Text Localization in Natural Image with Cascaded Convolutional TextNetwork](https://arxiv.org/pdf/1603.09423.pdf)	| 0.86	| |
'16-CVPR	| 16/04/14	| [Multi-Oriented Text Detection with Fully Convolutional Networks](https://arxiv.org/pdf/1604.04018.pdf) | 0.83 | 0.54 | [`*TORCH(M)`](https://github.com/stupidZZ/FCN_Text)
'16-CVPR	| 16/04/22	| [Synthetic Data for Text Localisation in Natural Images](http://www.robots.ox.ac.uk/~ankush/textloc.pdf)	| 0.847 <br> (L)[0.8359](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=3820) | | [`CODE`](https://github.com/ankush-me/SynthText) <br> [`DB`](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
'16-arXiv	| 16/06/29	| [Scene Text Detection Via Holistic, Multi-Channel Prediction](https://arxiv.org/pdf/1606.09002.pdf)	|0.8433	| 0.6477 | 
'16-ECCV	| 16/09/12	| [Detecting Text in Natural Image with Connectionist Text Proposal Network](https://arxiv.org/pdf/1609.03605.pdf) |	[0.8215](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=13931)	| [0.6085](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=13930) | [`*CAFFE(M)`](https://github.com/tianzhi0549/CTPN) <br> [`CAFFE`](https://github.com/qingswu/CTPN) <br> [`TF(M)`](https://github.com/eragonruan/text-detection-ctpn) <br> [`TF`](https://github.com/Li-Ming-Fan/OCR-DETECTION-CTPN) <br> [`DEMO`](http://textdet.com/) <br> [`BLOG(CH)`](http://slade-ruan.me/2017/10/22/text-detection-ctpn/)
'17-AAAI	| 16/11/21	|[TextBoxes: A fast text detector with a single deep neural network](https://arxiv.org/pdf/1611.06779.pdf) |	0.85 <br> (L)[0.8767](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=21928) | | [`*CAFFE(M)`](https://github.com/MhLiao/TextBoxes) <br> [`TF`](https://github.com/shinjayne/shinTB) <br> [`BLOG(KR)`](http://jaynewho.com/post/6)
'18-TM	| 17/03/03	| [Arbitrary-Oriented Scene Text Detection via Rotation Proposals](https://arxiv.org/pdf/1703.01086.pdf)	|	[0.9125](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=15904)	| [0.8020](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=17393)	| [`*CAFFE`](https://github.com/mjq11302010044/RRPN)
'17-CVPR	| 17/03/04	| [Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection](https://arxiv.org/pdf/1703.01425.pdf) | | [0.7064](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=13007)
'17-CVPR	| 17/03/19	| [Detecting Oriented Text in Natural Images by Linking Segments](https://arxiv.org/pdf/1703.06520.pdf)	|	0.853	| 0.75	<br> (L)[0.7636](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=29245)| [`*TF(M)`](https://github.com/bgshih/seglink) <br> [`TF(M)`](https://github.com/dengdan/seglink) <br> [`SLIDE`](http://mclab.eic.hust.edu.cn/~xbai/SpotlightPPT/TextDetection-seglink-spotlight-CVPR17.pdf) <br> [`VIDEO`](https://www.youtube.com/watch?v=w0vZWUi-m0c) |
'17-arXiv	| 17/03/24 |	[Deep Direct Regression for Multi-Oriented Scene Text Detection](https://arxiv.org/pdf/1703.08289.pdf)	| 0.86 |  0.81	| 
'17-arXiv	| 17/04/03	| [Cascaded Segmentation-Detection Networks for Word-Level Text Spotting](https://arxiv.org/pdf/1704.00834.pdf)	|	0.86	| 0.71 |
'17-CVPR	| 17/04/11	| [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/pdf/1704.03155.pdf) | | 0.8072	<br> (L)[0.8038](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=29855) | [`TF(M)`](https://github.com/argman/EAST) <br> [`TF`](https://github.com/AKSHAYUBHAT/EAST) <br> [`PYTORCH(M)`](https://github.com/SakuraRiven/EAST) <br> [`PYTORCH`](https://github.com/songdejia/EAST) <br> [`DEMO`](http://east.zxytim.com/) <br> [`KERAS(M)`](https://github.com/kurapan/EAST) <br> [`VIDEO`](https://www.youtube.com/watch?v=o5asMTdhmvA)
'17-ICIP	| 17/05/15	| [WordFence: Text Detection in Natural Images with Border Awareness](https://arxiv.org/pdf/1705.05483.pdf) |	0.86 |
'17-arXiv	| 17/06/30	| [R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection](https://arxiv.org/pdf/1706.09579.pdf) | 0.8773	| 0.8254 | [`TF(M)`](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow) <br> [`CAFFE(M)`](https://github.com/beacandler/R2CNN)
'17-CVPR	| 17/07/21	| [Multi-scale FCN with Cascaded Instance Aware Segmentation for Arbitrary Oriented Word Spotting In The Wild](http://openaccess.thecvf.com/content_cvpr_2017/papers/He_Multi-Scale_FCN_With_CVPR_2017_paper.pdf)	|	0.85	| 0.63 | 
'17-arXiv	| 17/08/17	| [Deep Scene Text Detection with Connected Component Proposals](https://arxiv.org/pdf/1708.05133.pdf)	| 0.919 | 
'17-ICCV	| 17/08/22	| [WordSup: Exploiting Word Annotations for Character based Text Detection](https://arxiv.org/pdf/1708.06720) |	0.9064 |	0.7816 |
'17-ICCV	| 17/09/01	| [Single Shot Text Detector with Regional Attention](https://arxiv.org/pdf/1709.00138.pdf)	|	0.8704 |	[0.7691](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=18294) | [`*CAFFE(M)`](https://github.com/BestSonny/SSTD) <br> [`PYTORCH`](https://github.com/HotaekHan/SSTDNet) <br> [`VIDEO`](https://www.youtube.com/watch?v=oBWVgz685-k) 
'17-arXiv	| 17/09/11	| [Fused Text Segmentation Networks for Multi-oriented Scene Text Detection](https://arxiv.org/pdf/1709.03272.pdf) | | [0.8414](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=30447)	|
'17-ICCV	| 17/10/13	| [WeText: Scene Text Detection under Weak Supervision](https://arxiv.org/pdf/1710.04826.pdf)	| 0.869 <br> (L)[0.8313](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=20665) |
'17-ICCV	| 17/10/22	| [Self-organized Text Detection with Minimal Post-processing via Border Learning](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Self-Organized_Text_Detection_ICCV_2017_paper.pdf)	| 0.84 | | [`*KERAS(M)`](https://gitlab.com/rex-yue-wu/ISI-PPT-Text-Detector)
'17-ICDAR	| 17/11/11	| [Deep Residual Text Detection Network for Scene Text](https://arxiv.org/pdf/1711.04147.pdf)	|	0.9117 <br> (L)[0.8925](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=21966) | 
'18-AAAI	| 17/11/12	| [Feature Enhancement Network: A Refined Scene Text Detector](https://arxiv.org/pdf/1711.04249.pdf)	| [0.9161](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=28362) |
'17-arXiv	| 17/11/30	| [ArbiText: Arbitrary-Oriented Text Detection in Unconstrained Scene](https://arxiv.org/pdf/1711.11249.pdf)	|	| 0.759 |	
'18-AAAI	| 18/01/04	| [PixelLink: Detecting Scene Text via Instance Segmentation](https://arxiv.org/pdf/1801.01315.pdf)	|	0.881 | [0.8519](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=30576)	| [`*TF(M)`](https://github.com/ZJULearning/pixel_link) [`TF`](https://github.com/BowieHsu/tensorflow_ocr)
'18-CVPR	| 18/01/05	| [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/pdf/1801.01671.pdf)	| [0.925](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=1&m=34624)	| [0.8984](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=34625)	|	[`PYTORCH`](https://github.com/jiangxiluning/FOTS.PyTorch) <br> [`PYTORCH`](https://github.com/xieyufei1993/FOTS) <br> [`VIDEO`](https://www.youtube.com/watch?v=F7TTYlFr2QM) | 
'18-TIP	| 18/01/09	| [TextBoxes++: A Single-Shot Oriented Scene Text Detector](https://arxiv.org/pdf/1801.02765.pdf)	| 0.88	| 0.829 <br> (L)[0.8475](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=29660) | [`*CAFFE(M)`](https://github.com/MhLiao/TextBoxes_plusplus)
'18-CVPR	| 18/02/27	| [Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation](https://arxiv.org/pdf/1802.08948.pdf)	| 0.88	| 0.843 |[`*PYTORCH(M)`](https://github.com/lvpengyuan/corner)
'18-CVPR	| 18/03/09	| [An end-to-end TextSpotter with Explicit Alighment and Attention](https://arxiv.org/pdf/1803.03474.pdf)	| 0.9	| 0.87 |[`*CAFFE(M)`](https://github.com/tonghe90/textspotter)
'18-CVPR	| 18/03/14	| [Rotation-Sensitive Regression for Oriented Scene Text Detection](https://arxiv.org/pdf/1803.05265.pdf)	| 0.89	| 0.838 |	[`*CAFFE(M)`](https://github.com/MhLiao/RRD)
'18-arXiv	| 18/04/08	| [Detecting Multi-Oriented Text with Corner-based Region Proposals](https://arxiv.org/pdf/1804.02690.pdf) | 0.876	| 0.845 | [`*CAFFE(M)`](https://github.com/xhzdeng/crpn)
'18-arXiv	| 18/04/24	| [An Anchor-Free Region Proposal Network for Faster R-CNN based Text Detection Approaches](https://arxiv.org/pdf/1804.09003.pdf)	| 0.92	| 0.86	|
'18-IJCAI	| 18/05/03	| [IncepText: A New Inception-Text Module with Deformable PSROI Pooling for Multi-Oriented Scene Text Detection](https://arxiv.org/pdf/1805.01167.pdf)	| | [0.9047](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=34989) |	
'18-arXiv	| 18/06/07	| [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/pdf/1806.02559.pdf) | |		[0.8721](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=37493)	| [`PRJ`](https://github.com/whai362/PSENet)
'18-ECCV	| 18/07/04	| [TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](https://arxiv.org/pdf/1807.01544.pdf) | |	0.826	| [`PYTORCH`](https://github.com/princewang1994/TextSnake.pytorch)
'18-ECCV	| 18/07/06	| [Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes](https://arxiv.org/pdf/1807.02242.pdf) | 0.917	| 0.86 |
'18-ECCV	| 18/07/10	| [Accurate Scene Text Detection through Border Semantics Awareness and Bootstrapping](https://arxiv.org/pdf/1807.03547.pdf) |	0.892	|
'19-AAAI    | 18/11/21  | [Scene Text Detection with Supervised Pyramid Context Network](https://arxiv.org/pdf/1811.08605.pdf)  | 0.921 | 0.872 | 
'19-TIP | 18/12/04 | [TextField: Learning A Deep Direction Field for Irregular Scene Text Detection](https://arxiv.org/pdf/1812.01393.pdf) | | 0.824 | [`*CAFFE(M)`](https://github.com/YukangWang/TextField)
'19-CVPR | 19/03/21 | [Towards Robust Curve Text Detection with Conditional Spatial Expansion](https://arxiv.org/pdf/1903.08836.pdf) | | | |  
'19-CVPR | 19/03/28 | [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/pdf/1903.12473.pdf) | | 0.857 | [`TF(M)`](https://github.com/liuheng92/tensorflow_PSENet)
'19-CVPR | 19/04/03 | [Character Region Awareness for Text Detection](https://arxiv.org/pdf/1904.01941.pdf) | 0.952 | 0.869 |[`*PYTORCH(M)`](https://github.com/clovaai/CRAFT-pytorch) <br> [`VIDEO`](https://www.youtube.com/watch?v=HI8MzpY8KMI) <br> [`PYTORCH`](https://github.com/guruL/Character-Region-Awareness-for-Text-Detection-) <br> [`TF(M)`](https://github.com/namedysx/CRAFT-tensorflow) <br> [`KERAS`](https://github.com/RubanSeven/CRAFT_keras) <br> [`BLOG_CH`](https://medium.com/@xiaosean5408/craft簡介-character-region-awareness-for-text-detection-a5c782408f00) <br> [`BLOG_KR`](https://data-newbie.tistory.com/187) <br> [`BLOG_KR`](https://medium.com/qandastudy/character-region-awareness-for-text-detection-craft-review-a7542779e037) <br> [`BLOG_KR`](https://github.com/chullhwan-song/Reading-Paper/issues/136)|  
'19-CVPR | 19/04/13 | [Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes Screen reader support enabled](https://arxiv.org/pdf/1904.06535.pdf) | | 0.877 | |  
'19-CVPR | 19/06/16 | [Learning Shape-Aware Embedding for Scene Text Detection](http://jiaya.me/papers/textdetection_cvpr19.pdf) | | 0.877 | |  
'19-CVPR | 19/06/16 | [Arbitrary Shape Scene Text Detection with Adaptive Text Region Representation](https://arxiv.org/pdf/1905.05980.pdf) | 0.917 | 0.876 | |  
'19-ICCV | 19/08/16 | [Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/abs/1908.05900) | | 0.829 | |  
'19-ICCV | 19/09/02 | [Geometry Normalization Networks for Accurate Scene Text Detection](https://arxiv.org/abs/1909.00794) | | 0.8852 | |  
'19-AAAI | 19/11/20 | [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947) | | 0.847 | |  
<p align='center'>
<img src = '/detection_ic13_results.png' height = '550px'>
<img src = '/detection_ic15_results.png' height = '550px'>
</p>

## Text Recognition
* Papers are sorted by published date.
* IC is shorts for ICDAR.
* Score is word-accuracy for recognition task.
  * For results on IC03, IC13, and IC15 dataset, papers used different numbers of samples per paper,  
  but we did not distinguish between them
* `*CODE` means official code and `CODE(M)` means that trained model is provided.  

*Conf.* | *Date* | *Title* | *SVT* | *IIIT5k* | *IC03* | *IC13* | *Resources* |
:---: | :---: |:--- | :---: | :---: | :---: | :---: | :---: |
'15-ICLR	| 14/12/18	| [Deep structured output learning for unconstrained text recognition](https://arxiv.org/pdf/1412.5903.pdf) |	0.717 | |0.896	| 0.818 | [`TF`](https://github.com/AlexandreSev/Structured_Data) <br> [`SLIDE`](https://www.robots.ox.ac.uk/~vgg/publications/2015/Jaderberg15a/presentation.pdf) <br> [`VIDEO`](https://www.youtube.com/watch?v=NYkG38RCoRg)
'16-IJCV	| 15/05/07	| [Reading text in the wild with convolutional neural networks](https://arxiv.org/pdf/1412.1842.pdf)	| 0.807	|	| 	0.933	| 0.908 | [`KERAS`](https://github.com/mathDR/reading-text-in-the-wild)
'16-AAAI	| 15/06/14	| [Reading Scene Text in Deep Convolutional Sequences](https://arxiv.org/pdf/1506.04395.pdf)
'17-TPAMI	| 15/07/21	| [An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)	| 0.808	| 0.782 | 0.894	| 0.867 | [`TORCH(M)`](https://github.com/bgshih/crnn)  <br> [`TF`](https://github.com/weinman/cnn_lstm_ctc_ocr)  <br> [`TF`](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow) <br> [`TF`](https://github.com/MaybeShewill-CV/CRNN_Tensorflow) <br> [`TF`](https://github.com/bai-shang/OCR_TF_CRNN_CTC) <br> [`PYTORCH`](https://github.com/meijieru/crnn.pytorch)  <br> [`PYTORCH(M)`](https://github.com/BelBES/crnn-pytorch) <br> [`BLOG(KR)`](https://medium.com/@mldevhong/%EB%85%BC%EB%AC%B8-%EB%B2%88%EC%97%AD-rcnn-an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-f6456886d6f8)
'16-CVPR	| 16/03/09  | [Recursive Recurrent Nets with Attention Modeling for OCR in the Wild](https://arxiv.org/pdf/1603.03101.pdf) |	0.807 | 0.784	| 0.887	| 0.9 |
'16-CVPR	| 16/03/12	| [Robust scene text recognition with automatic rectification](https://arxiv.org/pdf/1603.03915.pdf) |	0.819	| 0.819 | 0.901	| 0.886 | [`PYTORCH`](https://github.com/marvis/ocr_attention) <br> [`PYTORCH`](https://github.com/WarBean/tps_stn_pytorch) 
'16-CVPR	| 16/06/27	| [CNN-N-Gram for Handwriting Word Recognition](https://www.cs.tau.ac.il/~wolf/papers/CNNNGram.pdf) |	0.8362 | | | | [`VIDEO`](https://www.youtube.com/watch?v=czc2Ipm3Bis)
'16-BMVC	| 16/09/19	| [STAR-Net: A SpaTial Attention Residue Network for Scene Text Recognition](http://www.visionlab.cs.hku.hk/publications/wliu_bmvc16.pdf) |	0.836	| 0.833	|	0.899	| 0.891 |
'17-arXiv	| 17/07/27	| [STN-OCR: A single Neural Network for Text Detection and Text Recognition](https://arxiv.org/pdf/1707.08831.pdf) | 0.798	| 0.86	| |	0.903 | [`*MXNET(M)`](https://github.com/Bartzi/stn-ocr) <br> [`PRJ`](https://bartzi.de/research/stn-ocr) <br> [`BLOG`](https://medium.com/@Synced/stn-ocr-a-single-neural-network-for-text-detection-and-text-recognition-220debe6ded4)
'17-IJCAI	| 17/08/19	| [Learning to Read Irregular Text with Attention Mechanisms](https://faculty.ist.psu.edu/zzhou/paper/IJCAI17-IrregularText.pdf) |
'17-arXiv	| 17/09/06	| [Scene Text Recognition with Sliding Convolutional Character Models](https://arxiv.org/pdf/1709.01727.pdf) |	0.765	| 0.816	| 0.845 |	0.852 |
'17-ICCV	| 17/09/07	| [Focusing Attention: Towards Accurate Text Recognition in Natural Images](https://arxiv.org/pdf/1709.02054.pdf) |	0.859 | 0.874 | 0.942 |	0.933 |
'18-CVPR	| 17/11/12	| [AON: Towards Arbitrarily-Oriented Text Recognition](https://arxiv.org/pdf/1711.04226.pdf) |	0.828	|0.87	| 0.915	||[`TF`](https://github.com/huizhang0110/AON)
'17-NIPS	| 17/12/04	| [Gated Recurrent Convolution Neural Network for OCR](https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)	| 0.815	| 0.808 | 0.978 | | [`*TORCH(M)`](https://github.com/Jianfeng1991/GRCNN-for-OCR)
'18-AAAI	| 18/01/04	| [Char-Net: A Character-Aware Neural Network for Distorted Scene Text Recognition](http://www.visionlab.cs.hku.hk/publications/wliu_aaai18.pdf) |	0.844	| 0.836	| 0.915 |	0.908 |
'18-AAAI	| 18/01/04	| [SqueezedText: A Real-time Scene Text Recognition by Binary Convolutional Encoder-decoder Network](https://pdfs.semanticscholar.org/0e59/f7d7e9c9380b425a94038c7a2500b2f6063a.pdf) | |	0.87	| 0.931 |	0.929 |
'18-CVPR	| 18/05/09  | [Edit Probability for Scene Text Recognition](https://arxiv.org/pdf/1805.03384.pdf) | 0.875 | 0.883 | 0.946 | 0.944 |
'18-TPAMI	| 18/06/25	| [ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](http://122.205.5.5:8071/UpLoadFiles/Papers/ASTER_PAMI18.pdf) |	0.936	| 0.934 |	0.945	| 0.918	| [`*TF(M)`](https://github.com/bgshih/aster) <br> [`PYTORCH`](https://github.com/ayumiymk/aster.pytorch)
'18-ECCV	| 18/09/08	| [Synthetically Supervised Feature Learning for Scene Text Recognition](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Liu_Synthetically_Supervised_Feature_ECCV_2018_paper.pdf) |	0.871	| 0.894	| 	0.947	| 0.94 |
'19-AAAI    | 18/09/18  | [Scene Text Recognition from Two-Dimensional Perspective](https://arxiv.org/pdf/1809.06508.pdf)   | 0.821 | 0.92 | | 0.914 |
'19-AAAI    | 18/11/02  | [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/pdf/1811.00751.pdf)   | 0.845 | 0.915 | | 0.91 | [`*TORCH(M)`](https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition)  
'19-CVPR    | 18/12/14  | [ESIR: End-to-end Scene Text Recognition via Iterative Image Rectification](https://arxiv.org/pdf/1812.05824.pdf)   | 0.902 | 0.933 |  | 0.913 | [PRJ](https://github.com/fnzhan/ESIR)  
'19-PR    | 19/01/10  | [MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition](https://arxiv.org/pdf/1901.03003.pdf)   | 0.883 | 0.912 | 0.950 | 0.924 | [`*PYTORCH(M)`](https://github.com/Canjie-Luo/MORAN_v2)  
'19-ICCV | 19/04/03 | [What is wrong with scene text recognition model comparisons? dataset and model analysis](https://arxiv.org/pdf/1904.01906.pdf) | 0.875 | | 0.949 | 0.936 | [`*PYTORCH(M)`](https://github.com/clovaai/deep-text-recognition-benchmark) <br> [`BLOG_KR`](https://data-newbie.tistory.com/156)
'19-CVPR | 19/04/18 | [Aggregation Cross-Entropy for Sequence Recognition](https://arxiv.org/pdf/1904.08364.pdf) | 0.826 | 0.823 | 0.921 | 0.897 | [`*PYTORCH`](https://github.com/summerlvsong/Aggregation-Cross-Entropy) | 
'19-CVPR | 19/06/16 | [Sequence-to-Sequence Domain Adaptation Network for Robust Text Image Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Sequence-To-Sequence_Domain_Adaptation_Network_for_Robust_Text_Image_Recognition_CVPR_2019_paper.pdf) | 0.845 | 0.838 | 0.921 | 0.918 | |
'19-ICCV | 19/08/06 | [Symmetry-constrained Rectification Network for Scene Text Recognition](https://arxiv.org/abs/1908.01957) | 0.889 | 0.944 | 0.95  | 0.939 |  
'20-AAAI | 19/12/28 | [TextScanner: Reading Characters in Order for Robust Scene Text Recognition](https://arxiv.org/abs/1912.12422) | 0.895 | 0.926 |       | 0.925 |  
'20-AAAI | 19/12/21 | [Decoupled Attention Network for Text Recognition](https://arxiv.org/abs/1912.10205) | 0.892 | 0.943 | 0.95  | 0.939 | [`*PYTORCH(M)`](https://github.com/Wang-Tianwei/Decoupled-attention-network)  
'20-AAAI | 20/02/04 | [GTC: Guided Training of CTC](https://arxiv.org/abs/2002.01276) | 0.929 | 0.955 | 0.952 | 0.943 |  

<p align='center'>
<img src = '/recognition_ic13_results.png' height = '550px'>
<img src = '/recognition_iiit5k_results.png' height = '550px'>
</p>

## End-to-End Text Recognition
* Papers are sorted by published date.
* IC is shorts for ICDAR.
* Score is F1-score for generic task. 
    * (L) stands for score in [leader-board](http://rrc.cvc.uab.es/).
* `*CODE` means official code and `CODE(M)` means that trained model is provided.

*Conf.* | *Date* | *Title* | *IC03* | *IC13* | *IC15* | *Resources* |
:---: | :---: |:--- | :---: | :---: | :---: | :---: |
'12-ICPR    | 12/11/11  | [End-to-end text recognition with convolutional neural networks](https://ai.stanford.edu/~ang/papers/ICPR12-TextRecognitionConvNeuralNets.pdf)    | 0.67 | | | [`*CODE`](http://cs.stanford.edu/people/twangcat/ICPR2012_code/SceneTextCNN_demo.tar)
'14-ECCV    | 14/09/06  | [Deep Features for Text Spotting](https://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14/jaderberg14.pdf) | 0.75  | | | [`PRJ`](http://www.robots.ox.ac.uk/~vgg/research/text/) <br> [`MATLAB`](https://bitbucket.org/jaderberg/eccv2014_textspotting)
'15-IJCV    | 15/05/07  | [Reading Text in the Wild with Convolutional Neural Networks](https://arxiv.org/pdf/1412.1842.pdf)    | 0.70  | 0.77 | | [`KERAS`](https://github.com/mathDR/reading-text-in-the-wild)
'15-TPAMI   | 15/10/30  | [Real-time Lexicon-free Scene Text Localization and Recognition](http://cmp.felk.cvut.cz/~neumalu1/Neumann_TPAMI2015.pdf) | | 0.542 | 0.156 |
'16-arXiv   | 16/04/10  | [TextProposals: a Text-specific Selective Search Algorithm for Word Spotting in the Wild](https://arxiv.org/pdf/1604.02619.pdf) | | 0.6843 | 0.4718 <br> (L)[0.533](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=4&m=7807) | [`*CAFFE(M)`](https://github.com/lluisgomez/TextProposals)
'17-AAAI    | 16/11/21  | [TextBoxes: A fast text detector with a single deep neural network](https://arxiv.org/pdf/1611.06779.pdf) | | 0.84 | | [`TF`](https://github.com/shinjayne/shinTB) <br> [`*CAFFE(M)`](https://github.com/MhLiao/TextBoxes) <br> [`BLOG_KR`](http://jaynewho.com/post/6)
'17-ICCV    | 17/07/13  | [Towards End-to-end Text Spotting with Convolution Recurrent Neural Network](https://arxiv.org/pdf/1707.03985.pdf) | | 0.8459 | | [`VIDEO`](https://www.youtube.com/watch?v=j0guWqBJ0lA)
'17-ICCV    | 17/10/22  | [Deep TextSpotter An End-to-End Trainable Scene Text Localization and Recognition Framework](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busta_Deep_TextSpotter_An_ICCV_2017_paper.pdf) | | 0.77 | 0.47 | [`VIDEO`](https://www.youtube.com/watch?v=VcNSQGO0j7s) <br> [`*CAFFE(M)`](https://github.com/MichalBusta/DeepTextSpotter)
'18-CVPR    | 18/01/05  | [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/pdf/1801.01671.pdf) | | [0.8477](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=4&m=34627) | [0.6533](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=4&m=34626) | [`VIDEO`](https://www.youtube.com/watch?v=F7TTYlFr2QM) <br> [`TF(M)`](https://github.com/Pay20Y/FOTS_TF)
'18-TIP     | 18/01/09  | [TextBoxes++: A Single-Shot Oriented Scene Text Detector](https://arxiv.org/pdf/1801.02765.pdf) | | [0.8465](http://rrc.cvc.uab.es/?ch=2&com=evaluation&view=method_info&task=4&m=27895) | 0.519 | [`*CAFFE(M)`](https://github.com/MhLiao/TextBoxes_plusplus)
'18-CVPR    | 18/03/09  | [An end-to-end TextSpotter with Explicit Alignment and Attention](https://arxiv.org/pdf/1803.03474.pdf) | | 0.86 | 0.63 | [`*CAFFE(M)`](https://github.com/tonghe90/textspotter)
'18-TPAMI   | 18/06/25  | [ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](http://122.205.5.5:8071/UpLoadFiles/Papers/ASTER_PAMI18.pdf) | | | 0.64 | [`*TF(M)`](https://github.com/bgshih/aster)
'18-ECCV    | 18/07/06  | [Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes](https://arxiv.org/pdf/1807.02242.pdf) | | 0.865 | 0.624 |
'19-ICCV    | 19/08/24  | [Towards Unconstrained End-to-End Text Spotting](https://arxiv.org/pdf/1908.09231.pdf) | | | 0.6994 | [`BLOG_KR`](https://www.notion.so/Towards-Unconstrained-End-to-End-Text-Spotting-0c66c692950f458e9a1323db2a79d143)  
'19-ICCV    | 19/10/17  | [Convolutional Character Networks](https://arxiv.org/abs/1910.07954) | | | 0.7108 | [`*PYTORCH(M)`](https://github.com/MalongTech/research-charnet)  
'19-ICCV    | 19/10/27  | [TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf) | | | 0.6537 |  
'20-AAAI    | 19/11/21  | [All You Need Is Boundary: Toward Arbitrary-Shaped Text Spotting](https://arxiv.org/pdf/1911.09550) | | 0.841 | 0.641 |  
'20-AAAI    | 20/02/12  | [Text Perceptron: Towards End-to-End Arbitrary-Shaped Text Spotting](https://arxiv.org/abs/2002.06820) | | 0.858 | 0.651 |  
<p align='center'>
<img src = '/end2end_ic13_ic15_results.png' height = '400px'>
</p>

## Others
* Papers are sorted by published date.
* `*CODE` means official code and `CODE(M)` means that trained model is provided.

*Conf.* | *Date* | *Title* | *Description* | *Resources* |
:---: | :---: |:--- | :---: | :---: |
'14-NIPS	| 14/06/09  |   [Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition](https://arxiv.org/pdf/1406.2227.pdf) |	Dataset | [`PRJ`](http://www.robots.ox.ac.uk/~vgg/data/text/)
'17-ECCV	| 17/02/13  |	[End-to-End Interpretation of the French Street Name Signs Dataset](https://arxiv.org/pdf/1702.03970.pdf) |	Dataset (FSNS) | [`*TF(M)`](https://github.com/tensorflow/models/tree/master/research/attention_ocr)
'17-arXiv	| 17/04/11	|	[Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/pdf/1704.03549.pdf) |	FSNS | [`*TF(M)`](https://github.com/tensorflow/models/tree/master/research/attention_ocr) <br> [`TF`](https://github.com/da03/Attention-OCR) <br> [`TF`](https://github.com/emedvedev/attention-ocr) <br> [`LUA`](https://github.com/da03/torch-Attention-OCR) <br> [`BLOG_KR`](https://norman3.github.io/papers/docs/attention_ocr.html)
'17-CVPR	| 17/07/21	|	[Unambiguous Text Localization and Retrieval for Cluttered Scenes](http://openaccess.thecvf.com/content_cvpr_2017/papers/Rong_Unambiguous_Text_Localization_CVPR_2017_paper.pdf) |	Text Retrieval
'17-AAAI	| 17/10/22	|	[Detection and Recognition of Text Embedded in Online Images via Neural Context Models](http://s-space.snu.ac.kr/bitstream/10371/116866/1/aaai2017_cameraready.pdf) |	Dataset | [`PRJ`](https://github.com/cmkang/CTSN)
'18-CVPR	| 17/11/17	|	[Separating Style and Content for Generalized Style Transfer](https://arxiv.org/pdf/1711.06454.pdf) |	Font Style
'17-arXiv	| 17/12/06	|	[Detecting Curve Text in the Wild New Dataset and New Solution](https://arxiv.org/pdf/1712.02170.pdf) |	Dataset (CTW 1500) | [`PRJ`](https://github.com/Yuliang-Liu/Curve-Text-Detector)
'18-AAAI	| 17/12/14	|	[SEE: Towards Semi-Supervised End-to-End Scene Text Recognition](https://arxiv.org/pdf/1712.05404.pdf) |	FSNS | [`PRJ`](https://bartzi.de/research/see) <br> [`*CHAINER(M)`](https://github.com/Bartzi/see)
'17-CVPR	| 18/06/07	|	[Learning to Extract Semantic Structure from Documents Using Multimodal Fully Convolutional Neural Networks](https://arxiv.org/pdf/1706.02337.pdf) |	Document Layout | [`PRJ`](http://personal.psu.edu/xuy111/projects/cvpr2017_doc.html)
'18-CVPR	| 18/06/19	|	[DocUNet: Document Image Unwarping via A Stacked U-Net](http://www.juew.org/publication/DocUNet.pdf) |	Document Dewarping | [`PRJ`](http://www3.cs.stonybrook.edu/~cvl/docunet.html)
'18-CVPR	| 18/06/19	|	[Document Enhancement using Visibility Detection](http://webee.technion.ac.il/~ayellet/Ps/18-KKT.pdf) |	Document Enhancement | [`PRJ`](http://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/VisibilityDetection/)
'18-IJCAI	| 18/06/22	|	[Multi-Task Handwritten Document Layout Analysis](https://arxiv.org/pdf/1806.08852.pdf) |	Document Layout
'18-ECCV	| 18/07/09	|	[Verisimilar Image Synthesis for Accurate Detection and Recognition of Texts in Scenes](https://arxiv.org/pdf/1807.03021.pdf) |	Dataset | [`PRJ`](https://github.com/fnzhan/Verisimilar-Image-Synthesis-for-Accurate-Detection-and-Recognition-of-Texts-in-Scenes)
'19-AAAI	| 18/12/03  |	[EnsNet: Ensconce Text in the Wild](https://arxiv.org/pdf/1812.00723.pdf) | Text Removal |	[`DB`](https://github.com/HCIILAB/Scene-Text-Removal)  
'19-CVPR	| 18/12/14  |	[Spatial Fusion GAN for Image Synthesis](https://arxiv.org/pdf/1812.05840.pdf) | Dataset |	[`DB`](https://github.com/fnzhan/SF-GAN)  
'19-AAAI	| 19/01/27  |	[Hierarchical Encoder with Auxiliary Supervision for Table-to-text Generation: Learning Better Representation for Tables](https://www.aaai.org/Papers/AAAI/2019/AAAI-LiuT.3205.pdf) | TableToText |	  
'19-AAAI	| 19/01/27  |	[A Radical-aware Attention-based Model for Chinese Text Classification](https://www.aaai.org/Papers/AAAI/2019/AAAI-TaoH.5441.pdf) | Chinese Character Classification |	  
'19-CVPR | 19/02/25 | [Handwriting Recognition in Low-resource Scripts using Adversarial Learning](https://arxiv.org/pdf/1811.01396.pdf) | Handwritting Recognition | [`TF`](https://github.com/AyanKumarBhunia/Handwriting_Recogition_using_Adversarial_Learning)
'19-CVPR	| 19/03/27  |	[Tightness-aware Evaluation Protocol for Scene Text Detection](https://arxiv.org/pdf/1904.00813.pdf) | Evaluation |	[`CODE`](https://github.com/Yuliang-Liu/TIoU-metric)  
'19-ICCV | 19/05/31 | [Scene Text Visual Question Answering](https://arxiv.org/pdf/1905.13648v2) | Dataset | [`ICDAR_DB`](https://rrc.cvc.uab.es/?ch=11)
'19-CVPR	| 19/06/16  |	[DynTypo: Example-based Dynamic Text Effects Transfer](https://menyifang.github.io/projects/DynTypo/DynTypo_files/Paper_DynTypo_CVPR19.pdf) | Text Effects |	[`PRJ`](https://menyifang.github.io/projects/DynTypo/DynTypo.html) <br> [`VIDEO`](https://youtu.be/FkFQ6bV1s-o) 
'19-CVPR | 19/06/16 | [Typography with Decor: Intelligent Text Style Transfer](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Typography_With_Decor_Intelligent_Text_Style_Transfer_CVPR_2019_paper.pdf) | Text Effects | [`*PYTORCH(M)`](https://daooshee.github.io/Typography2019/) 
'19-CVPR | 19/06/16 | [An Alternative Deep Feature Approach to Line Level Keyword Spotting](http://openaccess.thecvf.com/content_CVPR_2019/papers/Retsinas_An_Alternative_Deep_Feature_Approach_to_Line_Level_Keyword_Spotting_CVPR_2019_paper.pdf) | Kyeword Spotting
'19-ICCV | 19/07/23 | [GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition](https://arxiv.org/abs/1907.09653) | Domain Adaptation |  
'19-ICCV | 19/09/17 | [Chinese Street View Text: Large-scale Chinese Text Reading with Partially Supervised Learning](https://arxiv.org/abs/1909.07808) | Dataset | [`ICDAR_DB`](https://rrc.cvc.uab.es/?ch=16)
'19-ICCV | 19/10/02 | [Large-scale Tag-based Font Retrieval with Generative Feature Learning](https://arxiv.org/pdf/1909.02072.pdf) | Font Retrieval |  
'19-ICCV | 19/10/27 | [TextPlace: Visual Place Recognition and Topological Localization Through Reading Scene Texts](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hong_TextPlace_Visual_Place_Recognition_and_Topological_Localization_Through_Reading_Scene_ICCV_2019_paper.pdf) | Place Recognition | [`DB`](https://github.com/ziyanghong/dataset)
'19-ICCV | 19/10/27 | [DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_ICCV_2019_paper.pdf) | Document Dewarping | [`*PYTORCH(M)`](https://github.com/cvlab-stonybrook/DewarpNet)

## Other lists
* OCR Paper Curation
    * [HCIILAB-Detection](https://github.com/HCIILAB/Scene-Text-Detection)
    * [HCIILAB-Recognition](https://github.com/HCIILAB/Scene-Text-Recognition)
    * [HCIILAB-End2End](https://github.com/HCIILAB/Scene-Text-End2end)
    * [whitelok](https://github.com/whitelok/image-text-localization-recognition)
    * [tangzhenyu](https://github.com/tangzhenyu/Scene-Text-Understanding)
    * [wanghaisheng](https://github.com/wanghaisheng/awesome-ocr)
    * [ChanChiChoi](https://github.com/ChanChiChoi/awesome-ocr/blob/master/README.md)
    * [handong1587](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-ocr.md)
    * [hs105](https://github.com/hs105/Deep-Learning-for-OCR)

## Tutorial Materials
* Lecture slides
    * [Irregular Text Detection and Recognition (CBDAR2019 keynote)](http://122.205.5.5:8071/~xbai/Talk_slice/IrregularText-CBDAR2019.pptx)
    * [Deep Neural Networks for Scene Text Reading (IC17 Keynote)](http://u-pat.org/ICDAR2017/keynotes/ICDAR2017_Keynote_Prof_Bai.pdf)
    * [Oriented Scene Text Detection Revisited (VALSE17 Invited Talk)](http://cloud.eic.hust.edu.cn:8071/~xbai/Talk_slice/Oriented-Scene-Text-Detection-Revisited_VALSE2017.pdf)
    * [Scene Text Detection and Recognition (Joint course of Megvii Inc. & Peking Univ.)](https://zsc.github.io/megvii-pku-dl-course/slides/Lecture7(Text%20Detection%20and%20Recognition_20171031).pdf)
    * [Classic Text Detectors](https://www.slideshare.net/anyline_io/text-detection-strategies)
* Survey Paper
    * [Scene text detection and recognition: recent advances and future trends](https://www.researchgate.net/profile/Xiang_Bai4/publication/286945604_Scene_text_detection_and_recognition_recent_advances_and_future_trends/links/57f720f408ae886b8981d364/Scene-text-detection-and-recognition-recent-advances-and-future-trends.pdf)

## Acknowledgment
* This work is done by OCR team in Clova AI powered by NAVER-LINE. NAVER-LINE is an Asian top internet company and develops Clova, a cloud-based AI-assistant platform.
* This repository is scheduled to be updated regularly in accordance with schedules of major AI conferences.
