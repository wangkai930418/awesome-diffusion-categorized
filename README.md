<!-- The superlink doesn't support uppercases -->

# Awesome Diffusion Categorized

## Contents

<!-- - [Accelerate](#accelerate) -->
- [Image Restoration](#image-restoration)
- [Colorization](#colorization)
- [Face Restoration](#face-restoration)
- [Storytelling](#storytelling)
- [Virtual Try On](#try-on)
- [Drag Edit](#drag-edit)
- [Diffusion Inversion](#diffusion-models-inversion)
- [Text-Guided Editing](#text-guided-image-editing)
- [Continual Learning](#continual-learning)
- [Remove Concept](#remove-concept)
- [New Concept Learning](#new-concept-learning)
- [T2I augmentation](#t2i-diffusion-model-augmentation)
- [Spatial Control](#spatial-control)
- [Image Translation](#i2i-translation)
- [Seg & Detect & Track](#segmentation-detection-tracking)
- [Adding Conditions](#additional-conditions)
- [Few-Shot](#few-shot)
- [Inpainting](#sd-inpaint)
- [Layout](#layout-generation)
- [Text Generation](#text-generation)
- [Super Resolution](#super-resolution)
- [Video Generation](#video-generation)
- [Video Editing](#video-editing)


## Image Restoration

**Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model** \
[[ICLR 2023 oral](https://arxiv.org/abs/2212.00490)]
[[Project](https://wyhuai.github.io/ddnm.io/)]
[[Code](https://github.com/wyhuai/DDNM)]

**Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild** \
[[CVPR 2024](https://arxiv.org/abs/2401.13627)]
[[Project](https://supir.xpixel.group/)]
[[Code](https://github.com/Fanghua-Yu/SUPIR)]

**Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model** \
[[CVPR 2024](https://arxiv.org/abs/2403.11157)]
[[Project](https://isee-laboratory.github.io/DiffUIR/)]
[[Code](https://github.com/iSEE-Laboratory/DiffUIR)]

**Zero-Reference Low-Light Enhancement via Physical Quadruple Priors** \
[[CVPR 2024](https://arxiv.org/abs/2403.12933)]
[[Project](https://daooshee.github.io/QuadPrior-Website/)]
[[Code](https://github.com/daooshee/QuadPrior/)]

**From Posterior Sampling to Meaningful Diversity in Image Restoration** \
[[ICLR 2024](https://arxiv.org/abs/2310.16047)]
[[Project](https://noa-cohen.github.io/MeaningfulDiversityInIR/)]
[[Code](https://github.com/noa-cohen/MeaningfulDiversityInIR)]

**Generative Diffusion Prior for Unified Image Restoration and Enhancement** \
[[CVPR 2023](https://arxiv.org/abs/2304.01247)]
[[Project](https://generativediffusionprior.github.io/)]
[[Code](https://github.com/Fayeben/GenerativeDiffusionPrior)]

**MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration** \
[[ECCV 2024](https://arxiv.org/abs/2407.10833)]
[[Project](https://renyulin-f.github.io/MoE-DiffIR.github.io/)]
[[Code](https://github.com/renyulin-f/MoE-DiffIR)]

**Image Restoration with Mean-Reverting Stochastic Differential Equations** \
[[ICML 2023](https://arxiv.org/abs/2301.11699)]
[[Project](https://algolzw.github.io/ir-sde/index.html)]
[[Code](https://github.com/Algolzw/image-restoration-sde)]

**PhoCoLens: Photorealistic and Consistent Reconstruction in Lensless Imaging** \
[[NeurIPS 2024 Spotlight](https://arxiv.org/abs/2409.17996)]
[[Project](https://phocolens.github.io/)]
[[Code](https://github.com/PhoCoLens)]

**Denoising Diffusion Models for Plug-and-Play Image Restoration** \
[[CVPR 2023 Workshop NTIRE](https://arxiv.org/abs/2305.08995)]
[[Project](https://yuanzhi-zhu.github.io/DiffPIR/)]
[[Code](https://github.com/yuanzhi-zhu/DiffPIR)]

**Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing** \
[[Website](https://arxiv.org/abs/2407.01521)]
[[Project](https://daps-inverse-problem.github.io/)]
[[Code](https://github.com/zhangbingliang2019/DAPS)]

**Solving Video Inverse Problems Using Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2409.02574)]
[[Project](https://solving-video-inverse.github.io/main/)]
[[Code](https://github.com/solving-video-inverse/codes)]

**Learning Efficient and Effective Trajectories for Differential Equation-based Image Restoration** \
[[Website](https://arxiv.org/abs/2410.04811)]
[[Project](https://zhu-zhiyu.github.io/FLUX-IR/)]
[[Code](https://github.com/ZHU-Zhiyu/FLUX-IR)]

**AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion** \
[[Website](https://arxiv.org/abs/2310.10123)]
[[Project](https://jiangyitong.github.io/AutoDIR_webpage/)]
[[Code](https://github.com/jiangyitong/AutoDIR)]

**FlowIE: Efficient Image Enhancement via Rectified Flow** \
[[CVPR 2024 oral](https://arxiv.org/abs/2406.00508)]
[[Code](https://github.com/EternalEvan/FlowIE)]

**ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting** \
[[NeurIPS 2023 (Spotlight)](https://arxiv.org/abs/2307.12348)]
[[Code](https://github.com/zsyOAOA/ResShift)]

**GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration** \
[[ICML 2023 oral](https://arxiv.org/abs/2301.12686)]
[[Code](https://github.com/sony/gibbsddrm)]

**Image Restoration by Denoising Diffusion Models with Iteratively Preconditioned Guidance** \
[[CVPR 2024](https://arxiv.org/abs/2312.16519)]
[[Code](https://github.com/tirer-lab/DDPG)]

**DiffIR: Efficient Diffusion Model for Image Restoration** \
[[ICCV 2023](https://arxiv.org/abs/2303.09472)]
[[Code](https://github.com/Zj-BinXia/DiffIR)]

**LightenDiffusion: Unsupervised Low-Light Image Enhancement with Latent-Retinex Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2407.08939)]
[[Code](https://github.com/JianghaiSCU/LightenDiffusion)]

**Rethinking Video Deblurring with Wavelet-Aware Dynamic Transformer and Diffusion Model** \
[[ECCV 2024](https://arxiv.org/abs/2408.13459)]
[[Code](https://github.com/Chen-Rao/VD-Diff)]

**DAVI: Diffusion Prior-Based Amortized Variational Inference for Noisy Inverse Problem** \
[[ECCV 2024](https://arxiv.org/abs/2407.16125)]
[[Code](https://github.com/mlvlab/DAVI)]

**Low-Light Image Enhancement with Wavelet-based Diffusion Models** \
[[SIGGRAPH Asia 2023](https://arxiv.org/abs/2306.00306)]
[[Code](https://github.com/JianghaiSCU/Diffusion-Low-Light)]

**Residual Denoising Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2308.13712)]
[[Code](https://github.com/nachifur/RDDM)]

**Diff-Plugin: Revitalizing Details for Diffusion-based Low-level Tasks** \
[[CVPR 2024](https://arxiv.org/abs/2403.00644)]
[[Code](https://github.com/yuhaoliu7456/Diff-Plugin)]

**Deep Equilibrium Diffusion Restoration with Parallel Sampling** \
[[CVPR 2024](https://arxiv.org/abs/2311.11600)]
[[Code](https://github.com/caojiezhang/deqir)]

**ReFIR: Grounding Large Restoration Models with Retrieval Augmentation** \
[[NeurIPS 2024](https://arxiv.org/abs/2410.05601)]
[[Code](https://github.com/csguoh/ReFIR)]

**Zero-Shot Adaptation for Approximate Posterior Sampling of Diffusion Models in Inverse Problems** \
[[Website](https://arxiv.org/abs/2407.11288)]
[[Code](https://github.com/ualcalar17/ZAPS)]

**UniProcessor: A Text-induced Unified Low-level Image Processor** \
[[Website](https://arxiv.org/abs/2407.20928)]
[[Code](https://github.com/IntMeGroup/UniProcessor)]


**Refusion: Enabling Large-Size Realistic Image Restoration with Latent-Space Diffusion Models** \
[[CVPR 2023 Workshop NTIRE](https://arxiv.org/abs/2304.08291)]
[[Code](https://github.com/Algolzw/image-restoration-sde)]

**Equipping Diffusion Models with Differentiable Spatial Entropy for Low-Light Image Enhancement** \
[[CVPR 2024 Workshop NTIRE](https://arxiv.org/abs/2404.09735)]
[[Code](https://github.com/shermanlian/spatial-entropy-loss)]

**PnP-Flow: Plug-and-Play Image Restoration with Flow Matching** \
[[Website](https://arxiv.org/abs/2410.02423)]
[[Code](https://github.com/annegnx/PnP-Flow)]

**Deep Data Consistency: a Fast and Robust Diffusion Model-based Solver for Inverse Problems** \
[[Website](https://arxiv.org/abs/2405.10748)]
[[Code](https://github.com/Hanyu-Chen373/DeepDataConsistency)]

**Learning A Coarse-to-Fine Diffusion Transformer for Image Restoration** \
[[Website](https://arxiv.org/abs/2308.08730)]
[[Code](https://github.com/wlydlut/C2F-DFT)]

**Stimulating the Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling** \
[[Website](https://arxiv.org/abs/2307.03992)]
[[Code](https://github.com/Li-Tong-621/DMID)]

**Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2307.00619)]
[[Code](https://github.com/liturout/psld)]

**Sagiri: Low Dynamic Range Image Enhancement with Generative Diffusion Prior** \
[[Website](https://arxiv.org/abs/2406.09389)]
[[Code](https://github.com/ztMotaLee/Sagiri)]

**Frequency Compensated Diffusion Model for Real-scene Dehazing** \
[[Website](https://arxiv.org/abs/2308.10510)]
[[Code](https://github.com/W-Jilly/frequency-compensated-diffusion-model-pytorch)]

**Efficient Image Deblurring Networks based on Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.05907)]
[[Code](https://github.com/bnm6900030/swintormer)]

**Blind Image Restoration via Fast Diffusion Inversion** \
[[Website](https://arxiv.org/abs/2405.19572)]
[[Code](https://github.com/hamadichihaoui/BIRD)]

**DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.16749)]
[[Code](https://github.com/sun-umn/DMPlug)]

**Accelerating Diffusion Models for Inverse Problems through Shortcut Sampling** \
[[Website](https://arxiv.org/abs/2305.16965)]
[[Code](https://github.com/GongyeLiu/SSD)]

**Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration** \
[[Website](https://arxiv.org/abs/2406.18516)]
[[Code](https://github.com/KangLiao929/Noise-DA/)]

**Unlimited-Size Diffusion Restoration** \
[[Website](https://arxiv.org/abs/2303.00354)]
[[Code](https://github.com/wyhuai/DDNM/tree/main/hq_demo)]

**VmambaIR: Visual State Space Model for Image Restoration** \
[[Website](https://arxiv.org/abs/2403.11423)]
[[Code](https://github.com/AlphacatPlus/VmambaIR)]

**Using diffusion model as constraint: Empower Image Restoration Network Training with Diffusion Model** \
[[Website](https://arxiv.org/abs/2406.19030)]
[[Code](https://github.com/JosephTiTan/DiffLoss)]

**TIP: Text-Driven Image Processing with Semantic and Restoration Instructions** \
[[ECCV 2024](https://arxiv.org/abs/2312.11595)]
[[Project](https://chenyangqiqi.github.io/tip/)]


**Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model** \
[[ICCV 2023](https://arxiv.org/abs/2308.13164)]

**Multiscale Structure Guided Diffusion for Image Deblurring** \
[[ICCV 2023](https://arxiv.org/abs/2212.01789)]

**Boosting Image Restoration via Priors from Pre-trained Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.06793)]

**Unpaired Photo-realistic Image Deraining with Energy-informed Diffusion Model** \
[[Website](https://arxiv.org/abs/2407.17193)]

**Particle-Filtering-based Latent Diffusion for Inverse Problems** \
[[Website](https://arxiv.org/abs/2408.13868)]

**Bayesian Conditioned Diffusion Models for Inverse Problem** \
[[Website](https://arxiv.org/abs/2406.09768)]

**ReCo-Diff: Explore Retinex-Based Condition Strategy in Diffusion Model for Low-Light Image Enhancement** \
[[Website](https://arxiv.org/abs/2312.12826)]

**Multimodal Prompt Perceiver: Empower Adaptiveness, Generalizability and Fidelity for All-in-One Image Restoration** \
[[Website](https://arxiv.org/abs/2312.02918)]

**Tell Me What You See: Text-Guided Real-World Image Denoising**\
[[Website](https://arxiv.org/abs/2312.10191)]

**Zero-LED: Zero-Reference Lighting Estimation Diffusion Model for Low-Light Image Enhancement** \
[[Website](https://arxiv.org/abs/2403.02879)]

**Prototype Clustered Diffusion Models for Versatile Inverse Problems** \
[[Website](https://arxiv.org/abs/2407.09768)]

**AGLLDiff: Guiding Diffusion Models Towards Unsupervised Training-free Real-world Low-light Image Enhancement** \
[[Website](https://arxiv.org/abs/2407.14900)]

**Taming Generative Diffusion for Universal Blind Image Restoration** \
[[Website](https://arxiv.org/abs/2408.11287)]

**Efficient Image Restoration through Low-Rank Adaptation and Stable Diffusion XL** \
[[Website](https://arxiv.org/abs/2408.17060)]

**Empirical Bayesian image restoration by Langevin sampling with a denoising diffusion implicit prior** \
[[Website](https://arxiv.org/abs/2409.04384)]


**Data-free Distillation with Degradation-prompt Diffusion for Multi-weather Image Restoration** \
[[Website](https://arxiv.org/abs/2409.03455)]

**FreeEnhance: Tuning-Free Image Enhancement via Content-Consistent Noising-and-Denoising Process** \
[[Website](https://arxiv.org/abs/2409.07451)]

**Diffusion State-Guided Projected Gradient for Inverse Problems** \
[[Website](https://arxiv.org/abs/2410.03463)]

**InstantIR: Blind Image Restoration with Instant Generative Reference** \
[[Website](https://arxiv.org/abs/2410.06551)]

**Score-Based Variational Inference for Inverse Problems** \
[[Website](https://arxiv.org/abs/2410.05646)]


## Colorization

**Control Color: Multimodal Diffusion-based Interactive Image Colorization** \
[[Website](https://arxiv.org/abs/2402.10855)]
[[Project](https://zhexinliang.github.io/Control_Color/)]
[[Code](https://github.com/ZhexinLiang/Control-Color)]

**Multimodal Semantic-Aware Automatic Colorization with Diffusion Prior** \
[[Website](https://arxiv.org/abs/2404.16678)]
[[Project](https://servuskk.github.io/ColorDiff-Image/)]
[[Code](https://github.com/servuskk/ColorDiff-Image)]

**ColorizeDiffusion: Adjustable Sketch Colorization with Reference Image and Text** \
[[Website](https://arxiv.org/abs/2401.01456)]
[[Code](https://github.com/tellurion-kanata/colorizeDiffusion)]

**Diffusing Colors: Image Colorization with Text Guided Diffusion** \
[[SIGGRAPH Asia 2023](https://arxiv.org/abs/2312.04145)]
[[Project](https://pub.res.lightricks.com/diffusing-colors/)]


**DiffColor: Toward High Fidelity Text-Guided Image Colorization with Diffusion Models** \
[[Website](https://arxiv.org/abs/2308.01655)]


## Face Restoration

**DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior** \
[[Website](https://arxiv.org/abs/2308.15070)]
[[Project](https://0x3f3f3f3fun.github.io/projects/diffbir/)]
[[Code](https://github.com/XPixelGroup/DiffBIR)]

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration** \
[[CVPR 2023](https://arxiv.org/abs/2303.06885)]
[[Code](https://github.com/Kaldwin0106/DR2_Drgradation_Remover)]

**PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance** \
[[NeurIPS 2023](https://arxiv.org/abs/2309.10810)]
[[Code](https://github.com/pq-yang/pgdiff)]

**DifFace: Blind Face Restoration with Diffused Error Contraction** \
[[Website](https://arxiv.org/abs/2312.15736)]
[[Code](https://github.com/zsyOAOA/DifFace)]

**Towards Real-World Blind Face Restoration with Generative Diffusion Prior** \
[[Website](https://arxiv.org/abs/2312.15736)]
[[Code](https://github.com/chenxx89/BFRffusion)]

**Towards Unsupervised Blind Face Restoration using Diffusion Prior** \
[[Website](https://arxiv.org/abs/2410.04618)]
[[Project](https://dt-bfr.github.io/)]

**DiffBFR: Bootstrapping Diffusion Model Towards Blind Face Restoration** \
[[Website](https://arxiv.org/abs/2305.04517)]

**CLR-Face: Conditional Latent Refinement for Blind Face Restoration Using Score-Based Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.06106)]

**DiffMAC: Diffusion Manifold Hallucination Correction for High Generalization Blind Face Restoration** \
[[Website](https://arxiv.org/abs/2403.10098)]

**Gaussian is All You Need: A Unified Framework for Solving Inverse Problems via Diffusion Posterior Sampling** \
[[Website](https://arxiv.org/abs/2409.08906)]

**Overcoming False Illusions in Real-World Face Restoration with Multi-Modal Guided Diffusion Model** \
[[Website](https://arxiv.org/abs/2410.04161)]


## Storytelling

⭐⭐**Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2306.00973)]
[[Project](https://haoningwu3639.github.io/StoryGen_Webpage/)]
[[Code](https://github.com/haoningwu3639/StoryGen)] 

⭐⭐**Training-Free Consistent Text-to-Image Generation** \
[[SIGGRAPH 2024](https://arxiv.org/abs/2402.03286)]
[[Project](https://consistory-paper.github.io/)] 
[[Code](https://github.com/kousw/experimental-consistory)]

**The Chosen One: Consistent Characters in Text-to-Image Diffusion Models** \
[[SIGGRAPH 2024](https://arxiv.org/abs/2311.10093)] 
[[Project](https://omriavrahami.com/the-chosen-one/)] 
[[Code](https://github.com/ZichengDuan/TheChosenOne)]

**AutoStudio: Crafting Consistent Subjects in Multi-turn Interactive Image Generation** \
[[Website](https://arxiv.org/abs/2406.01388)]
[[Project](https://howe183.github.io/AutoStudio.io/)]
[[Code](https://github.com/donahowe/AutoStudio)]

**StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation** \
[[Website](https://arxiv.org/abs/2405.01434)]
[[Project](https://storydiffusion.github.io/)]
[[Code](https://github.com/HVision-NKU/StoryDiffusion)]

**StoryGPT-V: Large Language Models as Consistent Story Visualizers** \
[[Website](https://arxiv.org/abs/2312.02252)]
[[Project](https://storygpt-v.s3.amazonaws.com/index.html)]
[[Code](https://github.com/xiaoqian-shen/StoryGPT-V)] 

**Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation** \
[[Website](https://arxiv.org/abs/2307.06940)]
[[Project](https://ailab-cvc.github.io/Animate-A-Story/)]
[[Code](https://github.com/AILab-CVC/Animate-A-Story)] 

**TaleCrafter: Interactive Story Visualization with Multiple Characters** \
[[Website](https://arxiv.org/abs/2305.18247)]
[[Project](https://ailab-cvc.github.io/TaleCrafter/)]
[[Code](https://github.com/AILab-CVC/TaleCrafter)] 

**Story-Adapter: A Training-free Iterative Framework for Long Story Visualization** \
[[Website](https://arxiv.org/abs/2410.06244)]
[[Project](https://jwmao1.github.io/storyadapter/)]
[[Code](https://github.com/jwmao1/story-adapter)] 

**StoryImager: A Unified and Efficient Framework for Coherent Story Visualization and Completion** \
[[ECCV 2024](https://arxiv.org/abs/2404.05979)]
[[Code](https://github.com/tobran/StoryImager)]

**Make-A-Story: Visual Memory Conditioned Consistent Story Generation** \
[[CVPR 2023](https://arxiv.org/abs/2211.13319)]
[[Code](https://github.com/ubc-vision/Make-A-Story)]

**StoryMaker: Towards Holistic Consistent Characters in Text-to-image Generation** \
[[Website](https://arxiv.org/abs/2409.12576)]
[[Code](https://github.com/RedAIGC/StoryMaker)]


**SEED-Story: Multimodal Long Story Generation with Large Language Model** \
[[Website](https://arxiv.org/abs/2407.08683)]
[[Code](https://github.com/TencentARC/SEED-Story)]

**Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2211.10950)]
[[Code](https://github.com/xichenpan/ARLDM)]

**Masked Generative Story Transformer with Character Guidance and Caption Augmentation** \
[[Website](https://arxiv.org/abs/2403.08502)]
[[Code](https://github.com/chrispapa2000/maskgst)]

**StoryBench: A Multifaceted Benchmark for Continuous Story Visualization** \
[[Website](https://arxiv.org/abs/2308.11606)]
[[Code](https://github.com/google/storybench)]

**Boosting Consistency in Story Visualization with Rich-Contextual Conditional Diffusion Models** \
[[Website](https://arxiv.org/abs/2407.02482)]
[[Code](https://github.com/muzishen/RCDMs)]

**DreamStory: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion** \
[[Website](https://arxiv.org/abs/2407.12899)]
[[Project](https://dream-xyz.github.io/dreamstory)]

**MagicScroll: Nontypical Aspect-Ratio Image Generation for Visual Storytelling via Multi-Layered Semantic-Aware Denoising** \
[[Website](https://arxiv.org/abs/2312.10899)]
[[Project](https://magicscroll.github.io/)]

**Causal-Story: Local Causal Attention Utilizing Parameter-Efficient Tuning For Visual Story Synthesis** \
[[ICASSP 2024](https://arxiv.org/abs/2309.09553)]

**CogCartoon: Towards Practical Story Visualization** \
[[Website](https://arxiv.org/abs/2312.10718)]

**Generating coherent comic with rich story using ChatGPT and Stable Diffusion** \
[[Website](https://arxiv.org/abs/2305.11067)]

**Improved Visual Story Generation with Adaptive Context Modeling** \
[[Website](https://arxiv.org/abs/2305.16811)]

**Make-A-Storyboard: A General Framework for Storyboard with Disentangled and Merged Control** \
[[Website](https://arxiv.org/abs/2312.07549)]

**Zero-shot Generation of Coherent Storybook from Plain Text Story using Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.03900)]

**Evolving Storytelling: Benchmarks and Methods for New Character Customization with Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.11852)]

**ORACLE: Leveraging Mutual Information for Consistent Character Generation with LoRAs in Diffusion Models** \
[[Website](https://arxiv.org/abs/2406.02820)]

**Storynizor: Consistent Story Generation via Inter-Frame Synchronized and Shuffled ID Injection** \
[[Website](https://arxiv.org/abs/2409.19624)]



## Try On

**TryOnDiffusion: A Tale of Two UNets** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_TryOnDiffusion_A_Tale_of_Two_UNets_CVPR_2023_paper.html)]
[[Website](https://arxiv.org/abs/2306.08276)]
[[Project](https://tryondiffusion.github.io/)]
[[Official Code](https://github.com/tryonlabs/tryondiffusion)] 
[[Unofficial Code](https://github.com/fashn-AI/tryondiffusion)] 

**StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On** \
[[CVPR 2024](https://arxiv.org/abs/2312.01725)]
[[Project](https://rlawjdghek.github.io/StableVITON/)]
[[Code](https://github.com/rlawjdghek/stableviton)] 

**VTON-HandFit: Virtual Try-on for Arbitrary Hand Pose Guided by Hand Priors Embedding** \
[[Website](https://arxiv.org/abs/2408.12340)]
[[Project](https://vton-handfit.github.io/)]
[[Code](https://github.com/VTON-HandFit/VTON-HandFit)] 

**IMAGDressing-v1: Customizable Virtual Dressing** \
[[Website](https://arxiv.org/abs/2407.12705)]
[[Project](https://imagdressing.github.io/)]
[[Code](https://github.com/muzishen/IMAGDressing)] 

**OutfitAnyone: Ultra-high Quality Virtual Try-On for Any Clothing and Any Person** \
[[Website](https://arxiv.org/abs/2407.16224)]
[[Project](https://humanaigc.github.io/outfit-anyone/)]
[[Code](https://github.com/HumanAIGC/OutfitAnyone)] 


**ViViD: Video Virtual Try-on using Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.11794)]
[[Project](https://becauseimbatman0.github.io/ViViD)]
[[Code](https://github.com/BecauseImBatman0/ViViD)] 

**GaussianVTON: 3D Human Virtual Try-ON via Multi-Stage Gaussian Splatting Editing with Image Prompting** \
[[Website](https://arxiv.org/abs/2405.07472)]
[[Project](https://haroldchen19.github.io/gsvton/)]
[[Code](https://github.com/HaroldChen19/GaussianVTON)] 

**Street TryOn: Learning In-the-Wild Virtual Try-On from Unpaired Person Images** \
[[Website](https://arxiv.org/abs/2311.16094)]
[[Project](https://cuiaiyu.github.io/StreetTryOn/)]
[[Code](https://github.com/cuiaiyu/street-tryon-benchmark)] 

**From Parts to Whole: A Unified Reference Framework for Controllable Human Image Generation** \
[[Website](https://arxiv.org/abs/2404.15267)]
[[Project](https://huanngzh.github.io/Parts2Whole/)]
[[Code](https://github.com/huanngzh/Parts2Whole)] 

**PICTURE: PhotorealistIC virtual Try-on from UnconstRained dEsigns** \
[[Website](https://arxiv.org/abs/2312.04534)]
[[Project](https://ningshuliang.github.io/2023/Arxiv/index.html)]
[[Code](https://github.com/ningshuliang/PICTURE)] 

**StableGarment: Garment-Centric Generation via Stable Diffusion** \
[[Website](https://arxiv.org/abs/2403.10783)]
[[Project](https://raywang335.github.io/stablegarment.github.io/)] 
[[Code](https://github.com/logn-2024/StableGarment)]

**Improving Diffusion Models for Virtual Try-on** \
[[Website](https://arxiv.org/abs/2403.05139)]
[[Project](https://idm-vton.github.io/)] 
[[Code](https://github.com/yisol/IDM-VTON)]

**D4-VTON: Dynamic Semantics Disentangling for Differential Diffusion based Virtual Try-On** \
[[ECCV 2024](https://arxiv.org/abs/2407.15111)]
[[Code](https://github.com/Jerome-Young/D4-VTON)]

**Improving Virtual Try-On with Garment-focused Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2409.08258)]
[[Code](https://github.com/siqi0905/GarDiff/tree/master)]

**Texture-Preserving Diffusion Models for High-Fidelity Virtual Try-On** \
[[CVPR 2024](https://arxiv.org/abs/2404.01089)]
[[Code](https://github.com/gal4way/tpd)]

**Taming the Power of Diffusion Models for High-Quality Virtual Try-On with Appearance Flow**  \
[[ACM MM 2023](https://arxiv.org/abs/2308.06101)]
[[Code](https://github.com/bcmi/DCI-VTON-Virtual-Try-On)] 

**LaDI-VTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On** \
[[ACM MM 2023](https://arxiv.org/abs/2305.13501)]
[[Code](https://github.com/miccunifi/ladi-vton)] 

**OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on** \
[[Website](https://arxiv.org/abs/2403.01779)]
[[Code](https://github.com/levihsu/OOTDiffusion)] 

**CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Model** \
[[Website](https://arxiv.org/abs/2407.15886)]
[[Code](https://github.com/Zheng-Chong/CatVTON)] 


**DreamPaint: Few-Shot Inpainting of E-Commerce Items for Virtual Try-On without 3D Modeling** \
[[Website](https://arxiv.org/abs/2305.01257)]
[[Code](https://github.com/EmergingUnicorns/DeepPaint)] 

**CAT-DM: Controllable Accelerated Virtual Try-on with Diffusion Model** \
[[Website](https://arxiv.org/abs/2311.18405)]
[[Code](https://github.com/zengjianhao/cat-dm)] 

**MV-VTON: Multi-View Virtual Try-On with Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.17364)]
[[Code](https://github.com/hywang2002/MV-VTON)]

**M&M VTO: Multi-Garment Virtual Try-On and Editing** \
[[CVPR 2024 Highlight](https://arxiv.org/abs/2406.04542)]
[[Project](https://mmvto.github.io/)]

**WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2407.10625)]
[[Project](https://wildvidfit-project.github.io/)]

**Tunnel Try-on: Excavating Spatial-temporal Tunnels for High-quality Virtual Try-on in Videos** \
[[Website](https://arxiv.org/abs/2404.17571)]
[[Project](https://mengtingchen.github.io/tunnel-try-on-page/)]

**Masked Extended Attention for Zero-Shot Virtual Try-On In The Wild** \
[[Website](https://arxiv.org/abs/2406.15331)]
[[Project](https://nadavorzech.github.io/max4zero.github.io/)]

**Diffuse to Choose: Enriching Image Conditioned Inpainting in Latent Diffusion Models for Virtual Try-All** \
[[Website](https://arxiv.org/abs/2401.13795)]
[[Project](https://diffuse2choose.github.io/)]

**Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment** \
[[Website](https://arxiv.org/abs/2403.12965)]
[[Project](https://mengtingchen.github.io/wear-any-way-page/)]

**VITON-DiT: Learning In-the-Wild Video Try-On from Human Dance Videos via Diffusion Transformers** \
[[Website](https://arxiv.org/abs/2405.18326)]
[[Project](https://zhengjun-ai.github.io/viton-dit-page/)]

**AnyFit: Controllable Virtual Try-on for Any Combination of Attire Across Any Scenario** \
[[Website](https://arxiv.org/abs/2405.18172)]
[[Project](https://colorful-liyu.github.io/anyfit-page/)]

**FLDM-VTON: Faithful Latent Diffusion Model for Virtual Try-on** \
[[IJCAI 2024](https://arxiv.org/abs/2404.14162)]

**GraVITON: Graph based garment warping with attention guided inversion for Virtual-tryon** \
[[Website](https://arxiv.org/abs/2406.02184)]

**WarpDiffusion: Efficient Diffusion Model for High-Fidelity Virtual Try-on** \
[[Website](https://arxiv.org/abs/2312.03667)]

**Product-Level Try-on: Characteristics-preserving Try-on with Realistic Clothes Shading and Wrinkles** \
[[Website](https://arxiv.org/abs/2401.11239)]

**Mobile Fitting Room: On-device Virtual Try-on via Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.01877)]

**Time-Efficient and Identity-Consistent Virtual Try-On Using A Variant of Altered Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.07371)]

**ACDG-VTON: Accurate and Contained Diffusion Generation for Virtual Try-On** \
[[Website](https://arxiv.org/abs/2403.13951)]

**ShoeModel: Learning to Wear on the User-specified Shoes via Diffusion Model** \
[[Website](https://arxiv.org/abs/2404.04833)]

**AnyDesign: Versatile Area Fashion Editing via Mask-Free Diffusion** \
[[Website](https://arxiv.org/abs/2408.11553)]

**DPDEdit: Detail-Preserved Diffusion Models for Multimodal Fashion Image Editing** \
[[Website](https://arxiv.org/abs/2409.01086)]


## Drag Edit

**DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models** \
[[ICLR 2024](https://openreview.net/forum?id=OEL4FJMg1b)] 
[[Website](https://arxiv.org/abs/2307.02421)] 
[[Project](https://mc-e.github.io/project/DragonDiffusion/)] 
[[Code](https://github.com/MC-E/DragonDiffusion)]

**Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2305.10973)] 
[[Project](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)] 
[[Code](https://github.com/XingangPan/DragGAN)]


**Readout Guidance: Learning Control from Diffusion Features** \
[[CVPR 2024 Highlight](https://arxiv.org/abs/2312.02150)] 
[[Project](https://readout-guidance.github.io/)] 
[[Code](https://github.com/google-research/readout_guidance)]

**FreeDrag: Feature Dragging for Reliable Point-based Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2307.04684)] 
[[Project](https://lin-chen.site/projects/freedrag/)] 
[[Code](https://github.com/LPengYang/FreeDrag)]

**DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2306.14435)] 
[[Project](https://yujun-shi.github.io/projects/dragdiffusion.html)] 
[[Code](https://github.com/Yujun-Shi/DragDiffusion)]

**InstaDrag: Lightning Fast and Accurate Drag-based Image Editing Emerging from Videos** \
[[Website](https://arxiv.org/abs/2405.13722)] 
[[Project](https://instadrag.github.io/)] 
[[Code](https://github.com/magic-research/InstaDrag)]

**GoodDrag: Towards Good Practices for Drag Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.07206)] 
[[Project](https://gooddrag.github.io/)] 
[[Code](https://github.com/zewei-Zhang/GoodDrag)]

**Repositioning the Subject within Image** \
[[Website](https://arxiv.org/abs/2401.16861)] 
[[Project](https://yikai-wang.github.io/seele/)] 
[[Code](https://github.com/Yikai-Wang/ReS)]

**Drag-A-Video: Non-rigid Video Editing with Point-based Interaction** \
[[Website](https://arxiv.org/abs/2312.02936)] 
[[Project](https://drag-a-video.github.io/)]
[[Code](https://github.com/tyshiwo1/drag-a-video)]

**DragAnything: Motion Control for Anything using Entity Representation** \
[[Website](https://arxiv.org/abs/2403.07420)] 
[[Project](https://weijiawu.github.io/draganything_page/)]
[[Code](https://github.com/showlab/DragAnything)]

**InstantDrag: Improving Interactivity in Drag-based Image Editing** \
[[Website](https://arxiv.org/abs/2409.08857)] 
[[Project](https://joonghyuk.com/instantdrag-web/)]
[[Code](https://github.com/alex4727/InstantDrag)]

**DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2402.02583)] 
[[Code](https://github.com/MC-E/DragonDiffusion)]

**Drag Your Noise: Interactive Point-based Editing via Diffusion Semantic Propagation** \
[[CVPR 2024](https://arxiv.org/abs/2404.01050)] 
[[Code](https://github.com/haofengl/DragNoise)]

**DragVideo: Interactive Drag-style Video Editing** \
[[ECCV 2024](https://arxiv.org/abs/2312.02216)] 
[[Code](https://github.com/rickyskywalker/dragvideo-official)]

**RotationDrag: Point-based Image Editing with Rotated Diffusion Features** \
[[Website](https://arxiv.org/abs/2401.06442)] 
[[Code](https://github.com/Tony-Lowe/RotationDrag)]

**TrackGo: A Flexible and Efficient Method for Controllable Video Generation** \
[[Website](https://arxiv.org/abs/2408.11475)] 
[[Project](https://zhtjtcz.github.io/TrackGo-Page/)] 

**DragText: Rethinking Text Embedding in Point-based Image Editing** \
[[Website](https://arxiv.org/abs/2407.17843)] 
[[Project](https://micv-yonsei.github.io/dragtext2025/)] 

**FastDrag: Manipulate Anything in One Step** \
[[Website](https://arxiv.org/abs/2405.15769)] 
[[Project](https://fastdrag-site.github.io/)] 

**DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory** \
[[Website](https://arxiv.org/abs/2308.08089)] 
[[Project](https://www.microsoft.com/en-us/research/project/dragnuwa/)] 

**StableDrag: Stable Dragging for Point-based Image Editing** \
[[Website](https://arxiv.org/abs/2403.04437)] 
[[Project](https://stabledrag.github.io/)]

**DiffUHaul: A Training-Free Method for Object Dragging in Images** \
[[Website](https://arxiv.org/abs/2406.01594)] 
[[Project](https://omriavrahami.com/diffuhaul/)]

**RegionDrag: Fast Region-Based Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2407.18247)] 

**Motion Guidance: Diffusion-Based Image Editing with Differentiable Motion Estimators** \
[[Website](https://arxiv.org/abs/2401.18085)] 

**Combing Text-based and Drag-based Editing for Precise and Flexible Image Editing** \
[[Website](https://arxiv.org/abs/2410.03097)] 


## Diffusion Models Inversion

⭐⭐⭐**Null-text Inversion for Editing Real Images using Guided Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Mokady_NULL-Text_Inversion_for_Editing_Real_Images_Using_Guided_Diffusion_Models_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2211.09794)] 
[[Project](https://null-text-inversion.github.io/)] 
[[Code](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)]

⭐⭐**Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code** \
[[ICLR 2024](https://openreview.net/forum?id=FoMZ4ljhVw)] 
[[Website](https://arxiv.org/abs/2310.01506)] 
[[Project](https://cure-lab.github.io/PnPInversion/)] 
[[Code](https://github.com/cure-lab/DirectInversion/tree/main)] 

⭐**Inversion-Based Creativity Transfer with Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Inversion-Based_Style_Transfer_With_Diffusion_Models_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2211.13203)] 
[[Code](https://github.com/zyxElsa/InST)] 

⭐**EDICT: Exact Diffusion Inversion via Coupled Transformations** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wallace_EDICT_Exact_Diffusion_Inversion_via_Coupled_Transformations_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2211.12446)] 
[[Code](https://github.com/salesforce/edict)] 

⭐**Improving Negative-Prompt Inversion via Proximal Guidance** \
[[Website](https://arxiv.org/abs/2306.05414)] 
[[Code](https://github.com/phymhan/prompt-to-prompt)] 



**An Edit Friendly DDPM Noise Space: Inversion and Manipulations** \
[[CVPR 2024](https://arxiv.org/abs/2304.06140)] 
[[Project](https://inbarhub.github.io/DDPM_inversion/)] 
[[Code](https://github.com/inbarhub/DDPM_inversion)]
[[Demo](https://huggingface.co/spaces/LinoyTsaban/edit_friendly_ddpm_inversion)]

**Dynamic Prompt Learning: Addressing Cross-Attention Leakage for Text-Based Image Editing** \
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/72801)] 
[[Website](https://arxiv.org/abs/2309.15664)] 
[[Code](https://github.com/wangkai930418/DPL)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=5UXXhVI08r)]  -->

**Inversion-Free Image Editing with Natural Language** \
[[CVPR 2024](https://arxiv.org/abs/2312.04965)] 
[[Project](https://sled-group.github.io/InfEdit/index.html)] 
[[Code](https://github.com/sled-group/InfEdit)] 

**LEDITS++: Limitless Image Editing using Text-to-Image Models** \
[[CVPR 2024](https://arxiv.org/abs/2311.16711)]
[[Project](https://leditsplusplus-project.static.hf.space/index.html)] 
[[Code](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ledits_pp)] 

**Noise Map Guidance: Inversion with Spatial Context for Real Image Editing** \
[[ICLR 2024](https://openreview.net/forum?id=mhgm0IXtHw)] 
[[Website](https://arxiv.org/abs/2402.04625)] 
[[Code](https://github.com/hansam95/nmg)] 


**ReNoise: Real Image Inversion Through Iterative Noising** \
[[ECCV 2024](https://arxiv.org/abs/2403.14602)] 
[[Project](https://garibida.github.io/ReNoise-Inversion/)] 
[[Code](https://github.com/garibida/ReNoise-Inversion)] 

**IterInv: Iterative Inversion for Pixel-Level T2I Models** \
[[NeurIPS-W 2023](https://neurips.cc/virtual/2023/74859)] 
[[Openreview](https://openreview.net/forum?id=mSGmzVo0aS)] 
[[NeuripsW](https://neurips.cc/virtual/2023/workshop/66539#wse-detail-74859)]
[[Website](https://arxiv.org/abs/2310.19540)] 
[[Code](https://github.com/Tchuanm/IterInv)] 

**DICE: Discrete Inversion Enabling Controllable Editing for Multinomial Diffusion and Masked Generative Models** \
[[Website](https://arxiv.org/abs/2410.08207)] 
[[Project](https://hexiaoxiao-cs.github.io/DICE/)] 
[[Code](https://github.com/hexiaoxiao-cs/DICE)] 

**Object-aware Inversion and Reassembly for Image Editing** \
[[Website](https://arxiv.org/abs/2310.12149)] 
[[Project](https://aim-uofa.github.io/OIR-Diffusion/)] 
[[Code](https://github.com/aim-uofa/OIR)] 

**A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_A_Latent_Space_of_Stochastic_Diffusion_Models_for_Zero-Shot_Image_ICCV_2023_paper.pdf)] 
[[Code](https://github.com/humansensinglab/cycle-diffusion)] 

**Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2403.11105)] 
[[Code](https://github.com/leeruibin/SPDInv)] 

**LocInv: Localization-aware Inversion for Text-Guided Image Editing** \
[[CVPR 2024 AI4CC workshop](https://arxiv.org/abs/2405.01496)] 
[[Code](https://github.com/wangkai930418/DPL)] 

**Accelerating Diffusion Models for Inverse Problems through Shortcut Sampling** \
[[IJCAI 2024](https://arxiv.org/abs/2305.16965)] 
[[Code](https://github.com/gongyeliu/ssd)] 

**StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing** \
[[Website](https://arxiv.org/abs/2303.15649)] 
[[Code](https://github.com/sen-mao/StyleDiffusion)] 

**Generating Non-Stationary Textures using Self-Rectification** \
[[Website](https://arxiv.org/abs/2401.02847)] 
[[Code](https://github.com/xiaorongjun000/Self-Rectification)] 

**Exact Diffusion Inversion via Bi-directional Integration Approximation** \
[[Website](https://arxiv.org/abs/2307.10829)] 
[[Code](https://github.com/guoqiang-zhang-x/BDIA)] 


**Fixed-point Inversion for Text-to-image diffusion models** \
[[Website](https://arxiv.org/abs/2312.12540)] 
[[Code](https://github.com/dvirsamuel/FPI)] 

**Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing** \
[[Website](https://arxiv.org/abs/2403.09468)] 
[[Code](https://github.com/furiosa-ai/eta-inversion)] 

**Effective Real Image Editing with Accelerated Iterative Diffusion Inversion** \
[[ICCV 2023 Oral](https://openaccess.thecvf.com/content/ICCV2023/html/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2309.04907)]

**BELM: Bidirectional Explicit Linear Multi-step Sampler for Exact Inversion in Diffusion Models** \
[[NeurIPS 2024](https://arxiv.org/abs/2410.07273)] 

**BARET : Balanced Attention based Real image Editing driven by Target-text Inversion** \
[[WACV 2024](https://arxiv.org/abs/2312.05482)] 

**Wavelet-Guided Acceleration of Text Inversion in Diffusion-Based Image Editing** \
[[ICASSP 2024](https://arxiv.org/abs/2401.09794)]

**Task-Oriented Diffusion Inversion for High-Fidelity Text-based Editing** \
[[Website](https://arxiv.org/abs/2408.13395)] 

**Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.16807)] 

**Direct Inversion: Optimization-Free Text-Driven Real Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2211.07825)] 

**SimInversion: A Simple Framework for Inversion-Based Text-to-Image Editing** \
[[Website](https://arxiv.org/abs/2409.10476)]

**Prompt Tuning Inversion for Text-Driven Image Editing Using Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.04441)]

**KV Inversion: KV Embeddings Learning for Text-Conditioned Real Image Action Editing** \
[[Website](https://arxiv.org/abs/2309.16608)]

**Tuning-Free Inversion-Enhanced Control for Consistent Image Editing** \
[[Website](https://arxiv.org/abs/2312.14611)]

**LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance** \
[[Website](https://arxiv.org/abs/2307.00522)]


## Text Guided Image Editing
⭐⭐⭐**Prompt-to-Prompt Image Editing with Cross Attention Control** \
[[ICLR 2023](https://openreview.net/forum?id=_CDixzkzeyb)] 
[[Website](https://arxiv.org/abs/2211.09794)] 
[[Project](https://prompt-to-prompt.github.io/)] 
[[Code](https://github.com/google/prompt-to-prompt)] 
[[Replicate Demo](https://replicate.com/cjwbw/prompt-to-prompt)]

⭐⭐⭐**Zero-shot Image-to-Image Translation** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2302.03027)] 
[[Project](https://pix2pixzero.github.io/)] 
[[Code](https://github.com/pix2pixzero/pix2pix-zero)] 
[[Replicate Demo](https://replicate.com/cjwbw/pix2pix-zero)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/v0.16.0/api/pipelines/stable_diffusion/pix2pix_zero)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py)]

⭐⭐**InstructPix2Pix: Learning to Follow Image Editing Instructions** \
[[CVPR 2023 (Highlight)](https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2211.09800)] 
[[Project](https://www.timothybrooks.com/instruct-pix2pix/)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/v0.13.0/en/api/pipelines/stable_diffusion/pix2pix)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py)] 
[[Official Code](https://github.com/timothybrooks/instruct-pix2pix)]
[[Dataset](http://instruct-pix2pix.eecs.berkeley.edu/)]

⭐⭐**Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2211.12572)]
[[Project](https://pnp-diffusion.github.io/sm/index.html)] 
[[Code](https://github.com/MichalGeyer/plug-and-play)]
[[Dataset](https://www.dropbox.com/sh/8giw0uhfekft47h/AAAF1frwakVsQocKczZZSX6La?dl=0)]
[[Replicate Demo](https://replicate.com/daanelson/plug_and_play_image_translation)] 
[[Demo](https://huggingface.co/spaces/hysts/PnP-diffusion-features)] 

⭐**DiffEdit: Diffusion-based semantic image editing with mask guidance** \
[[ICLR 2023](https://openreview.net/forum?id=3lge0p5o-M-)] 
[[Website](https://arxiv.org/abs/2210.11427)] 
[[Unofficial Code](https://paperswithcode.com/paper/diffedit-diffusion-based-semantic-image)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/diffedit)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py)] 

⭐**Imagic: Text-Based Real Image Editing with Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kawar_Imagic_Text-Based_Real_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2210.09276)] 
[[Project](https://imagic-editing.github.io/)] 
[[Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community#imagic-stable-diffusion)]

⭐**Inpaint Anything: Segment Anything Meets Image Inpainting** \
[[Website](https://arxiv.org/abs/2304.06790)] 
[[Code 1](https://github.com/geekyutao/Inpaint-Anything)] 
[[Code 2](https://github.com/sail-sg/EditAnything)] 

**MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_MasaCtrl_Tuning-Free_Mutual_Self-Attention_Control_for_Consistent_Image_Synthesis_and_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2304.08465)] 
[[Project](https://ljzycmd.github.io/projects/MasaCtrl/)] 
[[Code](https://github.com/TencentARC/MasaCtrl)] 
[[Demo](https://huggingface.co/spaces/TencentARC/MasaCtrl)]

**Collaborative Score Distillation for Consistent Visual Synthesis** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/73044)] 
[[Website](https://arxiv.org/abs/2307.04787)] 
[[Project](https://subin-kim-cv.github.io/CSD/)] 
[[Code](https://github.com/subin-kim-cv/CSD)]
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=0tEjORCGFD)]  -->

**Visual Instruction Inversion: Image Editing via Visual Prompting** \
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70612)] 
[[Website](https://arxiv.org/abs/2307.14331)] 
[[Project](https://thaoshibe.github.io/visii/)] 
[[Code](https://github.com/thaoshibe/visii)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=l9BsCh8ikK)]  -->

**Energy-Based Cross Attention for Bayesian Context Update in Text-to-Image Diffusion Models** \
[[NeurIPS 2023](https://openreview.net/forum?id=lOCHMGO6ow)] 
[[Website](https://arxiv.org/abs/2306.09869)] 
[[Code](https://github.com/EnergyAttention/Energy-Based-CrossAttention)] 

**Localizing Object-level Shape Variations with Text-to-Image Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Patashnik_Localizing_Object-Level_Shape_Variations_with_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html
)] 
[[Website](https://arxiv.org/abs/2303.11306)] 
[[Project](https://orpatashnik.github.io/local-prompt-mixing/)] 
[[Code](https://github.com/orpatashnik/local-prompt-mixing)]

**Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance** \
[[Website](https://arxiv.org/abs/2210.05559)] 
[[Code1](https://github.com/chenwu98/unified-generative-zoo)] 
[[Code2](https://github.com/chenwu98/cycle-diffusion)] 
[[Diffusers Code](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cycle_diffusion)] 

**PAIR-Diffusion: Object-Level Image Editing with Structure-and-Appearance Paired Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.17546)] 
[[Project](https://vidit98.github.io/publication/conference-paper/pair_diff.html)] 
[[Code](https://github.com/Picsart-AI-Research/PAIR-Diffusion)] 
[[Demo](https://huggingface.co/spaces/PAIR/PAIR-Diffusion)]

**SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models** \
[[CVPR 2024](https://arxiv.org/abs/2312.06739)] 
[[Project](https://yuzhou914.github.io/SmartEdit/)] 
[[Code](https://github.com/TencentARC/SmartEdit)] 

**Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2311.18608)] 
[[Project](https://hyelinnam.github.io/CDS/)] 
[[Code](https://github.com/HyelinNAM/CDS)] 

**Text-Driven Image Editing via Learnable Regions** \
[[CVPR 2024](https://arxiv.org/abs/2311.16432)] 
[[Project](https://yuanze-lin.me/LearnableRegions_page/)] 
[[Code](https://github.com/yuanze-lin/Learnable_Regions)] 

**Motion Guidance: Diffusion-Based Image Editing with Differentiable Motion Estimators** \
[[ICLR 2024](https://arxiv.org/abs/2401.18085)] 
[[Project](https://dangeng.github.io/motion_guidance/)] 
[[Code](https://github.com/dangeng/motion_guidance/)] 


**TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models** \
[[SIGGRAPH Asia 2024](https://arxiv.org/abs/2408.00735)]
[[Project](https://turboedit-paper.github.io/)] 
[[Code](https://github.com/GiilDe/turbo-edit)] 

**Invertible Consistency Distillation for Text-Guided Image Editing in Around 7 Steps** \
[[NeurIPS 2024](https://arxiv.org/abs/2406.14539)] 
[[Project](https://yandex-research.github.io/invertible-cd/)] 
[[Code](https://github.com/yandex-research/invertible-cd)] 

**Zero-shot Image Editing with Reference Imitation** \
[[Website](https://arxiv.org/abs/2406.07547)] 
[[Project](https://xavierchen34.github.io/MimicBrush-Page/)] 
[[Code](https://github.com/ali-vilab/MimicBrush)] 


**MultiBooth: Towards Generating All Your Concepts in an Image from Text** \
[[Website](https://arxiv.org/abs/2404.14239)] 
[[Project](https://multibooth.github.io/)] 
[[Code](https://github.com/chenyangzhu1/MultiBooth)] 

**Infusion: Preventing Customized Text-to-Image Diffusion from Overfitting** \
[[Website](https://arxiv.org/abs/2404.14007)] 
[[Project](https://zwl666666.github.io/infusion/)] 
[[Code](https://github.com/zwl666666/infusion)] 

**StyleBooth: Image Style Editing with Multimodal Instruction** \
[[Website](https://arxiv.org/abs/2404.12154)] 
[[Project](https://ali-vilab.github.io/stylebooth-page/)] 
[[Code](https://github.com/modelscope/scepter)] 

**SwapAnything: Enabling Arbitrary Object Swapping in Personalized Visual Editing** \
[[Website](https://arxiv.org/abs/2404.05717)] 
[[Project](https://swap-anything.github.io/)] 
[[Code](https://github.com/eric-ai-lab/swap-anything)] 

**EditVal: Benchmarking Diffusion Based Text-Guided Image Editing Methods** \
[[Website](https://arxiv.org/abs/2310.02426)] 
[[Project](https://deep-ml-research.github.io/editval/#home)] 
[[Code](https://github.com/deep-ml-research/editval_code)] 

**InstructEdit: Improving Automatic Masks for Diffusion-based Image Editing With User Instructions** \
[[Website](https://arxiv.org/abs/2305.18047)] 
[[Project](https://qianwangx.github.io/InstructEdit/)] 
[[Code](https://github.com/QianWangX/InstructEdit)] 

**MDP: A Generalized Framework for Text-Guided Image Editing by Manipulating the Diffusion Path** \
[[Website](https://arxiv.org/abs/2303.16765)] 
[[Project](https://qianwangx.github.io/MDP-Diffusion/)] 
[[Code](https://github.com/QianWangX/MDP-Diffusion)] 

**HIVE: Harnessing Human Feedback for Instructional Visual Editing** \
[[Website](https://arxiv.org/abs/2303.09618)] 
[[Project](https://shugerdou.github.io/hive/)] 
[[Code](https://github.com/salesforce/HIVE)] 

**FaceStudio: Put Your Face Everywhere in Seconds** \
[[Website](https://arxiv.org/abs/2312.02663)] 
[[Project](https://icoz69.github.io/facestudio/)] 
[[Code](https://github.com/xyynafc/FaceStudio)] 

**Smooth Diffusion: Crafting Smooth Latent Spaces in Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.04410)] 
[[Project](https://shi-labs.github.io/Smooth-Diffusion/)] 
[[Code](https://github.com/SHI-Labs/Smooth-Diffusion)] 

**FreeEdit: Mask-free Reference-based Image Editing with Multi-modal Instruction** \
[[Website](https://arxiv.org/abs/2409.18071)] 
[[Project](https://github.com/hrz2000/FreeEdit)] 
[[Code](https://freeedit.github.io/)] 

**MAG-Edit: Localized Image Editing in Complex Scenarios via Mask-Based Attention-Adjusted Guidance** \
[[Website](https://arxiv.org/abs/2312.11396)] 
[[Project](https://mag-edit.github.io/)] 
[[Code](https://github.com/HelenMao/MAG-Edit)] 

**LIME: Localized Image Editing via Attention Regularization in Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.09256)]
[[Project](https://enis.dev/LIME/)] 
[[Code](https://github.com/enisimsar/LIME)] 

**MirrorDiffusion: Stabilizing Diffusion Process in Zero-shot Image Translation by Prompts Redescription and Beyond** \
[[Website](https://arxiv.org/abs/2401.03221)] 
[[Project](https://mirrordiffusion.github.io/)] 
[[Code](https://github.com/MirrorDiffusion/MirrorDiffusion)] 

**Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control** \
[[Website](https://arxiv.org/abs/2405.12970)] 
[[Project](https://faceadapter.github.io/face-adapter.github.io/)] 
[[Code](https://github.com/FaceAdapter/Face-Adapter)] 

**FastEdit: Fast Text-Guided Single-Image Editing via Semantic-Aware Diffusion Fine-Tuning** \
[[Website](https://arxiv.org/abs/2408.03355)]
[[Project](https://fastedit-sd.github.io/)] 
[[Code](https://github.com/JasonCodeMaker/FastEdit)] 

**Delta Denoising Score** \
[[Website](https://arxiv.org/abs/2304.07090)] 
[[Project](https://delta-denoising-score.github.io/)] 
[[Code](https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb)] 


**UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2210.09477)] 
[[Code](https://github.com/xuduo35/UniTune)]
 
**Learning to Follow Object-Centric Image Editing Instructions Faithfully** \
[[EMNLP 2023](https://arxiv.org/abs/2310.19145)] 
[[Code](https://github.com/tuhinjubcse/faithfuledits_emnlp2023)] 

**GroupDiff: Diffusion-based Group Portrait Editing** \
[[ECCV 2024](https://arxiv.org/abs/2409.14379)] 
[[Code](https://github.com/yumingj/GroupDiff)] 

**TiNO-Edit: Timestep and Noise Optimization for Robust Diffusion-Based Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2404.11120)] 
[[Code](https://github.com/SherryXTChen/TiNO-Edit)] 

**ZONE: Zero-Shot Instruction-Guided Local Editing** \
[[CVPR 2024](https://arxiv.org/abs/2312.16794)]
[[Code](https://github.com/lsl001006/ZONE)] 

**Focus on Your Instruction: Fine-grained and Multi-instruction Image Editing by Attention Modulation** \
[[CVPR 2024](https://arxiv.org/abs/2312.10113)] 
[[Code](https://github.com/guoqincode/focus-on-your-instruction)] 


**DreamSampler: Unifying Diffusion Sampling and Score Distillation for Image Manipulation** \
[[ECCV 2024](https://arxiv.org/abs/2403.11415)]
[[Code](https://github.com/dreamsampler/dream-sampler)] 

**FlexiEdit: Frequency-Aware Latent Refinement for Enhanced Non-Rigid Editing** \
[[ECCV 2024](https://arxiv.org/abs/2407.17850)] 
[[Code](https://github.com/kookie12/FlexiEdit)] 

**Guide-and-Rescale: Self-Guidance Mechanism for Effective Tuning-Free Real Image Editing** \
[[ECCV 2024](https://arxiv.org/abs/2409.01322)] 
[[Code](https://github.com/FusionBrainLab/Guide-and-Rescale)] 

**Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks** \
[[AAAI 2024](https://arxiv.org/abs/2401.07709)]
[[Code](https://github.com/xiaotianqing/instdiffedit)] 

**FISEdit: Accelerating Text-to-image Editing via Cache-enabled Sparse Diffusion Inference** \
[[AAAI 2024](https://arxiv.org/abs/2305.17423)]
[[Code](https://github.com/pku-dair/hetu)] 

**Face Aging via Diffusion-based Editing**\
[[BMVC 2023](https://arxiv.org/abs/2309.11321)]
[[Code](https://github.com/MunchkinChen/FADING)] 

**FlexEdit: Marrying Free-Shape Masks to VLLM for Flexible Image Editing** \
[[Website](https://arxiv.org/abs/2408.12429)] 
[[Code](https://github.com/a-new-b/flex_edit)] 

**Specify and Edit: Overcoming Ambiguity in Text-Based Image Editing** \
[[Website](https://arxiv.org/abs/2407.20232)] 
[[Code](https://github.com/fabvio/SANE)] 

**PostEdit: Posterior Sampling for Efficient Zero-Shot Image Editing** \
[[Website](https://arxiv.org/abs/2410.04844)] 
[[Code](https://github.com/TFNTF/PostEdit)] 

**Move and Act: Enhanced Object Manipulation and Background Integrity for Image Editing** \
[[Website](https://arxiv.org/abs/2405.14785)] 
[[Code](https://github.com/YangLing0818/EditWorld)] 

**EditWorld: Simulating World Dynamics for Instruction-Following Image Editing** \
[[Website](https://arxiv.org/abs/2407.17847)] 
[[Code](https://github.com/mobiushy/move-act)] 

**ClickDiffusion: Harnessing LLMs for Interactive Precise Image Editing** \
[[Website](https://arxiv.org/abs/2404.04376)] 
[[Code](https://github.com/poloclub/clickdiffusion)] 

**Differential Diffusion: Giving Each Pixel Its Strength** \
[[Website](https://arxiv.org/abs/2306.00950)] 
[[Code](https://github.com/exx8/differential-diffusion)] 


**Ground-A-Score: Scaling Up the Score Distillation for Multi-Attribute Editing** \
[[Website](https://arxiv.org/abs/2403.13551)]
[[Code](https://github.com/ground-a-score/ground-a-score)] 


**InstructDiffusion: A Generalist Modeling Interface for Vision Tasks** \
[[Website](https://arxiv.org/abs/2309.03895)]
[[Code](https://github.com/cientgu/instructdiffusion)] 

**Region-Aware Diffusion for Zero-shot Text-driven Image Editing** \
[[Website](https://arxiv.org/abs/2302.11797v1)] 
[[Code](https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model)] 

**Forgedit: Text Guided Image Editing via Learning and Forgetting** \
[[Website](https://arxiv.org/abs/2309.10556)] 
[[Code](https://github.com/witcherofresearch/Forgedit)] 

**AdapEdit: Spatio-Temporal Guided Adaptive Editing Algorithm for Text-Based Continuity-Sensitive Image Editing** \
[[Website](https://arxiv.org/abs/2312.08019)] 
[[Code](https://github.com/anonymouspony/adap-edit)] 

**An Item is Worth a Prompt: Versatile Image Editing with Disentangled Control** \
[[Website](https://arxiv.org/abs/2403.04880)]
[[Code](https://github.com/collovlabs/d-edit)] 

**FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.11895)]
[[Code](https://github.com/thermal-dynamics/freediff)] 

**Unified Diffusion-Based Rigid and Non-Rigid Editing with Text and Image Guidance** \
[[Website](https://arxiv.org/abs/2401.02126)] 
[[Code](https://github.com/kihensarn/ti-guided-edit)] 

**SpecRef: A Fast Training-free Baseline of Specific Reference-Condition Real Image Editing** \
[[Website](https://arxiv.org/abs/2401.03433)] 
[[Code](https://github.com/jingjiqinggong/specp2p)] 

**PromptFix: You Prompt and We Fix the Photo** \
[[Website](https://arxiv.org/abs/2405.16785)] 
[[Code](https://github.com/yeates/PromptFix)] 

**FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation** \
[[Website](https://arxiv.org/abs/2408.00998)]
[[Code](https://github.com/XiangGao1102/FBSDiff)] 


**Conditional Score Guidance for Text-Driven Image-to-Image Translation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/71103)] 
[[Website](https://arxiv.org/abs/2305.18007)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=cBS5CU96Jq)]  -->

**Emu Edit: Precise Image Editing via Recognition and Generation Tasks** \
[[CVPR 2024](https://arxiv.org/abs/2311.10089)] 
[[Project](https://emu-edit.metademolab.com/)]

**ByteEdit: Boost, Comply and Accelerate Generative Image Editing** \
[[ECCV 2024](https://arxiv.org/abs/2404.04860)] 
[[Project](https://byte-edit.github.io/)]

**Watch Your Steps: Local Image and Scene Editing by Text Instructions** \
[[ECCV 2024](https://arxiv.org/abs/2308.08947)]
[[Project](https://ashmrz.github.io/WatchYourSteps/)] 

**TurboEdit: Instant text-based image editing** \
[[ECCV 2024](https://arxiv.org/abs/2408.08332)]
[[Project](https://betterze.github.io/TurboEdit/)] 


**MultiEdits: Simultaneous Multi-Aspect Editing with Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2406.00985)]
[[Project](https://mingzhenhuang.com/projects/MultiEdits.html)] 

**Unified Editing of Panorama, 3D Scenes, and Videos Through Disentangled Self-Attention Injection** \
[[Website](https://arxiv.org/abs/2405.16823)]
[[Project](https://unifyediting.github.io/)] 

**Editable Image Elements for Controllable Synthesis** \
[[Website](https://arxiv.org/abs/2404.16029)]
[[Project](https://jitengmu.github.io/Editable_Image_Elements/)] 

**SGEdit: Bridging LLM with Text2Image Generative Model for Scene Graph-based Image Editing** \
[[Website](https://arxiv.org/abs/2410.11815)]
[[Project](https://bestzzhang.github.io/SGEdit/)] 


**ReGeneration Learning of Diffusion Models with Rich Prompts for Zero-Shot Image Translation** \
[[Website](https://arxiv.org/abs/2305.04651)] 
[[Project](https://yupeilin2388.github.io/publication/ReDiffuser)]

**GANTASTIC: GAN-based Transfer of Interpretable Directions for Disentangled Image Editing in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.19645)]
[[Project](https://gantastic.github.io/)]

**MoEController: Instruction-based Arbitrary Image Manipulation with Mixture-of-Expert Controllers** \
[[Website](https://arxiv.org/abs/2309.04372)]
[[Project](https://oppo-mente-lab.github.io/moe_controller/)]

**FlexEdit: Flexible and Controllable Diffusion-based Object-centric Image Editing** \
[[Website](https://arxiv.org/abs/2403.18605)]
[[Project](https://flex-edit.github.io/)]

**GeoDiffuser: Geometry-Based Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.14403)]
[[Project](https://ivl.cs.brown.edu/research/geodiffuser.html)]

**SOEDiff: Efficient Distillation for Small Object Editing** \
[[Website](https://arxiv.org/abs/2405.09114)]
[[Project](https://soediff.github.io/)]

**Click2Mask: Local Editing with Dynamic Mask Generation** \
[[Website](https://arxiv.org/abs/2409.08272)]
[[Project](https://omeregev.github.io/click2mask/)]

**Iterative Multi-granular Image Editing using Diffusion Models** \
[[WACV 2024](https://arxiv.org/abs/2309.00613)] 


**Text-to-image Editing by Image Information Removal** \
[[WACV 2024](https://arxiv.org/abs/2305.17489)]


**TexSliders: Diffusion-Based Texture Editing in CLIP Space** \
[[SIGGRAPH 2024](https://arxiv.org/abs/2405.00672)]

**Custom-Edit: Text-Guided Image Editing with Customized Diffusion Models** \
[[CVPR 2023 AI4CC Workshop](https://arxiv.org/abs/2305.15779)] 

**Learning Feature-Preserving Portrait Editing from Generated Pairs** \
[[Website](https://arxiv.org/abs/2407.20455)]

**EmoEdit: Evoking Emotions through Image Manipulation** \
[[Website](https://arxiv.org/abs/2405.12661)]

**DM-Align: Leveraging the Power of Natural Language Instructions to Make Changes to Images** \
[[Website](https://arxiv.org/abs/2404.18020)]

**LayerDiffusion: Layered Controlled Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.18676)]

**iEdit: Localised Text-guided Image Editing with Weak Supervision** \
[[Website](https://arxiv.org/abs/2305.05947)]

**User-friendly Image Editing with Minimal Text Input: Leveraging Captioning and Injection Techniques** \
[[Website](https://arxiv.org/abs/2306.02717)]

**PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing** \
[[Website](https://arxiv.org/abs/2306.16894)]

**PRedItOR: Text Guided Image Editing with Diffusion Prior** \
[[Website](https://arxiv.org/abs/2302.07979v2)]


**FEC: Three Finetuning-free Methods to Enhance Consistency for Real Image Editing** \
[[Website](https://arxiv.org/abs/2309.14934)]

**The Blessing of Randomness: SDE Beats ODE in General Diffusion-based Image Editing** \
[[Website](https://arxiv.org/abs/2311.01410)]

**Image Translation as Diffusion Visual Programmers** \
[[Website](https://arxiv.org/abs/2312.16794)]

**Latent Inversion with Timestep-aware Sampling for Training-free Non-rigid Editing** \
[[Website](https://arxiv.org/abs/2402.08601)]

**LoMOE: Localized Multi-Object Editing via Multi-Diffusion** \
[[Website](https://arxiv.org/abs/2403.00437)]

**Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing** \
[[Website](https://arxiv.org/abs/2403.03431)]


**DiffChat: Learning to Chat with Text-to-Image Synthesis Models for Interactive Image Creation** \
[[Website](https://arxiv.org/abs/2403.04997)]

**InstructGIE: Towards Generalizable Image Editing** \
[[Website](https://arxiv.org/abs/2403.05018)]

**LASPA: Latent Spatial Alignment for Fast Training-free Single Image Editing** \
[[Website](https://arxiv.org/abs/2403.12585)]


**Uncovering the Text Embedding in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.01154)]

**Tuning-Free Adaptive Style Incorporation for Structure-Consistent Text-Driven Style Transfer** \
[[Website](https://arxiv.org/abs/2404.06835)]


**Enhancing Text-to-Image Editing via Hybrid Mask-Informed Fusion** \
[[Website](https://arxiv.org/abs/2405.15313)]

**Text Guided Image Editing with Automatic Concept Locating and Forgetting** \
[[Website](https://arxiv.org/abs/2405.19708)]

**The Curious Case of End Token: A Zero-Shot Disentangled Image Editing using CLIP** \
[[Website](https://arxiv.org/abs/2406.00457)]

**LIPE: Learning Personalized Identity Prior for Non-rigid Image Editing** \
[[Website](https://arxiv.org/abs/2406.17236)]


**Achieving Complex Image Edits via Function Aggregation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2408.08495)]

**Prompt-Softbox-Prompt: A free-text Embedding Control for Image Editing** \
[[Website](https://arxiv.org/abs/2408.13623)]

**InverseMeetInsert: Robust Real Image Editing via Geometric Accumulation Inversion in Guided Diffusion Models** \
[[Website](https://arxiv.org/abs/2409.11734)]

**PixLens: A Novel Framework for Disentangled Evaluation in Diffusion-Based Image Editing with Object Detection + SAM** \
[[Website](https://arxiv.org/abs/2410.05710)]

**Augmentation-Driven Metric for Balancing Preservation and Modification in TextGuided Image Editing** \
[[Website](https://arxiv.org/abs/2410.11374)]


## Continual Learning

**RGBD2: Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Lei_RGBD2_Generative_Scene_Synthesis_via_Incremental_View_Inpainting_Using_RGBD_CVPR_2023_paper.pdf)] 
[[Website](https://arxiv.org/abs/2212.05993)] 
[[Project](https://jblei.site/proj/rgbd-diffusion)] 
[[Code](https://github.com/Karbo123/RGBD-Diffusion)]

**Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning** \
[[ECCV 2024 Oral](https://arxiv.org/abs/2409.01128)] 
[[Code](https://github.com/jinglin-liang/DDDR)]

**CLoG: Benchmarking Continual Learning of Image Generation Models** \
[[Website](https://arxiv.org/abs/2406.04584)] 
[[Code](https://github.com/linhaowei1/CLoG)]

**Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models** \
[[Website](https://arxiv.org/abs/2305.10120)] 
[[Code](https://github.com/clear-nus/selective-amnesia)]

**Continual Learning of Diffusion Models with Generative Distillation** \
[[Website](https://arxiv.org/abs/2311.14028)] 
[[Code](https://github.com/atenrev/difussion_continual_learning)]

**Prompt-Based Exemplar Super-Compression and Regeneration for Class-Incremental Learning** \
[[Website](https://arxiv.org/abs/2311.18266)] 
[[Code](https://github.com/KerryDRX/ESCORT)]

**Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA** \
[[TMLR](https://arxiv.org/abs/2304.06027)] 
[[Project](https://jamessealesmith.github.io/continual-diffusion/)]

**Class-Incremental Learning using Diffusion Model for Distillation and Replay** \
[[ICCV 2023 VCL workshop best paper](https://arxiv.org/abs/2306.17560)] 

**Create Your World: Lifelong Text-to-Image Diffusion** \
[[Website](https://arxiv.org/abs/2309.04430)] 

**Low-Rank Continual Personalization of Diffusion Models** \
[[Website](https://arxiv.org/pdf/2410.04891)] 

**Mining Your Own Secrets: Diffusion Classifier Scores for Continual Personalization of Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.00700)] 

**Online Continual Learning of Video Diffusion Models From a Single Video Stream** \
[[Website](https://arxiv.org/abs/2406.04814)] 

**Exploring Continual Learning of Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.15342)] 

**DiracDiffusion: Denoising and Incremental Reconstruction with Assured Data-Consistency** \
[[Website](https://arxiv.org/abs/2303.14353)] 

**DiffusePast: Diffusion-based Generative Replay for Class Incremental Semantic Segmentation** \
[[Website](https://arxiv.org/abs/2308.01127)] 

**Continual Diffusion with STAMINA: STack-And-Mask INcremental Adapters** \
[[Website](https://arxiv.org/abs/2311.18763)] 

**Premonition: Using Generative Models to Preempt Future Data Changes in Continual Learning** \
[[Website](https://arxiv.org/abs/2403.07356)] 

**MuseumMaker: Continual Style Customization without Catastrophic Forgetting** \
[[Website](https://arxiv.org/abs/2404.16612)] 


## Remove Concept

**Ablating Concepts in Text-to-Image Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Kumari_Ablating_Concepts_in_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2303.13516)] 
[[Project](https://www.cs.cmu.edu/~concept-ablation/)] 
[[Code](https://github.com/nupurkmr9/concept-ablation)]

**Erasing Concepts from Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2303.07345)] 
[[Project](https://erasing.baulab.info/)] 
[[Code](https://github.com/rohitgandikota/erasing)]

**Paint by Inpaint: Learning to Add Image Objects by Removing Them First** \
[[Website](https://arxiv.org/abs/2404.18212)] 
[[Project](https://rotsteinnoam.github.io/Paint-by-Inpaint/)] 
[[Code](https://github.com/RotsteinNoam/Paint-by-Inpaint)]

**One-dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications** \
[[Website](https://arxiv.org/abs/2312.16145)] 
[[Project](https://lyumengyao.github.io/projects/spm)] 
[[Code](https://github.com/Con6924/SPM)]

**Editing Massive Concepts in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.13807)] 
[[Project](https://silentview.github.io/EMCID/)] 
[[Code](https://github.com/SilentView/EMCID)]

**STEREO: Towards Adversarially Robust Concept Erasing from Text-to-Image Generation Models** \
[[Website](https://arxiv.org/abs/2408.16807)] 
[[Project](https://koushiksrivats.github.io/robust-concept-erasing/)] 
[[Code](https://github.com/koushiksrivats/robust-concept-erasing)]


**Towards Safe Self-Distillation of Internet-Scale Text-to-Image Diffusion Models** \
[[ICML 2023 workshop](https://arxiv.org/abs/2307.05977v1)] 
[[Code](https://github.com/nannullna/safe-diffusion)]

**Reliable and Efficient Concept Erasure of Text-to-Image Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2407.12383)] 
[[Code](https://github.com/CharlesGong12/RECE)]

**Safeguard Text-to-Image Diffusion Models with Human Feedback Inversion** \
[[ECCV 2024](https://arxiv.org/abs/2407.21032)] 
[[Code](https://github.com/nannullna/safeguard-hfi)]

**ObjectAdd: Adding Objects into Image via a Training-Free Diffusion Modification Fashion** \
[[Website](https://arxiv.org/abs/2404.17230)] 
[[Code](https://github.com/potato-kitty/objectadd)]

**Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.17591)] 
[[Code](https://github.com/SHI-Labs/Forget-Me-Not)]

**Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.15234)] 
[[Code](https://github.com/OPTML-Group/AdvUnlearn)]

**ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning** \
[[Website](https://arxiv.org/abs/2405.19237)] 
[[Code](https://github.com/ruchikachavhan/concept-prune)]

**Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models** \
[[Website](https://arxiv.org/abs/2305.10120)] 
[[Code](https://github.com/clear-nus/selective-amnesia)]

**Add-SD: Rational Generation without Manual Reference** \
[[Website](https://arxiv.org/abs/2407.21016)] 
[[Code](https://github.com/ylingfeng/Add-SD)]

**MACE: Mass Concept Erasure in Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.06135)] 

**Unstable Unlearning: The Hidden Risk of Concept Resurgence in Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.08074)] 

**Direct Unlearning Optimization for Robust and Safe Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2407.21035)] 

**Prompt Sliders for Fine-Grained Control, Editing and Erasing of Concepts in Diffusion Model** \
[[Website](https://arxiv.org/pdf/2409.16535)] 

**Erasing Concepts from Text-to-Image Diffusion Models with Few-shot Unlearning** \
[[Website](https://arxiv.org/abs/2405.07288)] 

**Geom-Erasing: Geometry-Driven Removal of Implicit Concept in Diffusion Models** \
[[Website](https://arxiv.org/abs/2310.05873)] 

**Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers** \
[[Website](https://arxiv.org/abs/2311.17717)] 

**All but One: Surgical Concept Erasing with Model Preservation in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.12807)] 

**EraseDiff: Erasing Data Influence in Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.05779)] 

**UnlearnCanvas: A Stylized Image Dataset to Benchmark Machine Unlearning for Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.11846)] 

**Removing Undesirable Concepts in Text-to-Image Generative Models with Learnable Prompts** \
[[Website](https://arxiv.org/abs/2402.11846)] 

**R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2405.16341)] 

**Pruning for Robust Concept Erasing in Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.16534)] 

**Unlearning Concepts from Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2407.14209)] 

**EIUP: A Training-Free Approach to Erase Non-Compliant Concepts Conditioned on Implicit Unsafe Prompts** \
[[Website](https://arxiv.org/abs/2408.01014)] 

**Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning** \
[[Website](https://arxiv.org/abs/2410.05664)] 


## New Concept Learning

⭐⭐⭐**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation** \
[[CVPR 2023 Honorable Mention](https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2208.12242)] 
[[Project](https://dreambooth.github.io/)] 
[[Official Dataset](https://github.com/google/dreambooth)]
[[Unofficial Code](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/training/dreambooth)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)] 

⭐⭐⭐**An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion** \
[[ICLR 2023 top-25%](https://openreview.net/forum?id=NAQvF08TcyG)] 
[[Website](https://arxiv.org/abs/2208.01618)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/training/text_inversion)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion)] 
[[Code](https://github.com/rinongal/textual_inversion)]


⭐⭐**Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2212.04488)] 
[[Project](https://www.cs.cmu.edu/~custom-diffusion/)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/custom_diffusion)] 
[[Code](https://github.com/adobe-research/custom-diffusion)]

⭐⭐**ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement** \
[[ECCV 2024](https://arxiv.org/abs/2407.07197)] 
[[Project](https://moatifbutt.github.io/colorpeel/)]
[[Code](https://github.com/moatifbutt/color-peel)]

⭐⭐**ReVersion: Diffusion-Based Relation Inversion from Images** \
[[Website](https://arxiv.org/abs/2303.13495)] 
[[Project](https://ziqihuangg.github.io/projects/reversion.html)]
[[Code](https://github.com/ziqihuangg/ReVersion)]

⭐**SINE: SINgle Image Editing with Text-to-Image Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_SINE_SINgle_Image_Editing_With_Text-to-Image_Diffusion_Models_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2212.04489)] 
[[Project](https://zhang-zx.github.io/SINE/)] 
[[Code](https://github.com/zhang-zx/SINE)] 

⭐**Break-A-Scene: Extracting Multiple Concepts from a Single Image** \
[[SIGGRAPH Asia 2023](https://arxiv.org/abs/2305.16311)] 
[[Project](https://omriavrahami.com/break-a-scene/)]
[[Code](https://github.com/google/break-a-scene)]

⭐**Concept Decomposition for Visual Exploration and Inspiration** \
[[SIGGRAPH Asia 2023](https://arxiv.org/abs/2305.18203)] 
[[Project](https://inspirationtree.github.io/inspirationtree/)] 
[[Code](https://github.com/google/inspiration_tree)]

**Cones: Concept Neurons in Diffusion Models for Customized Generation** \
[[ICML 2023 Oral](https://icml.cc/virtual/2023/oral/25582)] 
[[ICML 2023 Oral](https://dl.acm.org/doi/10.5555/3618408.3619298)] 
[[Website](https://arxiv.org/abs/2303.05125)] 
[[Code](https://github.com/Johanan528/Cones)]

**BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70870)] 
[[Website](https://arxiv.org/abs/2305.14720)] 
[[Project](https://dxli94.github.io/BLIP-Diffusion-website/)]
[[Code](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion)]

**Inserting Anybody in Diffusion Models via Celeb Basis** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/71823)] 
[[Website](https://arxiv.org/abs/2306.00926)] 
[[Project](https://celeb-basis.github.io/)] 
[[Code](https://github.com/ygtxr1997/celebbasis)]

**Controlling Text-to-Image Diffusion by Orthogonal Finetuning** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/72033)] 
[[Website](https://arxiv.org/abs/2306.07280)] 
[[Project](https://oft.wyliu.com/)] 
[[Code](https://github.com/Zeju1997/oft)]

**Photoswap: Personalized Subject Swapping in Images** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70336)] 
[[Website](https://arxiv.org/abs/2305.18286)] 
[[Project](https://photoswap.github.io/)] 
[[Code](https://github.com/eric-ai-lab/photoswap)]

**Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/71844)] 
[[Website](https://arxiv.org/abs/2305.18292)] 
[[Project](https://showlab.github.io/Mix-of-Show/)] 
[[Code](https://github.com/TencentARC/Mix-of-Show)]

**ITI-GEN: Inclusive Text-to-Image Generation** \
[[ICCV 2023 Oral](https://arxiv.org/abs/2309.05569)] 
[[Website](https://arxiv.org/abs/2309.05569)] 
[[Project](https://czhang0528.github.io/iti-gen)] 
[[Code](https://github.com/humansensinglab/ITI-GEN)]

**Unsupervised Compositional Concepts Discovery with Text-to-Image Generative Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Unsupervised_Compositional_Concepts_Discovery_with_Text-to-Image_Generative_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2306.05357)] 
[[Project](https://energy-based-model.github.io/unsupervised-concept-discovery/)]
[[Code](https://github.com/nanlliu/Unsupervised-Compositional-Concepts-Discovery)]

**ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation** \
[[ICCV 2023 Oral](https://openaccess.thecvf.com/content/ICCV2023/html/Wei_ELITE_Encoding_Visual_Concepts_into_Textual_Embeddings_for_Customized_Text-to-Image_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2302.13848)] 
[[Code](https://github.com/csyxwei/ELITE)]
<!-- [[Demo](https://huggingface.co/spaces/ELITE-library/ELITE)] -->

**A Neural Space-Time Representation for Text-to-Image Personalization** \
[[SIGGRAPH Asia 2023](https://arxiv.org/abs/2305.15391)] 
[[Project](https://neuraltextualinversion.github.io/NeTI/)] 
[[Code](https://github.com/NeuralTextualInversion/NeTI)]

**Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2302.12228)] 
[[Project](https://tuning-encoder.github.io/)] 
[[Code](https://github.com/mkshing/e4t-diffusion)]

**Is This Loss Informative? Speeding Up Textual Inversion with Deterministic Objective Evaluation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/71329)] 
[[Website](https://arxiv.org/abs/2302.04841)] 
[[Code](https://github.com/yandex-research/DVAR)]

**ConceptExpress: Harnessing Diffusion Models for Single-image Unsupervised Concept Extraction** \
[[ECCV 2024](https://arxiv.org/abs/2407.07077)] 
[[Project](https://haoosz.github.io/ConceptExpress/)] 
[[Code](https://github.com/haoosz/ConceptExpress)]

**Face2Diffusion for Fast and Editable Face Personalization** \
[[CVPR 2024](https://arxiv.org/abs/2403.05094)] 
[[Project](https://mapooon.github.io/Face2DiffusionPage/)] 
[[Code](https://github.com/mapooon/Face2Diffusion)]

**Identity Decoupling for Multi-Subject Personalization of Text-to-Image Models** \
[[CVPR 2024](https://arxiv.org/abs/2404.04243)] 
[[Project](https://mudi-t2i.github.io/)] 
[[Code](https://github.com/agwmon/MuDI)]

**CapHuman: Capture Your Moments in Parallel Universes** \
[[CVPR 2024](https://arxiv.org/abs/2402.00627)] 
[[Project](https://caphuman.github.io/)] 
[[Code](https://github.com/VamosC/CapHumanf)]

**Style Aligned Image Generation via Shared Attention** \
[[CVPR 2024](https://arxiv.org/abs/2312.02133)] 
[[Project](https://style-aligned-gen.github.io/)] 
[[Code](https://github.com/google/style-aligned/)]

**FreeCustom: Tuning-Free Customized Image Generation for Multi-Concept Composition** \
[[CVPR 2024](https://arxiv.org/abs/2405.13870v1)] 
[[Project](https://aim-uofa.github.io/FreeCustom/)] 
[[Code](https://github.com/aim-uofa/FreeCustom)]

**DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization** \
[[CVPR 2024](https://arxiv.org/abs/2402.09812)] 
[[Project](https://ku-cvlab.github.io/DreamMatcher/)] 
[[Code](https://github.com/KU-CVLAB/DreamMatcher)]

**Material Palette: Extraction of Materials from a Single Image** \
[[CVPR 2024](https://arxiv.org/abs/2311.17060)] 
[[Project](https://astra-vision.github.io/MaterialPalette/)] 
[[Code](https://github.com/astra-vision/MaterialPalette)]

**Learning Continuous 3D Words for Text-to-Image Generation** \
[[CVPR 2024](https://arxiv.org/abs/2402.08654)] 
[[Project](https://ttchengab.github.io/continuous_3d_words/)] 
[[Code](https://github.com/ttchengab/continuous_3d_words_code/)]

**ConceptBed: Evaluating Concept Learning Abilities of Text-to-Image Diffusion Models** \
[[AAAI 2024](https://arxiv.org/abs/2306.04695)] 
[[Project](https://conceptbed.github.io/)] 
[[Code](https://github.com/conceptbed/evaluations)]

**The Hidden Language of Diffusion Models** \
[[ICLR 2024](https://arxiv.org/abs/2306.00966)] 
[[Project](https://hila-chefer.github.io/Conceptor/)] 
[[Code](https://github.com/hila-chefer/Conceptor)]

**ZeST: Zero-Shot Material Transfer from a Single Image** \
[[ECCV 2024](https://arxiv.org/abs/2404.06425)] 
[[Project](https://ttchengab.github.io/zest/)] 
[[Code](https://github.com/ttchengab/zest_code)]

**UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization** \
[[Website](https://arxiv.org/abs/2408.05939)] 
[[Project](https://aigcdesigngroup.github.io/UniPortrait-Page/)] 
[[Code](https://github.com/junjiehe96/UniPortrait)]

**MagicFace: Training-free Universal-Style Human Image Customized Synthesis** \
[[Website](https://arxiv.org/abs/2408.07433)] 
[[Project](https://codegoat24.github.io/MagicFace/)] 
[[Code](https://github.com/CodeGoat24/MagicFace)]

**LCM-Lookahead for Encoder-based Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2404.03620)] 
[[Project](https://lcm-lookahead.github.io/)] 
[[Code](https://github.com/OrLichter/lcm-lookahead)]

**AITTI: Learning Adaptive Inclusive Token for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2406.12805)] 
[[Project](https://itsmag11.github.io/AITTI/)] 
[[Code](https://github.com/itsmag11/AITTI)]

**MS-Diffusion: Multi-subject Zero-shot Image Personalization with Layout Guidance** \
[[Website](https://arxiv.org/abs/2406.07209)] 
[[Project](https://ms-diffusion.github.io/)] 
[[Code](https://github.com/MS-Diffusion/MS-Diffusion)]

**ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance** \
[[Website](https://arxiv.org/abs/2405.17532)] 
[[Project](https://classdiffusion.github.io/)] 
[[Code](https://github.com/Rbrq03/ClassDiffusion)]

**MasterWeaver: Taming Editability and Identity for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2405.05806)] 
[[Project](https://masterweaver.github.io/)] 
[[Code](https://github.com/csyxwei/MasterWeaver)]

**Customizing Text-to-Image Models with a Single Image Pair** \
[[Website](https://arxiv.org/abs/2405.01536)] 
[[Project](https://paircustomization.github.io/)] 
[[Code](https://github.com/PairCustomization/PairCustomization)]

**DisEnvisioner: Disentangled and Enriched Visual Prompt for Customized Image Generation** \
[[Website](https://arxiv.org/abs/2410.02067)] 
[[Project](https://disenvisioner.github.io/)] 
[[Code](https://github.com/EnVision-Research/DisEnvisioner)]

**ConsistentID: Portrait Generation with Multimodal Fine-Grained Identity Preserving** \
[[Website](https://arxiv.org/abs/2404.16771)] 
[[Project](https://ssugarwh.github.io/consistentid.github.io/)] 
[[Code](https://github.com/JackAILab/ConsistentID)]

**ID-Aligner: Enhancing Identity-Preserving Text-to-Image Generation with Reward Feedback Learning** \
[[Website](https://arxiv.org/abs/2404.15449)] 
[[Project](https://idaligner.github.io/)] 
[[Code](https://github.com/Weifeng-Chen/ID-Aligner)]

**CharacterFactory: Sampling Consistent Characters with GANs for Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.15677)] 
[[Project](https://qinghew.github.io/CharacterFactory/)] 
[[Code](https://github.com/qinghew/CharacterFactory)]

**Customizing Text-to-Image Diffusion with Camera Viewpoint Control** \
[[Website](https://arxiv.org/abs/2404.12333)] 
[[Project](https://customdiffusion360.github.io/)] 
[[Code](https://github.com/customdiffusion360/custom-diffusion360)]

**Harmonizing Visual and Textual Embeddings for Zero-Shot Text-to-Image Customization** \
[[Website](https://arxiv.org/abs/2403.14155)] 
[[Project](https://ldynx.github.io/harmony-zero-t2i/)] 
[[Code](https://github.com/ldynx/harmony-zero-t2i)]

**StyleDrop: Text-to-Image Generation in Any Style** \
[[Website](https://arxiv.org/abs/2306.00983)] 
[[Project](https://styledrop.github.io/)] 
[[Code](https://github.com/zideliu/StyleDrop-PyTorch)]


**FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention** \
[[Website](https://arxiv.org/abs/2305.10431)] 
[[Project](https://fastcomposer.mit.edu/)] 
[[Code](https://github.com/mit-han-lab/fastcomposer)]
<!-- [[Demo](https://2acfe10ec96df6f2b0.gradio.live/)] -->

**AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning** \
[[Website](https://arxiv.org/abs/2307.04725)] 
[[Project](https://animatediff.github.io/)] 
[[Code](https://github.com/guoyww/animatediff/)]

**Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning**\
[[Website](https://arxiv.org/abs/2307.11410)] 
[[Project](https://oppo-mente-lab.github.io/subject_diffusion/)] 
[[Code](https://github.com/OPPO-Mente-Lab/Subject-Diffusion)]

**Highly Personalized Text Embedding for Image Manipulation by Stable Diffusion** \
[[Website](https://arxiv.org/abs/2303.08767)] 
[[Project](https://hiper0.github.io/)] 
[[Code](https://github.com/HiPer0/HiPer)]



**DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Positive-Negative Prompt-Tuning** \
[[Website](https://arxiv.org/abs/2211.11337)] 
[[Project](https://www.sysu-hcp.net/projects/dreamartist/index.html)] 
[[Code](https://github.com/7eu7d7/DreamArtist-stable-diffusion)]

**SingleInsert: Inserting New Concepts from a Single Image into Text-to-Image Models for Flexible Editing** \
[[Website](https://arxiv.org/abs/2310.08094)] 
[[Project](https://jarrentwu1031.github.io/SingleInsert-web/)] 
[[Code](https://github.com/JarrentWu1031/SingleInsert)]

**CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs//2310.19784)] 
[[Project](https://jiangyzy.github.io/CustomNet/)] 
[[Code](https://github.com/TencentARC/CustomNet)]

**When StyleGAN Meets Stable Diffusion: a W+ Adapter for Personalized Image Generation** \
[[Website](https://arxiv.org/abs/2311.17461)] 
[[Project](https://csxmli2016.github.io/projects/w-plus-adapter/)] 
[[Code](https://github.com/csxmli2016/w-plus-adapter)]

**InstantID: Zero-shot Identity-Preserving Generation in Seconds** \
[[Website](https://arxiv.org/abs/2401.07519)] 
[[Project](https://instantid.github.io/)] 
[[Code](https://github.com/InstantID/InstantID)]

**PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding** \
[[Website](https://arxiv.org/abs/2312.04461)] 
[[Project](https://photo-maker.github.io/)] 
[[Code](https://github.com/TencentARC/PhotoMaker)]




**CatVersion: Concatenating Embeddings for Diffusion-Based Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2311.14631)] 
[[Project](https://royzhao926.github.io/CatVersion-page/)] 
[[Code](https://github.com/RoyZhao926/CatVersion)]

**DreamDistribution: Prompt Distribution Learning for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.14216)] 
[[Project](https://briannlongzhao.github.io/DreamDistribution/)] 
[[Code](https://github.com/briannlongzhao/DreamDistribution)]


**λ-ECLIPSE: Multi-Concept Personalized Text-to-Image Diffusion Models by Leveraging CLIP Latent Space** \
[[Website](https://arxiv.org/abs/2402.05195)] 
[[Project](https://eclipse-t2i.github.io/Lambda-ECLIPSE/)] 
[[Code](https://github.com/eclipse-t2i/lambda-eclipse-inference)]

**Viewpoint Textual Inversion: Unleashing Novel View Synthesis with Pretrained 2D Diffusion Models** \
[[Website](https://arxiv.org/abs/2309.07986)] 
[[Project](https://jmhb0.github.io/viewneti/)] 
[[Code](https://github.com/jmhb0/view_neti)]

**Gen4Gen: Generative Data Pipeline for Generative Multi-Concept Composition** \
[[Website](https://arxiv.org/abs/2402.15504)] 
[[Project](https://danielchyeh.github.io/Gen4Gen/)] 
[[Code](https://github.com/louisYen/Gen4Gen)]

**StableIdentity: Inserting Anybody into Anywhere at First Sight** \
[[Website](https://arxiv.org/abs/2401.15975)] 
[[Project](https://qinghew.github.io/StableIdentity/)] 
[[Code](https://github.com/qinghew/StableIdentity)]

**DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Model** \
[[Website](https://arxiv.org/abs/2402.17412)] 
[[Project](https://diffusekrona.github.io/)] 
[[Code](https://github.com/IBM/DiffuseKronA)]

**Direct Consistency Optimization for Compositional Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2402.12004)] 
[[Project](https://dco-t2i.github.io/)] 
[[Code](https://github.com/kyungmnlee/dco)]

**TextBoost: Towards One-Shot Personalization of Text-to-Image Models via Fine-tuning Text Encoder** \
[[Website](https://arxiv.org/abs/2409.08248)] 
[[Project](https://textboost.github.io/)] 
[[Code](https://github.com/nahyeonkaty/textboost)]

**EZIGen: Enhancing zero-shot subject-driven image generation with precise subject encoding and decoupled guidance** \
[[Website](https://arxiv.org/abs/2409.08091)] 
[[Project](https://zichengduan.github.io/pages/EZIGen/index.html)] 
[[Code](https://github.com/ZichengDuan/EZIGen)]

**OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.10983)] 
[[Project](https://kongzhecn.github.io/omg-project/)] 
[[Code](https://github.com/kongzhecn/OMG/)]

**MoMA: Multimodal LLM Adapter for Fast Personalized Image Generation** \
[[Website](https://arxiv.org/abs/2404.05674)] 
[[Project](https://moma-adapter.github.io/)] 
[[Code](https://github.com/bytedance/MoMA/tree/main)]

**ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs** \
[[Website](https://arxiv.org/abs/2311.13600)] 
[[Project](https://ziplora.github.io/)] 
[[Code](https://github.com/mkshing/ziplora-pytorch)]

**CSGO: Content-Style Composition in Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2311.13600)] 
[[Project](https://csgo-gen.github.io/)] 
[[Code](https://github.com/instantX-research/CSGO)]

**DreamSteerer: Enhancing Source Image Conditioned Editability using Personalized Diffusion Models** \
[[NeurIPS 2024](https://arxiv.org/abs/2410.11208)] 
[[Code](https://github.com/Dijkstra14/DreamSteerer)]

**Customized Generation Reimagined: Fidelity and Editability Harmonized** \
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06727.pdf)] 
[[Code](https://github.com/jinjianRick/DCI_ICO)]

**Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning** \
[[ECCV 2024](https://arxiv.org/abs/2407.06642)] 
[[Code](https://github.com/wfanyue/DPG-T2I-Personalization)]

**High-fidelity Person-centric Subject-to-Image Synthesis** \
[[CVPR 2024](https://arxiv.org/abs/2311.10329)] 
[[Code](https://github.com/codegoat24/face-diffuser)] 

**ProSpect: Expanded Conditioning for the Personalization of Attribute-aware Image Generation** \
[[SIGGRAPH Asia 2023](https://arxiv.org/abs/2305.16225)] 
[[Code](https://github.com/zyxElsa/ProSpect)]

**Multiresolution Textual Inversion** \
[[NeurIPS 2022 workshop](https://arxiv.org/abs/2211.17115)] 
[[Code](https://github.com/giannisdaras/multires_textual_inversion)]


**Compositional Inversion for Stable Diffusion Models** \
[[AAAI 2024](https://arxiv.org/abs/2312.08048)] 
[[Code](https://github.com/zhangxulu1996/Compositional-Inversion)]

**Decoupled Textual Embeddings for Customized Image Generation** \
[[AAAI 2024](https://arxiv.org/abs/2312.11826)] 
[[Code](https://github.com/PrototypeNx/DETEX)]

**TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation** \
[[Website](https://arxiv.org/abs/2410.05591)] 
[[Code](https://github.com/KwonGihyun/TweedieMix)]

**Resolving Multi-Condition Confusion for Finetuning-Free Personalized Image Generation** \
[[Website](https://arxiv.org/abs/2409.17920)] 
[[Code](https://github.com/hqhQAQ/MIP-Adapter)]

**Concept Conductor: Orchestrating Multiple Personalized Concepts in Text-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2408.03632)] 
[[Code](https://github.com/Nihukat/Concept-Conductor)]

**RectifID: Personalizing Rectified Flow with Anchored Classifier Guidance** \
[[Website](https://arxiv.org/abs/2405.14677)] 
[[Code](https://github.com/feifeiobama/RectifID)]

**PuLID: Pure and Lightning ID Customization via Contrastive Alignment** \
[[Website](https://arxiv.org/abs/2404.16022)] 
[[Code](https://github.com/ToTheBeginning/PuLID)]

**Cross Initialization for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2312.15905)] 
[[Code](https://github.com/lyupang/crossinitialization)]

**Enhancing Detail Preservation for Customized Text-to-Image Generation: A Regularization-Free Approach** \
[[Website](https://arxiv.org/abs/2305.13579)] 
[[Code](https://github.com/drboog/profusion)]

**SVDiff: Compact Parameter Space for Diffusion Fine-Tuning** \
[[Website](https://arxiv.org/abs/2303.11305)] 
[[Code](https://github.com/mkshing/svdiff-pytorch)]


**ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2306.00971)] 
[[Code](https://github.com/haoosz/vico)]

**AerialBooth: Mutual Information Guidance for Text Controlled Aerial View Synthesis from a Single Image** \
[[Website](https://arxiv.org/abs/2311.15040)] 
[[Code](https://github.com/Xiang-cd/unet-finetune)]

**A Closer Look at Parameter-Efficient Tuning in Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.15478)] 
[[Code](https://github.com/divyakraman/AerialBooth2023)]


**Controllable Textual Inversion for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2304.05265)] 
[[Code](https://github.com/jnzju/COTI)]

**Cross-domain Compositing with Pretrained Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.10167)] 
[[Code](https://github.com/cross-domain-compositing/cross-domain-compositing)] 

**Concept-centric Personalization with Large-scale Diffusion Priors** \
[[Website](https://arxiv.org/abs/2312.08195)] 
[[Code](https://github.com/PRIV-Creation/Concept-centric-Personalization)] 

**Customization Assistant for Text-to-image Generation** \
[[Website](https://arxiv.org/abs/2312.03045)] 
[[Code](https://github.com/drboog/profusion)] 

**Cross Initialization for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2312.15905)] 
[[Code](https://github.com/lyupang/crossinitialization)] 

**Cones 2: Customizable Image Synthesis with Multiple Subjects** \
[[Website](https://arxiv.org/abs/2305.19327v1)] 
[[Code](https://github.com/ali-vilab/cones-v2)] 

**LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.11627)] 
[[Code](https://github.com/Young98CN/LoRA_Composer)] 

**AttenCraft: Attention-guided Disentanglement of Multiple Concepts for Text-to-Image Customization** \
[[Website](https://arxiv.org/abs/2405.17965)] 
[[Code](https://github.com/junjie-shentu/AttenCraft)] 

**CusConcept: Customized Visual Concept Decomposition with Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.00398)] 
[[Code](https://github.com/xzLcan/CusConcept)] 

**HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation** \
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01439.pdf)] 
[[Project](https://sites.google.com/view/hybridbooth)] 

**Language-Informed Visual Concept Learning** \
[[ICLR 2024](https://arxiv.org/abs/2312.03587)] 
[[Project](https://ai.stanford.edu/~yzzhang/projects/concept-axes/)] 

**Key-Locked Rank One Editing for Text-to-Image Personalization** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2305.01644)] 
[[Project](https://research.nvidia.com/labs/par/Perfusion/)] 

**Diffusion in Style** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Everaert_Diffusion_in_Style_ICCV_2023_paper.pdf)] 
[[Project](https://ivrl.github.io/diffusion-in-style/)] 

**RealCustom: Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization** \
[[CVPR 2024](https://arxiv.org/abs/2403.00483)] 
[[Project](https://corleone-huang.github.io/realcustom/)] 

**RealCustom++: Representing Images as Real-Word for Real-Time Customization** \
[[Website](https://arxiv.org/abs/2408.09744)] 
[[Project](https://corleone-huang.github.io/RealCustom_plus_plus/)] 

**Personalized Residuals for Concept-Driven Text-to-Image Generation** \
[[CVPR 2024](https://arxiv.org/abs/2405.12978)] 
[[Project](https://cusuh.github.io/personalized-residuals/)] 

**LogoSticker: Inserting Logos into Diffusion Models for Customized Generation** \
[[ECCV 2024](https://arxiv.org/abs/2407.13752)] 
[[Project](https://mingkangz.github.io/logosticker/)] 

**InstructBooth: Instruction-following Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2312.03011)] 
[[Project](https://sites.google.com/view/instructbooth)] 

**AttnDreamBooth: Towards Text-Aligned Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2406.05000)] 
[[Project](https://attndreambooth.github.io/)] 

**MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation** \
[[Website](https://arxiv.org/abs/2404.11565)] 
[[Project](https://snap-research.github.io/mixture-of-attention/)] 


**PortraitBooth: A Versatile Portrait Model for Fast Identity-preserved Personalization** \
[[Website](https://arxiv.org/abs/2312.06354)] 
[[Project](https://portraitbooth.github.io/)] 

**Subject-driven Text-to-Image Generation via Apprenticeship Learning** \
[[Website](https://arxiv.org/abs/2304.00186)] 
[[Project](https://open-vision-language.github.io/suti/)] 

**Orthogonal Adaptation for Modular Customization of Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.02432)] 
[[Project](https://ryanpo.com/ortha/)] 

**Diffusion in Diffusion: Cyclic One-Way Diffusion for Text-Vision-Conditioned Generation** \
[[Website](https://arxiv.org/abs/2306.08247)] 
[[Project](https://bigaandsmallq.github.io/COW/)] 


**HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2307.06949)] 
[[Project](https://hyperdreambooth.github.io/)] 


**Domain-Agnostic Tuning-Encoder for Fast Personalization of Text-To-Image Models** \
[[Website](https://arxiv.org/abs/2307.06925)] 
[[Project](https://datencoder.github.io/)] 

**$P+$: Extended Textual Conditioning in Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2303.09522)] 
[[Project](https://prompt-plus.github.io/)] 

**PhotoVerse: Tuning-Free Image Customization with Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2309.05793)] 
[[Project](https://photoverse2d.github.io/)] 

**InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning** \
[[Website](https://arxiv.org/abs/2304.03411)] 
[[Project](https://jshi31.github.io/InstantBooth/)] 

**Total Selfie: Generating Full-Body Selfies** \
[[Website](https://arxiv.org/abs/2308.14740)] 
[[Project](https://homes.cs.washington.edu/~boweiche/project_page/totalselfie/)] 



**DreamTuner: Single Image is Enough for Subject-Driven Generation** \
[[Website](https://arxiv.org/abs/2312.13691)] 
[[Project](https://dreamtuner-diffusion.github.io/)] 


**PALP: Prompt Aligned Personalization of Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2401.06105)] 
[[Project](https://prompt-aligned.github.io/)] 

**TextureDreamer: Image-guided Texture Synthesis through Geometry-aware Diffusion** \
[[CVPR 2024](https://arxiv.org/abs/2401.09416)] 
[[Project](https://texturedreamer.github.io/)] 

**Visual Style Prompting with Swapping Self-Attention** \
[[Website](https://arxiv.org/abs/2402.12974)] 
[[Project](https://curryjung.github.io/VisualStylePrompt/)] 

**Infinite-ID: Identity-preserved Personalization via ID-semantics Decoupling Paradigm** \
[[Website](https://arxiv.org/abs/2403.11781)] 
[[Project](https://infinite-id.github.io/)] 

**Non-confusing Generation of Customized Concepts in Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.06914)] 
[[Project](https://clif-official.github.io/clif/)] 

**HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation** \
[[Website](https://arxiv.org/abs/2410.08192)] 
[[Project](https://sites.google.com/view/hybridboot/)] 


**ComFusion: Personalized Subject Generation in Multiple Specific Scenes From Single Image** \
[[ECCV 2024](https://arxiv.org/abs/2402.11849)] 

**Concept Weaver: Enabling Multi-Concept Fusion in Text-to-Image Models** \
[[CVPR 2024](https://arxiv.org/abs/2404.03913)] 

**JeDi: Joint-Image Diffusion Models for Finetuning-Free Personalized Text-to-Image Generation** \
[[CVPR 2024](https://arxiv.org/abs/2407.06187)] 

**DreamStyler: Paint by Style Inversion with Text-to-Image Diffusion Models** \
[[AAAI 2024](https://arxiv.org/abs/2309.06933)] 

**FreeTuner: Any Subject in Any Style with Training-free Diffusion** \
[[Website](https://arxiv.org/abs/2405.14201)] 

**Towards Prompt-robust Face Privacy Protection via Adversarial Decoupling Augmentation Framework** \
[[Website](https://arxiv.org/abs/2305.03980)] 

**InstaStyle: Inversion Noise of a Stylized Image is Secretly a Style Adviser** \
[[Website](https://arxiv.org/abs/2311.15040)] 

**DisenBooth: Disentangled Parameter-Efficient Tuning for Subject-Driven Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2305.03374)] 

**Taming Encoder for Zero Fine-tuning Image Customization with Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2304.02642)] 

**Gradient-Free Textual Inversion** \
[[Website](https://arxiv.org/abs/2304.05818)] 

**Identity Encoder for Personalized Diffusion** \
[[Website](https://arxiv.org/abs/2304.07429)] 

**Unified Multi-Modal Latent Diffusion for Joint Subject and Text Conditional Image Generation** \
[[Website](https://arxiv.org/abs/2303.09319)] 

**ELODIN: Naming Concepts in Embedding Spaces** \
[[Website](https://arxiv.org/abs/2303.04001)] 

**Generate Anything Anywhere in Any Scene** \
[[Website](https://arxiv.org/abs/2306.17154)] 

**Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model** \
[[Website](https://arxiv.org/abs/2306.07596)] 

**Face0: Instantaneously Conditioning a Text-to-Image Model on a Face** \
[[Website](https://arxiv.org/abs/2306.06638v1)] 

**MagiCapture: High-Resolution Multi-Concept Portrait Customization** \
[[Website](https://arxiv.org/abs/2309.06895)] 

**A Data Perspective on Enhanced Identity Preservation for Diffusion Personalization** \
[[Website](https://arxiv.org/abs/2311.04315)] 

**DIFFNAT: Improving Diffusion Image Quality Using Natural Image Statistics** \
[[Website](https://arxiv.org/abs/2311.09753)] 

**An Image is Worth Multiple Words: Multi-attribute Inversion for Constrained Text-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2311.11919)] 

**Lego: Learning to Disentangle and Invert Concepts Beyond Object Appearance in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.13833)] 

**Memory-Efficient Personalization using Quantized Diffusion Model** \
[[Website](https://arxiv.org/abs/2401.04339)] 

**BootPIG: Bootstrapping Zero-shot Personalized Image Generation Capabilities in Pretrained Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.13974)] 

**Pick-and-Draw: Training-free Semantic Guidance for Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2401.16762)] 

**Object-Driven One-Shot Fine-tuning of Text-to-Image Diffusion with Prototypical Embedding** \
[[Website](https://arxiv.org/abs/2401.15708)] 

**SeFi-IDE: Semantic-Fidelity Identity Embedding for Personalized Diffusion-Based Generation** \
[[Website](https://arxiv.org/abs/2402.00631)] 

**Visual Concept-driven Image Generation with Text-to-Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2402.11487)] 


**IDAdapter: Learning Mixed Features for Tuning-Free Personalization of Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2403.13535)] 

**MM-Diff: High-Fidelity Image Personalization via Multi-Modal Condition Integration** \
[[Website](https://arxiv.org/abs/2403.15059)] 

**DreamSalon: A Staged Diffusion Framework for Preserving Identity-Context in Editable Face Generation** \
[[Website](https://arxiv.org/abs/2403.19235)] 

**OneActor: Consistent Character Generation via Cluster-Conditioned Guidance** \
[[Website](https://arxiv.org/abs/2404.10267)] 

**StyleMaster: Towards Flexible Stylized Image Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.15287)] 

**Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks** \
[[Website](https://arxiv.org/abs/2405.19931)] 

**Inv-Adapter: ID Customization Generation via Image Inversion and Lightweight Adapter** \
[[Website](https://arxiv.org/abs/2406.02881)] 

**PaRa: Personalizing Text-to-Image Diffusion via Parameter Rank Reduction** \
[[Website](https://arxiv.org/abs/2406.05641)] 

**AlignIT: Enhancing Prompt Alignment in Customization of Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2406.18893)] 

**Layout-and-Retouch: A Dual-stage Framework for Improving Diversity in Personalized Image Generation** \
[[Website](https://arxiv.org/abs/2407.09779)] 

**PreciseControl: Enhancing Text-To-Image Diffusion Models with Fine-Grained Attribute Control** \
[[Website](https://arxiv.org/abs/2408.05083)] 

**MagicID: Flexible ID Fidelity Generation System** \
[[Website](https://arxiv.org/abs/2408.09248)] 

**CoRe: Context-Regularized Text Embedding Learning for Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2408.15914)] 

**ArtiFade: Learning to Generate High-quality Subject from Blemished Images** \
[[Website](https://arxiv.org/abs/2409.03745)] 

**CustomContrast: A Multilevel Contrastive Perspective For Subject-Driven Text-to-Image Customization** \
[[Website](https://arxiv.org/abs/2409.05606)] 

**Fusion is all you need: Face Fusion for Customized Identity-Preserving Image Synthesis** \
[[Website](https://arxiv.org/abs/2409.19111)] 

**Event-Customized Image Generation** \
[[Website](https://arxiv.org/abs/2410.02483)] 


## T2I Diffusion Model augmentation

⭐⭐⭐**Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2301.13826)] 
[[Project](https://yuval-alaluf.github.io/Attend-and-Excite/)] 
[[Official Code](https://github.com/yuval-alaluf/Attend-and-Excite)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_and_excite.py)] 
[[Diffusers doc](https://huggingface.co/docs/diffusers/api/pipelines/attend_and_excite)] 
[[Replicate Demo](https://replicate.com/daanelson/attend-and-excite)]

**SEGA: Instructing Diffusion using Semantic Dimensions** \
[[NeurIPS 2023](https://openreview.net/forum?id=KIPAIy329j&referrer=%5Bthe%20profile%20of%20Patrick%20Schramowski%5D(%2Fprofile%3Fid%3D~Patrick_Schramowski1))] 
[[Website](https://arxiv.org/abs/2301.12247)] 
[[Code](https://github.com/ml-research/semantic-image-editing)]
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/semantic_stable_diffusion/pipeline_semantic_stable_diffusion.py)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion)] 

**Improving Sample Quality of Diffusion Models Using Self-Attention Guidance** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Hong_Improving_Sample_Quality_of_Diffusion_Models_Using_Self-Attention_Guidance_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2210.00939)] 
[[Project](https://ku-cvlab.github.io/Self-Attention-Guidance/)] 
[[Code Official](https://github.com/KU-CVLAB/Self-Attention-Guidance)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/self_attention_guidance)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_sag.py)]
<!-- [[Demo](https://huggingface.co/spaces/susunghong/Self-Attention-Guidance)] -->

**Expressive Text-to-Image Generation with Rich Text** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ge_Expressive_Text-to-Image_Generation_with_Rich_Text_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2304.06720)] 
[[Project](https://rich-text-to-image.github.io/)] 
[[Code](https://github.com/SongweiGe/rich-text-to-image)]
[[Demo](https://huggingface.co/spaces/songweig/rich-text-to-image)]

**Editing Implicit Assumptions in Text-to-Image Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Orgad_Editing_Implicit_Assumptions_in_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2303.08084)] 
[[Project](https://time-diffusion.github.io/)] 
[[Code](https://github.com/bahjat-kawar/time-diffusion)]
[[Demo](https://huggingface.co/spaces/bahjat-kawar/time-diffusion)]

**ElasticDiffusion: Training-free Arbitrary Size Image Generation** \
[[CVPR 2024](https://arxiv.org/abs/2311.18822)] 
[[Project](https://elasticdiffusion.github.io/)] 
[[Code](https://github.com/moayedhajiali/elasticdiffusion-official)]
[[Demo](https://replicate.com/moayedhajiali/elasticdiffusion)]

**MagicFusion: Boosting Text-to-Image Generation Performance by Fusing Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_MagicFusion_Boosting_Text-to-Image_Generation_Performance_by_Fusing_Diffusion_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2303.13126)] 
[[Project](https://magicfusion.github.io/)]
[[Code](https://github.com/MagicFusion/MagicFusion.github.io)]

**Discriminative Class Tokens for Text-to-Image Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Schwartz_Discriminative_Class_Tokens_for_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2303.17155)] 
[[Project](https://vesteinn.github.io/disco/)]
[[Code](https://github.com/idansc/discriminative_class_tokens)]

**Compositional Visual Generation with Composable Diffusion Models** \
[[ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6940_ECCV_2022_paper.php)] 
[[Website](https://arxiv.org/abs/2206.01714)] 
[[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)] 
[[Code](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)]

**DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models** \
[[ICCV 2023](https://arxiv.org/abs/2402.19481)] 
[[Project](https://hanlab.mit.edu/projects/distrifusion)]
[[Code](https://github.com/mit-han-lab/distrifuser)]
[[Blog](https://hanlab.mit.edu/blog/distrifusion)] 


**ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/72054)] 
[[Website](https://arxiv.org/abs/2304.05977)] 
[[Code](https://github.com/THUDM/ImageReward)]
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=JVzeOYEx6d)]  -->

**Diffusion Self-Guidance for Controllable Image Generation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70344)] 
[[Website](https://arxiv.org/abs/2306.00986)] 
[[Project](https://dave.ml/selfguidance/)] 
[[Code](https://github.com/Sainzerjj/Free-Guidance-Diffusion)]

**DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/72425)] 
[[Website](https://arxiv.org/abs/2306.14685)] 
[[Code](https://github.com/ximinng/DiffSketcher)]

**Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/72543)] 
[[Website](https://arxiv.org/abs/2306.08877)] 
[[Code](https://github.com/RoyiRa/Syntax-Guided-Generation)]

**DemoFusion: Democratising High-Resolution Image Generation With No $$$** \
[[CVPR 2024](https://arxiv.org/abs/2311.16973)] 
[[Project](https://ruoyidu.github.io/demofusion/demofusion.html)] 
[[Code](https://github.com/PRIS-CV/DemoFusion)] 

**Towards Effective Usage of Human-Centric Priors in Diffusion Models for Text-based Human Image Generation** \
[[CVPR 2024](https://arxiv.org/abs/2403.05239)] 
[[Project](https://hcplayercvpr2024.github.io/)] 
[[Code](https://github.com/hcplayercvpr2024/hcplayer)] 

**Divide & Bind Your Attention for Improved Generative Semantic Nursing**\
[[BMVC 2023 Oral](https://arxiv.org/abs/2307.10864)] 
[[Project](https://sites.google.com/view/divide-and-bind)] 
[[Code](https://github.com/boschresearch/Divide-and-Bind)] 

**Make It Count: Text-to-Image Generation with an Accurate Number of Objects** \
[[Website](https://arxiv.org/abs/2406.10210)] 
[[Project](https://make-it-count-paper.github.io//)] 
[[Code](https://github.com/Litalby1/make-it-count)]

**OmniBooth: Learning Latent Control for Image Synthesis with Multi-modal Instruction** \
[[Website](https://arxiv.org/abs/2410.04932)] 
[[Project](https://len-li.github.io/omnibooth-web/)] 
[[Code](https://github.com/EnVision-Research/OmniBooth)]

**Margin-aware Preference Optimization for Aligning Diffusion Models without Reference** \
[[Website](https://arxiv.org/abs/2406.06424)] 
[[Project](https://mapo-t2i.github.io/)] 
[[Code](https://github.com/mapo-t2i/mapo)]

**Step-aware Preference Optimization: Aligning Preference with Denoising Performance at Each Step** \
[[Website](https://arxiv.org/abs/2406.04314)] 
[[Project](https://rockeycoss.github.io/spo.github.io/)] 
[[Code](https://github.com/RockeyCoss/SPO)]

**Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2403.07860)] 
[[Project](https://shihaozhaozsh.github.io/LaVi-Bridge/)] 
[[Code](https://github.com/ShihaoZhaoZSH/LaVi-Bridge)]

**CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching** \
[[Website](https://arxiv.org/abs/2404.03653)] 
[[Project](https://caraj7.github.io/comat/)] 
[[Code](https://github.com/CaraJ7/CoMat)]

**Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions** \
[[Website](https://arxiv.org/abs/2403.17064)] 
[[Project](https://compvis.github.io/attribute-control/)] 
[[Code](https://github.com/CompVis/attribute-control)]

**Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance** \
[[Website](https://arxiv.org/abs/2403.17377)] 
[[Project](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/)] 
[[Code](https://github.com/KU-CVLAB/Perturbed-Attention-Guidance)]

**Real-World Image Variation by Aligning Diffusion Inversion Chain** \
[[Website](https://arxiv.org/abs/2305.18729)] 
[[Project](https://rival-diff.github.io/)] 
[[Code](https://github.com/julianjuaner/RIVAL/)]

**FreeU: Free Lunch in Diffusion U-Net** \
[[Website](https://arxiv.org/abs/2309.11497)] 
[[Project](https://chenyangsi.top/FreeU/)] 
[[Code](https://github.com/ChenyangSi/FreeU)]

**ConceptLab: Creative Generation using Diffusion Prior Constraints** \
[[Website](https://arxiv.org/abs/2308.02669)] 
[[Project](https://kfirgoldberg.github.io/ConceptLab/)] 
[[Code](https://github.com/kfirgoldberg/ConceptLab)]

**Aligning Text-to-Image Diffusion Models with Reward Backpropagationn** \
[[Website](https://arxiv.org/abs/2310.03739)] 
[[Project](https://align-prop.github.io/)] 
[[Code](https://github.com/mihirp1998/AlignProp/)]

**Mini-DALLE3: Interactive Text to Image by Prompting Large Language Models** \
[[Website](https://arxiv.org/abs/2310.07653)] 
[[Project](https://minidalle3.github.io/)] 
[[Code](https://github.com/Zeqiang-Lai/Mini-DALLE3)]

**ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2310.07702)] 
[[Project](https://yingqinghe.github.io/scalecrafter/)] 
[[Code](https://github.com/YingqingHe/ScaleCrafter)]

**One More Step: A Versatile Plug-and-Play Module for Rectifying Diffusion Schedule Flaws and Enhancing Low-Frequency Controls** \
[[Website](https://arxiv.org/abs/2311.15744)] 
[[Project](https://jabir-zheng.github.io/OneMoreStep/)] 
[[Code](https://github.com/mhh0318/OneMoreStep)]

**TokenCompose: Grounding Diffusion with Token-level Supervision**\
[[Website](https://arxiv.org/abs/2312.03626)] 
[[Project](https://mlpc-ucsd.github.io/TokenCompose/)] 
[[Code](https://github.com/mlpc-ucsd/TokenCompose)]

**DiffusionGPT: LLM-Driven Text-to-Image Generation System** \
[[Website](https://arxiv.org/abs/2401.10061)] 
[[Project](https://diffusiongpt.github.io/)] 
[[Code](https://github.com/DiffusionGPT/DiffusionGPT)]

**Decompose and Realign: Tackling Condition Misalignment in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.14408)] 
[[Project](https://wileewang.github.io/Decompose-and-Realign/)] 
[[Code](https://github.com/EnVision-Research/Decompose-and-Realign)]

**Taiyi-Diffusion-XL: Advancing Bilingual Text-to-Image Generation with Large Vision-Language Model Support** \
[[Website](https://arxiv.org/abs/2401.14688)] 
[[Project](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B)] 
[[Code](https://github.com/IDEA-CCNL/Fooocus-Taiyi-XL)]

**ECLIPSE: A Resource-Efficient Text-to-Image Prior for Image Generations** \
[[Website](https://arxiv.org/abs/2312.04655)] 
[[Project](https://eclipse-t2i.vercel.app/)] 
[[Code](https://github.com/eclipse-t2i/eclipse-inference)]

**MuLan: Multimodal-LLM Agent for Progressive Multi-Object Diffusion** \
[[Website](https://arxiv.org/abs/2402.12741)] 
[[Project](https://measure-infinity.github.io/mulan/)] 
[[Code](https://github.com/measure-infinity/mulan-code)]

**ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.02084)] 
[[Project](https://res-adapter.github.io/)] 
[[Code](https://github.com/bytedance/res-adapter)]

**Stylus: Automatic Adapter Selection for Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.18928)] 
[[Project](https://stylus-diffusion.github.io/)] 
[[Code](https://github.com/stylus-diffusion/stylus)]

**MaxFusion: Plug&Play Multi-Modal Generation in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.09977)] 
[[Project](https://nithin-gk.github.io/maxfusion.github.io/)] 
[[Code](https://github.com/Nithin-GK/MaxFusion)]

**Iterative Object Count Optimization for Text-to-image Diffusion Models** \
[[Website](https://arxiv.org/abs/2408.11721)] 
[[Project](https://ozzafar.github.io/count_token/)] 
[[Code](https://github.com/ozzafar/discriminative_class_tokens_for_counting)]

**ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment** \
[[Website](https://arxiv.org/abs/2403.05135)] 
[[Project](https://ella-diffusion.github.io/)] 
[[Code](https://github.com/ELLA-Diffusion/ELLA)]

**HiPrompt: Tuning-free Higher-Resolution Generation with Hierarchical MLLM Prompts** \
[[Website](https://arxiv.org/abs/2409.02919)] 
[[Project](https://liuxinyv.github.io/HiPrompt/)] 
[[Code](https://github.com/Liuxinyv/HiPrompt)]


**TheaterGen: Character Management with LLM for Consistent Multi-turn Image Generation** \
[[Website](https://arxiv.org/abs/2404.18919)] 
[[Project](https://howe140.github.io/theatergen.io/)] 
[[Code](https://github.com/donahowe/Theatergen)]

**SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models** \
[[ACM MM 2023 Oral](https://arxiv.org/abs/2305.05189)] 
[[Code](https://github.com/Qrange-group/SUR-adapter)]

**Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models** \
[[ICLR 2024](https://arxiv.org/abs/2402.05375)] 
[[Code](https://github.com/sen-mao/SuppressEOT)]

**Dynamic Prompt Optimizing for Text-to-Image Generation** \
[[CVPR 2024](https://arxiv.org/abs/2404.04095)] 
[[Code](https://github.com/Mowenyii/PAE)]

**Tackling the Singularities at the Endpoints of Time Intervals in Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.08381)] 
[[Code](https://github.com/PangzeCheung/SingDiffusion)]

**Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance** \
[[CVPR 2024](https://arxiv.org/abs/2404.05384)] 
[[Code](https://github.com/SmilesDZgk/S-CFG)]

**InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization** \
[[CVPR 2024](https://arxiv.org/abs/2404.04650)] 
[[Code](https://github.com/xiefan-guo/initno)]

**Object-Conditioned Energy-Based Attention Map Alignment in Text-to-Image Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2404.07389)] 
[[Code](https://github.com/YasminZhang/EBAMA/tree/master)]

**Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function** \
[[NeurIPS 2024](https://arxiv.org/abs/2409.19967)] 
[[Code](https://github.com/I2-Multimedia-Lab/Magnet)]

**SePPO: Semi-Policy Preference Optimization for Diffusion Alignment** \
[[Website](https://arxiv.org/abs/2410.05255)] 
[[Code](https://github.com/DwanZhang-AI/SePPO)]

**Prompt-Consistency Image Generation (PCIG): A Unified Framework Integrating LLMs, Knowledge Graphs, and Controllable Diffusion Models** \
[[Website](https://arxiv.org/abs/2406.16333)] 
[[Code](https://github.com/TruthAI-Lab/PCIG)]

**Improving Long-Text Alignment for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.11817)] 
[[Code](https://github.com/luping-liu/LongAlign)]

**Diffusion-RPO: Aligning Diffusion Models through Relative Preference Optimization** \
[[Website](https://arxiv.org/abs/2406.06382)] 
[[Code](https://github.com/yigu1008/Diffusion-RPO)]

**RealisHuman: A Two-Stage Approach for Refining Malformed Human Parts in Generated Images** \
[[Website](https://arxiv.org/abs/2409.03644)] 
[[Code](https://github.com/Wangbenzhi/RealisHuman)]

**Aggregation of Multi Diffusion Models for Enhancing Learned Representations** \
[[Website](https://arxiv.org/abs/2410.01262)] 
[[Code](https://github.com/hammour-steak/amdm)]

**AID: Attention Interpolation of Text-to-Image Diffusion** \
[[Website](https://arxiv.org/abs/2403.17924)] 
[[Code](https://github.com/QY-H00/attention-interpolation-diffusion)]

**FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis** \
[[Website](https://arxiv.org/abs/2403.12963)] 
[[Code](https://github.com/LeonHLJ/FouriScale)]

**ORES: Open-vocabulary Responsible Visual Synthesis** \
[[Website](https://arxiv.org/abs/2308.13785)] 
[[Code](https://github.com/kodenii/ores)]

**Fair Diffusion: Instructing Text-to-Image Generation Models on Fairness** \
[[Website](https://arxiv.org/abs/2302.10893)] 
[[Code](https://github.com/ml-research/fair-diffusion)]

**Understanding and Mitigating Compositional Issues in Text-to-Image Generative Models** \
[[Website](https://arxiv.org/abs/2406.07844)] 
[[Code](https://github.com/ArmanZarei/Mitigating-T2I-Comp-Issues)]

**IterComp: Iterative Composition-Aware Feedback Learning from Model Gallery for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2410.07171)] 
[[Code](https://github.com/YangLing0818/IterComp)]

**InstructG2I: Synthesizing Images from Multimodal Attributed Graphs** \
[[Website](https://arxiv.org/abs/2410.07157)] 
[[Code](https://github.com/PeterGriffinJin/InstructG2I)]


**Detector Guidance for Multi-Object Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2306.02236)] 
[[Code](https://github.com/luping-liu/Detector-Guidance)]

**Designing a Better Asymmetric VQGAN for StableDiffusion** \
[[Website](https://arxiv.org/abs/2306.04632)] 
[[Code](https://github.com/buxiangzhiren/Asymmetric_VQGAN)]

**FABRIC: Personalizing Diffusion Models with Iterative Feedback** \
[[Website](https://arxiv.org/abs/2307.10159)] 
[[Code](https://github.com/sd-fabric/fabric)]


**Prompt-Free Diffusion: Taking "Text" out of Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.16223)] 
[[Code](https://github.com/SHI-Labs/Prompt-Free-Diffusion)] 

**Progressive Text-to-Image Diffusion with Soft Latent Direction** \
[[Website](https://arxiv.org/abs/2309.09466)] 
[[Code](https://github.com/babahui/progressive-text-to-image)] 

**Hypernymy Understanding Evaluation of Text-to-Image Models via WordNet Hierarchy** \
[[Website](https://arxiv.org/abs/2310.09247)] 
[[Code](https://github.com/yandex-research/text-to-img-hypernymy)]

**TraDiffusion: Trajectory-Based Training-Free Image Generation** \
[[Website](https://arxiv.org/abs/2408.09739)] 
[[Code](https://github.com/och-mac/TraDiffusion)]

**If at First You Don’t Succeed, Try, Try Again:Faithful Diffusion-based Text-to-Image Generation by Selection** \
[[Website](https://arxiv.org/abs/2305.13308)] 
[[Code](https://github.com/ExplainableML/ImageSelect)]

**LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts** \
[[Website](https://arxiv.org/abs/2310.10640)] 
[[Code](https://github.com/hananshafi/llmblueprint)]

**Making Multimodal Generation Easier: When Diffusion Models Meet LLMs** \
[[Website](https://arxiv.org/abs/2310.08949)] 
[[Code](https://github.com/zxy556677/EasyGen)]

**Enhancing Diffusion Models with Text-Encoder Reinforcement Learning** \
[[Website](https://arxiv.org/abs/2311.15657)] 
[[Code](https://github.com/chaofengc/texforce)]

**AltDiffusion: A Multilingual Text-to-Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2308.09991)] 
[[Code](https://github.com/superhero-7/AltDiffusion)]

**It is all about where you start: Text-to-image generation with seed selection** \
[[Website](https://arxiv.org/abs/2304.14530)] 
[[Code](https://github.com/dvirsamuel/SeedSelect)]

**End-to-End Diffusion Latent Optimization Improves Classifier Guidance** \
[[Website](https://arxiv.org/abs/2303.13703)] 
[[Code](https://github.com/salesforce/doodl)]

**Correcting Diffusion Generation through Resampling** \
[[Website](https://arxiv.org/abs/2312.06038)] 
[[Code](https://github.com/ucsb-nlp-chang/diffusion_resampling)]

**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs** \
[[Website](https://arxiv.org/abs/2401.11708)] 
[[Code](https://github.com/YangLing0818/RPG-DiffusionMaster)]

**A User-Friendly Framework for Generating Model-Preferred Prompts in Text-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2402.12760)] 
[[Code](https://github.com/naylenv/uf-fgtg)] 

**PromptCharm: Text-to-Image Generation through Multi-modal Prompting and Refinement** \
[[Website](https://arxiv.org/abs/2403.04014)] 
[[Code](https://github.com/ma-labo/promptcharm)] 

**Enhancing Semantic Fidelity in Text-to-Image Synthesis: Attention Regulation in Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.06381)] 
[[Code](https://github.com/YaNgZhAnG-V5/attention_regulation)] 

**Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2403.07860)] 
[[Code](https://github.com/ShihaoZhaoZSH/LaVi-Bridge)] 

**LightIt: Illumination Modeling and Control for Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.10615)] 
[[Project](https://peter-kocsis.github.io/LightIt/)] 

**ComfyGen: Prompt-Adaptive Workflows for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2410.01731)] 
[[Project](https://comfygen-paper.github.io/)] 

**LLM4GEN: Leveraging Semantic Representation of LLMs for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2407.00737)] 
[[Project](https://xiaobul.github.io/LLM4GEN/)] 

**RefDrop: Controllable Consistency in Image or Video Generation via Reference Feature Guidance** \
[[Website](https://arxiv.org/abs/2405.17661)] 
[[Project](https://sbyebss.github.io/refdrop/)] 

**UniFL: Improve Stable Diffusion via Unified Feedback Learning** \
[[Website](https://arxiv.org/abs/2404.05595)] 
[[Project](https://uni-fl.github.io/)] 

**Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2403.16990)] 
[[Project](https://omer11a.github.io/bounded-attention/)] 

**Semantic Guidance Tuning for Text-To-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.15964)] 
[[Project](https://korguy.github.io/)] 

**Amazing Combinatorial Creation: Acceptable Swap-Sampling for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2310.01819)] 
[[Project](https://asst2i.github.io/anon/)] 

**Image Anything: Towards Reasoning-coherent and Training-free Multi-modal Image Generation** \
[[Website](https://arxiv.org/abs/2401.17664)] 
[[Project](https://vlislab22.github.io/ImageAnything/)] 

**Make a Cheap Scaling: A Self-Cascade Diffusion Model for Higher-Resolution Adaptation** \
[[Website](https://arxiv.org/abs/2402.10491)] 
[[Project](https://guolanqing.github.io/Self-Cascade/)] 

**FineDiffusion: Scaling up Diffusion Models for Fine-grained Image Generation with 10,000 Classes** \
[[Website](https://arxiv.org/abs/2402.18331)] 
[[Project](https://finediffusion.github.io/)] 

**Lazy Diffusion Transformer for Interactive Image Editing** \
[[Website](https://arxiv.org/abs/2404.12382)] 
[[Project](https://lazydiffusion.github.io/)] 

**Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis** \
[[Website](https://arxiv.org/abs/2404.13686)] 
[[Project](https://hyper-sd.github.io/)] 

**Concept Arithmetics for Circumventing Concept Inhibition in Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.13706)] 
[[Project](https://cs-people.bu.edu/vpetsiuk/arc/)] 

**Norm-guided latent space exploration for text-to-image generation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70922)] 
[[Website](https://arxiv.org/abs/2306.08687)] 

**Improving Diffusion-Based Image Synthesis with Context Prediction** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70058)] 
[[Website](https://arxiv.org/abs/2401.02015)] 

**GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections** \
[[ECCV 2024](https://arxiv.org/abs/2408.12352)] 

**MultiGen: Zero-shot Image Generation from Multi-modal Prompt** \
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01296.pdf)] 

**On Mechanistic Knowledge Localization in Text-to-Image Generative Models** \
[[ICML 2024](https://arxiv.org/abs/2405.01008)] 

**Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation** \
[[NeurIPS 2024](https://arxiv.org/abs/2410.00447)]

**A Cat Is A Cat (Not A Dog!): Unraveling Information Mix-ups in Text-to-Image Encoders through Causal Analysis and Embedding Optimization** \
[[Website](https://arxiv.org/abs/2410.00321)] 


**Diffusion Model Alignment Using Direct Preference Optimization** \
[[Website](https://arxiv.org/abs/2311.12908)] 

**Exposure Diffusion: HDR Image Generation by Consistent LDR denoising** \
[[Website](https://arxiv.org/abs/2405.14304)] 

**Information Theoretic Text-to-Image Alignment** \
[[Website](https://arxiv.org/abs/2405.20759)] 

**Diffscaler: Enhancing the Generative Prowess of Diffusion Transformers** \
[[Website](https://arxiv.org/abs/2404.09976)] 

**Object-Attribute Binding in Text-to-Image Generation: Evaluation and Control** \
[[Website](https://arxiv.org/abs/2404.13766)] 

**Aligning Diffusion Models by Optimizing Human Utility** \
[[Website](https://arxiv.org/abs/2404.04465)] 

**Instruct-Imagen: Image Generation with Multi-modal Instruction** \
[[Website](https://arxiv.org/abs/2401.01952)] 

**CONFORM: Contrast is All You Need For High-Fidelity Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.06059)] 

**MaskDiffusion: Boosting Text-to-Image Consistency with Conditional Mask** \
[[Website](https://arxiv.org/abs/2309.04399)] 

**Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images** \
[[Website](https://arxiv.org/abs/2308.16582)] 

**Text2Layer: Layered Image Generation using Latent Diffusion Model** \
[[Website](https://arxiv.org/abs/2307.09781)] 

**Stimulating the Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling** \
[[Website](https://arxiv.org/abs/2307.03992)] 

**A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation** \
[[Website](https://arxiv.org/abs/2310.16656)] 

**UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion** \
[[Website](https://arxiv.org/abs/2401.13388)] 


**Improving Compositional Text-to-image Generation with Large Vision-Language Models** \
[[Website](https://arxiv.org/abs/2310.06311)] 

**Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else** \
[[Website](https://arxiv.org/abs/2310.07419)] 

**Unseen Image Synthesis with Diffusion Models** \
[[Website](https://arxiv.org/abs/2310.09213)] 

**AnyLens: A Generative Diffusion Model with Any Rendering Lens** \
[[Website](https://arxiv.org/abs/2311.17609)] 

**Seek for Incantations: Towards Accurate Text-to-Image Diffusion Synthesis through Prompt Engineering** \
[[Website](https://arxiv.org/abs/2401.06345)] 

**Text2Street: Controllable Text-to-image Generation for Street Views** \
[[Website](https://arxiv.org/abs/2402.04504)] 

**Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2402.10210)] 

**Contrastive Prompts Improve Disentanglement in Text-to-Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2402.13490)] 

**Debiasing Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.14577)] 

**Stochastic Conditional Diffusion Models for Semantic Image Synthesis** \
[[Website](https://arxiv.org/abs/2402.16506)] 

**Referee Can Play: An Alternative Approach to Conditional Generation via Model Inversion** \
[[Website](https://arxiv.org/abs/2402.16305)] 

**Transparent Image Layer Diffusion using Latent Transparency** \
[[Website](https://arxiv.org/abs/2402.17113)] 

**Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2402.17245)] 

**HanDiffuser: Text-to-Image Generation With Realistic Hand Appearances** \
[[Website](https://arxiv.org/abs/2403.01693)] 

**StereoDiffusion: Training-Free Stereo Image Generation Using Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.04965)] 

**Make Me Happier: Evoking Emotions Through Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.08255)] 

**Zippo: Zipping Color and Transparency Distributions into a Single Diffusion Model** \
[[Website](https://arxiv.org/abs/2403.11077)] 

**LayerDiff: Exploring Text-guided Multi-layered Composable Image Synthesis via Layer-Collaborative Diffusion Model** \
[[Website](https://arxiv.org/abs/2403.11929)] 

**AGFSync: Leveraging AI-Generated Feedback for Preference Optimization in Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2403.13352)] 

**U-Sketch: An Efficient Approach for Sketch to Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.18425)] 

**ECNet: Effective Controllable Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.18417)] 

**TextCraftor: Your Text Encoder Can be Image Quality Controller** \
[[Website](https://arxiv.org/abs/2403.18978)] 

**Prompt Optimizer of Text-to-Image Diffusion Models for Abstract Concept Understanding** \
[[Website](https://arxiv.org/abs/2404.11589)] 

**Towards Better Text-to-Image Generation Alignment via Attention Modulation** \
[[Website](https://arxiv.org/abs/2404.13899)] 

**Towards Understanding the Working Mechanism of Text-to-Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2405.15330)] 

**SG-Adapter: Enhancing Text-to-Image Generation with Scene Graph Guidance** \
[[Website](https://arxiv.org/abs/2405.15321)] 

**Improving Geo-diversity of Generated Images with Contextualized Vendi Score Guidance** \
[[Website](https://arxiv.org/abs/2406.04551)] 

**Lost in Translation: Latent Concept Misalignment in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2408.00230)] 

**FRAP: Faithful and Realistic Text-to-Image Generation with Adaptive Prompt Weighting** \
[[Website](https://arxiv.org/abs/2408.11706)] 

**Foodfusion: A Novel Approach for Food Image Composition via Diffusion Models** \
[[Website](https://arxiv.org/abs/2408.14135)] 

**SPDiffusion: Semantic Protection Diffusion for Multi-concept Text-to-image Generation** \
[[Website](https://arxiv.org/abs/2409.01327)] 

**Training-Free Sketch-Guided Diffusion with Latent Optimization** \
[[Website](https://arxiv.org/abs/2409.00313)] 

**Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization** \
[[Website](https://arxiv.org/abs/2410.03190)] 

**Sparse Repellency for Shielded Generation in Text-to-image Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.06025)] 

**Training-free Diffusion Model Alignment with Sampling Demons** \
[[Website](https://arxiv.org/abs/2410.05760)] 

**MinorityPrompt: Text to Minority Image Generation via Prompt Optimization** \
[[Website](https://arxiv.org/abs/2410.07838)] 


## Spatial Control

**MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation** \
[[ICML 2023](https://icml.cc/virtual/2023/poster/23809)] 
[[ICML 2023](https://dl.acm.org/doi/10.5555/3618408.3618482)] 
[[Website](https://arxiv.org/abs/2302.08113)] 
[[Project](https://multidiffusion.github.io/)] 
[[Code](https://github.com/omerbt/MultiDiffusion)]
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_panorama.py)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/panorama)] 
[[Replicate Demo](https://replicate.com/omerbt/multidiffusion)]

**SceneComposer: Any-Level Semantic Image Synthesis** \
[[CVPR 2023 Highlight](https://openaccess.thecvf.com/content/CVPR2023/papers/Zeng_SceneComposer_Any-Level_Semantic_Image_Synthesis_CVPR_2023_paper.pdf)] 
[[Website](https://arxiv.org/abs/2211.11742)] 
[[Project](https://zengyu.me/scenec/)] 
[[Code](https://github.com/zengxianyu/scenec)]

**GLIGEN: Open-Set Grounded Text-to-Image Generation** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_GLIGEN_Open-Set_Grounded_Text-to-Image_Generation_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2301.07093)] 
[[Code](https://github.com/gligen/GLIGEN)]
[[Demo](https://huggingface.co/spaces/gligen/demo)]

**Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis** \
[[ICLR 2023](https://openreview.net/forum?id=PUIqjT4rzq7)] 
[[Website](https://arxiv.org/abs/2212.05032)] 
[[Project](https://weixi-feng.github.io/structure-diffusion-guidance/)] 
[[Code](https://github.com/shunk031/training-free-structured-diffusion-guidance)]

**Visual Programming for Text-to-Image Generation and Evaluation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/69940)] 
[[Website](https://arxiv.org/abs/2305.15328)] 
[[Project](https://vp-t2i.github.io/)] 
[[Code](https://github.com/j-min/VPGen)]


**GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation** \
[[ICLR 2024](https://openreview.net/forum?id=xBfQZWeDRH)] 
[[Website](https://arxiv.org/abs/2306.04607)] 
[[Project](https://kaichen1998.github.io/projects/geodiffusion/)] 
[[Code](https://github.com/KaiChen1998/GeoDiffusion/tree/main)]


**ReCo: Region-Controlled Text-to-Image Generation** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_ReCo_Region-Controlled_Text-to-Image_Generation_CVPR_2023_paper.pdf)] 
[[Website](https://arxiv.org/abs/2211.15518)] 
[[Code](https://github.com/microsoft/ReCo)]

**Harnessing the Spatial-Temporal Attention of Diffusion Models for High-Fidelity Text-to-Image Synthesis** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Harnessing_the_Spatial-Temporal_Attention_of_Diffusion_Models_for_High-Fidelity_Text-to-Image_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2304.03869)] 
[[Code](https://github.com/UCSB-NLP-Chang/Diffusion-SpaceTime-Attn)]

**BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Xie_BoxDiff_Text-to-Image_Synthesis_with_Training-Free_Box-Constrained_Diffusion_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2307.10816)] 
[[Code](https://github.com/Sierkinhane/BoxDiff)]

**Dense Text-to-Image Generation with Attention Modulation** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_Dense_Text-to-Image_Generation_with_Attention_Modulation_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2308.12964)] 
[[Code](https://github.com/naver-ai/densediffusion)]

**LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models** \
[[Website](https://arxiv.org/abs/2305.13655)] 
[[Project](https://llm-grounded-diffusion.github.io/)] 
[[Code](https://github.com/TonyLianLong/LLM-groundedDiffusion)]
[[Demo](https://huggingface.co/spaces/longlian/llm-grounded-diffusion)]
[[Blog](https://bair.berkeley.edu/blog/2023/05/23/lmd/)] 

**StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control** \
[[CVPR 2024](https://arxiv.org/abs/2403.09055)] 
[[Code](https://github.com/ironjr/StreamMultiDiffusion)]
[[Project](https://jaerinlee.com/research/streammultidiffusion)] 

**MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis** \
[[CVPR 2024](https://arxiv.org/abs/2402.05408)] 
[[Project](https://migcproject.github.io/)] 
[[Code](https://github.com/limuloo/MIGC)]

**Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language** \
[[Website](https://arxiv.org/abs/2406.20085)] 
[[Project](https://yichengchen24.github.io/projects/autocherrypicker/)] 
[[Code](https://github.com/yichengchen24/ACP)]

**Training-Free Layout Control with Cross-Attention Guidance** \
[[Website](https://arxiv.org/abs/2304.03373)] 
[[Project](https://hohonu-vicml.github.io/DirectedDiffusion.Page/)] 
[[Code](https://github.com/hohonu-vicml/DirectedDiffusion)]


**Directed Diffusion: Direct Control of Object Placement through Attention Guidance** \
[[Website](https://arxiv.org/abs/2302.13153)] 
[[Project](https://silent-chen.github.io/layout-guidance/)] 
[[Code](https://github.com/silent-chen/layout-guidance)]

**Grounded Text-to-Image Synthesis with Attention Refocusing** \
[[Website](https://arxiv.org/abs/2306.05427)] 
[[Project](https://attention-refocusing.github.io/)] 
[[Code](https://github.com/Attention-Refocusing/attention-refocusing)]

**eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers** \
[[Website](https://arxiv.org/abs/2211.01324)] 
[[Project](https://research.nvidia.com/labs/dir/eDiff-I/)] 
[[Code](https://github.com/cloneofsimo/paint-with-words-sd)]

**LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2304.03373)] 
[[Project](https://layoutllm-t2i.github.io/)] 
[[Code](https://github.com/LayoutLLM-T2I/LayoutLLM-T2I)]

**Compositional Text-to-Image Synthesis with Attention Map Control of Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.13921)] 
[[Project](https://oppo-mente-lab.github.io/compositional_t2i/)] 
[[Code](https://github.com/OPPO-Mente-Lab/attention-mask-control)]

**R&B: Region and Boundary Aware Zero-shot Grounded Text-to-image Generation** \
[[Website](https://arxiv.org/abs/2310.08872)] 
[[Project](https://sagileo.github.io/Region-and-Boundary/)] 
[[Code](https://github.com/StevenShaw1999/RnB)]

**FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition** \
[[Website](https://arxiv.org/abs/2312.07536)] 
[[Project](https://genforce.github.io/freecontrol/)] 
[[Code](https://github.com/genforce/freecontrol)]

**InteractDiffusion: Interaction Control in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.05849)] 
[[Project](https://jiuntian.github.io/interactdiffusion/)] 
[[Code](https://github.com/jiuntian/interactdiffusion)]

**Ranni: Taming Text-to-Image Diffusion for Accurate Instruction Following** \
[[Website](https://arxiv.org/abs/2311.17002)] 
[[Project](https://ranni-t2i.github.io/Ranni/)] 
[[Code](https://github.com/ali-vilab/Ranni)]

**InstanceDiffusion: Instance-level Control for Image Generation** \
[[Website](https://arxiv.org/abs/2402.03290)] 
[[Project](https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/)] 
[[Code](https://github.com/frank-xwang/InstanceDiffusion)]

**Coarse-to-Fine Latent Diffusion for Pose-Guided Person Image Synthesis** \
[[CVPR 2024](https://arxiv.org/abs/2402.18078)] 
[[Code](https://github.com/YanzuoLu/CFLD)]

**NoiseCollage: A Layout-Aware Text-to-Image Diffusion Model Based on Noise Cropping and Merging** \
[[CVPR 2024](https://arxiv.org/abs/2403.03485)] 
[[Code](https://github.com/univ-esuty/noisecollage)]

**Masked-Attention Diffusion Guidance for Spatially Controlling Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2308.06027)] 
[[Code](https://github.com/endo-yuki-t/MAG)]

**Rethinking The Training And Evaluation of Rich-Context Layout-to-Image Generation** \
[[Website](https://arxiv.org/abs/2409.04847)] 
[[Code](https://github.com/cplusx/rich_context_L2I/tree/main)]

**Enhancing Object Coherence in Layout-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2311.10522)] 
[[Code](https://github.com/CodeGoat24/EOCNet)]

**DivCon: Divide and Conquer for Progressive Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2403.06400)] 
[[Code](https://github.com/DivCon-gen/DivCon)]

**RealCompo: Dynamic Equilibrium between Realism and Compositionality Improves Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.12908)] 
[[Code](https://github.com/YangLing0818/RealCompo)]

**StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control** \
[[Website](https://arxiv.org/abs/2403.09055)] 
[[Code](https://github.com/ironjr/StreamMultiDiffusion)]


**Layered Rendering Diffusion Model for Zero-Shot Guided Image Synthesis** \
[[ECCV 2024](https://arxiv.org/abs/2311.18435)] 
[[Project](https://qizipeng.github.io/LRDiff_projectPage/)] 


**ReCorD: Reasoning and Correcting Diffusion for HOI Generation** \
[[ACM MM 2024](https://arxiv.org/abs/2407.17911)] 
[[Project](https://alberthkyhky.github.io/ReCorD/)] 

**Compositional Text-to-Image Generation with Dense Blob Representations** \
[[Website](https://arxiv.org/abs/2405.08246)] 
[[Project](https://blobgen-2d.github.io/)] 

**GroundingBooth: Grounding Text-to-Image Customization** \
[[Website](https://arxiv.org/abs/2409.08520)] 
[[Project](https://groundingbooth.github.io/)] 

**Check, Locate, Rectify: A Training-Free Layout Calibration System for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2311.15773)] 
[[Project](https://simm-t2i.github.io/SimM/)] 

**ReGround: Improving Textual and Spatial Grounding at No Cost** \
[[Website](https://arxiv.org/abs/2403.13589)] 
[[Project](https://re-ground.github.io/)] 

**DetDiffusion: Synergizing Generative and Perceptive Models for Enhanced Data Generation and Perception** \
[[CVPR 2024](https://arxiv.org/abs/2403.13304)] 

**Guided Image Synthesis via Initial Image Editing in Diffusion Model** \
[[ACM MM 2023](https://arxiv.org/abs/2305.03382)] 

**Training-free Composite Scene Generation for Layout-to-Image Synthesis** \
[[ECCV 2024](https://arxiv.org/abs/2407.13609)] 

**LSReGen: Large-Scale Regional Generator via Backward Guidance Framework** \
[[Website](https://arxiv.org/abs/2407.15066)] 

**Enhancing Prompt Following with Visual Control Through Training-Free Mask-Guided Diffusion** \
[[Website](https://arxiv.org/abs/2404.14768)] 

**Draw Like an Artist: Complex Scene Generation with Diffusion Model via Composition, Painting, and Retouching** \
[[Website](https://arxiv.org/abs/2408.13858)] 

**Enhancing Image Layout Control with Loss-Guided Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.14101)] 

**GLoD: Composing Global Contexts and Local Details in Image Generation** \
[[Website](https://arxiv.org/abs/2404.15447)] 

**A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis** \
[[Website](https://arxiv.org/abs/2306.14544)] 

**Controllable Text-to-Image Generation with GPT-4** \
[[Website](https://arxiv.org/abs/2305.18583)] 

**Localized Text-to-Image Generation for Free via Cross Attention Control** \
[[Website](https://arxiv.org/abs/2306.14636)] 

**Training-Free Location-Aware Text-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2304.13427)] 

**Composite Diffusion | whole >= \Sigma parts** \
[[Website](https://arxiv.org/abs/2307.13720)] 

**Continuous Layout Editing of Single Images with Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.13078)] 

**Zero-shot spatial layout conditioning for text-to-image diffusion models** \
[[Website](https://arxiv.org/abs/2306.13754)] 

**Obtaining Favorable Layouts for Multiple Object Generation** \
[[Website](https://arxiv.org/abs/2405.00791)] 

**LoCo: Locally Constrained Training-Free Layout-to-Image Synthesis**\
[[Website](https://arxiv.org/abs/2311.12342)] 

**Self-correcting LLM-controlled Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.16090)] 

**Joint Generative Modeling of Scene Graphs and Images via Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.01130)] 

**Spatial-Aware Latent Initialization for Controllable Image Generation** \
[[Website](https://arxiv.org/abs/2401.16157)] 

**Layout-to-Image Generation with Localized Descriptions using ControlNet with Cross-Attention Control** \
[[Website](https://arxiv.org/abs/2402.13404)] 

**ObjBlur: A Curriculum Learning Approach With Progressive Object-Level Blurring for Improved Layout-to-Image Generation** \
[[Website](https://arxiv.org/abs/2404.07564)] 

**The Crystal Ball Hypothesis in diffusion models: Anticipating object positions from initial noise** \
[[Website](https://arxiv.org/abs/2406.01970)] 

**Zero-Painter: Training-Free Layout Control for Text-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2406.04032)] 

**SpotActor: Training-Free Layout-Controlled Consistent Image Generation** \
[[Website](https://arxiv.org/abs/2409.04801)] 

**IFAdapter: Instance Feature Control for Grounded Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2409.08240)] 

**Scribble-Guided Diffusion for Training-free Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2409.08026)] 


## I2I translation

⭐⭐⭐**SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** \
[[ICLR 2022](https://openreview.net/forum?id=aBsCjcPu_tE)] 
[[Website](https://arxiv.org/abs/2108.01073)] 
[[Project](https://sde-image-editing.github.io/)] 
[[Code](https://github.com/ermongroup/SDEdit)] 

⭐⭐⭐**DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation** \
[[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html)]
[[Website](https://arxiv.org/abs/2110.02711)]
[[Code](https://github.com/gwang-kim/DiffusionCLIP)]

**CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation** \
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/69913)]
[[Website](https://arxiv.org/abs/2310.13165)]
[[Project](https://cyclenetweb.github.io/)] 
[[Code](https://github.com/sled-group/cyclenet)]
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=z9d9DsjAPH)] -->

**DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations** \
[[CVPR 2024](https://arxiv.org/abs/2403.06951)]
[[Project](https://tianhao-qi.github.io/DEADiff/)]
[[Code](https://github.com/Tianhao-Qi/DEADiff_code)]

**Diffusion-based Image Translation using Disentangled Style and Content Representation** \
[[ICLR 2023](https://openreview.net/forum?id=Nayau9fwXU)]
[[Website](https://arxiv.org/abs/2209.15264)]
[[Code](https://github.com/cyclomon/DiffuseIT)]

**FlexIT: Towards Flexible Semantic Image Translation** \
[[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Couairon_FlexIT_Towards_Flexible_Semantic_Image_Translation_CVPR_2022_paper.html)]
[[Website](https://arxiv.org/abs/2203.04705)]
[[Code](https://github.com/facebookresearch/semanticimagetranslation)]

**Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Zero-Shot_Contrastive_Loss_for_Text-Guided_Diffusion_Image_Style_Transfer_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.08622)]
[[Code](https://github.com/YSerin/ZeCon)]


**E2GAN: Efficient Training of Efficient GANs for Image-to-Image Translation** \
[[ICML 2024](https://arxiv.org/abs/2401.06127)]
[[Project](https://yifanfanfanfan.github.io/e2gan/)]
[[Code](https://github.com/Yifanfanfanfan/Yifanfanfanfan.github.io/tree/main/e2gan)]

**Eye-for-an-eye: Appearance Transfer with Semantic Correspondence in Diffusion Models** \
[[Website](https://arxiv.org/abs/2406.07008)]
[[Project](https://sooyeon-go.github.io/eye_for_an_eye/)] 
[[Code](https://github.com/sooyeon-go/eye_for_an_eye)]

**Cross-Image Attention for Zero-Shot Appearance Transfer** \
[[Website](https://arxiv.org/abs/2311.03335)]
[[Project](https://garibida.github.io/cross-image-attention/)]
[[Code](https://github.com/garibida/cross-image-attention)]

**Diffusion Guided Domain Adaptation of Image Generators** \
[[Website](https://arxiv.org/abs/2212.04473)]
[[Project](https://styleganfusion.github.io/)] 
[[Code](https://github.com/KunpengSong/styleganfusion)]

**Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.12092)]
[[Project](https://sliders.baulab.info/)]
[[Code](https://github.com/rohitgandikota/sliders)]

**FreeStyle: Free Lunch for Text-guided Style Transfer using Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.15636)]
[[Project](https://freestylefreelunch.github.io/)]
[[Code](https://github.com/FreeStyleFreeLunch/FreeStyle)] 

**FilterPrompt: Guiding Image Transfer in Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.13263)]
[[Project](https://meaoxixi.github.io/FilterPrompt/)]
[[Code](https://github.com/Meaoxixi/FilterPrompt)] 

**Every Pixel Has its Moments: Ultra-High-Resolution Unpaired Image-to-Image Translation via Dense Normalization** \
[[ECCV 2024](https://arxiv.org/abs/2407.04245)]
[[Code](https://github.com/Kaminyou/Dense-Normalization)] 

**One-Shot Structure-Aware Stylized Image Synthesis** \
[[CVPR 2024](https://arxiv.org/abs/2402.17275)]
[[Code](https://github.com/hansam95/osasis)] 

**BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2205.07680)]
[[Code](https://github.com/xuekt98/BBDM)]

**Spectrum Translation for Refinement of Image Generation (STIG) Based on Contrastive Learning and Spectral Filter Profile** \
[[AAAI 2024](https://arxiv.org/abs/2403.05093)]
[[Code](https://github.com/ykykyk112/STIG)] 

**Frequency-Controlled Diffusion Model for Versatile Text-Guided Image-to-Image Translation** \
[[AAAI 2024](https://arxiv.org/abs/2407.03006)]
[[Code](https://github.com/XiangGao1102/FCDiffusion)] 

**ZePo: Zero-Shot Portrait Stylization with Faster Sampling** \
[[ACM MM 2024](https://arxiv.org/abs/2408.05492)]
[[Code](https://github.com/liujin112/ZePo)]

**Improving Diffusion-based Image Translation using Asymmetric Gradient Guidance** \
[[Website](https://arxiv.org/abs/2306.04396)]
[[Code](https://github.com/submissionanon18/agg)]

**Enabling Local Editing in Diffusion Models by Joint and Individual Component Analysis** \
[[Website](https://arxiv.org/abs/2408.16845)]
[[Code](https://zelaki.github.io/localdiff/)]

**PixWizard: Versatile Image-to-Image Visual Assistant with Open-Language Instructions** \
[[Website](https://arxiv.org/abs/2409.15278)]
[[Code](https://github.com/AFeng-x/PixWizard)]


**GEM: Boost Simple Network for Glass Surface Segmentation via Segment Anything Model and Data Synthesis** \
[[Website](https://arxiv.org/abs/2401.15282)]
[[Code](https://github.com/isbrycee/GEM-Glass-Segmentor)]

**CreativeSynth: Creative Blending and Synthesis of Visual Arts based on Multimodal Diffusion** \
[[Website](https://arxiv.org/abs/2401.14066)] 
[[Code](https://github.com/haha-lisa/creativesynth)] 

**PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering** \
[[Website](https://arxiv.org/abs/2403.05053)] 
[[Code](https://github.com/CodeGoat24/PrimeComposer)] 

**One-Step Image Translation with Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2403.12036)] 
[[Code](https://github.com/GaParmar/img2img-turbo)] 

**D2Styler: Advancing Arbitrary Style Transfer with Discrete Diffusion Methods** \
[[Website](https://arxiv.org/abs/2408.03558)] 
[[Code](https://github.com/Onkarsus13/D2Styler)] 

**StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_StyleDiffusion_Controllable_Disentangled_Style_Transfer_via_Diffusion_Models_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2308.07863)]

**ControlStyle: Text-Driven Stylized Image Generation Using Diffusion Priors** \
[[ACM MM 2023](https://arxiv.org/abs/2311.05463)]

**High-Fidelity Diffusion-based Image Editing** \
[[AAAI 2024](https://arxiv.org/abs/2312.15707)]

**EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models** \
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02096.pdf)]

**Harnessing the Latent Diffusion Model for Training-Free Image Style Transfer** \
[[Website](https://arxiv.org/abs/2410.01366)]

**UniHDA: Towards Universal Hybrid Domain Adaptation of Image Generators** \
[[Website](https://arxiv.org/abs/2401.12596)]

**Regularized Distribution Matching Distillation for One-step Unpaired Image-to-Image Translation** \
[[Website](https://arxiv.org/abs/2406.14762)]

**TEXTOC: Text-driven Object-Centric Style Transfer** \
[[Website](https://arxiv.org/abs/2408.08461)]

**Seed-to-Seed: Image Translation in Diffusion Seed Space** \
[[Website](https://arxiv.org/abs/2409.00654)]

**Diffusion-Based Image-to-Image Translation by Noise Correction via Prompt Interpolation** \
[[Website](https://arxiv.org/abs/2409.08077)]

## Segmentation Detection Tracking
**odise: open-vocabulary panoptic segmentation with text-to-image diffusion modelss** \
[[CVPR 2023 Highlight](https://arxiv.org/abs/2303.04803)] 
[[Project](https://jerryxu.net/ODISE/)] 
[[Code](https://github.com/NVlabs/ODISE)]
[[Demo](https://huggingface.co/spaces/xvjiarui/ODISE)]


**LD-ZNet: A Latent Diffusion Approach for Text-Based Image Segmentation** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zbinden_Stochastic_Segmentation_with_Conditional_Categorical_Diffusion_Models_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.12343)] 
[[Project](https://koutilya-pnvr.github.io/LD-ZNet/)]
[[Code](https://github.com/koutilya-pnvr/LD-ZNet)]

**Text-Image Alignment for Diffusion-Based Perception** \
[[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kondapaneni_Text-Image_Alignment_for_Diffusion-Based_Perception_CVPR_2024_paper.html)] 
[[Website](https://arxiv.org/abs/2310.00031)] 
[[Project](https://www.vision.caltech.edu/tadp/)] 
[[Code](https://github.com/damaggu/TADP)]

**Stochastic Segmentation with Conditional Categorical Diffusion Models**\
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zbinden_Stochastic_Segmentation_with_Conditional_Categorical_Diffusion_Models_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.08888)] 
[[Code](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)]

**DDP: Diffusion Model for Dense Visual Prediction**\
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ji_DDP_Diffusion_Model_for_Dense_Visual_Prediction_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.17559)]
[[Code](https://github.com/JiYuanFeng/DDP)]

**DiffusionDet: Diffusion Model for Object Detection** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2211.09788)] 
[[Code](https://github.com/shoufachen/diffusiondet)]

**OVTrack: Open-Vocabulary Multiple Object Tracking** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_OVTrack_Open-Vocabulary_Multiple_Object_Tracking_CVPR_2023_paper.html
)] 
[[Website](https://arxiv.org/abs/2304.08408)] 
[[Project](https://www.vis.xyz/pub/ovtrack/)] 

**SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/71719)] 
[[Website](https://arxiv.org/abs/2312.12425)] 
[[Code](https://github.com/MengyuWang826/SegRefiner)]

**DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction** \
[[CVPR 2024](https://arxiv.org/abs/2403.02075)] 
[[Project](https://diffmot.github.io/)] 
[[Code](https://github.com/Kroery/DiffMOT)]

**Zero-Shot Image Segmentation via Recursive Normalized Cut on Diffusion Features** \
[[Website](https://arxiv.org/abs/2406.02842)] 
[[Project](https://diffcut-segmentation.github.io/)] 
[[Code](https://github.com/PaulCouairon/DiffCut)]

**Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion** \
[[Website](https://arxiv.org/abs/2308.12469)] 
[[Project](https://sites.google.com/view/diffseg/home)] 
[[Code](https://github.com/PotatoTian/DiffSeg)]

**InstaGen: Enhancing Object Detection by Training on Synthetic Dataset** \
[[Website](https://arxiv.org/abs/2402.05937)] 
[[Project](https://fcjian.github.io/InstaGen/)] 
[[Code](https://github.com/fcjian/InstaGen)]

**InvSeg: Test-Time Prompt Inversion for Semantic Segmentation** \
[[Website](https://arxiv.org/abs/2410.11473)] 
[[Project](https://jylin8100.github.io/InvSegProject/)] 
[[Code](https://github.com/jyLin8100/InvSeg)]


**Exploring Phrase-Level Grounding with Text-to-Image Diffusion Model** \
[[ECCV 2024](https://arxiv.org/abs/2407.05352)] 
[[Code](https://github.com/nini0919/DiffPNG)]

**ConsistencyTrack: A Robust Multi-Object Tracker with a Generation Strategy of Consistency Model** \
[[Website](https://arxiv.org/abs/2408.15548)] 
[[Code](https://github.com/Tankowa/ConsistencyTrack)]

**SemFlow: Binding Semantic Segmentation and Image Synthesis via Rectified Flow** \
[[Website](https://arxiv.org/abs/2405.20282)] 
[[Code](https://github.com/wang-chaoyang/SemFlow)]

**Delving into the Trajectory Long-tail Distribution for Muti-object Tracking** \
[[Website](https://arxiv.org/abs/2403.04700)] 
[[Code](https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT)]

**Zero-Shot Video Semantic Segmentation based on Pre-Trained Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.16947)] 
[[Code](https://github.com/QianWangX/VidSeg_diffusion)]

**Scribble Hides Class: Promoting Scribble-Based Weakly-Supervised Semantic Segmentation with Its Class Label** \
[[Website](https://arxiv.org/abs/2402.17555)] 
[[Code](https://github.com/Zxl19990529/Class-driven-Scribble-Promotion-Network)]

**Personalize Segment Anything Model with One Shot** \
[[Website](https://arxiv.org/abs/2305.03048)] 
[[Code](https://github.com/ZrrSkywalker/Personalize-SAM)]

**DiffusionTrack: Diffusion Model For Multi-Object Tracking** \
[[Website](https://arxiv.org/abs/2308.09905)] 
[[Code](https://github.com/rainbowluocs/diffusiontrack)]


**MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation** \
[[Website](https://arxiv.org/abs/2309.13042)] 
[[Code](https://github.com/Jiahao000/MosaicFusion)]

**A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting** \
[[Website](https://arxiv.org/abs/2401.10227)]
[[Code](https://github.com/segments-ai/latent-diffusion-segmentation)]

**Beyond Generation: Harnessing Text to Image Models for Object Detection and Segmentation** \
[[Website](https://arxiv.org/abs/2309.05956)] 
[[Code](https://github.com/gyhandy/Text2Image-for-Detection)]

**UniGS: Unified Representation for Image Generation and Segmentation** \
[[Website](https://arxiv.org/abs/2312.01985)] 
[[Code](https://github.com/qqlu/Entity)]

**Placing Objects in Context via Inpainting for Out-of-distribution Segmentation**\
[[Website](https://arxiv.org/abs/2402.16392)] 
[[Code](https://github.com/naver/poc)]

**MaskDiffusion: Exploiting Pre-trained Diffusion Models for Semantic Segmentation** \
[[Website](https://arxiv.org/abs/2403.11194)] 
[[Code](https://github.com/Valkyrja3607/MaskDiffusion)]

**Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation** \
[[Website](https://arxiv.org/abs/2403.12042)] 
[[Code](https://github.com/buxiangzhiren/VD-IT)]

**Open-Vocabulary Attention Maps with Token Optimization for Semantic Segmentation in Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.14291)] 
[[Code](https://github.com/vpulab/ovam)]

**EmerDiff: Emerging Pixel-level Semantic Knowledge in Diffusion Models** \
[[ICLR 2024](https://openreview.net/forum?id=YqyTXmF8Y2)]
[[Website](https://arxiv.org/abs/2401.11739)] 
[[Project](https://kmcode1.github.io/Projects/EmerDiff/)]


**Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation** \
[[CVPR 2024](https://arxiv.org/abs/2404.06542)] 
[[Project](https://aimagelab.github.io/freeda/)] 

**FreeSeg-Diff: Training-Free Open-Vocabulary Segmentation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.20105)] 
[[Project](https://bcorrad.github.io/freesegdiff/)]


**DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.11681)] 
[[Project](https://weijiawu.github.io/DiffusionMask/)] 

**Diffusion-based Image Translation with Label Guidance for Domain Adaptive Semantic Segmentation** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_Diffusion-based_Image_Translation_with_Label_Guidance_for_Domain_Adaptive_Semantic_ICCV_2023_paper.html
)] 
[[Website](https://arxiv.org/abs/2308.12350)] 

**SDDGR: Stable Diffusion-based Deep Generative Replay for Class Incremental Object Detection** \
[[CVPR 2024](https://arxiv.org/abs/2402.17323)] 

**Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers** \
[[ECCV 2024](https://arxiv.org/abs/2407.08394)] 

**Unleashing the Potential of the Diffusion Model in Few-shot Semantic Segmentation** \
[[NeurIPS 2024](https://arxiv.org/abs/2410.02369)] 

**Generalization by Adaptation: Diffusion-Based Domain Extension for Domain-Generalized Semantic Segmentation** \
[[WACV 2024](https://arxiv.org/abs/2312.01850)] 

**Boosting Few-Shot Detection with Large Language Models and Layout-to-Image Synthesis** \
[[ACCV 2024](https://arxiv.org/abs/2410.06841)] 

**A Simple Background Augmentation Method for Object Detection with Diffusion Model** \
[[Website](https://arxiv.org/abs/2408.00350)] 

**Unveiling the Power of Diffusion Features For Personalized Segmentation and Retrieval** \
[[Website](https://arxiv.org/abs/2405.18025)] 

**SLiMe: Segment Like Me** \
[[Website](https://arxiv.org/abs/2309.03179)] 

**ASAM: Boosting Segment Anything Model with Adversarial Tuning** \
[[Website](https://arxiv.org/abs/2405.00256)] 

**Diffusion Features to Bridge Domain Gap for Semantic Segmentation** \
[[Website](https://arxiv.org/abs/2406.00777)] 

**MaskDiff: Modeling Mask Distribution with Diffusion Probabilistic Model for Few-Shot Instance Segmentation** \
[[Website](https://arxiv.org/abs/2303.05105)] 

**DiffusionSeg: Adapting Diffusion Towards Unsupervised Object Discovery** \
[[Website](https://arxiv.org/abs/2303.09813)] 

**Ref-Diff: Zero-shot Referring Image Segmentation with Generative Models** \
[[Website](https://arxiv.org/abs/2308.16777)] 

**Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter** \
[[Website](https://arxiv.org/abs/2309.02773)] 

**Attention as Annotation: Generating Images and Pseudo-masks for Weakly Supervised Semantic Segmentation with Diffusion** \
[[Website](https://arxiv.org/abs/2309.01369v1)] 

**From Text to Mask: Localizing Entities Using the Attention of Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2309.04109)] 

**Factorized Diffusion Architectures for Unsupervised Image Generation and Segmentation** \
[[Website](https://arxiv.org/abs/2309.15726)] 

**Patch-based Selection and Refinement for Early Object Detection** \
[[Website](https://arxiv.org/abs/2311.02274)] 

**TrackDiffusion: Multi-object Tracking Data Generation via Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.00651)] 

**Towards Granularity-adjusted Pixel-level Semantic Annotation** \
[[Website](https://arxiv.org/abs/2312.02420)] 

**Gen2Det: Generate to Detect** \
[[Website](https://arxiv.org/abs/2312.04566)] 

**Bridging Generative and Discriminative Models for Unified Visual Perception with Diffusion Priors** \
[[Website](https://arxiv.org/abs/2401.16459)] 

**ConsistencyDet: Robust Object Detector with Denoising Paradigm of Consistency Model** \
[[Website](https://arxiv.org/abs/2404.07773)] 

**Diverse Generation while Maintaining Semantic Coordination: A Diffusion-Based Data Augmentation Method for Object Detection** \
[[Website](https://arxiv.org/abs/2408.02891)] 

**Generative Edge Detection with Stable Diffusion** \
[[Website](https://arxiv.org/abs/2410.03080)] 


## Additional conditions 

⭐⭐⭐**Adding Conditional Control to Text-to-Image Diffusion Models** \
[[ICCV 2023 best paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2302.05543)] 
[[Official Code](https://github.com/lllyasviel/controlnet)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)] 


⭐⭐**T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.08453)] 
[[Official Code](https://github.com/TencentARC/T2I-Adapter)]
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/t2i_adapter)] 


**SketchKnitter: Vectorized Sketch Generation with Diffusion Models** \
[[ICLR 2023 Spotlight](https://openreview.net/forum?id=4eJ43EN2g6l&noteId=fxpTz_vCdO)]
[[ICLR 2023 Spotlight](https://iclr.cc/virtual/2023/poster/11832)]
[[Website](https://openreview.net/pdf?id=4eJ43EN2g6l)]
[[Code](https://github.com/XDUWQ/SketchKnitter/tree/75ded224e91f5ecf7e225c031b32cb97508443b9)]

**Freestyle Layout-to-Image Synthesis** \
[[CVPR 2023 highlight](https://openaccess.thecvf.com/content/CVPR2023/html/Xue_Freestyle_Layout-to-Image_Synthesis_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2303.14412)] 
[[Project](https://essunny310.github.io/FreestyleNet/)] 
[[Code](https://github.com/essunny310/freestylenet)]

**Collaborative Diffusion for Multi-Modal Face Generation and Editing** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Collaborative_Diffusion_for_Multi-Modal_Face_Generation_and_Editing_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2304.10530)] 
[[Project](https://ziqihuangg.github.io/projects/collaborative-diffusion.html)] 
[[Code](https://github.com/ziqihuangg/Collaborative-Diffusion)]

**HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ju_HumanSD_A_Native_Skeleton-Guided_Diffusion_Model_for_Human_Image_Generation_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2304.04269)]
[[Project](https://idea-research.github.io/HumanSD/)] 
[[Code]](https://github.com/IDEA-Research/HumanSD) 


**FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2303.09833)] 
[[Code](https://github.com/vvictoryuki/freedom)]


**Sketch-Guided Text-to-Image Diffusion Models** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2211.13752)] 
[[Project](https://sketch-guided-diffusion.github.io/)] 
[[Code]](https://github.com/Mikubill/sketch2img) 

**Adversarial Supervision Makes Layout-to-Image Diffusion Models Thrive** \
[[ICLR 2024](https://arxiv.org/abs/2401.08815)] 
[[Project](https://yumengli007.github.io/ALDM/)] 
[[Code](https://github.com/boschresearch/ALDM)]

**IPAdapter-Instruct: Resolving Ambiguity in Image-based Conditioning using Instruct Prompts** \
[[Website](https://arxiv.org/abs/2408.03209)] 
[[Project](https://unity-research.github.io/IP-Adapter-Instruct.github.io/)] 
[[Code](https://github.com/unity-research/IP-Adapter-Instruct)]

**ControlNeXt: Powerful and Efficient Control for Image and Video Generation** \
[[Website](https://arxiv.org/abs/2408.06070)] 
[[Project](https://pbihao.github.io/projects/controlnext/index.html)] 
[[Code](https://github.com/dvlab-research/ControlNeXt)]

**Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance** \
[[Website](https://arxiv.org/abs/2406.07540)] 
[[Project](https://genforce.github.io/ctrl-x/)] 
[[Code](https://github.com/genforce/ctrl-x)]

**Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model** \
[[Website](https://arxiv.org/abs/2404.09967)] 
[[Project](https://ctrl-adapter.github.io/)] 
[[Code](https://github.com/HL-hanlin/Ctrl-Adapter)]

**IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2308.06721)] 
[[Project](https://ip-adapter.github.io/)] 
[[Code](https://github.com/tencent-ailab/ip-adapter)]

**A Simple Approach to Unifying Diffusion-based Conditional Generation** \
[[Website](https://arxiv.org/abs/2410.11439)] 
[[Project](https://lixirui142.github.io/unicon-diffusion/)] 
[[Code](https://github.com/lixirui142/UniCon)]

**HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion** \
[[Website](https://arxiv.org/abs/2310.08579)] 
[[Project](https://snap-research.github.io/HyperHuman/)] 
[[Code](https://github.com/snap-research/HyperHuman)]

**Late-Constraint Diffusion Guidance for Controllable Image Synthesis** \
[[Website](https://arxiv.org/abs/2305.11520)] 
[[Project](https://alonzoleeeooo.github.io/LCDG/)] 
[[Code](https://github.com/AlonzoLeeeooo/LCDG)]


**Composer: Creative and controllable image synthesis with composable conditions** \
[[Website](https://arxiv.org/abs/2302.09778)] 
[[Project](https://damo-vilab.github.io/composer-page/)] 
[[Code](https://github.com/damo-vilab/composer)]

**DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.15194)] 
[[Project](https://sungnyun.github.io/diffblender/)] 
[[Code](https://github.com/sungnyun/diffblender)]

**Cocktail: Mixing Multi-Modality Controls for Text-Conditional Image Generation** \
[[Website](https://arxiv.org/abs/2303.09833)] 
[[Project](https://mhh0318.github.io/cocktail/)] 
[[Code](https://github.com/mhh0318/Cocktail)]

**UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild** \
[[Website](https://arxiv.org/abs/2305.11147)] 
[[Project](https://canqin001.github.io/UniControl-Page/)] 
[[Code](https://github.com/salesforce/UniControl)]

**Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.16322)] 
[[Project](https://shihaozhaozsh.github.io/unicontrolnet/)] 
[[Code](https://github.com/ShihaoZhaoZSH/Uni-ControlNet)]

**LooseControl: Lifting ControlNet for Generalized Depth Conditioning** \
[[Website](https://arxiv.org/abs/2312.03079)] 
[[Project](https://shariqfarooq123.github.io/loose-control/)] 
[[Code](https://github.com/shariqfarooq123/LooseControl)]

**X-Adapter: Adding Universal Compatibility of Plugins for Upgraded Diffusion Model** \
[[Website](https://arxiv.org/abs/2312.03079)] 
[[Project](https://showlab.github.io/X-Adapter/)] 
[[Code](https://github.com/showlab/X-Adapter)]

**ControlNet-XS: Designing an Efficient and Effective Architecture for Controlling Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.06573)] 
[[Project](https://vislearn.github.io/ControlNet-XS/)] 
[[Code](https://github.com/vislearn/ControlNet-XS)]

**ViscoNet: Bridging and Harmonizing Visual and Textual Conditioning for ControlNet** \
[[Website](https://arxiv.org/abs/2312.03154)] 
[[Project](https://soon-yau.github.io/visconet/)] 
[[Code](https://github.com/soon-yau/visconet)]

**SCP-Diff: Photo-Realistic Semantic Image Synthesis with Spatial-Categorical Joint Prior** \
[[Website](https://arxiv.org/abs/2403.09638)] 
[[Project](https://air-discover.github.io/SCP-Diff/)] 
[[Code](https://github.com/AIR-DISCOVER/SCP-Diff-Toolkit)]


**Compose and Conquer: Diffusion-Based 3D Depth Aware Composable Image Synthesis** \
[[ICLR 2024](https://arxiv.org/abs/2401.09048)] 
[[Code](https://github.com/tomtom1103/compose-and-conquer/)]

**It's All About Your Sketch: Democratising Sketch Control in Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.07234)] 
[[Code](https://github.com/subhadeepkoley/DemoSketch2RGB)]

**Universal Guidance for Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.07121)] 
[[Code](https://github.com/arpitbansal297/Universal-Guided-Diffusion)]

**Late-Constraint Diffusion Guidance for Controllable Image Synthesis** \
[[Website](https://arxiv.org/abs/2305.11520)] 
[[Code]](https://github.com/AlonzoLeeeooo/LCDG) 

**Meta ControlNet: Enhancing Task Adaptation via Meta Learning** \
[[Website](https://arxiv.org/abs/2312.01255)] 
[[Code](https://github.com/JunjieYang97/Meta-ControlNet)]

**Local Conditional Controlling for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.08768)] 
[[Code](https://github.com/YibooZhao/Local-Control)]

**KnobGen: Controlling the Sophistication of Artwork in Sketch-Based Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.01595)] 
[[Code](https://github.com/aminK8/KnobGen)]


**Modulating Pretrained Diffusion Models for Multimodal Image Synthesis** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2302.12764)] 
[[Project](https://mcm-diffusion.github.io/)] 

**SpaText: Spatio-Textual Representation for Controllable Image Generation**\
[[CVPR 2023](https://arxiv.org/abs/2211.14305)] 
[[Project]](https://omriavrahami.com/spatext/) 

**CCM: Adding Conditional Controls to Text-to-Image Consistency Models** \
[[ICML 2024](https://arxiv.org/abs/2312.06971)] 
[[Project](https://swiftforce.github.io/CCM/)] 

**Dreamguider: Improved Training free Diffusion-based Conditional Generation** \
[[Website](https://arxiv.org/abs/2406.02549)] 
[[Project](https://nithin-gk.github.io/dreamguider.github.io/)] 

**ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback** \
[[Website](https://arxiv.org/abs/2404.07987)] 
[[Project](https://liming-ai.github.io/ControlNet_Plus_Plus/)] 

**AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2406.18958)] 
[[Project](https://any-control.github.io/)] 

**BeyondScene: Higher-Resolution Human-Centric Scene Generation With Pretrained Diffusion** \
[[Website](https://arxiv.org/abs/2404.04544)] 
[[Project](https://janeyeon.github.io/beyond-scene/)] 

**FineControlNet: Fine-level Text Control for Image Generation with Spatially Aligned Text Control Injection** \
[[Website](https://arxiv.org/abs/2312.09252)] 
[[Project](https://samsunglabs.github.io/FineControlNet-project-page/)] 

**Control4D: Dynamic Portrait Editing by Learning 4D GAN from 2D Diffusion-based Editor** \
[[Website](https://arxiv.org/abs/2305.20082)] 
[[Project](https://control4darxiv.github.io/)] 

**SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing** \
[[Website](https://arxiv.org/abs/2312.11392)] 
[[Project](https://scedit.github.io/)] 

**CTRLorALTer: Conditional LoRAdapter for Efficient 0-Shot Control & Altering of T2I Models** \
[[Website](https://arxiv.org/abs/2405.07913)] 
[[Project](https://compvis.github.io/LoRAdapter/)] 

**AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2406.18958)] 
[[Project](https://any-control.github.io/)] 

**Sketch-Guided Scene Image Generation** \
[[Website](https://arxiv.org/abs/2407.06469)] 

**SSMG: Spatial-Semantic Map Guided Diffusion Model for Free-form Layout-to-Image Generation** \
[[Website](https://arxiv.org/abs/2308.10156)] 

**Conditioning Diffusion Models via Attributes and Semantic Masks for Face Generation** \
[[Website](https://arxiv.org/abs/2306.00914)] 

**Integrating Geometric Control into Text-to-Image Diffusion Models for High-Quality Detection Data Generation via Text Prompt** \
[[Website](https://arxiv.org/abs/2306.04607)] 

**Adding 3D Geometry Control to Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.08103)] 

**LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation** \
[[Website]](https://arxiv.org/abs/2302.08908) 

**JointNet: Extending Text-to-Image Diffusion for Dense Distribution Modeling** \
[[Website]](https://arxiv.org/abs/2310.06347) 

**ViscoNet: Bridging and Harmonizing Visual and Textual Conditioning for ControlNet** \
[[Website]](https://arxiv.org/abs/2312.03154) 

**Do You Guys Want to Dance: Zero-Shot Compositional Human Dance Generation with Multiple Persons** \
[[Website]](https://arxiv.org/abs/2401.13363) 

**Mask-ControlNet: Higher-Quality Image Generation with An Additional Mask Prompt** \
[[Website]](https://arxiv.org/abs/2404.05331) 

**FlexEControl: Flexible and Efficient Multimodal Control for Text-to-Image Generation** \
[[Website]](https://arxiv.org/abs/2405.04834) 

**Stable-Pose: Leveraging Transformers for Pose-Guided Text-to-Image Generation** \
[[Website]](https://arxiv.org/abs/2406.02485) 

**Label-free Neural Semantic Image Synthesis** \
[[Website]](https://arxiv.org/abs/2407.01790) 

## Few-Shot 
**Discriminative Diffusion Models as Few-shot Vision and Language Learners** \
[[Website](https://arxiv.org/abs/2305.10722)] 
[[Code](https://github.com/eric-ai-lab/dsd)]

**Few-Shot Diffusion Models** \
[[Website](https://arxiv.org/abs/2205.15463)] 
[[Code](https://github.com/georgosgeorgos/few-shot-diffusion-models)]

**Few-shot Semantic Image Synthesis with Class Affinity Transfer** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Careil_Few-Shot_Semantic_Image_Synthesis_With_Class_Affinity_Transfer_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2304.02321)] 


**DiffAlign : Few-shot learning using diffusion based synthesis and alignment** \
[[Website](https://arxiv.org/abs/2212.05404)] 

**Few-shot Image Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2211.03264)] 

**Lafite2: Few-shot Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2210.14124)] 


## SD-inpaint

**Paint by Example: Exemplar-based Image Editing with Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Paint_by_Example_Exemplar-Based_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.html
)] 
[[Website](https://arxiv.org/abs/2211.13227)] 
[[Code](https://github.com/Fantasy-Studio/Paint-by-Example)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/paint_by_example)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/paint_by_example/pipeline_paint_by_example.py)] 


**GLIDE: Towards photorealistic image generation and editing with text-guided diffusion model** \
[[ICML 2022 Spotlight](https://icml.cc/virtual/2022/spotlight/16340)] 
[[Website](https://arxiv.org/abs/2112.10741)] 
[[Code](https://github.com/openai/glide-text2im)]

**Blended Diffusion for Text-driven Editing of Natural Images** \
[[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.html)] 
[[Website](https://arxiv.org/abs/2111.14818)] 
[[Project](https://omriavrahami.com/blended-diffusion-page/)]
[[Code](https://github.com/omriav/blended-diffusion)]

**Blended Latent Diffusion** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2206.02779)] 
[[Project](https://omriavrahami.com/blended-latent-diffusion-page/)]
[[Code](https://github.com/omriav/blended-latent-diffusion)]

**TF-ICON: Diffusion-Based Training-Free Cross-Domain Image Composition** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_TF-ICON_Diffusion-Based_Training-Free_Cross-Domain_Image_Composition_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2307.12493)] 
[[Project](https://shilin-lu.github.io/tf-icon.github.io/)]
[[Code](https://github.com/Shilin-LU/TF-ICON)]

**Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Imagen_Editor_and_EditBench_Advancing_and_Evaluating_Text-Guided_Image_Inpainting_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2212.06909)] 
[[Code](https://github.com/fenglinglwb/PSM)]


**Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models** \
[[ICML 2023](https://icml.cc/virtual/2023/poster/24127)] 
[[Website](https://arxiv.org/abs/2304.03322)] 
[[Code](https://github.com/ucsb-nlp-chang/copaint)]

**Coherent and Multi-modality Image Inpainting via Latent Space Optimization** \
[[Website](https://arxiv.org/abs/2407.08019)] 
[[Project](https://pilot-page.github.io/)] 
[[Code](https://github.com/Lingzhi-Pan/PILOT)]

**Inst-Inpaint: Instructing to Remove Objects with Diffusion Models** \
[[Website](https://arxiv.org/abs/2304.03246)] 
[[Project](http://instinpaint.abyildirim.com/)] 
[[Code](https://github.com/abyildirim/inst-inpaint)]
[[Demo](https://huggingface.co/spaces/abyildirim/inst-inpaint)]

**Anywhere: A Multi-Agent Framework for Reliable and Diverse Foreground-Conditioned Image Inpainting** \
[[Website](https://arxiv.org/abs/2404.18598)] 
[[Project](https://anywheremultiagent.github.io/)]
[[Code](https://github.com/Sealical/anywhere-multi-agent)] 

**CLIPAway: Harmonizing Focused Embeddings for Removing Objects via Diffusion Models** \
[[Website](https://arxiv.org/abs/2406.09368)] 
[[Project](https://yigitekin.github.io/CLIPAway/)]
[[Code](https://github.com/YigitEkin/CLIPAway)] 

**AnyDoor: Zero-shot Object-level Image Customization** \
[[Website](https://arxiv.org/abs/2307.09481)] 
[[Project](https://damo-vilab.github.io/AnyDoor-Page/)]
[[Code](https://github.com/damo-vilab/AnyDoor)] 

**A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting** \
[[Website](https://arxiv.org/abs/2312.03594)] 
[[Project](https://powerpaint.github.io/)]
[[Code](https://github.com/open-mmlab/mmagic/tree/main/projects/powerpaint)] 

**Follow-Your-Canvas: Higher-Resolution Video Outpainting with Extensive Content Generation** \
[[Website](https://arxiv.org/abs/2409.01055)] 
[[Project](https://follow-your-canvas.github.io/)]
[[Code](https://github.com/mayuelala/FollowYourCanvas)] 

**Towards Language-Driven Video Inpainting via Multimodal Large Language Models** \
[[Website](https://arxiv.org/abs/2401.10226)]
[[Project](https://jianzongwu.github.io/projects/rovi/)]
[[Code](https://github.com/jianzongwu/Language-Driven-Video-Inpainting)]

**Reflecting Reality: Enabling Diffusion Models to Produce Faithful Mirror Reflections** \
[[Website](https://arxiv.org/abs/2409.14677)]
[[Project](https://val.cds.iisc.ac.in/reflecting-reality.github.io/)]
[[Code](https://github.com/val-iisc/Reflecting-Reality)]

**Improving Text-guided Object Inpainting with Semantic Pre-inpainting**\
[[ECCV 2024](https://arxiv.org/abs/2409.08260)] 
[[Code](https://github.com/Nnn-s/CATdiffusion)] 


**FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior** \
[[ECCV 2024](https://arxiv.org/abs/2407.04947)] 
[[Code](https://github.com/aim-uofa/FreeCompose)] 

**360-Degree Panorama Generation from Few Unregistered NFoV Images** \
[[ACM MM 2023](https://arxiv.org/abs/2308.14686)] 
[[Code](https://github.com/shanemankiw/Panodiff)] 

**Delving Globally into Texture and Structure for Image Inpainting**\
[[ACM MM 2022](https://arxiv.org/abs/2209.08217)] 
[[Code](https://github.com/htyjers/DGTS-Inpainting)]

**ControlEdit: A MultiModal Local Clothing Image Editing Method** \
[[Website](https://arxiv.org/abs/2409.14720)] 
[[Code](https://github.com/cd123-cd/ControlEdit)]


**Training-and-prompt-free General Painterly Harmonization Using Image-wise Attention Sharing** \
[[Website](https://arxiv.org/abs/2404.12900)] 
[[Code](https://github.com/BlueDyee/TF-GPH)]

**What to Preserve and What to Transfer: Faithful, Identity-Preserving Diffusion-based Hairstyle Transfer** \
[[Website](https://arxiv.org/abs/2408.16450)] 
[[Code](https://github.com/cychungg/HairFusion)]

**Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model** \
[[Website](https://arxiv.org/abs/2407.16982)] 
[[Code](https://github.com/OpenGVLab/Diffree)]

**Structure Matters: Tackling the Semantic Discrepancy in Diffusion Models for Image Inpainting** \
[[Website](https://arxiv.org/abs/2403.19898)] 
[[Code](https://github.com/htyjers/StrDiffusion)]

**Reference-based Image Composition with Sketch via Structure-aware Diffusion Model** \
[[Website](https://arxiv.org/abs/2304.09748)] 
[[Code](https://github.com/kangyeolk/Paint-by-Sketch)]

**Image Inpainting via Iteratively Decoupled Probabilistic Modeling** \
[[Website](https://arxiv.org/abs/2212.02963)] 
[[Code](https://github.com/fenglinglwb/PSM)]

**ControlCom: Controllable Image Composition using Diffusion Model** \
[[Website](https://arxiv.org/abs/2308.10040)] 
[[Code]](https://github.com/bcmi/ControlCom-Image-Composition) 

**Uni-paint: A Unified Framework for Multimodal Image Inpainting with Pretrained Diffusion Model** \
[[Website](https://arxiv.org/abs/2310.07222)] 
[[Code](https://github.com/ysy31415/unipaint)] 

**MAGICREMOVER: TUNING-FREE TEXT-GUIDED IMAGE INPAINTING WITH DIFFUSION MODELS** \
[[Website](https://arxiv.org/abs/2310.02848)] 
[[Code](https://github.com/exisas/Magicremover)] 

**HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.14091)] 
[[Code](https://github.com/Picsart-AI-Research/HD-Painter)] 

**BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion** \
[[Website](https://arxiv.org/abs/2403.06976)] 
[[Code](https://github.com/TencentARC/BrushNet)] 

**Sketch-guided Image Inpainting with Partial Discrete Diffusion Process** \
[[Website](https://arxiv.org/abs/2404.11949)] 
[[Code](https://github.com/vl2g/Sketch-Inpainting)] 

**ReMOVE: A Reference-free Metric for Object Erasure** \
[[Website](https://arxiv.org/abs/2409.00707)] 
[[Code](https://github.com/chandrasekaraditya/ReMOVE)] 

**MotionCom: Automatic and Motion-Aware Image Composition with LLM and Video Diffusion Prior** \
[[Website](https://arxiv.org/abs/2409.10090)] 
[[Code](https://github.com/weijing-tao/MotionCom)] 

**AddMe: Zero-shot Group-photo Synthesis by Inserting People into Scenes** \
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03028.pdf)] 
[[Project](https://addme-awesome.github.io/page/)]

**Text2Place: Affordance-aware Text Guided Human Placement** \
[[ECCV 2024](https://arxiv.org/abs/2407.15446)] 
[[Project](https://rishubhpar.github.io/Text2Place/)]

**IMPRINT: Generative Object Compositing by Learning Identity-Preserving Representation** \
[[CVPR 2024](https://arxiv.org/abs/2403.10701)] 
[[Project](https://song630.github.io/IMPRINT-Project-Page/)]

**Matting by Generation** \
[[SIGGRAPH 2024](https://arxiv.org/abs/2407.21017)] 
[[Project](https://lightchaserx.github.io/matting-by-generation/)]

**Taming Latent Diffusion Model for Neural Radiance Field Inpainting** \
[[Website](https://arxiv.org/abs/2404.09995)] 
[[Project](https://hubert0527.github.io/MALD-NeRF/)]

**SmartMask: Context Aware High-Fidelity Mask Generation for Fine-grained Object Insertion and Layout Control** \
[[Website](https://arxiv.org/abs/2312.05039)] 
[[Project](https://smartmask-gen.github.io/)]

**Towards Stable and Faithful Inpainting** \
[[Website](https://arxiv.org/abs/2312.04831)] 
[[Project](https://yikai-wang.github.io/asuka/)]

**Magic Fixup: Streamlining Photo Editing by Watching Dynamic Videos** \
[[Website](https://arxiv.org/abs/2403.13044)] 
[[Project](https://magic-fixup.github.io/)]

**ObjectDrop: Bootstrapping Counterfactuals for Photorealistic Object Removal and Insertion** \
[[Website](https://arxiv.org/abs/2403.18818)] 
[[Project](https://objectdrop.github.io/)]

**TALE: Training-free Cross-domain Image Composition via Adaptive Latent Manipulation and Energy-guided Optimization** \
[[ACM MM 2024](https://arxiv.org/abs/2408.03637)] 

**Semantically Consistent Video Inpainting with Conditional Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.00251)] 

**Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention**\
[[Website](https://arxiv.org/abs/2312.03556)] 

**Outline-Guided Object Inpainting with Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.16421)] 

**SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model** \
[[Website](https://arxiv.org/abs/2212.05034)] 

**Gradpaint: Gradient-Guided Inpainting with Diffusion Models** \
[[Website](https://arxiv.org/abs/2309.09614)] 

**Infusion: Internal Diffusion for Video Inpainting** \
[[Website](https://arxiv.org/abs/2311.01090)] 

**Rethinking Referring Object Removal** \
[[Website](https://arxiv.org/abs/2403.09128)] 

**Tuning-Free Image Customization with Image and Text Guidance** \
[[Website](https://arxiv.org/abs/2403.12658)] 

**VIP: Versatile Image Outpainting Empowered by Multimodal Large Language Model** \
[[Website](https://arxiv.org/abs/2406.01059)] 

**FaithFill: Faithful Inpainting for Object Completion Using a Single Reference Image** \
[[Website](https://arxiv.org/abs/2406.07865)] 

**InsertDiffusion: Identity Preserving Visualization of Objects through a Training-Free Diffusion Architecture** \
[[Website](https://arxiv.org/abs/2407.10592)] 

**Thinking Outside the BBox: Unconstrained Generative Object Compositing** \
[[Website](https://arxiv.org/abs/2409.04559)] 

**Content-aware Tile Generation using Exterior Boundary Inpainting** \
[[Website](https://arxiv.org/abs/2409.14184)] 

**AnyLogo: Symbiotic Subject-Driven Diffusion System with Gemini Status** \
[[Website](https://arxiv.org/abs/2409.17740)] 

## Layout Generation

**LayoutDM: Discrete Diffusion Model for Controllable Layout Generation** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.08137)]
[[Project](https://cyberagentailab.github.io/layout-dm/)] 
[[Code](https://github.com/CyberAgentAILab/layout-dm)]

**Desigen: A Pipeline for Controllable Design Template Generation** \
[[CVPR 2024](https://arxiv.org/abs/2403.09093)]
[[Project](https://whaohan.github.io/desigen/)]
[[Code](https://github.com/whaohan/desigen)]

**DLT: Conditioned layout generation with Joint Discrete-Continuous Diffusion Layout Transformer** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Levi_DLT_Conditioned_layout_generation_with_Joint_Discrete-Continuous_Diffusion_Layout_Transformer_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.03755)]
[[Code](https://github.com/wix-incubator/DLT)]

**LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_LayoutDiffusion_Improving_Graphic_Layout_Generation_by_Discrete_Diffusion_Probabilistic_Models_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.11589)]
[[Code](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion)]

**Desigen: A Pipeline for Controllable Design Template Generation** \
[[CVPR 2024](https://arxiv.org/abs/2403.09093)] 
[[Code](https://github.com/whaohan/desigen)] 

**LayoutDM: Transformer-based Diffusion Model for Layout Generation** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chai_LayoutDM_Transformer-Based_Diffusion_Model_for_Layout_Generation_CVPR_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2305.02567)] 

**Unifying Layout Generation with a Decoupled Diffusion Model** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Hui_Unifying_Layout_Generation_With_a_Decoupled_Diffusion_Model_CVPR_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.05049)]

**PLay: Parametrically Conditioned Layout Generation using Latent Diffusion** \
[[ICML 2023](https://dl.acm.org/doi/10.5555/3618408.3618624)]
[[Website](https://arxiv.org/abs/2301.11529)]

**Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints** \
[[ICLR 2024](https://arxiv.org/abs/2402.04754)]

**CGB-DM: Content and Graphic Balance Layout Generation with Transformer-based Diffusion Model** \
[[Website](https://arxiv.org/abs/2407.15233)]

**Diffusion-based Document Layout Generation** \
[[Website](https://arxiv.org/abs/2303.10787)]

**Dolfin: Diffusion Layout Transformers without Autoencoder** \
[[Website](https://arxiv.org/abs/2310.16305)]

**LayoutFlow: Flow Matching for Layout Generation** \
[[Website](https://arxiv.org/abs/2403.18187)]

**Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Model** \
[[Website](https://arxiv.org/abs/2409.16689)]


## Text Generation

⭐⭐**TextDiffuser: Diffusion Models as Text Painters** \
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70636)]
[[Website](https://arxiv.org/abs/2305.10855)]
[[Project](https://jingyechen.github.io/textdiffuser/)] 
[[Code](https://github.com/microsoft/unilm/tree/master/textdiffuser)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=ke3RgcDmfO)] -->

⭐⭐**TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering** \
[[ECCV 2024 Oral](https://arxiv.org/abs/2311.16465)]
[[Project](https://jingyechen.github.io/textdiffuser2/)] 
[[Code](https://github.com/microsoft/unilm/tree/master/textdiffuser-2)] 

**GlyphControl: Glyph Conditional Control for Visual Text Generation** \
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70191)]
[[Website](https://arxiv.org/abs/2305.18259)]
[[Code](https://github.com/AIGText/GlyphControl-release)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=thPI8hrA4V)] -->

**DiffUTE: Universal Text Editing Diffusion Model** \
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71364)]
[[Website](https://arxiv.org/abs/2305.10825)]
[[Code](https://github.com/chenhaoxing/DiffUTE)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=XKeSauhUdJ)] -->

**Word-As-Image for Semantic Typography** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2303.01818)]
[[Project](https://wordasimage.github.io/Word-As-Image-Page/)] 
[[Code](https://github.com/Shiriluz/Word-As-Image)] 

**Kinetic Typography Diffusion Model** \
[[ECCV 2024](https://arxiv.org/abs/2407.10476)]
[[Project](https://seonmip.github.io/kinety/)] 
[[Code](https://github.com/SeonmiP/KineTy)] 

**Dynamic Typography: Bringing Text to Life via Video Diffusion Prior** \
[[Website](https://arxiv.org/abs/2404.11614)]
[[Project](https://animate-your-word.github.io/demo/)] 
[[Code](https://github.com/zliucz/animate-your-word)] 

**JoyType: A Robust Design for Multilingual Visual Text Creation** \
[[Website](https://arxiv.org/abs/2409.17524)]
[[Project](https://jdh-algo.github.io/JoyType/)] 
[[Code](https://github.com/jdh-algo/JoyType)] 

**UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.04884)]
[[Project](https://udifftext.github.io/)] 
[[Code](https://github.com/ZYM-PKU/UDiffText)] 

**One-Shot Diffusion Mimicker for Handwritten Text Generation** \
[[ECCV 2024](https://arxiv.org/abs/2409.04004)]
[[Code](https://github.com/dailenson/One-DM)] 

**DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution** \
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02357.pdf)]
[[Code](https://github.com/shreygithub/DCDM)] 

**HFH-Font: Few-shot Chinese Font Synthesis with Higher Quality, Faster Speed, and Higher Resolution** \
[[SIGGRAPH Asia 2024](https://arxiv.org/abs/2410.06488)]
[[Code](https://github.com/grovessss/HFH-Font)] 

**Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model** \
[[AAAI 2024](https://arxiv.org/abs/2312.12232)]
[[Code](https://github.com/ecnuljzhang/brush-your-text)] 

**FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning** \
[[AAAI 2024](https://arxiv.org/abs/2312.12142)]
[[Code](https://github.com/yeungchenwa/FontDiffuser)] 

**Text Image Inpainting via Global Structure-Guided Diffusion Models** \
[[AAAI 2024](https://arxiv.org/abs/2401.14832)]
[[Code](https://github.com/blackprotoss/GSDM)] 

**Ambigram generation by a diffusion model** \
[[ICDAR 2023](https://arxiv.org/abs/2306.12049)]
[[Code](https://github.com/univ-esuty/ambifusion)] 

**Scene Text Image Super-resolution based on Text-conditional Diffusion Models** \
[[WACV 2024](https://arxiv.org/abs/2311.09759)]
[[Code](https://github.com/toyotainfotech/stisr-tcdm)] 

**Leveraging Text Localization for Scene Text Removal via Text-aware Masked Image Modeling** \
[[ECCV 2024](https://arxiv.org/abs/2409.13431)]
[[Code](https://github.com/wzx99/TMIM)] 

**VitaGlyph: Vitalizing Artistic Typography with Flexible Dual-branch Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.01738)]
[[Code](https://github.com/Carlofkl/VitaGlyph)] 

**Visual Text Generation in the Wild** \
[[Website](https://arxiv.org/abs/2407.14138)]
[[Code](https://github.com/alibabaresearch/advancedliteratemachinery)] 

**Deciphering Oracle Bone Language with Diffusion Models** \
[[Website](https://arxiv.org/abs/2406.00684)]
[[Code](https://github.com/guanhaisu/OBSD)] 

**High Fidelity Scene Text Synthesis** \
[[Website](https://arxiv.org/abs/2405.14701)]
[[Code](https://github.com/CodeGoat24/DreamText)] 

**AnyText: Multilingual Visual Text Generation And Editing** \
[[Website](https://arxiv.org/abs/2311.03054)]
[[Code](https://github.com/tyxsspa/AnyText)] 

**Few-shot Calligraphy Style Learning** \
[[Website](https://arxiv.org/abs/2404.17199)]
[[Code](https://github.com/kono-dada/xysffusion)] 

**GlyphDraw2: Automatic Generation of Complex Glyph Posters with Diffusion Models and Large Language Models** \
[[Website](https://arxiv.org/abs/2407.02252)]
[[Code](https://github.com/OPPO-Mente-Lab/GlyphDraw2)] 

**DiffusionPen: Towards Controlling the Style of Handwritten Text Generation** \
[[Website](https://arxiv.org/abs/2409.06065)]
[[Code](https://github.com/koninik/DiffusionPen)] 

**AmbiGen: Generating Ambigrams from Pre-trained Diffusion Model** \
[[Website](https://arxiv.org/abs/2312.02967)]
[[Project](https://raymond-yeh.com/AmbiGen/)] 

**UniVG: Towards UNIfied-modal Video Generation** \
[[Website](https://arxiv.org/abs/2401.09084)]
[[Project](https://univg-baidu.github.io/)] 

**FontStudio: Shape-Adaptive Diffusion Model for Coherent and Consistent Font Effect Generation** \
[[Website](https://arxiv.org/abs/2406.08392)]
[[Project](https://font-studio.github.io/)] 

**DECDM: Document Enhancement using Cycle-Consistent Diffusion Models** \
[[WACV 2024](https://arxiv.org/abs/2311.09625)]

**SceneTextGen: Layout-Agnostic Scene Text Image Synthesis with Diffusion Models** \
[[Website](https://arxiv.org/abs/2406.01062)]

**AnyTrans: Translate AnyText in the Image with Large Scale Models** \
[[Website](https://arxiv.org/abs/2406.11432)]

**ARTIST: Improving the Generation of Text-rich Images by Disentanglement** \
[[Website](https://arxiv.org/abs/2406.12044)]

**Improving Text Generation on Images with Synthetic Captions** \
[[Website](https://arxiv.org/abs/2406.00505)]

**CustomText: Customized Textual Image Generation using Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.12531)]

**VecFusion: Vector Font Generation with Diffusion** \
[[Website](https://arxiv.org/abs/2312.10540)]

**Typographic Text Generation with Off-the-Shelf Diffusion Model** \
[[Website](https://arxiv.org/abs/2402.14314)]

**Font Style Interpolation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.14311)]

**Refining Text-to-Image Generation: Towards Accurate Training-Free Glyph-Enhanced Image Generation** \
[[Website](https://arxiv.org/abs/2403.16422)]

**DiffCJK: Conditional Diffusion Model for High-Quality and Wide-coverage CJK Character Generation** \
[[Website](https://arxiv.org/abs/2404.05212)]

**CLII: Visual-Text Inpainting via Cross-Modal Predictive Interaction** \
[[Website](https://arxiv.org/abs/2407.16204)]

**Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2409.00786)]

**Text Image Generation for Low-Resource Languages with Dual Translation Learning** \
[[Website](https://arxiv.org/abs/2409.17747)]

**Decoupling Layout from Glyph in Online Chinese Handwriting Generation** \
[[Website](https://arxiv.org/abs/2410.02309)]

**Empowering Backbone Models for Visual Text Generation with Input Granularity Control and Glyph-Aware Training** \
[[Website](https://arxiv.org/abs/2410.04439)]


## Super Resolution

**ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting** \
[[NeurIPS 2023 spotlight](https://nips.cc/virtual/2023/poster/71244)] 
[[Website](https://arxiv.org/abs/2307.12348)] 
[[Project](https://zsyoaoa.github.io/projects/resshift/)] 
[[Code](https://github.com/zsyoaoa/resshift)] 

**Image Super-Resolution via Iterative Refinement** \
[[TPAMI](https://ieeexplore.ieee.org/document/9887996)] 
[[Website](https://arxiv.org/abs/2104.07636)] 
[[Project](https://iterative-refinement.github.io/)] 
[[Code](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] 

**DiffIR: Efficient Diffusion Model for Image Restoration**\
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_DiffIR_Efficient_Diffusion_Model_for_Image_Restoration_ICCV_2023_paper.pdf)]
[[Website](https://arxiv.org/abs/2303.09472)] 
[[Code](https://github.com/Zj-BinXia/DiffIR)]

**Kalman-Inspired Feature Propagation for Video Face Super-Resolution** \
[[ECCV 2024](https://arxiv.org/abs/2408.05205)] 
[[Project](https://jnjaby.github.io/projects/KEEP/)] 
[[Code](https://github.com/jnjaby/KEEP)] 

**AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation** \
[[Website](https://arxiv.org/abs/2404.01717)] 
[[Project](https://nju-pcalab.github.io/projects/AddSR/)] 
[[Code](https://github.com/NJU-PCALab/AddSR)] 

**Exploiting Diffusion Prior for Real-World Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2305.07015)] 
[[Project](https://iceclear.github.io/projects/stablesr/)] 
[[Code](https://github.com/IceClear/StableSR)] 

**SinSR: Diffusion-Based Image Super-Resolution in a Single Step** \
[[CVPR 2024](https://arxiv.org/abs/2311.14760)] 
[[Code](https://github.com/wyf0912/SinSR)] 

**CDFormer:When Degradation Prediction Embraces Diffusion Model for Blind Image Super-Resolution** \
[[CVPR 2024](https://arxiv.org/abs/2405.07648)] 
[[Code](https://github.com/I2-Multimedia-Lab/CDFormer)] 

**Taming Diffusion Prior for Image Super-Resolution with Domain Shift SDEs** \
[[NeurIPS 2024](https://arxiv.org/abs/2409.17778)] 
[[Code](https://github.com/QinpengCui/DoSSR)] 

**SeeClear: Semantic Distillation Enhances Pixel Condensation for Video Super-Resolution** \
[[NeurIPS 2024](https://arxiv.org/abs/2410.05799)] 
[[Code](https://github.com/Tang1705/SeeClear-NeurIPS24)] 

**Iterative Token Evaluation and Refinement for Real-World Super-Resolution** \
[[AAAI 2024](https://arxiv.org/abs/2312.05616)] 
[[Code](https://github.com/chaofengc/ITER)] 

**Distillation-Free One-Step Diffusion for Real-World Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2410.04224)] 
[[Code](https://github.com/JianzeLi-114/DFOSD)] 

**Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors** \
[[Website](https://arxiv.org/abs/2409.17058)] 
[[Code](https://github.com/ArcticHare105/S3Diff)] 

**One Step Diffusion-based Super-Resolution with Time-Aware Distillation** \
[[Website](https://arxiv.org/abs/2408.07476)] 
[[Code](https://github.com/LearningHx/TAD-SR)] 

**One-Step Effective Diffusion Network for Real-World Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2406.08177)] 
[[Code](https://github.com/cswry/OSEDiff)] 

**Binarized Diffusion Model for Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2406.05723)] 
[[Code](https://github.com/zhengchen1999/BI-DiffSR)] 

**Does Diffusion Beat GAN in Image Super Resolution?** \
[[Website](https://arxiv.org/abs/2405.17261)] 
[[Code](https://github.com/yandex-research/gan_vs_diff_sr)] 

**PatchScaler: An Efficient Patch-independent Diffusion Model for Super-Resolution** \
[[Website](https://arxiv.org/abs/2405.17158)] 
[[Code](https://github.com/yongliuy/PatchScaler)] 

**DeeDSR: Towards Real-World Image Super-Resolution via Degradation-Aware Stable Diffusion** \
[[Website](https://arxiv.org/abs/2404.00661)] 
[[Code](https://github.com/bichunyang419/DeeDSR)] 

**Image Super-resolution Via Latent Diffusion: A Sampling-space Mixture Of Experts And Frequency-augmented Decoder Approach** \
[[Website](https://arxiv.org/abs/2310.12004)] 
[[Code](https://github.com/amandaluof/moe_sr)] 

**Pixel-Aware Stable Diffusion for Realistic Image Super-resolution and Personalized Stylization** \
[[Website](https://arxiv.org/abs/2308.14469)] 
[[Code](https://github.com/yangxy/PASD)] 

**DSR-Diff: Depth Map Super-Resolution with Diffusion Model** \
[[Website](https://arxiv.org/abs/2311.09919)] 
[[Code](https://github.com/shiyuan7/DSR-Diff)] 

**SAM-DiffSR: Structure-Modulated Diffusion Model for Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2402.17133)] 
[[Code](https://github.com/lose4578/SAM-DiffSR)] 

**XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2403.05049)] 
[[Code](https://github.com/qyp2000/XPSR)] 

**Self-Adaptive Reality-Guided Diffusion for Artifact-Free Super-Resolution** \
[[Website](https://arxiv.org/abs/2403.16643)] 
[[Code](https://github.com/ProAirVerse/Self-Adaptive-Guidance-Diffusion)] 

**BlindDiff: Empowering Degradation Modelling in Diffusion Models for Blind Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2403.10211)] 
[[Code](https://github.com/lifengcs/BlindDiff)] 

**HSR-Diff: Hyperspectral Image Super-Resolution via Conditional Diffusion Models**\
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_HSR-Diff_Hyperspectral_Image_Super-Resolution_via_Conditional_Diffusion_Models_ICCV_2023_paper.pdf)]
[[Website](https://arxiv.org/abs/2306.12085)] 

**Text-guided Explorable Image Super-resolution** \
[[CVPR 2024](https://arxiv.org/abs/2403.01124)] 

**Arbitrary-Scale Image Generation and Upsampling using Latent Diffusion Model and Implicit Neural Decoder** \
[[CVPR 2024](https://arxiv.org/abs/2403.10255)] 

**Enhancing Hyperspectral Images via Diffusion Model and Group-Autoencoder Super-resolution Network** \
[[AAAI 2024](https://arxiv.org/abs/2402.17285)] 

**Detail-Enhancing Framework for Reference-Based Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2405.00431)] 

**You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation** \
[[Website](https://arxiv.org/abs/2401.17258)] 

**Solving Diffusion ODEs with Optimal Boundary Conditions for Better Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2305.15357)] 

**Dissecting Arbitrary-scale Super-resolution Capability from Pre-trained Diffusion Generative Models** \
[[Website](https://arxiv.org/abs/2306.00714)] 

**YODA: You Only Diffuse Areas. An Area-Masked Diffusion Approach For Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2308.07977)]

**Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution -- a Non-Denoising Model** \
[[Website](https://arxiv.org/abs/2311.02358)]

**TDDSR: Single-Step Diffusion with Two Discriminators for Super Resolution** \
[[Website](https://arxiv.org/abs/2410.07663)]

**Image Super-Resolution with Text Prompt Diffusio** \
[[Website](https://arxiv.org/abs/2311.14282)]

**DifAugGAN: A Practical Diffusion-style Data Augmentation for GAN-based Single Image Super-resolution** \
[[Website](https://arxiv.org/abs/2311.18508)]

**DREAM: Diffusion Rectification and Estimation-Adaptive Models** \
[[Website](https://arxiv.org/abs/2312.00210)]

**Inflation with Diffusion: Efficient Temporal Adaptation for Text-to-Video Super-Resolution** \
[[Website](https://arxiv.org/abs/2401.10404)]

**Adaptive Multi-modal Fusion of Spatially Variant Kernel Refinement with Diffusion Model for Blind Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2403.05808)]

**CasSR: Activating Image Power for Real-World Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2403.11451)]

**Learning Spatial Adaptation and Temporal Coherence in Diffusion Models for Video Super-Resolution** \
[[Website](https://arxiv.org/abs/2403.17000)]

**Frequency-Domain Refinement with Multiscale Diffusion for Super Resolution** \
[[Website](https://arxiv.org/abs/2405.10014)]



## Video Generation 

**Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators**  \
[[ICCV 2023 Oral](https://openaccess.thecvf.com/content/ICCV2023/html/Khachatryan_Text2Video-Zero_Text-to-Image_Diffusion_Models_are_Zero-Shot_Video_Generators_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.13439)]
[[Project](https://text2video-zero.github.io/)]
[[Code](https://github.com/Picsart-AI-Research/Text2Video-Zero)]






**SinFusion: Training Diffusion Models on a Single Image or Video** \
[[ICML 2023](https://icml.cc/virtual/2023/poster/24630)]
[[Website](https://arxiv.org/abs/2211.11743)]
[[Project](http://yaniv.nikankin.com/sinfusion/)] 
[[Code](https://github.com/yanivnik/sinfusion-code)]


**Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf)]
[[Website](https://arxiv.org/abs/2304.08818)]
[[Project](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)] 
[[Code](https://github.com/srpkdyy/VideoLDM)]


**ZIGMA: A DiT-style Zigzag Mamba Diffusion Model**  \
[[ECCV 2024](https://arxiv.org/abs/2403.13802)]
[[Project](https://taohu.me/zigma/)]
[[Code](https://github.com/CompVis/zigma)]

**MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation** \
[[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/944618542d80a63bbec16dfbd2bd689a-Abstract-Conference.html)]
[[Website](https://arxiv.org/abs/2205.09853)]
[[Project](https://mask-cond-video-diffusion.github.io/)] 
[[Code](https://github.com/voletiv/mcvd-pytorch)]
<!-- [[NeurIPS 2022](https://openreview.net/forum?id=hX5Ia-ION8Y)] -->


**GLOBER: Coherent Non-autoregressive Video Generation via GLOBal Guided Video DecodER** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/71560)]
[[Website](https://arxiv.org/abs/2309.13274)]
[[Code](https://github.com/iva-mzsun/glober)]
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=TRbklCR2ZW)] -->

**Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70404)]
[[Website](https://arxiv.org/abs/2309.14494)]
[[Code](https://github.com/SooLab/Free-Bloom)]
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=paa2OU5jN8)] -->

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.html)]
[[Website](https://arxiv.org/abs/2303.13744)]
[[Code](https://github.com/nihaomiao/CVPR23_LFDM)]

**FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation** \
[[CVPR 2023](https://arxiv.org/abs/2403.12962)]
[[Project](https://www.mmlab-ntu.com/project/fresco/)]
[[Code](https://github.com/williamyang1991/FRESCO)]


**TI2V-Zero: Zero-Shot Image Conditioning for Text-to-Video Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2404.16306)]
[[Project](https://merl.com/research/highlights/TI2V-Zero)] 
[[Code](https://github.com/merlresearch/TI2V-Zero)]

**Video Diffusion Models** \
[[ICLR 2022 workshop](https://openreview.net/forum?id=BBelR2NdDZ5)]
[[Website](https://arxiv.org/abs/2204.03458)]
[[Code](https://github.com/lucidrains/video-diffusion-pytorch)]
[[Project](https://video-diffusion.github.io/)] 

**PIA: Your Personalized Image Animator via Plug-and-Play Modules in Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2312.13964)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pia)]
[[Project](https://pi-animator.github.io/)]
[[Code](https://github.com/open-mmlab/PIA)] 

**IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation** \
[[ECCV 2024](https://arxiv.org/abs/2407.10937)]
[[Project](https://yhzhai.github.io/idol/)] 
[[Code](https://github.com/yhZhai/idol)]

**T2V-Turbo-v2: Enhancing Video Generation Model Post-Training through Data, Reward, and Conditional Guidance Design** \
[[Website](https://arxiv.org/abs/2410.05677)]
[[Project](https://t2v-turbo-v2.github.io/)] 
[[Code](https://github.com/Ji4chenLi/t2v-turbo)]

**Tora: Trajectory-oriented Diffusion Transformer for Video Generation** \
[[Website](https://arxiv.org/abs/2407.21705)]
[[Project](https://ali-videoai.github.io/tora_video/)] 
[[Code](https://github.com/ali-videoai/Tora)]

**MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling** \
[[Website](https://arxiv.org/abs/2409.16160)]
[[Project](https://menyifang.github.io/projects/MIMO/index.html)] 
[[Code](https://github.com/menyifang/MIMO)]

**MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequence** \
[[Website](https://arxiv.org/abs/2407.16655)]
[[Project](https://aim-uofa.github.io/MovieDreamer/)] 
[[Code](https://github.com/aim-uofa/MovieDreamer)]

**Video Diffusion Alignment via Reward Gradients** \
[[Website](https://arxiv.org/abs/2407.08737)]
[[Project](https://vader-vid.github.io/)] 
[[Code](https://github.com/mihirp1998/VADER)]

**Live2Diff: Live Stream Translation via Uni-directional Attention in Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2407.08701)]
[[Project](https://live2diff.github.io/)] 
[[Code](https://github.com/open-mmlab/Live2Diff)]

**Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models** \
[[Website](https://arxiv.org/abs/2407.15642)]
[[Project](https://maxin-cn.github.io/cinemo_project/)] 
[[Code](https://github.com/maxin-cn/Cinemo)]

**TVG: A Training-free Transition Video Generation Method with Diffusion Models** \
[[Website](https://arxiv.org/abs/2408.13413)]
[[Project](https://sobeymil.github.io/tvg.com/)] 
[[Code](https://github.com/SobeyMIL/TVG)]

**MIMAFace: Face Animation via Motion-Identity Modulated Appearance Feature Learning** \
[[Website](https://arxiv.org/abs/2409.15179)]
[[Project](https://mimaface2024.github.io/mimaface.github.io/)] 
[[Code](https://github.com/MIMAFace2024/MIMAFace)]

**MotionClone: Training-Free Motion Cloning for Controllable Video Generation** \
[[Website](https://arxiv.org/abs/2406.05338)]
[[Project](https://bujiazi.github.io/motionclone.github.io/)] 
[[Code](https://github.com/Bujiazi/MotionClone)]

**VEnhancer: Generative Space-Time Enhancement for Video Generation** \
[[Website](https://arxiv.org/abs/2407.07667)]
[[Project](https://vchitect.github.io/VEnhancer-project/)] 
[[Code](https://github.com/Vchitect/VEnhancer)]

**SF-V: Single Forward Video Generation Model** \
[[Website](https://arxiv.org/abs/2406.04324)]
[[Project](https://snap-research.github.io/SF-V/)] 
[[Code](https://github.com/snap-research/SF-V)]

**Pyramidal Flow Matching for Efficient Video Generative Modeling** \
[[Website](https://arxiv.org/abs/2410.05954)]
[[Project](https://pyramid-flow.github.io/)] 
[[Code](https://github.com/jy0205/Pyramid-Flow)]

**CoNo: Consistency Noise Injection for Tuning-free Long Video Diffusion** \
[[Website](https://arxiv.org/abs/2406.05082)]
[[Project](https://wxrui182.github.io/CoNo.github.io/)] 
[[Code](https://github.com/wxrui182/CoNo)]

**Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation** \
[[Website](https://arxiv.org/abs/2311.17117)]
[[Project](https://humanaigc.github.io/animate-anyone/)] 
[[Code](https://github.com/HumanAIGC/AnimateAnyone)]

**VideoTetris: Towards Compositional Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2406.04277)]
[[Project](https://videotetris.github.io/)] 
[[Code](https://github.com/YangLing0818/VideoTetris)]

**T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback** \
[[Website](https://arxiv.org/abs/2405.18750)]
[[Project](https://t2v-turbo.github.io/)] 
[[Code](https://github.com/Ji4chenLi/t2v-turbo)]

**ZeroSmooth: Training-free Diffuser Adaptation for High Frame Rate Video Generation** \
[[Website](https://arxiv.org/abs/2406.00908)]
[[Project](https://ssyang2020.github.io/zerosmooth.github.io/)] 
[[Code](https://github.com/ssyang2020/ZeroSmooth)]

**MotionBooth: Motion-Aware Customized Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2406.17758)]
[[Project](https://jianzongwu.github.io/projects/motionbooth/)] 
[[Code](https://github.com/jianzongwu/MotionBooth)]


**MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model** \
[[Website](https://arxiv.org/abs/2405.20222)]
[[Project](https://myniuuu.github.io/MOFA_Video/)] 
[[Code](https://github.com/MyNiuuu/MOFA-Video)]

**MotionDreamer: Zero-Shot 3D Mesh Animation from Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.20155)]
[[Project](https://lukas.uzolas.com/MotionDreamer/)] 
[[Code](https://github.com/lukasuz/MotionDreamer)]

**MotionCraft: Physics-based Zero-Shot Video Generation** \
[[Website](https://arxiv.org/abs/2405.13557)]
[[Project](https://mezzelfo.github.io/MotionCraft/)] 
[[Code](https://github.com/mezzelfo/MotionCraft)]

**MotionMaster: Training-free Camera Motion Transfer For Video Generation** \
[[Website](https://arxiv.org/abs/2404.15789)]
[[Project](https://sjtuplayer.github.io/projects/MotionMaster/)] 
[[Code](https://github.com/sjtuplayer/MotionMaster)]

**Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets** \
[[Website](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets)]
[[Project](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)]
[[Code](https://github.com/Stability-AI/generative-models)]

**Motion Inversion for Video Customization** \
[[Website](https://arxiv.org/abs/2403.20193)]
[[Project](https://wileewang.github.io/MotionInversion/)] 
[[Code](https://github.com/EnVision-Research/MotionInversion)]

**MagicAvatar: Multimodal Avatar Generation and Animation** \
[[Website](https://arxiv.org/abs/2308.14748)]
[[Project](https://magic-avatar.github.io/)] 
[[Code](https://github.com/magic-research/magic-avatar)]

**Progressive Autoregressive Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.08151)]
[[Project](https://desaixie.github.io/pa-vdm/)] 
[[Code](https://github.com/desaixie/pa_vdm)]

**TrailBlazer: Trajectory Control for Diffusion-Based Video Generation** \
[[Website](https://arxiv.org/abs/2401.00896)]
[[Project](https://hohonu-vicml.github.io/Trailblazer.Page/)] 
[[Code](https://github.com/hohonu-vicml/Trailblazer)]

**Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos** \
[[Website](https://arxiv.org/abs/2304.01186)]
[[Project](https://follow-your-pose.github.io/)] 
[[Code](https://github.com/mayuelala/FollowYourPose)]

**Breathing Life Into Sketches Using Text-to-Video Priors** \
[[Website](https://arxiv.org/abs/2311.13608)]
[[Project](https://livesketch.github.io/)] 
[[Code](https://github.com/yael-vinker/live_sketch)]

**Latent Video Diffusion Models for High-Fidelity Long Video Generation** \
[[Website](https://arxiv.org/abs/2211.13221)]
[[Project](https://yingqinghe.github.io/LVDM/)] 
[[Code](https://github.com/YingqingHe/LVDM)]


**Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance** \
[[Website](https://arxiv.org/abs/2306.00943)]
[[Project](https://doubiiu.github.io/projects/Make-Your-Video/)] 
[[Code](https://github.com/VideoCrafter/Make-Your-Video)]



**Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising** \
[[Website](https://arxiv.org/abs/2305.18264)]
[[Project](https://g-u-n.github.io/projects/gen-long-video/index.html)] 
[[Code](https://github.com/G-U-N/Gen-L-Video)]

**Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.13840)]
[[Project](https://controlavideo.github.io/)] 
[[Code](https://github.com/Weifeng-Chen/control-a-video)]

**VideoComposer: Compositional Video Synthesis with Motion Controllability** \
[[Website](https://arxiv.org/abs/2306.02018)]
[[Project](https://videocomposer.github.io/)] 
[[Code](https://github.com/damo-vilab/videocomposer)]

**DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion** \
[[Website](https://arxiv.org/abs/2304.06025)]
[[Project](https://grail.cs.washington.edu/projects/dreampose/)] 
[[Code](https://github.com/johannakarras/DreamPose)]

**LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2309.15103)]
[[Project](https://vchitect.github.io/LaVie-project/)] 
[[Code](https://github.com/Vchitect/LaVie)]

**Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2309.15818)]
[[Project](https://showlab.github.io/Show-1/)] 
[[Code](https://github.com/showlab/Show-1)]

**LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation** \
[[Website](https://arxiv.org/abs/2310.10769)]
[[Project](https://rq-wu.github.io/projects/LAMP/index.html)] 
[[Code](https://github.com/RQ-Wu/LAMP)]

**MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer** \
[[Website](https://arxiv.org/abs/2311.12052)]
[[Project](https://boese0601.github.io/magicdance/)] 
[[Code](https://github.com/Boese0601/MagicDance)]

**LLM-GROUNDED VIDEO DIFFUSION MODELS** \
[[Website](https://arxiv.org/abs/2309.17444)]
[[Project](https://llm-grounded-video-diffusion.github.io/)] 
[[Code](https://github.com/TonyLianLong/LLM-groundedVideoDiffusion)]

**FreeNoise: Tuning-Free Longer Video Diffusion Via Noise Rescheduling** \
[[Website](https://arxiv.org/abs/2310.15169)]
[[Project](http://haonanqiu.com/projects/FreeNoise.html)] 
[[Code](https://github.com/arthur-qiu/LongerCrafter)]

**VideoCrafter1: Open Diffusion Models for High-Quality Video Generation** \
[[Website](https://arxiv.org/abs/2310.19512)]
[[Project](https://ailab-cvc.github.io/videocrafter/)] 
[[Code](https://github.com/AILab-CVC/VideoCrafter)]

**VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2310.19512)]
[[Project](https://ailab-cvc.github.io/videocrafter2/)] 
[[Code](https://github.com/AILab-CVC/VideoCrafter)]

**VideoDreamer: Customized Multi-Subject Text-to-Video Generation with Disen-Mix Finetuning** \
[[Website](https://arxiv.org/abs/2311.00990)]
[[Project](https://videodreamer23.github.io/)] 
[[Code](https://github.com/videodreamer23/videodreamer23.github.io)]

**I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.04145)]
[[Project](https://i2vgen-xl.github.io/)] 
[[Code](https://github.com/damo-vilab/i2vgen-xl)]

**FusionFrames: Efficient Architectural Aspects for Text-to-Video Generation Pipeline** \
[[Website](https://arxiv.org/abs/2311.13073)]
[[Project](https://ai-forever.github.io/kandinsky-video/)] 
[[Code](https://github.com/ai-forever/KandinskyVideo)]

**MotionCtrl: A Unified and Flexible Motion Controller for Video Generation** \
[[Website](https://arxiv.org/abs/2312.03641)]
[[Project](https://wzhouxiff.github.io/projects/MotionCtrl/)] 
[[Code](https://github.com/TencentARC/MotionCtrl)]


**ART⋅V: Auto-Regressive Text-to-Video Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.18834)]
[[Project](https://warranweng.github.io/art.v/)] 
[[Code](https://github.com/WarranWeng/ART.V)]

**FlowZero: Zero-Shot Text-to-Video Synthesis with LLM-Driven Dynamic Scene Syntax** \
[[Website](https://arxiv.org/abs/2311.15813)]
[[Project](https://flowzero-video.github.io/)] 
[[Code](https://github.com/aniki-ly/FlowZero)]

**VideoBooth: Diffusion-based Video Generation with Image Prompts** \
[[Website](https://arxiv.org/abs/2312.00777)]
[[Project](https://vchitect.github.io/VideoBooth-project/)] 
[[Code](https://github.com/Vchitect/VideoBooth)]

**MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model** \
[[Website](https://arxiv.org/abs/2311.16498)]
[[Project](https://showlab.github.io/magicanimate/)] 
[[Code](https://github.com/magic-research/magic-animate)]

**LivePhoto: Real Image Animation with Text-guided Motion Control** \
[[Website](https://arxiv.org/abs/2312.02928)]
[[Project](https://xavierchen34.github.io/LivePhoto-Page/)] 
[[Code](https://github.com/XavierCHEN34/LivePhoto)]

**AnimateZero: Video Diffusion Models are Zero-Shot Image Animators** \
[[Website](https://arxiv.org/abs/2312.03793)]
[[Project](https://vvictoryuki.github.io/animatezero.github.io/)] 
[[Code](https://github.com/vvictoryuki/AnimateZero)]

**DreamVideo: Composing Your Dream Videos with Customized Subject and Motion** \
[[Website](https://arxiv.org/abs/2312.04433)]
[[Project](https://dreamvideo-t2v.github.io/)] 
[[Code](https://github.com/damo-vilab/i2vgen-xl)]

**Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2312.04483)]
[[Project](https://higen-t2v.github.io/)] 
[[Code](https://github.com/damo-vilab/i2vgen-xl)]

**DreaMoving: A Human Dance Video Generation Framework based on Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.05107)]
[[Project](https://dreamoving.github.io/dreamoving/)] 
[[Code](https://github.com/dreamoving/dreamoving-project)]

**Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution** \
[[Website](https://arxiv.org/abs/2312.06640)]
[[Project](https://shangchenzhou.com/projects/upscale-a-video/)] 
[[Code](https://github.com/sczhou/Upscale-A-Video)]

**FreeInit: Bridging Initialization Gap in Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.07537)]
[[Project](https://tianxingwu.github.io/pages/FreeInit/)] 
[[Code](https://github.com/TianxingWu/FreeInit)]

**Text2AC-Zero: Consistent Synthesis of Animated Characters using 2D Diffusion** \
[[Website](https://arxiv.org/abs/2312.07133)]
[[Project](https://abdo-eldesokey.github.io/text2ac-zero/)]
[[Code](https://github.com/abdo-eldesokey/text2ac-zero)]

**StyleCrafter: Enhancing Stylized Text-to-Video Generation with Style Adapter** \
[[Website](https://arxiv.org/abs/2312.00330)]
[[Project](https://gongyeliu.github.io/StyleCrafter.github.io/)]
[[Code](https://github.com/GongyeLiu/StyleCrafter)]

**A Recipe for Scaling up Text-to-Video Generation with Text-free Videos** \
[[Website](https://arxiv.org/abs/2312.15770)]
[[Project](https://tf-t2v.github.io/)]
[[Code](https://github.com/ali-vilab/i2vgen-xl)]

**FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis** \
[[Website](https://arxiv.org/abs/2312.17681)]
[[Project](https://jeff-liangf.github.io/projects/flowvid/)]
[[Code](https://github.com/Jeff-LiangF/FlowVid)]

**Moonshot: Towards Controllable Video Generation and Editing with Multimodal Conditions** \
[[Website](https://arxiv.org/abs/2401.01827)]
[[Project](https://showlab.github.io/Moonshot/)]
[[Code](https://github.com/salesforce/LAVIS)]

**Latte: Latent Diffusion Transformer for Video Generation** \
[[Website](https://arxiv.org/abs/2401.03048)]
[[Project](https://maxin-cn.github.io/latte_project/)]
[[Code](https://github.com/maxin-cn/Latte)]

**WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens** \
[[Website](https://arxiv.org/abs/2401.09985)]
[[Project](https://world-dreamer.github.io/)]
[[Code](https://github.com/JeffWang987/WorldDreamer)]

**SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.16933)]
[[Project](https://guoyww.github.io/projects/SparseCtrl/)] 
[[Code](https://github.com/guoyww/AnimateDiff#202312-animatediff-v3-and-sparsectrl)]

**Towards A Better Metric for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2401.07781)]
[[Project](https://showlab.github.io/T2VScore/)]
[[Code](https://github.com/showlab/T2VScore)] 

**AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning** \
[[Website](https://arxiv.org/abs/2402.00769)]
[[Project](https://animatelcm.github.io/)]
[[Code](https://github.com/G-U-N/AnimateLCM)] 

**Be-Your-Outpainter: Mastering Video Outpainting through Input-Specific Adaptation** \
[[Website](https://arxiv.org/abs/2403.13745)]
[[Project](https://be-your-outpainter.github.io/)]
[[Code](https://github.com/G-U-N/Be-Your-Outpainter)]

**UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control** \
[[Website](https://arxiv.org/abs/2403.02332)]
[[Project](https://unified-attention-control.github.io/)]
[[Code](https://github.com/XuweiyiChen/UniCtrl)] 

**VideoElevator: Elevating Video Generation Quality with Versatile Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.05438)]
[[Project](https://videoelevator.github.io/)]
[[Code](https://github.com/YBYBZhang/VideoElevator)] 

**ID-Animator: Zero-Shot Identity-Preserving Human Video Generation** \
[[Website](https://arxiv.org/abs/2404.15275)]
[[Project](https://id-animator.github.io/)]
[[Code](https://github.com/ID-Animator/ID-Animator)] 

**FlexiFilm: Long Video Generation with Flexible Conditions** \
[[Website](https://arxiv.org/abs/2404.18620)]
[[Project](https://y-ichen.github.io/FlexiFilm-Page/)]
[[Code](https://github.com/Y-ichen/FlexiFilm)] 

**FIFO-Diffusion: Generating Infinite Videos from Text without Training** \
[[Website](https://arxiv.org/abs/2405.11473)]
[[Project](https://jjihwan.github.io/projects/FIFO-Diffusion)]
[[Code](https://github.com/jjihwan/FIFO-Diffusion_public)] 

**TALC: Time-Aligned Captions for Multi-Scene Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2405.04682)]
[[Project](https://talc-mst2v.github.io/)]
[[Code](https://github.com/Hritikbansal/talc)] 

**CV-VAE: A Compatible Video VAE for Latent Generative Video Models** \
[[Website](https://arxiv.org/abs/2405.20279)]
[[Project](https://ailab-cvc.github.io/cvvae/index.html)]
[[Code](https://github.com/AILab-CVC/CV-VAE)] 

**MVOC: a training-free multiple video object composition method with diffusion models** \
[[Website](https://arxiv.org/abs/2406.15829)]
[[Project](https://sobeymil.github.io/mvoc.com/)]
[[Code](https://github.com/SobeyMIL/MVOC)] 

**Identifying and Solving Conditional Image Leakage in Image-to-Video Diffusion Model** \
[[Website](https://arxiv.org/abs/2406.15735)]
[[Project](https://cond-image-leak.github.io/)]
[[Code](https://github.com/thu-ml/cond-image-leakage/)] 

**Identifying and Solving Conditional Image Leakage in Image-to-Video Diffusion Model** \
[[Website](https://arxiv.org/abs/2406.16863)]
[[Project](http://haonanqiu.com/projects/FreeTraj.html)]
[[Code](https://github.com/arthur-qiu/FreeTraj)] 

**Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation** \
[[Website](https://arxiv.org/abs/2408.15239)]
[[Project](https://svd-keyframe-interpolation.github.io/)]
[[Code](https://github.com/jeanne-wang/svd_keyframe_interpolation)] 

**AMG: Avatar Motion Guided Video Generation** \
[[Website](https://arxiv.org/abs/2409.01502)]
[[Project](https://zshyang.github.io/amg-website/)]
[[Code](https://github.com/zshyang/amg)] 

**DiVE: DiT-based Video Generation with Enhanced Control** \
[[Website](https://arxiv.org/abs/2409.01595)]
[[Project](https://liautoad.github.io/DIVE/)]
[[Code](https://github.com/LiAutoAD/DIVE)] 

**MegActor-Σ: Unlocking Flexible Mixed-Modal Control in Portrait Animation with Diffusion Transformer** \
[[Website](https://arxiv.org/abs/2408.14975)]
[[Project](https://megactor-ops.github.io/)]
[[Code](https://github.com/megvii-research/MegActor)] 

**CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers** \
[[ICLR 2023](https://arxiv.org/abs/2205.15868)]
[[Code](https://github.com/THUDM/CogVideo)]


**CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer** \
[[Website](https://arxiv.org/abs/2408.06072)]
[[Code](https://github.com/THUDM/CogVideo)]

**Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing** \
[[ICLR 2024](https://arxiv.org/abs/2402.16627)]
[[Code](https://github.com/YangLing0818/ContextDiff)]

**SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces** \
[[ICLR 2024](https://arxiv.org/abs/2403.07711)]
[[Code](https://github.com/shim0114/SSM-Meets-Video-Diffusion-Models)]

**Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach** \
[[Website](https://arxiv.org/abs/2410.03160)]
[[Code](https://github.com/Yaofang-Liu/FVDM)]

**Real-Time Video Generation with Pyramid Attention Broadcast** \
[[Website](https://arxiv.org/abs/2408.12588)]
[[Code](https://github.com/NUS-HPC-AI-Lab/VideoSyso)]

**Co-Speech Gesture Video Generation via Motion-Decoupled Diffusion Model** \
[[Website](https://arxiv.org/abs/2404.01862)]
[[Code](https://github.com/thuhcsi/S2G-MDDiffusion)]

**Diffusion Probabilistic Modeling for Video Generation** \
[[Website](https://arxiv.org/abs/2203.09481)]
[[Code](https://github.com/buggyyang/RVD)]

**DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors** \
[[Website](https://arxiv.org/abs/2310.12190)]
[[Code](https://github.com/AILab-CVC/VideoCrafter)]

**VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation** \
[[Website](https://arxiv.org/abs/2303.08320)]
[[Code](https://github.com/modelscope/modelscope)]

**STDiff: Spatio-temporal Diffusion for Continuous Stochastic Video Prediction** \
[[Website](https://arxiv.org/abs/2312.06486)]
[[Code](https://github.com/xiye20/stdiffproject)]

**Vlogger: Make Your Dream A Vlog** \
[[Website](https://arxiv.org/abs/2401.09414)]
[[Code](https://github.com/zhuangshaobin/Vlogger)]

**Magic-Me: Identity-Specific Video Customized Diffusion** \
[[Website](https://arxiv.org/abs/2402.09368)]
[[Code](https://github.com/Zhen-Dong/Magic-Me)]

**VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.06098)]
[[Code](https://github.com/WangWenhao0716/VidProM)]

**EchoReel: Enhancing Action Generation of Existing Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.11535)]
[[Code](https://github.com/liujianzhi/echoreel)]

**StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text** \
[[Website](https://arxiv.org/abs/2403.14773)]
[[Code](https://github.com/Picsart-AI-Research/StreamingT2V)]

**TAVGBench: Benchmarking Text to Audible-Video Generation** \
[[Website](https://arxiv.org/abs/2404.14381)]
[[Code](https://github.com/OpenNLPLab/TAVGBench)]

**OD-VAE: An Omni-dimensional Video Compressor for Improving Latent Video Diffusion Model** \
[[Website](https://arxiv.org/abs/2409.01199)]
[[Code](https://github.com/PKU-YuanGroup/Open-Sora-Plan)]

**IV-Mixed Sampler: Leveraging Image Diffusion Models for Enhanced Video Synthesis** \
[[Website](https://arxiv.org/abs/2410.04171)]
[[Code](https://github.com/xie-lab-ml/IV-mixed-Sampler)]

**MAVIN: Multi-Action Video Generation with Diffusion Models via Transition Video Infilling** \
[[Website](https://arxiv.org/abs/2405.18003)]
[[Code](https://github.com/18445864529/MAVIN)]

**HARIVO: Harnessing Text-to-Image Models for Video Generation**
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06938.pdf)]
[[Project](https://kwonminki.github.io/HARIVO/)]

**Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners** \
[[CVPR 2024](https://arxiv.org/abs/2402.17723)]
[[Project](https://yzxing87.github.io/Seeing-and-Hearing/)]

**AtomoVideo: High Fidelity Image-to-Video Generation** \
[[CVPR 2024](https://arxiv.org/abs/2403.01800)]
[[Project](https://atomo-video.github.io/)]

**Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition** \
[[ICLR 2024](https://arxiv.org/abs/2403.14148)]
[[Project](https://sihyun.me/CMD/)] 

**TRIP: Temporal Residual Learning with Image Noise Prior for Image-to-Video Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.17005)]
[[Project](https://trip-i2v.github.io/TRIP/)]

**ZoLA: Zero-Shot Creative Long Animation Generation with Short Video Model** \
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06174.pdf)]
[[Project](https://gen-l-2.github.io/)]

**TCAN: Animating Human Images with Temporally Consistent Pose Guidance using Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2407.09012)]
[[Project](https://eccv2024tcan.github.io/)] 

**Training-free Long Video Generation with Chain of Diffusion Model Experts** \
[[Website](https://arxiv.org/abs/2408.13423)]
[[Project](https://confiner2025.github.io/)]

**FreeLong: Training-Free Long Video Generation with SpectralBlend Temporal Attention** \
[[Website](https://arxiv.org/abs/2407.19918)]
[[Project](https://freelongvideo.github.io/)]

**CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation** \
[[Website](https://arxiv.org/abs/2406.02509)]
[[Project](https://ir1d.github.io/CamCo/)]

**Hierarchical Patch Diffusion Models for High-Resolution Video Generation** \
[[Website](https://arxiv.org/abs/2406.07792)]
[[Project](https://snap-research.github.io/hpdm/)]

**I4VGen: Image as Stepping Stone for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2406.02230)]
[[Project](https://xiefan-guo.github.io/i4vgen/)]

**UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation** \
[[Website](https://arxiv.org/abs/2406.01188)]
[[Project](https://unianimate.github.io/)]

**Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control** \
[[Website](https://arxiv.org/abs/2405.17414)]
[[Project](https://collaborativevideodiffusion.github.io/)]

**Controllable Longer Image Animation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.17306)]
[[Project](https://wangqiang9.github.io/Controllable.github.io/)]

**AniClipart: Clipart Animation with Text-to-Video Priors** \
[[Website](https://arxiv.org/abs/2404.12347)]
[[Project](https://aniclipart.github.io/)]

**Spectral Motion Alignment for Video Motion Transfer using Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.15249)]
[[Project](https://geonyeong-park.github.io/spectral-motion-alignment/)]

**TimeRewind: Rewinding Time with Image-and-Events Video Diffusion** \
[[Website](https://arxiv.org/abs/2403.13800)]
[[Project](https://timerewind.github.io/)]

**VideoPoet: A Large Language Model for Zero-Shot Video Generation** \
[[Website](https://storage.googleapis.com/videopoet/paper.pdf)]
[[Project](https://sites.research.google/videopoet/)]

**PEEKABOO: Interactive Video Generation via Masked-Diffusion**\
[[Website](https://arxiv.org/abs/2312.07509)]
[[Project](https://jinga-lala.github.io/projects/Peekaboo/)]

**Searching Priors Makes Text-to-Video Synthesis Better** \
[[Website](https://arxiv.org/abs/2406.03215)]
[[Project](https://hrcheng98.github.io/Search_T2V/)] 

**Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2309.03549)]
[[Project](https://anonymous0x233.github.io/ReuseAndDiffuse/)] 

**Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning** \
[[Website](https://arxiv.org/abs/2311.10709)]
[[Project](https://emu-video.metademolab.com/)] 

**BIVDiff: A Training-Free Framework for General-Purpose Video Synthesis via Bridging Image and Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.02813)]
[[Project](https://bivdiff.github.io/)] 


**Imagen Video: High Definition Video Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2210.02303)]
[[Project](https://imagen.research.google/video/)] 

**MoVideo: Motion-Aware Video Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.11325)]
[[Project](https://jingyunliang.github.io/MoVideo/)] 


**Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer** \
[[Website](https://arxiv.org/abs/2311.17009)]
[[Project](https://diffusion-motion-transfer.github.io/)] 

**Smooth Video Synthesis with Noise Constraints on Diffusion Models for One-shot Video Tuning** \
[[Website](https://arxiv.org/abs/2311.17536)]
[[Project](https://github.com/SPengLiang/SmoothVideo)] 

**VideoAssembler: Identity-Consistent Video Generation with Reference Entities using Diffusion Model** \
[[Website](https://arxiv.org/abs/2311.17338)]
[[Project](https://videoassembler.github.io/videoassembler/)] 

**MicroCinema: A Divide-and-Conquer Approach for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2311.18829)]
[[Project](https://wangyanhui666.github.io/MicroCinema.github.io/)] 

**Generative Rendering: Controllable 4D-Guided Video Generation with 2D Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.01409)]
[[Project](https://primecai.github.io/generative_rendering/)] 

**GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation** \
[[Website](https://arxiv.org/abs/2312.04557)]
[[Project](https://www.shoufachen.com/gentron_website/)] 

**Customizing Motion in Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.04966)]
[[Project](https://joaanna.github.io/customizing_motion/)] 

**Photorealistic Video Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.06662)] 
[[Project](https://walt-video-diffusion.github.io/)] 

**VideoDrafter: Content-Consistent Multi-Scene Video Generation with LLM** \
[[Website](https://arxiv.org/abs/2401.01256)] 
[[Project](https://videodrafter.github.io/)] 

**Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.10474)]
[[Project](https://research.nvidia.com/labs/dir/pyoco/)] 

**ActAnywhere: Subject-Aware Video Background Generation** \
[[Website](https://arxiv.org/abs/2401.10822)]
[[Project](https://actanywhere.github.io/)] 

**Lumiere: A Space-Time Diffusion Model for Video Generation** \
[[Website](https://arxiv.org/abs/2401.12945)]
[[Project](https://lumiere-video.github.io/)] 

**InstructVideo: Instructing Video Diffusion Models with Human Feedback** \
[[Website](https://arxiv.org/abs/2312.12490)]
[[Project](https://instructvideo.github.io/)] 

**Boximator: Generating Rich and Controllable Motions for Video Synthesis** \
[[Website](https://arxiv.org/abs/2402.01566)]
[[Project](https://boximator.github.io/)] 

**Direct-a-Video: Customized Video Generation with User-Directed Camera Movement and Object Motion** \
[[Website](https://arxiv.org/abs/2402.03162)]
[[Project](https://direct-a-video.github.io/)] 

**ConsistI2V: Enhancing Visual Consistency for Image-to-Video Generation** \
[[Website](https://arxiv.org/abs/2402.04324)]
[[Project](https://tiger-ai-lab.github.io/ConsistI2V/)] 

**Tuning-Free Noise Rectification for High Fidelity Image-to-Video Generation** \
[[Website](https://arxiv.org/abs/2403.02827)]
[[Project](https://noise-rectification.github.io/)] 

**Audio-Synchronized Visual Animation** \
[[Website](https://arxiv.org/abs/2403.05659)]
[[Project](https://lzhangbj.github.io/projects/asva/asva.html)] 

**VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis** \
[[Website](https://arxiv.org/abs/2403.13501)]
[[Project](https://yumengli007.github.io/VSTAR/)] 

**S2DM: Sector-Shaped Diffusion Models for Video Generation** \
[[Website](https://arxiv.org/abs/2403.13408)]
[[Project](https://s2dm.github.io/S2DM/)] 

**AnimateZoo: Zero-shot Video Generation of Cross-Species Animation via Subject Alignment** \
[[Website](https://arxiv.org/abs/2404.04946)]
[[Project](https://justinxu0.github.io/AnimateZoo/)] 

**Disentangling Foreground and Background Motion for Enhanced Realism in Human Video Generation** \
[[Website](https://arxiv.org/abs/2405.16393)]
[[Project](https://liujl09.github.io/humanvideo_movingbackground/)] 

**Dance Any Beat: Blending Beats with Visuals in Dance Video Generation** \
[[Website](https://arxiv.org/abs/2405.09266)]
[[Project](https://dabfusion.github.io/)] 

**PoseCrafter: One-Shot Personalized Video Synthesis Following Flexible Pose Control** \
[[Website](https://arxiv.org/abs/2405.14582)]
[[Project](https://ml-gsai.github.io/PoseCrafter-demo/)] 

**Human4DiT: Free-view Human Video Generation with 4D Diffusion Transformer** \
[[Website](https://arxiv.org/abs/2405.17405)]
[[Project](https://human4dit.github.io/)] 

**Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation** \
[[Website](https://arxiv.org/abs/2406.01900)]
[[Project](https://follow-your-emoji.github.io/)] 

**FancyVideo: Towards Dynamic and Consistent Video Generation via Cross-frame Textual Guidance** \
[[Website](https://arxiv.org/abs/2408.08189)]
[[Project](https://fancyvideo.github.io/)] 

**CustomCrafter: Customized Video Generation with Preserving Motion and Concept Composition Abilities** \
[[Website](https://arxiv.org/abs/2408.13239)]
[[Project](https://customcrafter.github.io/)] 

**VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide** \
[[Website](https://arxiv.org/abs/2410.04364)]
[[Project](https://videoguide2025.github.io/)] 

**Grid Diffusion Models for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2404.00234)]

**SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction** \
[[Website](https://arxiv.org/abs/2310.20700)]

**GenRec: Unifying Video Generation and Recognition with Diffusion Models** \
[[Website](https://arxiv.org/abs/2408.15241)]


**Dual-Stream Diffusion Net for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2308.08316)]

**DisenStudio: Customized Multi-subject Text-to-Video Generation with Disentangled Spatial Control** \
[[Website](https://arxiv.org/abs/2405.12796)]

**SimDA: Simple Diffusion Adapter for Efficient Video Generation** \
[[Website](https://arxiv.org/abs/2308.09710)]

**VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2305.10874)]

**Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models** \
[[Website](https://arxiv.org/abs/2308.13812)]

**ConditionVideo: Training-Free Condition-Guided Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2310.07697)]

**LatentWarp: Consistent Diffusion Latents for Zero-Shot Video-to-Video Translation** \
[[Website](https://arxiv.org/abs/2311.00353)]

**Optimal Noise pursuit for Augmenting Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2311.00949)]

**Make Pixels Dance: High-Dynamic Video Generation** \
[[Website](https://arxiv.org/abs/2311.10982)]

**Video-Infinity: Distributed Long Video Generation** \
[[Website](https://arxiv.org/abs/2406.16260)]

**GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning** \
[[Website](https://arxiv.org/abs/2311.12631)]

**Highly Detailed and Temporal Consistent Video Stylization via Synchronized Multi-Frame Diffusion** \
[[Website](https://arxiv.org/abs/2311.14343)]

**Decouple Content and Motion for Conditional Image-to-Video Generation** \
[[Website](https://arxiv.org/abs/2311.14294)]

**X-Portrait: Expressive Portrait Animation with Hierarchical Motion Attention** \
[[Website](https://arxiv.org/abs/2403.15931)]

**F3-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis** \
[[Website](https://arxiv.org/abs/2312.03459)]

**MTVG : Multi-text Video Generation with Text-to-Video Models** \
[[Website](https://arxiv.org/abs/2312.04086)]

**VideoLCM: Video Latent Consistency Model** \
[[Website](https://arxiv.org/abs/2312.09109)]

**MotionAura: Generating High-Quality and Motion Consistent Videos using Discrete Diffusion** \
[[Website](https://arxiv.org/abs/2410.07659)]

**MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation** \
[[Website](https://arxiv.org/abs/2401.04468)]

**I2V-Adapter: A General Image-to-Video Adapter for Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.16693)]

**360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model** \
[[Website](https://arxiv.org/abs/2401.06578)]

**CustomVideo: Customizing Text-to-Video Generation with Multiple Subjects** \
[[Website](https://arxiv.org/abs/2401.09962)]

**Motion-Zero: Zero-Shot Moving Object Control Framework for Diffusion-Based Video Generation** \
[[Website](https://arxiv.org/abs/2401.10150)]

**Training-Free Semantic Video Composition via Pre-trained Diffusion Model** \
[[Website](https://arxiv.org/abs/2401.09195)]

**Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling** \
[[Website](https://arxiv.org/abs/2401.15977)]

**Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.16224)]

**Human Video Translation via Query Warping** \
[[Website](https://arxiv.org/abs/2402.12099)]

**Hybrid Video Diffusion Models with 2D Triplane and 3D Wavelet Representation** \
[[Website](https://arxiv.org/abs/2402.13729)]

**Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis** \
[[Website](https://arxiv.org/abs/2402.14797v1)]

**EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions** \
[[Website](https://arxiv.org/abs/2402.17485)]

**Context-aware Talking Face Video Generation** \
[[Website](https://arxiv.org/abs/2402.18092)]

**Pix2Gif: Motion-Guided Diffusion for GIF Generation** \
[[Website](https://arxiv.org/abs/2403.04634)]

**Intention-driven Ego-to-Exo Video Generation** \
[[Website](https://arxiv.org/abs/2403.09194)]

**AnimateDiff-Lightning: Cross-Model Diffusion Distillation** \
[[Website](https://arxiv.org/abs/2403.12706)]

**Frame by Familiar Frame: Understanding Replication in Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.19593)]

**Matten: Video Generation with Mamba-Attention** \
[[Website](https://arxiv.org/abs/2405.03025)]

**Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.04233)]

**ReVideo: Remake a Video with Motion and Content Control** \
[[Website](https://arxiv.org/abs/2405.13865)]

**VividPose: Advancing Stable Video Diffusion for Realistic Human Image Animation** \
[[Website](https://arxiv.org/abs/2405.18156)]

**SNED: Superposition Network Architecture Search for Efficient Video Diffusion Model** \
[[Website](https://arxiv.org/abs/2406.00195)]

**GVDIFF: Grounded Text-to-Video Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2407.01921)]

**Mobius: An High Efficient Spatial-Temporal Parallel Training Paradigm for Text-to-Video Generation Task** \
[[Website](https://arxiv.org/abs/2407.06617)]

**Contrastive Sequential-Diffusion Learning: An approach to Multi-Scene Instructional Video Synthesis** \
[[Website](https://arxiv.org/abs/2407.11814)]

**Multi-sentence Video Grounding for Long Video Generation** \
[[Website](https://arxiv.org/abs/2407.13219)]

**Fine-gained Zero-shot Video Sampling** \
[[Website](https://arxiv.org/abs/2407.21475)]

**Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data** \
[[Website](https://arxiv.org/abs/2408.10119)]

**xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations** \
[[Website](https://arxiv.org/abs/2408.12590)]

**EasyControl: Transfer ControlNet to Video Diffusion for Controllable Generation and Interpolation** \
[[Website](https://arxiv.org/abs/2408.13005)]

**Alignment is All You Need: A Training-free Augmentation Strategy for Pose-guided Video Generation** \
[[Website](https://arxiv.org/abs/2408.16506)]

**One-Shot Learning Meets Depth Diffusion in Multi-Object Videos** \
[[Website](https://arxiv.org/abs/2408.16704)]

**Denoising Reuse: Exploiting Inter-frame Motion Consistency for Efficient Video Latent Generation** \
[[Website](https://arxiv.org/abs/2409.12532)]

**S2AG-Vid: Enhancing Multi-Motion Alignment in Video Diffusion Models via Spatial and Syntactic Attention-Based Guidance** \
[[Website](https://arxiv.org/abs/2409.15259)]

**JVID: Joint Video-Image Diffusion for Visual-Quality and Temporal-Consistency in Video Generation** \
[[Website](https://arxiv.org/abs/2409.14149)]

**ImmersePro: End-to-End Stereo Video Synthesis Via Implicit Disparity Learning** \
[[Website](https://arxiv.org/abs/2410.00262)]

**COMUNI: Decomposing Common and Unique Video Signals for Diffusion-based Video Generation** \
[[Website](https://arxiv.org/abs/2410.01718)]

**Noise Crystallization and Liquid Noise: Zero-shot Video Generation using Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2410.05322)]

**BroadWay: Boost Your Text-to-Video Generation Model in a Training-free Way** \
[[Website](https://arxiv.org/abs/2410.06241)]



## Video Editing 

**FateZero: Fusing Attentions for Zero-shot Text-based Video Editing** \
[[ICCV 2023 Oral](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html
)]
[[Website](https://arxiv.org/abs/2303.09535)]
[[Project](https://fate-zero-edit.github.io/)] 
[[Code](https://github.com/ChenyangQiQi/FateZero)]

**Text2LIVE: Text-Driven Layered Image and Video Editing** \
[[ECCV 2022 Oral](https://arxiv.org/abs/2204.02491)]
[[Project](https://text2live.github.io/)] 
[[code](https://github.com/omerbt/Text2LIVE)]


**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding** \
[[CVPR 2023](https://arxiv.org/abs/2212.02802)]
[[Project](https://diff-video-ae.github.io/)] 
[[Code](https://github.com/man805/Diffusion-Video-Autoencoders)] 


**Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation** \
[[ICCV 2023](https://arxiv.org/abs/2212.11565)]
[[Project](https://tuneavideo.github.io/)]
[[Code](https://github.com/showlab/Tune-A-Video)]

**StableVideo: Text-driven Consistency-aware Diffusion Video Editing** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_StableVideo_Text-driven_Consistency-aware_Diffusion_Video_Editing_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2308.09592)]
[[Code](https://github.com/rese1f/stablevideo)] 

**Noise Calibration: Plug-and-play Content-Preserving Video Enhancement using Pre-trained Video Diffusion Models** \
[[ECCV 2024](https://arxiv.org/abs/2407.10285)]
[[Project](https://yangqy1110.github.io/NC-SDEdit/)] 
[[Code](https://github.com/yangqy1110/NC-SDEdit/)]

**Video-P2P: Video Editing with Cross-attention Control** \
[[Website](https://arxiv.org/abs/2303.04761)]
[[Project](https://video-p2p.github.io/)] 
[[Code](https://github.com/ShaoTengLiu/Video-P2P)]


**CoDeF: Content Deformation Fields for Temporally Consistent Video Processing** \
[[Website](https://arxiv.org/abs/2308.07926)]
[[Project](https://qiuyu96.github.io/CoDeF/)]
[[Code](https://github.com/qiuyu96/CoDeF)]

**MagicEdit: High-Fidelity and Temporally Coherent Video Editing**\
[[Website](https://arxiv.org/abs/2308.14749)]
[[Project](https://magic-edit.github.io/)] 
[[Code](https://github.com/magic-research/magic-edit)] 



**TokenFlow: Consistent Diffusion Features for Consistent Video Editing** \
[[Website](https://arxiv.org/abs/2307.10373)]
[[Project](https://diffusion-tokenflow.github.io/)] 
[[Code](https://github.com/omerbt/TokenFlow)]

**ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing** \
[[Website](https://arxiv.org/abs/2305.17098)]
[[Project](https://ml.cs.tsinghua.edu.cn/controlvideo/)] 
[[Code](https://github.com/thu-ml/controlvideo)]

**Make-A-Protagonist: Generic Video Editing with An Ensemble of Experts** \
[[Website](https://arxiv.org/abs/2305.08850)]
[[Project](https://make-a-protagonist.github.io/)] 
[[Code](https://github.com/Make-A-Protagonist/Make-A-Protagonist)]

**MotionDirector: Motion Customization of Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2310.08465)]
[[Project](https://showlab.github.io/MotionDirector/)] 
[[Code](https://github.com/showlab/MotionDirector)]

**EVA: Zero-shot Accurate Attributes and Multi-Object Video Editing** \
[[Website](https://arxiv.org/abs/2403.16111)]
[[Project](https://knightyxp.github.io/EVA/)] 
[[Code](https://github.com/knightyxp/EVA_Video_Edit)]



**RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models**\
[[Website](https://arxiv.org/abs/2312.04524)]
[[Project](https://rave-video.github.io/)] 
[[Code](https://github.com/rehg-lab/RAVE)]


**Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models**\
[[Website](https://arxiv.org/abs/2310.01107)]
[[Project](https://ground-a-video.github.io/)]
[[Code](https://github.com/Ground-A-Video/Ground-A-Video)] 

**MotionEditor: Editing Video Motion via Content-Aware Diffusion** \
[[Website](https://arxiv.org/abs/2311.18830)]
[[Project](https://francis-rings.github.io/MotionEditor/)]
[[Code](https://github.com/Francis-Rings/MotionEditor)] 

**VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.00845)]
[[Project](https://video-motion-customization.github.io/)]
[[Code](https://github.com/HyeonHo99/Video-Motion-Customization)] 

**MagicStick: Controllable Video Editing via Control Handle Transformations** \
[[Website](https://arxiv.org/abs/2312.03047)]
[[Project](https://magic-stick-edit.github.io/)]
[[Code](https://github.com/mayuelala/MagicStick)] 

**VidToMe: Video Token Merging for Zero-Shot Video Editing** \
[[Website](https://arxiv.org/abs/2312.10656)]
[[Project](https://vidtome-diffusion.github.io/)]
[[Code](https://github.com/lixirui142/VidToMe)] 


**VASE: Object-Centric Appearance and Shape Manipulation of Real Videos** \
[[Website](https://arxiv.org/abs/2401.02473)]
[[Project](https://helia95.github.io/vase-website/)]
[[Code](https://github.com/helia95/VASE)] 

**Neural Video Fields Editing** \
[[Website](https://arxiv.org/abs/2312.08882)]
[[Project](https://nvedit.github.io/)]
[[Code](https://github.com/Ysz2022/NVEdit)] 

**UniEdit: A Unified Tuning-Free Framework for Video Motion and Appearance Editing** \
[[Website](https://arxiv.org/abs/2402.13185v1)]
[[Project](https://jianhongbai.github.io/UniEdit/)]
[[Code](https://github.com/JianhongBai/UniEdit)] 

**MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion** \
[[Website](https://arxiv.org/abs/2405.20325)]
[[Project](https://francis-rings.github.io/MotionFollower/)]
[[Code](https://github.com/Francis-Rings/MotionFollower)] 


**Vid2Vid-zero: Zero-Shot Video Editing Using Off-the-Shelf Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.17599)]
[[Code](https://github.com/baaivision/vid2vid-zero)]


**DiffSLVA: Harnessing Diffusion Models for Sign Language Video Anonymization** \
[[Website](https://arxiv.org/abs/2311.16060)]
[[Code](https://github.com/Jeffery9707/DiffSLVA)] 

**LOVECon: Text-driven Training-Free Long Video Editing with ControlNet** \
[[Website](https://arxiv.org/abs/2310.09711)]
[[Code](https://github.com/zhijie-group/LOVECon)] 

**Pix2video: Video Editing Using Image Diffusion** \
[[Website](https://arxiv.org/abs/2303.12688)]
[[Code](https://github.com/G-U-N/Pix2Video.pytorch)] 

**E-Bench: Subjective-Aligned Benchmark Suite for Text-Driven Video Editing Quality Assessment** \
[[Website](https://arxiv.org/abs/2408.11481)]
[[Code](https://github.com/littlespray/E-Bench)] 

**Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer**\
[[Website](https://arxiv.org/abs/2305.05464)]
[[Code](https://github.com/haha-lisa/style-a-video)] 

**Flow-Guided Diffusion for Video Inpainting** \
[[Website](https://arxiv.org/abs/2311.15368)]
[[Code](https://github.com/nevsnev/fgdvi)] 

**Investigating the Effectiveness of Cross-Attention to Unlock Zero-Shot Editing of Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.05519)]
[[Code](https://github.com/sam-motamed/Video-Editing-X-Attention/)]

**Edit-Your-Motion: Space-Time Diffusion Decoupling Learning for Video Motion Editing** \
[[Website](https://arxiv.org/abs/2405.04496)]
[[Code](https://github.com/yiiizuo/Edit-Your-Motion)]

**COVE: Unleashing the Diffusion Feature Correspondence for Consistent Video Editing** \
[[Website](https://arxiv.org/abs/2406.08850)]
[[Code](https://github.com/wangjiangshan0725/COVE)]

**Shape-Aware Text-Driven Layered Video Editing** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Shape-Aware_Text-Driven_Layered_Video_Editing_CVPR_2023_paper.html
)]
[[Website](https://arxiv.org/abs/2301.13173)]
[[Project](https://text-video-edit.github.io/#)] 

**NaRCan: Natural Refined Canonical Image with Integration of Diffusion Prior for Video Editing** \
[[Website](https://arxiv.org/abs/2406.06523)]
[[Project](https://koi953215.github.io/NaRCan_page/)]

**Slicedit: Zero-Shot Video Editing With Text-to-Image Diffusion Models Using Spatio-Temporal Slices** \
[[Website](https://arxiv.org/abs/2405.12211)]
[[Project](https://matankleiner.github.io/slicedit/)]

**DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing** \
[[Website](https://arxiv.org/abs/2310.10624)]
[[Project](https://showlab.github.io/DynVideo-E/)]

**I2VEdit: First-Frame-Guided Video Editing via Image-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.16537)]
[[Project](https://i2vedit.github.io/)] 

**FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing** \
[[Website](https://arxiv.org/abs/2310.05922)]
[[Project](https://flatten-video-editing.github.io/)] 

**VidEdit: Zero-Shot and Spatially Aware Text-Driven Video Editing** \
[[Website](https://arxiv.org/abs//2306.08707)]
[[Project](https://videdit.github.io/)] 

**VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence** \
[[Website](https://arxiv.org/abs/2312.02087)]
[[Project](https://videoswap.github.io/)] 

**Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation** \
[[Website](https://arxiv.org/abs/2306.07954)]
[[Project](https://anonymous-31415926.github.io/)] 

**WAVE: Warping DDIM Inversion Features for Zero-shot Text-to-Video Editing**
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09682.pdf)]
[[Project](https://ree1s.github.io/wave/)]

**MeDM: Mediating Image Diffusion Models for Video-to-Video Translation with Temporal Correspondence Guidance** \
[[Website](https://arxiv.org/abs/2308.10079)]
[[Project](https://medm2023.github.io)]

**Customize-A-Video: One-Shot Motion Customization of Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.14780)]
[[Project](https://anonymous-314.github.io/)]

**DreamMotion: Space-Time Self-Similarity Score Distillation for Zero-Shot Video Editing** \
[[Website](https://arxiv.org/abs/2403.12002)]
[[Project](https://hyeonho99.github.io/dreammotion/)]

**DeCo: Decoupled Human-Centered Diffusion Video Editing with Motion Consistency** \
[[ECCV 2024](https://arxiv.org/abs/2408.07481)]

**Edit Temporal-Consistent Videos with Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2308.09091)]

**Streaming Video Diffusion: Online Video Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2405.19726)]

**Cut-and-Paste: Subject-Driven Video Editing with Attention Control** \
[[Website](https://arxiv.org/abs/2311.11697)]

**MagicProp: Diffusion-based Video Editing via Motion-aware Appearance Propagation** \
[[Website](https://arxiv.org/abs/2309.00908)]

**Dreamix: Video Diffusion Models Are General Video Editors** \
[[Website](https://arxiv.org/abs/2302.01329)]

**Towards Consistent Video Editing with Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.17431)]

**EVE: Efficient zero-shot text-based Video Editing with Depth Map Guidance and Temporal Consistency Constraints** \
[[Website](https://arxiv.org/abs/2308.10648)]

**CCEdit: Creative and Controllable Video Editing via Diffusion Models** \
[[Website](https://arxiv.org/abs/2309.16496)]

**Fuse Your Latents: Video Editing with Multi-source Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2310.16400)]

**FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier** \
[[Website](https://arxiv.org/abs/2311.09265)]

**VIDiff: Translating Videos via Multi-Modal Instructions with Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.18837)]

**RealCraft: Attention Control as A Solution for Zero-shot Long Video Editing** \
[[Website](https://arxiv.org/abs/2312.12635)]

**Object-Centric Diffusion for Efficient Video Editing** \
[[Website](https://arxiv.org/abs/2401.05735)]

**FastVideoEdit: Leveraging Consistency Models for Efficient Text-to-Video Editing** \
[[Website](https://arxiv.org/abs/2403.06269)]

**Video Editing via Factorized Diffusion Distillation** \
[[Website](https://arxiv.org/abs/2403.06269)]

**EffiVED:Efficient Video Editing via Text-instruction Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.11568)]

**Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion** \
[[Website](https://arxiv.org/abs/2403.14617)]

**GenVideo: One-shot Target-image and Shape Aware Video Editing using T2I Diffusion Models** \
[[Website](https://arxiv.org/abs/2404.12541)]

**Temporally Consistent Object Editing in Videos using Extended Attention** \
[[Website](https://arxiv.org/abs/2406.00272)]

**Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting** \
[[Website](https://arxiv.org/abs/2406.02541)]

**FRAG: Frequency Adapting Group for Diffusion Video Editing** \
[[Website](https://arxiv.org/abs/2406.06044)]

**InVi: Object Insertion In Videos Using Off-the-Shelf Diffusion Models** \
[[Website](https://arxiv.org/abs/2407.10958)]

**Text-based Talking Video Editing with Cascaded Conditional Diffusion** \
[[Website](https://arxiv.org/abs/2407.14841)]

**Reenact Anything: Semantic Video Motion Transfer Using Motion-Textual Inversion** \
[[Website](https://arxiv.org/abs/2408.00458)]


**Blended Latent Diffusion under Attention Control for Real-World Video Editing** \
[[Website](https://arxiv.org/abs/2409.03514)]

**EditBoard: Towards A Comprehensive Evaluation Benchmark for Text-based Video Editing Models** \
[[Website](https://arxiv.org/abs/2409.09668)]

**DNI: Dilutional Noise Initialization for Diffusion Video Editing** \
[[Website](https://arxiv.org/abs/2409.13037)]

**FreeMask: Rethinking the Importance of Attention Masks for Zero-Shot Video Editing** \
[[Website](https://arxiv.org/abs/2409.20500)]

**Replace Anyone in Videos** \
[[Website](https://arxiv.org/abs/2409.19911)]
