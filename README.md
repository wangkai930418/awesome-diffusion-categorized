<!-- The superlink doesn't support uppercases -->

# Awesome Diffusion Categorized

## Contents

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
<!-- - [Representation Learning](#representation-learning) -->
<!-- - [X2I/X2X](#x2i-x2x) -->
<!-- - [Semantic Intrinsic](#semantic-intrinsic) -->


## Try On

**TryOnDiffusion: A Tale of Two UNets** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_TryOnDiffusion_A_Tale_of_Two_UNets_CVPR_2023_paper.html)]
[[Website](https://arxiv.org/abs/2306.08276)]
[[Project](https://tryondiffusion.github.io/)]
[[Code](https://github.com/huggingface/diffusers/issues/5008)] 

**StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On** \
[[CVPR 2024](https://arxiv.org/abs/2312.01725)]
[[Project](https://rlawjdghek.github.io/StableVITON/)]
[[Code](https://github.com/rlawjdghek/stableviton?tab=readme-ov-file)] 

**Street TryOn: Learning In-the-Wild Virtual Try-On from Unpaired Person Images** \
[[Website](https://arxiv.org/abs/2311.16094)]
[[Project](https://cuiaiyu.github.io/StreetTryOn/)]
[[Code](https://github.com/cuiaiyu/street-tryon-benchmark)] 


**PICTURE: PhotorealistIC virtual Try-on from UnconstRained dEsigns** \
[[Website](https://arxiv.org/abs/2312.04534)]
[[Project](https://ningshuliang.github.io/2023/Arxiv/index.html)]
[[Code](https://github.com/ningshuliang/PICTURE)] 

**Taming the Power of Diffusion Models for High-Quality Virtual Try-On with Appearance Flow**  \
[[ACM MM 2023](https://arxiv.org/abs/2308.06101)]
[[Code](https://github.com/bcmi/DCI-VTON-Virtual-Try-On)] 

**LaDI-VTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On** \
[[ACM MM 2023](https://arxiv.org/abs/2305.13501)]
[[Code](https://github.com/miccunifi/ladi-vton)] 

**OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on** \
[[Website](https://arxiv.org/abs/2403.01779)]
[[Code](https://github.com/levihsu/OOTDiffusion)] 

**DreamPaint: Few-Shot Inpainting of E-Commerce Items for Virtual Try-On without 3D Modeling** \
[[Website](https://arxiv.org/abs/2305.01257)]
[[Code](https://github.com/EmergingUnicorns/DeepPaint)] 

**CAT-DM: Controllable Accelerated Virtual Try-on with Diffusion Model** \
[[Website](https://arxiv.org/abs/2311.18405)]
[[Code](https://github.com/zengjianhao/cat-dm)] 

**StableGarment: Garment-Centric Generation via Stable Diffusion** \
[[Website](https://arxiv.org/abs/2403.10783)]
[[Project](https://raywang335.github.io/stablegarment.github.io/)] 


**Diffuse to Choose: Enriching Image Conditioned Inpainting in Latent Diffusion Models for Virtual Try-All** \
[[Website](https://arxiv.org/abs/2401.13795)]
[[Project](https://diffuse2choose.github.io/)]

**Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment** \
[[Website](https://arxiv.org/abs/2403.12965)]
[[Project](https://mengtingchen.github.io/wear-any-way-page/)]

**WarpDiffusion: Efficient Diffusion Model for High-Fidelity Virtual Try-on** \
[[Website](https://arxiv.org/abs/2312.03667)]

**Product-Level Try-on: Characteristics-preserving Try-on with Realistic Clothes Shading and Wrinkles** \
[[Website](https://arxiv.org/abs/2401.11239)]

**Mobile Fitting Room: On-device Virtual Try-on via Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.01877)]

**Improving Diffusion Models for Virtual Try-on** \
[[Website](https://arxiv.org/abs/2403.05139)]

**Time-Efficient and Identity-Consistent Virtual Try-On Using A Variant of Altered Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.07371)]

**ACDG-VTON: Accurate and Contained Diffusion Generation for Virtual Try-On** \
[[Website](https://arxiv.org/abs/2403.13951)]

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


**FreeDrag: Feature Dragging for Reliable Point-based Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2307.04684)] 
[[Project](https://lin-chen.site/projects/freedrag/)] 
[[Code](https://github.com/LPengYang/FreeDrag)]

**DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2306.14435)] 
[[Project](https://yujun-shi.github.io/projects/dragdiffusion.html)] 
[[Code](https://github.com/Yujun-Shi/DragDiffusion)]

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

**DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing** \
[[CVPR 2024](https://arxiv.org/abs/2402.02583)] 
[[Code](https://github.com/MC-E/DragonDiffusion)]

**DragVideo: Interactive Drag-style Video Editing** \
[[Website](https://arxiv.org/abs/2312.02216)] 
[[Code](https://github.com/rickyskywalker/dragvideo-official)]

**RotationDrag: Point-based Image Editing with Rotated Diffusion Features** \
[[Website](https://arxiv.org/abs/2401.06442)] 
[[Code](https://github.com/Tony-Lowe/RotationDrag)]

**DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory** \
[[Website](https://arxiv.org/abs/2308.08089)] 
[[Project](https://www.microsoft.com/en-us/research/project/dragnuwa/)] 

**Readout Guidance: Learning Control from Diffusion Features** \
[[Website](https://arxiv.org/abs/2312.02150)] 
[[Project](https://readout-guidance.github.io/)] 

**StableDrag: Stable Dragging for Point-based Image Editing** \
[[Website](https://arxiv.org/abs/2403.04437)] 
[[Project](https://stabledrag.github.io/)]

**Motion Guidance: Diffusion-Based Image Editing with Differentiable Motion Estimators** \
[[Website](https://arxiv.org/abs/2401.18085)] 


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

**Noise Map Guidance: Inversion with Spatial Context for Real Image Editing** \
[[ICLR 2024](https://openreview.net/forum?id=mhgm0IXtHw)] 
[[Website](https://arxiv.org/abs/2402.04625)] 
[[Code](https://github.com/hansam95/nmg)] 

**IterInv: Iterative Inversion for Pixel-Level T2I Models** \
[[NeurIPS-W 2023](https://neurips.cc/virtual/2023/74859)] 
[[Openreview](https://openreview.net/forum?id=mSGmzVo0aS)] 
[[NeuripsW](https://neurips.cc/virtual/2023/workshop/66539#wse-detail-74859)]
[[Website](https://arxiv.org/abs/2310.19540)] 
[[Code](https://github.com/Tchuanm/IterInv)] 

**Object-aware Inversion and Reassembly for Image Editing** \
[[Website](https://arxiv.org/abs/2310.12149)] 
[[Project](https://aim-uofa.github.io/OIR-Diffusion/)] 
[[Code](https://github.com/aim-uofa/OIR)] 

**ReNoise: Real Image Inversion Through Iterative Noising** \
[[Website](https://arxiv.org/abs/2403.14602)] 
[[Project](https://garibida.github.io/ReNoise-Inversion/)] 
[[Code](https://github.com/garibida/ReNoise-Inversion)] 

**StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing** \
[[Website](https://arxiv.org/abs/2303.15649)] 
[[Code](https://github.com/sen-mao/StyleDiffusion)] 

**Generating Non-Stationary Textures using Self-Rectification** \
[[Website](https://arxiv.org/abs/2401.02847)] 
[[Code](https://github.com/xiaorongjun000/Self-Rectification)] 

**Accelerating Diffusion Models for Inverse Problems through Shortcut Sampling** \
[[Website](https://arxiv.org/abs/2305.16965)] 
[[Code](https://github.com/gongyeliu/ssd)] 

**Exact Diffusion Inversion via Bi-directional Integration Approximation** \
[[Website](https://arxiv.org/abs/2307.10829)] 
[[Code](https://github.com/guoqiang-zhang-x/BDIA)] 


**Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing** \
[[Website](https://arxiv.org/abs/2403.09468)] 
[[Code](https://github.com/furiosa-ai/eta-inversion)] 

**Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.11105)] 
[[Code](https://github.com/leeruibin/SPDInv)] 

**Effective Real Image Editing with Accelerated Iterative Diffusion Inversion** \
[[ICCV 2023 Oral](https://openaccess.thecvf.com/content/ICCV2023/html/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2309.04907)]

**Score-Based Diffusion Models as Principled Priors for Inverse Imaging** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Feng_Score-Based_Diffusion_Models_as_Principled_Priors_for_Inverse_Imaging_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2304.11751)] 


**BARET : Balanced Attention based Real image Editing driven by Target-text Inversion** \
[[WACV 2024](https://arxiv.org/abs/2312.05482)] 

**Wavelet-Guided Acceleration of Text Inversion in Diffusion-Based Image Editing** \
[[ICASSP 2024](https://arxiv.org/abs/2401.09794)]

**Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.16807)] 

**Direct Inversion: Optimization-Free Text-Driven Real Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2211.07825)] 

**Fixed-point Inversion for Text-to-image diffusion models** \
[[Website](https://arxiv.org/abs/2312.12540)] 


**Prompt Tuning Inversion for Text-Driven Image Editing Using Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.04441)]

**KV Inversion: KV Embeddings Learning for Text-Conditioned Real Image Action Editing** \
[[Website](https://arxiv.org/abs/2309.16608)]

**Tuning-Free Inversion-Enhanced Control for Consistent Image Editing** \
[[Website](https://arxiv.org/abs/2312.14611)]

**LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance** \
[[Website](https://arxiv.org/abs/2307.00522)]

**LEDITS++: Limitless Image Editing using Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2311.16711)]


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
[[Code](https://github.com/TencentARC/SmartEdit?tab=readme-ov-file)] 

**EditVal: Benchmarking Diffusion Based Text-Guided Image Editing Methods** \
[[Website](https://arxiv.org/abs/2310.02426)] 
[[Project](https://deep-ml-research.github.io/editval/#home)] 
[[Code](https://github.com/deep-ml-research/editval_code)] 

**InstructEdit: Improving Automatic Masks for Diffusion-based Image Editing With User Instructions** \
[[Website](https://arxiv.org/abs/2305.18047)] 
[[Project](https://qianwangx.github.io/InstructEdit/)] 
[[Code](https://github.com/QianWangX/InstructEdit)] 

**Text-Driven Image Editing via Learnable Regions** \
[[Website](https://arxiv.org/abs/2311.16432)] 
[[Project](https://yuanze-lin.me/LearnableRegions_page/)] 
[[Code](https://github.com/yuanze-lin/Learnable_Regions)] 

**Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing** \
[[Website](https://arxiv.org/abs/2311.18608)] 
[[Project](https://hyelinnam.github.io/CDS/)] 
[[Code](https://github.com/HyelinNAM/CDS)] 

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


**MAG-Edit: Localized Image Editing in Complex Scenarios via Mask-Based Attention-Adjusted Guidance** \
[[Website](https://arxiv.org/abs/2312.11396)] 
[[Project](https://mag-edit.github.io/)] 
[[Code](https://github.com/HelenMao/MAG-Edit)] 

**MirrorDiffusion: Stabilizing Diffusion Process in Zero-shot Image Translation by Prompts Redescription and Beyond** \
[[Website](https://arxiv.org/abs/2401.03221)] 
[[Project](https://mirrordiffusion.github.io/)] 
[[Code](https://github.com/MirrorDiffusion/MirrorDiffusion)] 

**Motion Guidance: Diffusion-Based Image Editing with Differentiable Motion Estimators** \
[[Website](https://arxiv.org/abs/2401.18085)] 
[[Project](https://dangeng.github.io/motion_guidance/)] 
[[Code](https://github.com/dangeng/motion_guidance/)] 

**UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2210.09477)] 
[[Code](https://github.com/xuduo35/UniTune)]
 
**Learning to Follow Object-Centric Image Editing Instructions Faithfully** \
[[EMNLP 2023](https://arxiv.org/abs/2310.19145)] 
[[Code](https://github.com/tuhinjubcse/faithfuledits_emnlp2023)] 

**Differential Diffusion: Giving Each Pixel Its Strength** \
[[Website](https://arxiv.org/abs/2306.00950)] 
[[Code](https://github.com/exx8/differential-diffusion)] 

**Region-Aware Diffusion for Zero-shot Text-driven Image Editing** \
[[Website](https://arxiv.org/abs/2302.11797v1)] 
[[Code](https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model)] 

**Forgedit: Text Guided Image Editing via Learning and Forgetting** \
[[Website](https://arxiv.org/abs/2309.10556)] 
[[Code](https://github.com/witcherofresearch/Forgedit)] 

**AdapEdit: Spatio-Temporal Guided Adaptive Editing Algorithm for Text-Based Continuity-Sensitive Image Editing** \
[[Website](https://arxiv.org/abs/2312.08019)] 
[[Code](https://github.com/anonymouspony/adap-edit)] 

**Focus on Your Instruction: Fine-grained and Multi-instruction Image Editing by Attention Modulation** \
[[Website](https://arxiv.org/abs/2312.10113)] 
[[Code](https://github.com/guoqincode/focus-on-your-instruction)] 

**Unified Diffusion-Based Rigid and Non-Rigid Editing with Text and Image Guidance** \
[[Website](https://arxiv.org/abs/2401.02126)] 
[[Code](https://github.com/kihensarn/ti-guided-edit)] 

**SpecRef: A Fast Training-free Baseline of Specific Reference-Condition Real Image Editing** \
[[Website](https://arxiv.org/abs/2401.03433)] 
[[Code](https://github.com/jingjiqinggong/specp2p)] 

**Conditional Score Guidance for Text-Driven Image-to-Image Translation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/71103)] 
[[Website](https://arxiv.org/abs/2305.18007)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=cBS5CU96Jq)]  -->

**LIME: Localized Image Editing via Attention Regularization in Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.09256)]
[[Project](https://enis.dev/LIME/)] 

**Watch Your Steps: Local Image and Scene Editing by Text Instructions** \
[[Website](https://arxiv.org/abs/2308.08947)]
[[Project](https://ashmrz.github.io/WatchYourSteps/)] 

**Delta Denoising Score** \
[[Website](https://arxiv.org/abs/2304.07090)] 
[[Project](https://delta-denoising-score.github.io/)] 

**ReGeneration Learning of Diffusion Models with Rich Prompts for Zero-Shot Image Translation** \
[[Website](https://arxiv.org/abs/2305.04651)] 
[[Project](https://yupeilin2388.github.io/publication/ReDiffuser)]

**Emu Edit: Precise Image Editing via Recognition and Generation Tasks** \
[[CVPR 2024](https://arxiv.org/abs/2311.10089)] 
[[Project](https://emu-edit.metademolab.com/)]

**MoEController: Instruction-based Arbitrary Image Manipulation with Mixture-of-Expert Controllers** \
[[Website](https://arxiv.org/abs/2309.04372)]
[[Project](https://oppo-mente-lab.github.io/moe_controller/)]

**Iterative Multi-granular Image Editing using Diffusion Models** \
[[WACV 2024](https://arxiv.org/abs/2309.00613)] 

**Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks** \
[[AAAI 2024](https://arxiv.org/abs/2401.07709)]

**Text-to-image Editing by Image Information Removal** \
[[WACV 2024](https://arxiv.org/abs/2305.17489)]

**Face Aging via Diffusion-based Editing**\
[[BMVC 2023](https://arxiv.org/abs/2309.11321)]

**Custom-Edit: Text-Guided Image Editing with Customized Diffusion Models** \
[[CVPR 2023 AI4CC Workshop](https://arxiv.org/abs/2305.15779)] 

**FISEdit: Accelerating Text-to-image Editing via Cache-enabled Sparse Diffusion Inference** \
[[Website](https://arxiv.org/abs/2305.17423)]

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

**InstructDiffusion: A Generalist Modeling Interface for Vision Tasks** \
[[Website](https://arxiv.org/abs/2309.03895)]

**FEC: Three Finetuning-free Methods to Enhance Consistency for Real Image Editing** \
[[Website](https://arxiv.org/abs/2309.14934)]

**The Blessing of Randomness: SDE Beats ODE in General Diffusion-based Image Editing** \
[[Website](https://arxiv.org/abs/2311.01410)]

**ZONE: Zero-Shot Instruction-Guided Local Editing** \
[[Website](https://arxiv.org/abs/2312.16794)]

**Image Translation as Diffusion Visual Programmers** \
[[Website](https://arxiv.org/abs/2312.16794)]

**Latent Inversion with Timestep-aware Sampling for Training-free Non-rigid Editing** \
[[Website](https://arxiv.org/abs/2402.08601)]

**LoMOE: Localized Multi-Object Editing via Multi-Diffusion** \
[[Website](https://arxiv.org/abs/2403.00437)]

**Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing** \
[[Website](https://arxiv.org/abs/2403.03431)]

**An Item is Worth a Prompt: Versatile Image Editing with Disentangled Control** \
[[Website](https://arxiv.org/abs/2403.04880)]

**DiffChat: Learning to Chat with Text-to-Image Synthesis Models for Interactive Image Creation** \
[[Website](https://arxiv.org/abs/2403.04997)]

**InstructGIE: Towards Generalizable Image Editing** \
[[Website](https://arxiv.org/abs/2403.05018)]

**DreamSampler: Unifying Diffusion Sampling and Score Distillation for Image Manipulation** \
[[Website](https://arxiv.org/abs/2403.11415)]

**LASPA: Latent Spatial Alignment for Fast Training-free Single Image Editing** \
[[Website](https://arxiv.org/abs/2403.12585)]

**Ground-A-Score: Scaling Up the Score Distillation for Multi-Attribute Editing** \
[[Website](https://arxiv.org/abs/2403.13551)]

## Continual Learning

**RGBD2: Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Lei_RGBD2_Generative_Scene_Synthesis_via_Incremental_View_Inpainting_Using_RGBD_CVPR_2023_paper.pdf)] 
[[Website](https://arxiv.org/abs/2212.05993)] 
[[Project](https://jblei.site/proj/rgbd-diffusion)] 
[[Code](https://github.com/Karbo123/RGBD-Diffusion)]


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
[[Website](https://arxiv.org/abs/2304.06027)] 
[[Project](https://jamessealesmith.github.io/continual-diffusion/)]

**Class-Incremental Learning using Diffusion Model for Distillation and Replay** \
[[ICCV 2023 VCL workshop best paper](https://arxiv.org/abs/2306.17560)] 

**Create Your World: Lifelong Text-to-Image Diffusion** \
[[Website](https://arxiv.org/abs/2309.04430)] 


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

**One-dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications** \
[[Website](https://arxiv.org/abs/2312.16145)] 
[[Project](https://lyumengyao.github.io/projects/spm)] 
[[Code](https://github.com/Con6924/SPM)]

**Editing Massive Concepts in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.13807)] 
[[Project](https://silentview.github.io/EMCID/)] 
[[Code](https://github.com/SilentView/EMCID)]

**Towards Safe Self-Distillation of Internet-Scale Text-to-Image Diffusion Models** \
[[ICML 2023 workshop](https://arxiv.org/abs/2307.05977v1)] 
[[Code](https://github.com/nannullna/safe-diffusion)]

**Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.17591)] 
[[Code](https://github.com/SHI-Labs/Forget-Me-Not)]


**Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models** \
[[Website](https://arxiv.org/abs/2305.10120)] 
[[Code](https://github.com/clear-nus/selective-amnesia)]

**MACE: Mass Concept Erasure in Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.06135)] 

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

<!-- ## Semantic Intrinsic

**Exploiting Diffusion Prior for Generalizable Pixel-Level Semantic Prediction** \
[[Website](https://arxiv.org/abs/2311.18832)] 
[[Project](https://shinying.github.io/dmp/)] 
[[Code](https://github.com/shinying/dmp)]

**Generative Models: What do they know? Do they know things? Let's find out!** \
[[Website](https://arxiv.org/abs/2311.17137)] 
[[Project](https://intrinsic-lora.github.io/)]  -->

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
[[Code](https://github.com/TencentARC/Mix-of-Show?tab=readme-ov-file)]

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


**Face2Diffusion for Fast and Editable Face Personalization** \
[[CVPR 2024](https://arxiv.org/abs/2403.05094)] 
[[Project](https://mapooon.github.io/Face2DiffusionPage/)] 
[[Code](https://github.com/mapooon/Face2Diffusion)]

**DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization** \
[[CVPR 2024](https://arxiv.org/abs/2402.09812)] 
[[Project](https://ku-cvlab.github.io/DreamMatcher/)] 
[[Code](https://github.com/KU-CVLAB/DreamMatcher)]

**ConceptBed: Evaluating Concept Learning Abilities of Text-to-Image Diffusion Models** \
[[AAAI 2024](https://arxiv.org/abs/2306.04695)] 
[[Project](https://conceptbed.github.io/)] 
[[Code](https://github.com/conceptbed/evaluations)]

**Harmonizing Visual and Textual Embeddings for Zero-Shot Text-to-Image Customization** \
[[Website](https://arxiv.org/abs/2403.14155)] 
[[Project](https://ldynx.github.io/harmony-zero-t2i/)] 
[[Code](https://github.com/ldynx/harmony-zero-t2i)]

**Material Palette: Extraction of Materials from a Single Image** \
[[Website](https://arxiv.org/abs/2311.17060)] 
[[Project](https://astra-vision.github.io/MaterialPalette/)] 
[[Code](https://github.com/astra-vision/MaterialPalette)]

**StyleDrop: Text-to-Image Generation in Any Style** \
[[Website](https://arxiv.org/abs/2306.00983)] 
[[Project](https://styledrop.github.io/)] 
[[Code](https://github.com/zideliu/StyleDrop-PyTorch)]

**Style Aligned Image Generation via Shared Attention** \
[[Website](https://arxiv.org/abs/2312.02133)] 
[[Project](https://style-aligned-gen.github.io/)] 
[[Code](https://github.com/google/style-aligned/)]

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

<!-- **Anti-DreamBooth: Protecting users from personalized text-to-image synthesis** \
[[ICCV 2023](https://arxiv.org/abs/2303.15433)] 
[[Code](https://github.com/VinAIResearch/Anti-DreamBooth)]
[[Project](https://anti-dreambooth.github.io/)]  -->

**DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Positive-Negative Prompt-Tuning** \
[[Website](https://arxiv.org/abs/2211.11337)] 
[[Project](https://www.sysu-hcp.net/projects/dreamartist/index.html)] 
[[Code](https://github.com/7eu7d7/DreamArtist-stable-diffusion)]

**The Hidden Language of Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.00966)] 
[[Project](https://hila-chefer.github.io/Conceptor/)] 
[[Code](https://github.com/hila-chefer/Conceptor)]

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


**The Chosen One: Consistent Characters in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.10093)] 
[[Project](https://omriavrahami.com/the-chosen-one/)] 
[[Code](https://github.com/ZichengDuan/TheChosenOne)]


**CatVersion: Concatenating Embeddings for Diffusion-Based Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2311.14631)] 
[[Project](https://royzhao926.github.io/CatVersion-page/)] 
[[Code](https://github.com/RoyZhao926/CatVersion)]

**DreamDistribution: Prompt Distribution Learning for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.14216)] 
[[Project](https://briannlongzhao.github.io/DreamDistribution/)] 
[[Code](https://github.com/briannlongzhao/DreamDistribution)]

**CapHuman: Capture Your Moments in Parallel Universes** \
[[Website](https://arxiv.org/abs/2402.00627)] 
[[Project](https://caphuman.github.io/)] 
[[Code](https://github.com/VamosC/CapHumanf)]

**λ-ECLIPSE: Multi-Concept Personalized Text-to-Image Diffusion Models by Leveraging CLIP Latent Space** \
[[Website](https://arxiv.org/abs/2402.05195)] 
[[Project](https://eclipse-t2i.github.io/Lambda-ECLIPSE/)] 
[[Code](https://github.com/eclipse-t2i/lambda-eclipse-inference)]

**Learning Continuous 3D Words for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2402.08654)] 
[[Project](https://ttchengab.github.io/continuous_3d_words/)] 
[[Code](https://github.com/ttchengab/continuous_3d_words_code/)]

**Viewpoint Textual Inversion: Unleashing Novel View Synthesis with Pretrained 2D Diffusion Models** \
[[Website](https://arxiv.org/abs/2309.07986)] 
[[Project](https://jmhb0.github.io/viewneti/)] 
[[Code](https://github.com/jmhb0/view_neti?tab=readme-ov-file)]

**Gen4Gen: Generative Data Pipeline for Generative Multi-Concept Composition** \
[[Website](https://arxiv.org/abs/2402.15504)] 
[[Project](https://danielchyeh.github.io/Gen4Gen/)] 
[[Code](https://github.com/louisYen/Gen4Gen)]

**DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Model** \
[[Website](https://arxiv.org/abs/2402.17412)] 
[[Project](https://diffusekrona.github.io/)] 
[[Code](https://github.com/IBM/DiffuseKronA)]

**OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.10983)] 
[[Project](https://kongzhecn.github.io/omg-project/)] 
[[Code](https://github.com/kongzhecn/OMG/)]

**ProSpect: Expanded Conditioning for the Personalization of Attribute-aware Image Generation** \
[[SIGGRAPH Asia 2023](https://arxiv.org/abs/2305.16225)] 
[[Code](https://github.com/zyxElsa/ProSpect)]

**Multiresolution Textual Inversion** \
[[NeurIPS 2022 workshop](https://arxiv.org/abs/2211.17115)] 
[[Code](https://github.com/giannisdaras/multires_textual_inversion)]


**Compositional Inversion for Stable Diffusion Models** \
[[AAAI 2024](https://arxiv.org/abs/2312.08048)] 
[[Code](https://github.com/zhangxulu1996/Compositional-Inversion)]

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
[[Code](https://github.com/drboog/profusion?tab=readme-ov-file)] 

**Cross Initialization for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2312.15905)] 
[[Code](https://github.com/lyupang/crossinitialization)] 


**High-fidelity Person-centric Subject-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2311.10329)] 
[[Code](https://github.com/codegoat24/face-diffuser)] 

**LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.11627)] 
[[Code](https://github.com/Young98CN/LoRA_Composer)] 

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

**ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs** \
[[Website](https://arxiv.org/abs/2311.13600)] 
[[Project](https://ziplora.github.io/)] 

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
[[Website](https://arxiv.org/abs/2401.09416)] 
[[Project](https://texturedreamer.github.io/)] 

**Direct Consistency Optimization for Compositional Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2402.12004)] 
[[Project](https://dco-t2i.github.io/)] 

**Visual Style Prompting with Swapping Self-Attention** \
[[Website](https://arxiv.org/abs/2402.12974)] 
[[Project](https://curryjung.github.io/VisualStylePrompt/)] 

**Infinite-ID: Identity-preserved Personalization via ID-semantics Decoupling Paradigm** \
[[Website](https://arxiv.org/abs/2403.11781)] 
[[Project](https://infinite-id.github.io/)] 


**DreamStyler: Paint by Style Inversion with Text-to-Image Diffusion Models** \
[[AAAI 2024](https://arxiv.org/abs/2309.06933)] 

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


**Cones 2: Customizable Image Synthesis with Multiple Subjects** \
[[Website](https://arxiv.org/abs/2305.19327v1)] 

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

**StableIdentity: Inserting Anybody into Anywhere at First Sight** \
[[Website](https://arxiv.org/abs/2401.15975)] 

**SeFi-IDE: Semantic-Fidelity Identity Embedding for Personalized Diffusion-Based Generation** \
[[Website](https://arxiv.org/abs/2402.00631)] 

**Visual Concept-driven Image Generation with Text-to-Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2402.11487)] 

**ComFusion: Personalized Subject Generation in Multiple Specific Scenes From Single Image** \
[[Website](https://arxiv.org/abs/2402.11849)] 

**IDAdapter: Learning Mixed Features for Tuning-Free Personalization of Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2403.13535)] 

**MM-Diff: High-Fidelity Image Personalization via Multi-Modal Condition Integration** \
[[Website](https://arxiv.org/abs/2403.15059)] 

<!-- ## Representation Learning

**Denoising Diffusion Autoencoders are Unified Self-supervised Learners**\
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_Denoising_Diffusion_Autoencoders_are_Unified_Self-supervised_Learners_ICCV_2023_paper.pdf)]
[[Code](github.com/FutureXiang/ddae)]

**Diffusion Model as Representation Learner**\
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Diffusion_Model_as_Representation_Learner_ICCV_2023_paper.pdf)]
[[Code](https://github.com/Adamdad/Repfusion)]

**Diffusion Models as Masked Autoencoders**\
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Diffusion_Models_as_Masked_Autoencoders_ICCV_2023_paper.pdf)]
[[Project](https://weichen582.github.io/diffmae.html)] -->



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
[[Code](https://github.com/moayedhajiali/elasticdiffusion-official?tab=readme-ov-file)]
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

**ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment** \
[[Website](https://arxiv.org/abs/2403.05135)] 
[[Project](https://ella-diffusion.github.io/)] 
[[Code](https://github.com/ELLA-Diffusion/ELLA)]


**SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models** \
[[ACM MM 2023 Oral](https://arxiv.org/abs/2305.05189)] 
[[Code](https://github.com/Qrange-group/SUR-adapter)]

**Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models** \
[[ICLR 2024](https://arxiv.org/abs/2402.05375)] 
[[Code](https://github.com/sen-mao/SuppressEOT)]

**Tackling the Singularities at the Endpoints of Time Intervals in Diffusion Models** \
[[CVPR 2024](https://arxiv.org/abs/2403.08381)] 
[[Code](https://github.com/PangzeCheung/SingDiffusion)]

**FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis** \
[[Website](https://arxiv.org/abs/2403.12963)] 
[[Code](https://github.com/LeonHLJ/FouriScale)]

**ORES: Open-vocabulary Responsible Visual Synthesis** \
[[Website](https://arxiv.org/abs/2308.13785)] 
[[Code](https://github.com/kodenii/ores)]

**Fair Diffusion: Instructing Text-to-Image Generation Models on Fairness** \
[[Website](https://arxiv.org/abs/2302.10893)] 
[[Code](https://github.com/ml-research/fair-diffusion)]

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

**Norm-guided latent space exploration for text-to-image generation** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70922)] 
[[Website](https://arxiv.org/abs/2306.08687)] 

**Improving Diffusion-Based Image Synthesis with Context Prediction** \
[[NeurIPS 2023](https://nips.cc/virtual/2023/poster/70058)] 
[[Website](https://arxiv.org/abs/2401.02015)] 

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

**StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control** \
[[CVPR 2024](https://arxiv.org/abs/2403.09055)] 
[[Code](https://github.com/ironjr/StreamMultiDiffusion)]

**Masked-Attention Diffusion Guidance for Spatially Controlling Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2308.06027)] 
[[Code](https://github.com/endo-yuki-t/MAG)]

**MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2402.05408)] 
[[Code](https://github.com/limuloo/MIGC)]

**DivCon: Divide and Conquer for Progressive Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2403.06400)] 
[[Code](https://github.com/DivCon-gen/DivCon)]

**RealCompo: Dynamic Equilibrium between Realism and Compositionality Improves Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.12908)] 
[[Code](https://github.com/YangLing0818/RealCompo)]

**StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control** \
[[Website](https://arxiv.org/abs/2403.09055)] 
[[Code](https://github.com/ironjr/StreamMultiDiffusion?tab=readme-ov-file)]

**InteractDiffusion: Interaction Control in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.05849)] 
[[Project](https://jiuntian.github.io/interactdiffusion/)] 

**Ranni: Taming Text-to-Image Diffusion for Accurate Instruction Following** \
[[Website](https://arxiv.org/abs/2311.17002)] 
[[Project](https://ranni-t2i.github.io/Ranni/)] 

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


**Enhancing Object Coherence in Layout-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2311.10522)] 

**LoCo: Locally Constrained Training-Free Layout-to-Image Synthesis**\
[[Website](https://arxiv.org/abs/2311.12342)] 

**Self-correcting LLM-controlled Diffusion Models** \
[[Website](https://arxiv.org/abs/2311.16090)] 

**Layered Rendering Diffusion Model for Zero-Shot Guided Image Synthesis** \
[[Website](https://arxiv.org/abs/2311.18435)] 

**Joint Generative Modeling of Scene Graphs and Images via Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.01130)] 

**Spatial-Aware Latent Initialization for Controllable Image Generation**
[[Website](https://arxiv.org/abs/2401.16157)] 

**Layout-to-Image Generation with Localized Descriptions using ControlNet with Cross-Attention Control** \
[[Website](https://arxiv.org/abs/2402.13404)] 


## I2I translation

⭐⭐⭐**SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** \
[[ICLR 2022](https://openreview.net/forum?id=aBsCjcPu_tE)] 
[[Website](https://arxiv.org/abs/2108.01073)] 
[[Project](https://sde-image-editing.github.io/)] 
[[Code](https://github.com/ermongroup/SDEdit)] 

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

**DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation** \
[[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html)]
[[Website](https://arxiv.org/abs/2110.02711)]
[[Code](https://github.com/gwang-kim/DiffusionCLIP)]

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

**BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2205.07680)]
[[Code](https://github.com/xuekt98/BBDM)]

**Improving Diffusion-based Image Translation using Asymmetric Gradient Guidance** \
[[Website](https://arxiv.org/abs/2306.04396)]
[[Code](https://github.com/submissionanon18/agg)]

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

**FreeStyle: Free Lunch for Text-guided Style Transfer using Diffusion Models** \
[[Website](https://arxiv.org/abs/2401.15636)]
[[Project](https://freestylefreelunch.github.io/)]

**StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_StyleDiffusion_Controllable_Disentangled_Style_Transfer_via_Diffusion_Models_ICCV_2023_paper.html)]
[[Website](https://arxiv.org/abs/2308.07863)]

**One-Shot Structure-Aware Stylized Image Synthesis** \
[[CVPR 2024](https://arxiv.org/abs/2402.17275)]

**ControlStyle: Text-Driven Stylized Image Generation Using Diffusion Priors** \
[[ACM MM 2023](https://arxiv.org/abs/2311.05463)]

**High-Fidelity Diffusion-based Image Editing** \
[[AAAI 2024](https://arxiv.org/abs/2312.15707)]

**Spectrum Translation for Refinement of Image Generation (STIG) Based on Contrastive Learning and Spectral Filter Profile** \
[[AAAI 2024](https://arxiv.org/abs/2403.05093)]

**E2GAN: Efficient Training of Efficient GANs for Image-to-Image Translation** \
[[Website](https://arxiv.org/abs/2401.06127)]

**UniHDA: Towards Universal Hybrid Domain Adaptation of Image Generators** \
[[Website](https://arxiv.org/abs/2401.12596)]


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


**Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion** \
[[Website](https://arxiv.org/abs/2308.12469)] 
[[Project](https://sites.google.com/view/diffseg/home)] 
[[Code](https://github.com/PotatoTian/DiffSeg)]

**InstaGen: Enhancing Object Detection by Training on Synthetic Dataset** \
[[Website](https://arxiv.org/abs/2402.05937)] 
[[Project](https://fcjian.github.io/InstaGen/)] 
[[Code](https://github.com/fcjian/InstaGen)]

**Delving into the Trajectory Long-tail Distribution for Muti-object Tracking** \
[[Website](https://arxiv.org/abs/2403.04700)] 
[[Code](https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT)]

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

**DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.11681)] 
[[Project](https://weijiawu.github.io/DiffusionMask/)] 

**Diffusion-based Image Translation with Label Guidance for Domain Adaptive Semantic Segmentation** \
[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_Diffusion-based_Image_Translation_with_Label_Guidance_for_Domain_Adaptive_Semantic_ICCV_2023_paper.html
)] 
[[Website](https://arxiv.org/abs/2308.12350)] 

**SDDGR: Stable Diffusion-based Deep Generative Replay for Class Incremental Object Detection** \
[[CVPR 2024](https://arxiv.org/abs/2402.17323)] 

**Generalization by Adaptation: Diffusion-Based Domain Extension for Domain-Generalized Semantic Segmentation** \
[[WACV 2024](https://arxiv.org/abs/2312.01850)] 

**SLiMe: Segment Like Me** \
[[Website](https://arxiv.org/abs/2309.03179)] 


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

## Additional conditions 

⭐⭐⭐**Adding Conditional Control to Text-to-Image Diffusion Models** \
[[ICCV 2023 best paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)] 
[[Website](https://arxiv.org/abs/2302.05543)] 
[[Official Code](https://github.com/lllyasviel/controlnet)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)] 

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

**HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion** \
[[Website](https://arxiv.org/abs/2310.08579)] 
[[Project](https://snap-research.github.io/HyperHuman/)] 
[[Code](https://github.com/snap-research/HyperHuman)]

**Late-Constraint Diffusion Guidance for Controllable Image Synthesis** \
[[Website](https://arxiv.org/abs/2305.11520)] 
[[Project](https://alonzoleeeooo.github.io/LCDG/)] 
[[Code](https://github.com/AlonzoLeeeooo/LCDG)]

**IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2308.06721)] 
[[Project](https://ip-adapter.github.io/)] 
[[Code](https://github.com/tencent-ailab/IP-Adapter)]

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

**T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.08453)] 
[[Code](https://github.com/TencentARC/T2I-Adapter)]

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



**Modulating Pretrained Diffusion Models for Multimodal Image Synthesis** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2302.12764)] 
[[Project](https://mcm-diffusion.github.io/)] 

**SpaText: Spatio-Textual Representation for Controllable Image Generation**\
[[CVPR 2023](https://arxiv.org/abs/2211.14305)] 
[[Project]](https://omriavrahami.com/spatext/) 

**CCM: Adding Conditional Controls to Text-to-Image Consistency Models** \
[[Website](https://arxiv.org/abs/2312.06971)] 
[[Project](https://swiftforce.github.io/CCM/)] 

**FineControlNet: Fine-level Text Control for Image Generation with Spatially Aligned Text Control Injection** \
[[Website](https://arxiv.org/abs/2312.09252)] 
[[Project](https://samsunglabs.github.io/FineControlNet-project-page/)] 

**Control4D: Dynamic Portrait Editing by Learning 4D GAN from 2D Diffusion-based Editor** \
[[Website](https://arxiv.org/abs/2305.20082)] 
[[Project](https://control4darxiv.github.io/)] 

**SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing** \
[[Website](https://arxiv.org/abs/2312.11392)] 
[[Project](https://scedit.github.io/)] 

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

**Inst-Inpaint: Instructing to Remove Objects with Diffusion Models** \
[[Website](https://arxiv.org/abs/2304.03246)] 
[[Project](http://instinpaint.abyildirim.com/)] 
[[Code](https://github.com/abyildirim/inst-inpaint)]
[[Demo](https://huggingface.co/spaces/abyildirim/inst-inpaint)]

**AnyDoor: Zero-shot Object-level Image Customization** \
[[Website](https://arxiv.org/abs/2307.09481)] 
[[Project](https://damo-vilab.github.io/AnyDoor-Page/)]
[[Code](https://github.com/damo-vilab/AnyDoor)] 

**A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting** \
[[Website](https://arxiv.org/abs/2312.03594)] 
[[Project](https://powerpaint.github.io/)]
[[Code](https://github.com/open-mmlab/mmagic/tree/main/projects/powerpaint)] 

**Towards Language-Driven Video Inpainting via Multimodal Large Language Models** \
[[Website](https://arxiv.org/abs/2401.10226)]
[[Project](https://jianzongwu.github.io/projects/rovi/)]
[[Code](https://github.com/jianzongwu/Language-Driven-Video-Inpainting)]

**360-Degree Panorama Generation from Few Unregistered NFoV Images** \
[[ACM MM 2023](https://arxiv.org/abs/2308.14686)] 
[[Code](https://github.com/shanemankiw/Panodiff)] 

**Delving Globally into Texture and Structure for Image Inpainting**\
[[ACM MM 2022](https://arxiv.org/abs/2209.08217)] 
[[Code](https://github.com/htyjers/DGTS-Inpainting)]

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

**IMPRINT: Generative Object Compositing by Learning Identity-Preserving Representation** \
[[CVPR 2024](https://arxiv.org/abs/2403.10701)] 
[[Project](https://song630.github.io/IMPRINT-Project-Page/)]

**SmartMask: Context Aware High-Fidelity Mask Generation for Fine-grained Object Insertion and Layout Control** \
[[Website](https://arxiv.org/abs/2312.05039)] 
[[Project](https://smartmask-gen.github.io/)]

**Towards Stable and Faithful Inpainting** \
[[Website](https://arxiv.org/abs/2312.04831)] 
[[Project](https://yikai-wang.github.io/asuka/)]

**Magic Fixup: Streamlining Photo Editing by Watching Dynamic Videos** \
[[Website](https://arxiv.org/abs/2403.13044)] 
[[Project](https://magic-fixup.github.io/)]

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

**Diffusion-based Document Layout Generation** \
[[Website](https://arxiv.org/abs/2303.10787)]

**Dolfin: Diffusion Layout Transformers without Autoencoder** \
[[Website](https://arxiv.org/abs/2310.16305)]



## Text Generation

**TextDiffuser: Diffusion Models as Text Painters** \
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70636)]
[[Website](https://arxiv.org/abs/2305.10855)]
[[Project](https://jingyechen.github.io/textdiffuser/)] 
[[Code](https://github.com/microsoft/unilm/tree/master/textdiffuser)] 
<!-- [[NeurIPS 2023](https://openreview.net/forum?id=ke3RgcDmfO)] -->

**TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering** \
[[Website](https://arxiv.org/abs/2311.16465)]
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

**UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models** \
[[Website](https://arxiv.org/abs/2312.04884)]
[[Project](https://udifftext.github.io/)] 
[[Code](https://github.com/ZYM-PKU/UDiffText?tab=readme-ov-file)] 

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

**AnyText: Multilingual Visual Text Generation And Editing** \
[[Website](https://arxiv.org/abs/2311.03054)]
[[Code](https://github.com/tyxsspa/AnyText)] 

**AmbiGen: Generating Ambigrams from Pre-trained Diffusion Model** \
[[Website](https://arxiv.org/abs/2312.02967)]
[[Project](https://raymond-yeh.com/AmbiGen/)] 

**UniVG: Towards UNIfied-modal Video Generation** \
[[Website](https://arxiv.org/abs/2401.09084)]
[[Project](https://univg-baidu.github.io/)] 

**DECDM: Document Enhancement using Cycle-Consistent Diffusion Models** \
[[WACV 2024](https://arxiv.org/abs/2311.09625)]

**VecFusion: Vector Font Generation with Diffusion** \
[[Website](https://arxiv.org/abs/2312.10540)]

**Typographic Text Generation with Off-the-Shelf Diffusion Model** \
[[Website](https://arxiv.org/abs/2402.14314)]

**Font Style Interpolation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.14311)]

**Refining Text-to-Image Generation: Towards Accurate Training-Free Glyph-Enhanced Image Generation** \
[[Website](https://arxiv.org/abs/2403.16422)]

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

**Exploiting Diffusion Prior for Real-World Image Super-Resolution** \
[[Website](https://arxiv.org/abs/2305.07015)] 
[[Project](https://iceclear.github.io/projects/stablesr/)] 
[[Code](https://github.com/IceClear/StableSR)] 

**SinSR: Diffusion-Based Image Super-Resolution in a Single Step** \
[[CVPR 2024](https://arxiv.org/abs/2311.14760)] 
[[Code](https://github.com/wyf0912/SinSR)] 

**Iterative Token Evaluation and Refinement for Real-World Super-Resolution** \
[[AAAI 2024](https://arxiv.org/abs/2312.05616)] 
[[Code](https://github.com/chaofengc/ITER)] 

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


<!-- ## X2I X2X

**GlueGen: Plug and Play Multi-modal Encoders for X-to-image Generation** \
[[ICCV 2023](https://arxiv.org/abs/2303.10056)] 
[[Code](https://github.com/salesforce/GlueGen)]

**CoDi: Any-to-Any Generation via Composable Diffusion** \
[[Website](https://arxiv.org/abs/2305.11846)]
[[NeurIPS 2023](https://neurips.cc/virtual/2023/poster/72964)]
[[Code](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)] 
[[Project](https://codi-gen.github.io/)]  -->



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

**Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets** \
[[Website](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets)]
[[Project](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)]
[[Code](https://github.com/Stability-AI/generative-models)]

**MagicAvatar: Multimodal Avatar Generation and Animation** \
[[Website](https://arxiv.org/abs/2308.14748)]
[[Project](https://magic-avatar.github.io/)] 
[[Code](https://github.com/magic-research/magic-avatar)]

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
[[Code](https://github.com/XuweiyiChen/UniCtrl?tab=readme-ov-file)] 

**VideoElevator: Elevating Video Generation Quality with Versatile Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2403.05438)]
[[Project](https://videoelevator.github.io/)]
[[Code](https://github.com/YBYBZhang/VideoElevator)] 

**Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing** \
[[ICLR 2024](https://arxiv.org/abs/2402.16627)]
[[Code](https://github.com/YangLing0818/ContextDiff)]

**SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces** \
[[ICLR 2024](https://arxiv.org/abs/2403.07711)]
[[Code](https://github.com/shim0114/SSM-Meets-Video-Diffusion-Models)]

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


**SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction** \
[[Website](https://arxiv.org/abs/2310.20700)]


**Dual-Stream Diffusion Net for Text-to-Video Generation** \
[[Website](https://arxiv.org/abs/2308.08316)]


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



**Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer**\
[[Website](https://arxiv.org/abs/2305.05464)]
[[Code](https://github.com/haha-lisa/style-a-video)] 

**Flow-Guided Diffusion for Video Inpainting** \
[[Website](https://arxiv.org/abs/2311.15368)]
[[Code](https://github.com/nevsnev/fgdvi)] 

**Shape-Aware Text-Driven Layered Video Editing** \
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Shape-Aware_Text-Driven_Layered_Video_Editing_CVPR_2023_paper.html
)]
[[Website](https://arxiv.org/abs/2301.13173)]
[[Project](https://text-video-edit.github.io/#)] 

**DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing** \
[[Website](https://arxiv.org/abs/2310.10624)]
[[Project](https://showlab.github.io/DynVideo-E/)]


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


**MeDM: Mediating Image Diffusion Models for Video-to-Video Translation with Temporal Correspondence Guidance** \
[[Website](https://arxiv.org/abs/2308.10079)]
[[Project](https://medm2023.github.io)]

**Customize-A-Video: One-Shot Motion Customization of Text-to-Video Diffusion Models** \
[[Website](https://arxiv.org/abs/2402.14780)]
[[Project](https://anonymous-314.github.io/)]

**DreamMotion: Space-Time Self-Similarity Score Distillation for Zero-Shot Video Editing** \
[[Website](https://arxiv.org/abs/2403.12002)]
[[Project](https://hyeonho99.github.io/dreammotion/)]

**Edit Temporal-Consistent Videos with Image Diffusion Model** \
[[Website](https://arxiv.org/abs/2308.09091)]


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
