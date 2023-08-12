<!-- The superlink doesn't support uppercases -->

- [Video Editing](#video-editing)
- [Diffusion Inversion](#diffusion-models-inversion)
- [Text-Guided Image Editing](#text-guided-image-editing)
- [Continual Learning](#continual-learning)
- [Remove Concept](#remove-concept)
- [New Concept Learning](#new-concept-learning)
- [Additional Conditions](#additional-conditions)
- [T2I Diffusion Model augmentation](#t2i-diffusion-model-augmentation)
- [Segmentation & Detection & Tracking](#segmentation-detection-tracking)
- [Few-Shot](#few-shot)
- [Drag Image Edit](#drag-image-edit)
- [SD Inpainting](#sd-inpaint)
- [Document Layout Generation](#document-layout-generation)
- [Super Resolution](#super-resolution)



## Video Editing 

⭐**FateZero: Fusing Attentions for Zero-shot Text-based Video Editing** \
[[ICCV 2023]](https://arxiv.org/abs/2303.09535) 
[[Code]](https://github.com/ChenyangQiQi/FateZero) 
[[Project](https://fate-zero-edit.github.io/)] 

⭐**Video-P2P: Video Editing with Cross-attention Control** \
[[Website]](https://arxiv.org/abs/2303.04761)
[[Code]](https://github.com/ShaoTengLiu/Video-P2P) 

⭐**Vid2Vid-zero: Zero-Shot Video Editing Using Off-the-Shelf Image Diffusion Models** \
[[Website]](https://arxiv.org/abs/2303.17599) 
[[Code]](https://github.com/baaivision/vid2vid-zero) 


**TokenFlow: Consistent Diffusion Features for Consistent Video Editing** \
[[Website]](https://arxiv.org/abs/2307.10373)
[[Code]](https://github.com/omerbt/TokenFlow)
[[Project](https://diffusion-tokenflow.github.io/)] 

**ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing** \
[[Website]](https://arxiv.org/abs/2305.17098)
[[Code]](https://github.com/thu-ml/controlvideo)
[[Project](https://ml.cs.tsinghua.edu.cn/controlvideo/)] 

**Make-A-Protagonist: Generic Video Editing with An Ensemble of Experts** \
[[Website]](https://arxiv.org/abs/2305.08850)
[[Code]](https://github.com/Make-A-Protagonist/Make-A-Protagonist)
[[Project](https://make-a-protagonist.github.io/)] 

**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding** \
[[CVPR 2023]](https://arxiv.org/abs/2212.02802) 
[[Code]](https://github.com/man805/Diffusion-Video-Autoencoders) 
[[Project](https://diff-video-ae.github.io/)] 

**VidEdit: Zero-Shot and Spatially Aware Text-Driven Video Editing** \
[[Website]](https://arxiv.org/abs//2306.08707)
[[Project](https://videdit.github.io/)] 

**Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation** \
[[Website]](https://arxiv.org/abs/2306.07954)
[[Project](https://anonymous-31415926.github.io/)] 

**Shape-Aware Text-Driven Layered Video Editing** \
[[CVPR 2023]](https://arxiv.org/abs/2301.13173) 
[[Project](https://text-video-edit.github.io/#)] 

**Pix2video: Video Editing Using Image Diffusion** \
[[Website]](https://arxiv.org/abs/2303.12688)

**Dreamix: Video Diffusion Models Are General Video Editors** \
[[Website]](https://arxiv.org/abs/2302.01329)

**Towards Consistent Video Editing with Text-to-Image Diffusion Models** \
[[Website]](https://arxiv.org/abs/2305.17431) 



## Diffusion Models Inversion

⭐⭐⭐**Null-text Inversion for Editing Real Images using Guided Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2211.09794)] 
[[Project](https://null-text-inversion.github.io/)] 
[[Code](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)]

⭐⭐**Improving Negative-Prompt Inversion via Proximal Guidance** \
[[Website](https://arxiv.org/abs/2306.05414)] 
[[Code](https://github.com/phymhan/prompt-to-prompt)] 

⭐**Accelerating Diffusion Models for Inverse Problems through Shortcut Sampling** \
[[Website](https://arxiv.org/abs/2305.16965)] 
[[Code](https://github.com/gongyeliu/ssd)] 

⭐**Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.16807)] 

**EDICT: Exact Diffusion Inversion via Coupled Transformations** \
[[Website](https://arxiv.org/abs/2211.12446)] 
[[Code](https://github.com/salesforce/edict)] 

**Inversion-Based Creativity Transfer with Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2211.13203)] 
[[Code](https://github.com/zyxElsa/InST)] 

**Direct Inversion: Optimization-Free Text-Driven Real Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2211.07825)] 


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
[[Demo](https://huggingface.co/spaces/pix2pix-zero-library/pix2pix-zero-demo)] 
[[Replicate Demo](https://replicate.com/cjwbw/pix2pix-zero)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/pix2pix_zero)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py)]

⭐⭐⭐**Null-text Inversion for Editing Real Images using Guided Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2211.09794)] 
[[Project](https://null-text-inversion.github.io/)] 
[[Code](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)]

⭐⭐**Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation** \
[[CVPR 2023](https://arxiv.org/abs/2211.12572)] 
[[Project](https://pnp-diffusion.github.io/sm/index.html)] 
[[Code](https://github.com/MichalGeyer/plug-and-play)]
[[Dataset](https://www.dropbox.com/sh/8giw0uhfekft47h/AAAF1frwakVsQocKczZZSX6La?dl=0)]
[[Replicate Demo](https://replicate.com/daanelson/plug_and_play_image_translation)] 
[[Demo](https://huggingface.co/spaces/hysts/PnP-diffusion-features)] 


⭐**Imagic: Text-Based Real Image Editing with Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2210.09276)] 
[[Project](https://imagic-editing.github.io/)] 
[[Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community#imagic-stable-diffusion)]

⭐**InstructPix2Pix: Learning to Follow Image Editing Instructions** \
[[CVPR 2023 (Highlight)](https://arxiv.org/abs/2211.09800)] 
[[Project](https://www.timothybrooks.com/instruct-pix2pix/)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/pix2pix)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py)] 
[[Official Code](https://github.com/timothybrooks/instruct-pix2pix)]
[[Dataset](http://instruct-pix2pix.eecs.berkeley.edu/)]

⭐**SINE: SINgle Image Editing with Text-to-Image Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2212.04489)] 
[[Project](https://zhang-zx.github.io/SINE/)] 
[[Code](https://github.com/zhang-zx/SINE)] 

⭐**Inpaint Anything: Segment Anything Meets Image Inpainting** \
[[Website](https://arxiv.org/abs/2304.06790)] 
[[Code 1](https://github.com/geekyutao/Inpaint-Anything)] 
[[Code 2](https://github.com/sail-sg/EditAnything)] 

⭐**SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** \
[[Website](https://arxiv.org/abs/2108.01073)] 
[[ICLR 2022](https://openreview.net/forum?id=aBsCjcPu_tE)] 
[[Project](https://sde-image-editing.github.io/)] 
[[Code](https://github.com/ermongroup/SDEdit)] 

⭐**DiffEdit: Diffusion-based semantic image editing with mask guidance** \
[[ICLR 2023](https://openreview.net/forum?id=3lge0p5o-M-)] 
[[Website](https://arxiv.org/abs/2210.11427)] 
[[Unofficial Code](https://paperswithcode.com/paper/diffedit-diffusion-based-semantic-image)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/diffedit)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_diffedit.py)] 

**MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing** \
[[ICCV 2023](https://arxiv.org/abs/2304.08465)] 
[[Project](https://ljzycmd.github.io/projects/MasaCtrl/)] 
[[Code](https://github.com/TencentARC/MasaCtrl)] 
[[Demo](https://huggingface.co/spaces/TencentARC/MasaCtrl)]

**An Edit Friendly DDPM Noise Space: Inversion and Manipulations** \
[[Website](https://arxiv.org/abs/2304.06140)] 
[[Code](https://github.com/inbarhub/DDPM_inversion)]
[[Project](https://inbarhub.github.io/DDPM_inversion/)] 
[[Demo](https://huggingface.co/spaces/LinoyTsaban/edit_friendly_ddpm_inversion)]

**InstructEdit: Improving Automatic Masks for Diffusion-based Image Editing With User Instructions** \
[[Website](https://arxiv.org/abs/2305.18047)] 
[[Code](https://github.com/QianWangX/InstructEdit)] 
[[Project](https://qianwangx.github.io/InstructEdit/)] 


**StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing** \
[[Website](https://arxiv.org/abs/2303.15649)] 
[[Code](https://github.com/sen-mao/StyleDiffusion)] 


**PAIR-Diffusion: Object-Level Image Editing with Structure-and-Appearance Paired Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.17546)] 
[[Code](https://github.com/Picsart-AI-Research/PAIR-Diffusion)] 
[[Demo](https://huggingface.co/spaces/PAIR/PAIR-Diffusion)]


**Localizing Object-level Shape Variations with Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.11306)] 
[[Project](https://orpatashnik.github.io/local-prompt-mixing/)] 
[[Code](https://github.com/orpatashnik/local-prompt-mixing)]

**ReGeneration Learning of Diffusion Models with Rich Prompts for Zero-Shot Image Translation** \
[[Website](https://arxiv.org/abs/2305.04651)] 
[[Project](https://yupeilin2388.github.io/publication/ReDiffuser)]


**Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance** \
[[Website](https://arxiv.org/abs/2210.05559)] 
[[Code1](https://github.com/chenwu98/unified-generative-zoo)] 
[[Code2](https://github.com/chenwu98/cycle-diffusion)] 
[[Diffusers Code](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cycle_diffusion)] 

**Delta Denoising Score** \
[[Website](https://arxiv.org/abs/2304.07090)] 
[[Project](https://delta-denoising-score.github.io/)] 

**Visual Instruction Inversion: Image Editing via Visual Prompting** \
[[Website](https://arxiv.org/abs/2307.14331)] 
[[Project](https://thaoshibe.github.io/visii/)] 

**MDP: A Generalized Framework for Text-Guided Image Editing by Manipulating the Diffusion Path** \
[[Website](https://arxiv.org/abs/2303.16765)] 
[[Code](https://github.com/QianWangX/MDP-Diffusion)] 

**Differential Diffusion: Giving Each Pixel Its Strength** \
[[Website](https://arxiv.org/abs/2306.00950)] 
[[Code](https://github.com/exx8/differential-diffusion)] 

**Conditional Score Guidance for Text-Driven Image-to-Image Translation** \
[[Website](https://arxiv.org/abs/2305.18007)] 

**Custom-Edit: Text-Guided Image Editing with Customized Diffusion Models** \
[[CVPR 2023 AI4CC Workshop](https://arxiv.org/abs/2305.15779)] 

**HIVE: Harnessing Human Feedback for Instructional Visual Editing** \
[[Website](https://arxiv.org/abs/2303.09618)] 
[[Code](https://github.com/salesforce/HIVE)] 

**Region-Aware Diffusion for Zero-shot Text-driven Image Editing** \
[[Website](https://arxiv.org/abs/2302.11797v1)] 
[[Code](https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model)] 

**UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2210.09477)] 
[[Code](https://github.com/xuduo35/UniTune)] 

**FISEdit: Accelerating Text-to-image Editing via Cache-enabled Sparse Diffusion Inference** \
[[Website](https://arxiv.org/abs/2305.17423)]

**LayerDiffusion: Layered Controlled Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.18676)]

**Text-to-image Editing by Image Information Removal** \
[[Website](https://arxiv.org/abs/2305.17489)]

**iEdit: Localised Text-guided Image Editing with Weak Supervision** \
[[Website](https://arxiv.org/abs/2305.05947)]

**Prompt Tuning Inversion for Text-Driven Image Editing Using Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.04441)]

**User-friendly Image Editing with Minimal Text Input: Leveraging Captioning and Injection Techniques** \
[[Website](https://arxiv.org/abs/2306.02717)]

**PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing** \
[[Website](https://arxiv.org/abs/2306.16894)]

**LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance** \
[[Website](https://arxiv.org/abs/2307.00522)]

**PRedItOR: Text Guided Image Editing with Diffusion Prior** \
[[Website](https://arxiv.org/abs/2302.07979v2)]

# Continual Learning

⭐**Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA** \
[[Website](https://arxiv.org/abs/2304.06027)] 
[[Project](https://jamessealesmith.github.io/continual-diffusion/)]

**Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models** \
[[Website](https://arxiv.org/abs/2305.10120)] 
[[Code](https://github.com/clear-nus/selective-amnesia)]

**RGBD2: Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models** \
[[Website](https://arxiv.org/abs/2212.05993)] 
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Lei_RGBD2_Generative_Scene_Synthesis_via_Incremental_View_Inpainting_Using_RGBD_CVPR_2023_paper.pdf)] 
[[Project](https://jblei.site/proj/rgbd-diffusion)] 
[[Code](https://github.com/Karbo123/RGBD-Diffusion)]

**Exploring Continual Learning of Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.15342)] 

**DiracDiffusion: Denoising and Incremental Reconstruction with Assured Data-Consistency** \
[[Website](https://arxiv.org/abs/2303.14353)] 

**Class-Incremental Learning using Diffusion Model for Distillation and Replay** \
[[Website](https://arxiv.org/abs/2306.17560)] 

**DiffusePast: Diffusion-based Generative Replay for Class Incremental Semantic Segmentation** \
[[Website](https://arxiv.org/abs/2308.01127)] 

# Remove Concept

**Ablating Concepts in Text-to-Image Diffusion Models** \
[[ICCV 2023](https://arxiv.org/abs/2303.13516)] 
[[Code](https://github.com/nupurkmr9/concept-ablation)]
[[Project](https://www.cs.cmu.edu/~concept-ablation/)] 

**Erasing Concepts from Diffusion Models** \
[[ICCV 2023](https://arxiv.org/abs/2303.07345)] 
[[Code](https://github.com/rohitgandikota/erasing)]
[[Project](https://erasing.baulab.info/)] 

**Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.17591)] 
[[Code](https://github.com/SHI-Labs/Forget-Me-Not)]

**Inst-Inpaint: Instructing to Remove Objects with Diffusion Models** \
[[Website](https://arxiv.org/abs/2304.03246)] 
[[Code](https://github.com/abyildirim/inst-inpaint)]
[[Project](http://instinpaint.abyildirim.com/)] 
[[Demo](https://huggingface.co/spaces/abyildirim/inst-inpaint)]

**Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models** \
[[Website](https://arxiv.org/abs/2305.10120)] 
[[Code](https://github.com/clear-nus/selective-amnesia)]

**Towards Safe Self-Distillation of Internet-Scale Text-to-Image Diffusion Models** \
[[ICML 2023 workshop](https://arxiv.org/abs/2307.05977v1)] 
[[Code](https://github.com/nannullna/safe-diffusion)]



# New Concept Learning
⭐⭐⭐**An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion** \
[[Website](https://arxiv.org/abs/2208.01618)] 
[[ICLR 2023 top-25%](https://openreview.net/forum?id=NAQvF08TcyG)] 
[[Code](https://github.com/rinongal/textual_inversion)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/training/text_inversion)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion)] 

⭐⭐⭐**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation** \
[[CVPR 2023](https://arxiv.org/abs/2208.12242)] 
[[Official Dataset](https://github.com/google/dreambooth)]
[[Unofficial Code](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)]
[[Project](https://dreambooth.github.io/)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/training/dreambooth)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)] 

⭐⭐**Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion** \
[[CVPR 2023](https://arxiv.org/abs/2212.04488)] 
[[Code](https://github.com/adobe-research/custom-diffusion)]
[[Project](https://www.cs.cmu.edu/~custom-diffusion/)] 
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/tree/main/examples/custom_diffusion)] 

⭐⭐**ReVersion: Diffusion-Based Relation Inversion from Images** \
[[Website](https://arxiv.org/abs/2303.13495)] 
[[Code](https://github.com/ziqihuangg/ReVersion)]
[[Project](https://ziqihuangg.github.io/projects/reversion.html)]

⭐**FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention** \
[[Website](https://arxiv.org/abs/2305.10431)] 
[[Code](https://github.com/mit-han-lab/fastcomposer)]
[[Demo](https://2acfe10ec96df6f2b0.gradio.live/)]
[[Project](https://fastcomposer.mit.edu/)] 

⭐**Enhancing Detail Preservation for Customized Text-to-Image Generation: A Regularization-Free Approach** \
[[Website](https://arxiv.org/abs/2305.13579)] 
[[Code](https://github.com/drboog/profusion)]


⭐**SINE: SINgle Image Editing with Text-to-Image Diffusion Models** \
[[CVPR 2023](https://arxiv.org/abs/2212.04489)] 
[[Project](https://zhang-zx.github.io/SINE/)] 
[[Code](https://github.com/zhang-zx/SINE)] 

⭐**SVDiff: Compact Parameter Space for Diffusion Fine-Tuning** \
[[Website](https://arxiv.org/abs/2303.11305)] 
[[Code](https://github.com/mkshing/svdiff-pytorch)]

⭐**A Neural Space-Time Representation for Text-to-Image Personalization** \
[[Website](https://arxiv.org/abs/2305.15391)] 
[[Code](https://github.com/NeuralTextualInversion/NeTI)]
[[Project](https://neuraltextualinversion.github.io/NeTI/)] 

⭐**Break-A-Scene: Extracting Multiple Concepts from a Single Image** \
[[Website](https://arxiv.org/abs/2305.16311)] 
[[Project](https://omriavrahami.com/break-a-scene/)]

⭐**Concept Decomposition for Visual Exploration and Inspiration** \
[[Website](https://arxiv.org/abs/2305.18203)] 
[[Project](https://inspirationtree.github.io/inspirationtree/)] 

⭐**AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning** \
[[Website](https://arxiv.org/abs/2307.04725)] 
[[Project](https://animatediff.github.io/)] 
[[Code](https://github.com/guoyww/animatediff/)]

**Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning**\
[[Website](https://arxiv.org/abs/2307.11410)] 
[[Project](https://oppo-mente-lab.github.io/subject_diffusion/)] 
[[Code](https://github.com/OPPO-Mente-Lab/Subject-Diffusion)]

**HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models** \
[[Website](https://arxiv.org/abs/2307.06949)] 
[[Project](https://hyperdreambooth.github.io/)] 

**Highly Personalized Text Embedding for Image Manipulation by Stable Diffusion** \
[[Website](https://arxiv.org/abs/2303.08767)] 
[[Code](https://github.com/HiPer0/HiPer)]
[[Project](https://hiper0.github.io/)] 

**BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing** \
[[Website](https://arxiv.org/abs/2305.14720)] 
[[Code](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion)]
[[Project](https://dxli94.github.io/BLIP-Diffusion-website/)]

**ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation** \
[[ICCV 2023](https://arxiv.org/abs/2302.13848)] 
[[Code](https://github.com/csyxwei/ELITE)]
[[Demo](https://huggingface.co/spaces/ELITE-library/ELITE)]

**Cones: Concept Neurons in Diffusion Models for Customized Generation** \
[[ICML 2023 oral](https://arxiv.org/abs/2303.05125)] 
[[Code](https://github.com/Johanan528/Cones)]


**Anti-DreamBooth: Protecting users from personalized text-to-image synthesis** \
[[ICCV 2023](https://arxiv.org/abs/2303.15433)] 
[[Code](https://github.com/VinAIResearch/Anti-DreamBooth)]
[[Project](https://anti-dreambooth.github.io/)] 


**DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Positive-Negative Prompt-Tuning** \
[[Website](https://arxiv.org/abs/2211.11337)] 
[[Code](https://github.com/7eu7d7/DreamArtist-stable-diffusion)]
[[Project](https://www.sysu-hcp.net/projects/dreamartist/index.html)] 



**Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2302.12228)] 
[[Code](https://github.com/mkshing/e4t-diffusion)]
[[Project](https://tuning-encoder.github.io/)] 

**The Hidden Language of Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.00966)] 
[[Code](https://github.com/hila-chefer/Conceptor)]
[[Project](https://hila-chefer.github.io/Conceptor/)] 

**Inserting Anybody in Diffusion Models via Celeb Basis** \
[[Website](https://arxiv.org/abs/2306.00926)] 
[[Code](https://github.com/ygtxr1997/celebbasis)]
[[Project](https://celeb-basis.github.io/)] 

**ConceptBed: Evaluating Concept Learning Abilities of Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.04695)] 
[[Code](https://github.com/conceptbed/evaluations)]
[[Project](https://conceptbed.github.io/)] 

**Controlling Text-to-Image Diffusion by Orthogonal Finetuning** \
[[Website](https://arxiv.org/abs/2306.07280)] 
[[Code](https://github.com/Zeju1997/oft)]
[[Project](https://oft.wyliu.com/)] 

**ProSpect: Expanded Conditioning for the Personalization of Attribute-aware Image Generation** \
[[Website](https://arxiv.org/abs/2305.16225)] 
[[Code](https://github.com/zyxElsa/ProSpect)]

**Diffusion in Diffusion: Cyclic One-Way Diffusion for Text-Vision-Conditioned Generation** \
[[Website](https://arxiv.org/abs/2306.08247)] 
[[Project](https://bigaandsmallq.github.io/COW/)] 

**ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2306.00971)] 
[[Code](https://github.com/haoosz/vico)]

**Subject-driven Text-to-Image Generation via Apprenticeship Learning** \
[[Website](https://arxiv.org/abs/2304.00186)] 
[[Project](https://open-vision-language.github.io/suti/)] 

**DisenBooth: Disentangled Parameter-Efficient Tuning for Subject-Driven Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2305.03374)] 

**Controllable Textual Inversion for Personalized Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2304.05265)] 
[[Code](https://github.com/jnzju/COTI)]

**Is This Loss Informative? Speeding Up Textual Inversion with Deterministic Objective Evaluation** \
[[Website](https://arxiv.org/abs/2302.04841)] 
[[Code](https://github.com/yandex-research/DVAR)]

**Multiresolution Textual Inversion** \
[[Neurips 2022 workshop](https://arxiv.org/abs/2211.17115)] 
[[Code](https://github.com/giannisdaras/multires_textual_inversion)]

**Key-Locked Rank One Editing for Text-to-Image Personalization** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2305.01644)] 
[[Project](https://research.nvidia.com/labs/par/Perfusion/)] 

**Towards Prompt-robust Face Privacy Protection via Adversarial Decoupling Augmentation Framework** \
[[Website](https://arxiv.org/abs/2305.03980)] 

**A Closer Look at Parameter-Efficient Tuning in Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.18181)] 
[[Code](https://github.com/Xiang-cd/unet-finetune)]

**Taming Encoder for Zero Fine-tuning Image Customization with Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2304.02642)] 

**Domain-Agnostic Tuning-Encoder for Fast Personalization of Text-To-Image Models** \
[[Website](https://arxiv.org/abs/2307.06925)] 
[[Project](https://datencoder.github.io/)] 

**$P+$: Extended Textual Conditioning in Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2303.09522)] 
[[Project](https://prompt-plus.github.io/)] 

**Photoswap: Personalized Subject Swapping in Images** \
[[Website](https://arxiv.org/abs/2305.18286)] 
[[Project](https://photoswap.github.io/)] 



**Gradient-Free Textual Inversion** \
[[Website](https://arxiv.org/abs/2304.05818)] 

**Identity Encoder for Personalized Diffusion** \
[[Website](https://arxiv.org/abs/2304.07429)] 

**InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning** \
[[Website](https://arxiv.org/abs/2304.03411)] 
[[Project](https://jshi31.github.io/InstantBooth/)] 

**Cross-domain Compositing with Pretrained Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.10167)] 
[[Code](https://github.com/cross-domain-compositing/cross-domain-compositing)] 


**Unified Multi-Modal Latent Diffusion for Joint Subject and Text Conditional Image Generation** \
[[Website](https://arxiv.org/abs/2303.09319)] 

**ELODIN: Naming Concepts in Embedding Spaces** \
[[Website](https://arxiv.org/abs/2303.04001)] 

**Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.18292)] 

**Cones 2: Customizable Image Synthesis with Multiple Subjects** \
[[Website](https://arxiv.org/abs/2305.19327v1)] 

**Generate Anything Anywhere in Any Scene** \
[[Website](https://arxiv.org/abs/2306.17154)] 

**Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model** \
[[Website](https://arxiv.org/abs/2306.07596)] 

**Face0: Instantaneously Conditioning a Text-to-Image Model on a Face** \
[[Website](https://arxiv.org/abs/2306.06638v1)] 




# Additional conditions 

⭐⭐⭐**Adding Conditional Control to Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.05543)] 
[[Official Code](https://github.com/lllyasviel/controlnet)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py)] 

⭐**GLIGEN: Open-Set Grounded Text-to-Image Generation** \
[[CVPR 2023](https://arxiv.org/abs/2301.07093)] 
[[Code](https://github.com/gligen/GLIGEN)]
[[Demo](https://huggingface.co/spaces/gligen/demo)]

⭐**T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.08453)] 
[[Code](https://github.com/TencentARC/T2I-Adapter)]

⭐**Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.16322)] 
[[Code](https://github.com/ShihaoZhaoZSH/Uni-ControlNet)]
[[Project](https://shihaozhaozsh.github.io/unicontrolnet/)] 

**Composer: Creative and controllable image synthesis with composable conditions** \
[[Website](https://arxiv.org/abs/2302.09778)] 
[[Code](https://github.com/damo-vilab/composer)]
[[Project](https://damo-vilab.github.io/composer-page/)] 

**DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.15194)] 
[[Code](https://github.com/sungnyun/diffblender)]
[[Project](https://sungnyun.github.io/diffblender/)] 

**Cocktail: Mixing Multi-Modality Controls for Text-Conditional Image Generation** \
[[Website](https://arxiv.org/abs/2303.09833)] 
[[Code](https://github.com/mhh0318/Cocktail)]
[[Project](https://mhh0318.github.io/cocktail/)] 

**UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild** \
[[Website](https://arxiv.org/abs/2305.11147)] 
[[Code](https://github.com/salesforce/UniControl)]
[[Project](https://canqin001.github.io/UniControl-Page/)] 

**FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model** \
[[ICCV 2023](https://arxiv.org/abs/2303.09833)] 
[[Code](https://github.com/vvictoryuki/freedom)]



**Freestyle Layout-to-Image Synthesis** \
[[CVPR 2023 highlight](https://arxiv.org/abs/2303.14412)] 
[[Code](https://github.com/essunny310/freestylenet)]
[[Project](https://essunny310.github.io/FreestyleNet/)] 

**Universal Guidance for Diffusion Models** \
[[Website](https://arxiv.org/abs/2302.07121)] 
[[Code](https://github.com/arpitbansal297/Universal-Guided-Diffusion)]

**Late-Constraint Diffusion Guidance for Controllable Image Synthesis** \
[[Website](https://arxiv.org/abs/2305.11520)] 
[[Code](https://github.com/AlonzoLeeeooo/LCDG)]
[[Project](https://alonzoleeeooo.github.io/LCDG/)] 

**Freestyle Layout-to-Image Synthesis** \
[[CVPR 2023 highlight]](https://arxiv.org/abs/2303.14412) 
[[Code]](https://github.com/essunny310/FreestyleNet) 
[[Project](https://essunny310.github.io/FreestyleNet/)] 

**Modulating Pretrained Diffusion Models for Multimodal Image Synthesis** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2302.12764)] 
[[Project](https://mcm-diffusion.github.io/)] 

**Sketch-Guided Text-to-Image Diffusion Models** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2211.13752)] 
[[Project](https://sketch-guided-diffusion.github.io/)] 
[[Code]](https://github.com/Mikubill/sketch2img) 

**Conditioning Diffusion Models via Attributes and Semantic Masks for Face Generation** \
[[Website](https://arxiv.org/abs/2306.00914)] 

**Control4D: Dynamic Portrait Editing by Learning 4D GAN from 2D Diffusion-based Editor** \
[[Website](https://arxiv.org/abs/2305.20082)] 
[[Project](https://control4darxiv.github.io/)] 

**Integrating Geometric Control into Text-to-Image Diffusion Models for High-Quality Detection Data Generation via Text Prompt** \
[[Website](https://arxiv.org/abs/2306.04607)] 

**Adding 3D Geometry Control to Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.08103)] 

**Continuous Layout Editing of Single Images with Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.13078)] 

**Zero-shot spatial layout conditioning for text-to-image diffusion models** \
[[Website](https://arxiv.org/abs/2306.13754)] 

**Composite Diffusion | whole >= \Sigma parts** \
[[Website](https://arxiv.org/abs/2307.13720)] 

**LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation** \
[[Website]](https://arxiv.org/abs/2302.08908) 

# T2I Diffusion Model augmentation
⭐⭐**Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2301.13826)] 
[[Official Code](https://github.com/yuval-alaluf/Attend-and-Excite)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_and_excite.py)] 
[[Diffusers doc](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/attend_and_excite)] 
[[Project](https://yuval-alaluf.github.io/Attend-and-Excite/)] 
[[Replicate Demo](https://replicate.com/daanelson/attend-and-excite)]

⭐**Improving Sample Quality of Diffusion Models Using Self-Attention Guidance** \
[[ICCV 2023](https://arxiv.org/abs/2210.00939)] 
[[Project](https://ku-cvlab.github.io/Self-Attention-Guidance/)] 
[[Code Official](https://github.com/KU-CVLAB/Self-Attention-Guidance)]
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_sag.py)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/self_attention_guidance)] 
[[Demo](https://huggingface.co/spaces/susunghong/Self-Attention-Guidance)]

⭐**Expressive Text-to-Image Generation with Rich Text** \
[[Website](https://arxiv.org/abs/2304.06720)] 
[[Project](https://rich-text-to-image.github.io/)] 
[[Code](https://github.com/SongweiGe/rich-text-to-image)]
[[Demo](https://huggingface.co/spaces/songweig/rich-text-to-image)]

⭐**MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation** \
[[ICML 2023](https://arxiv.org/abs/2302.08113)] 
[[Project](https://multidiffusion.github.io/)] 
[[Code](https://github.com/omerbt/MultiDiffusion)]
[[Demo](https://huggingface.co/spaces/weizmannscience/MultiDiffusion)]
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_panorama.py)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/panorama#multidiffusion-fusing-diffusion-paths-for-controlled-image-generation)] 
[[Replicate Demo](https://replicate.com/omerbt/multidiffusion)]

⭐**SEGA: Instructing Diffusion using Semantic Dimensions** \
[[Website](https://arxiv.org/abs/2301.12247)] 
[[Code](https://github.com/ml-research/semantic-image-editing)]
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/semantic_stable_diffusion/pipeline_semantic_stable_diffusion.py)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion)] 

**BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion** \
[[ICCV 2023](https://arxiv.org/abs/2307.10816)] 
[[Code](https://github.com/Sierkinhane/BoxDiff)]

**LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2304.03373)] 
[[Project](https://layoutllm-t2i.github.io/)] 
[[Code](https://github.com/LayoutLLM-T2I/LayoutLLM-T2I)]


**Fair Diffusion: Instructing Text-to-Image Generation Models on Fairness** \
[[Website](https://arxiv.org/abs/2302.10893)] 
[[Code](https://github.com/ml-research/fair-diffusion)]

**Directed Diffusion: Direct Control of Object Placement through Attention Guidance** \
[[Website](https://arxiv.org/abs/2302.13153)] 
[[Project](https://silent-chen.github.io/layout-guidance/)] 
[[Code](https://github.com/silent-chen/layout-guidance)]

**Training-Free Layout Control with Cross-Attention Guidance** \
[[Website](https://arxiv.org/abs/2304.03373)] 
[[Project](https://hohonu-vicml.github.io/DirectedDiffusion.Page/)] 
[[Code](https://github.com/hohonu-vicml/DirectedDiffusion)]

**Editing Implicit Assumptions in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.08084)] 
[[Project](https://time-diffusion.github.io/)] 
[[Demo](https://huggingface.co/spaces/bahjat-kawar/time-diffusion)]

**Real-World Image Variation by Aligning Diffusion Inversion Chain** \
[[Website](https://arxiv.org/abs/2305.18729)] 
[[Project](https://rival-diff.github.io/)] 
[[Code](https://github.com/julianjuaner/RIVAL/)]

**Harnessing the Spatial-Temporal Attention of Diffusion Models for High-Fidelity Text-to-Image Synthesis** \
[[Website](https://arxiv.org/abs/2304.03869)] 
[[Code](https://github.com/UCSB-NLP-Chang/Diffusion-SpaceTime-Attn)]

**SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models** \
[[Website](https://arxiv.org/abs/2305.05189)] 
[[Code](https://github.com/Qrange-group/SUR-adapter)]

**Detector Guidance for Multi-Object Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2306.02236)] 
[[Code](https://github.com/luping-liu/Detector-Guidance)]

**Designing a Better Asymmetric VQGAN for StableDiffusion** \
[[Website](https://arxiv.org/abs/2306.04632)] 
[[Code](https://github.com/buxiangzhiren/Asymmetric_VQGAN)]

**FABRIC: Personalizing Diffusion Models with Iterative Feedback** \
[[Website](https://arxiv.org/abs/2307.10159)] 
[[Code](https://github.com/sd-fabric/fabric)]

**ConceptLab: Creative Generation using Diffusion Prior Constraints** \
[[Website](https://arxiv.org/abs/2308.02669)] 
[[Code](https://github.com/kfirgoldberg/ConceptLab)]
[[Project](https://kfirgoldberg.github.io/ConceptLab/)] 


**StyleDrop: Text-to-Image Generation in Any Style** \
[[Website](https://arxiv.org/abs/2306.00983)] 
[[Project](https://styledrop.github.io/)] 

**Diffusion Self-Guidance for Controllable Image Generation** \
[[Website](https://arxiv.org/abs/2306.00986)] 
[[Project](https://dave.ml/selfguidance/)] 

**Divide & Bind Your Attention for Improved Generative Semantic Nursing**\
[[Website](https://arxiv.org/abs/2307.10864)] 
[[Project](https://sites.google.com/view/divide-and-bind)] 

**Text2Layer: Layered Image Generation using Latent Diffusion Model** \
[[Website](https://arxiv.org/abs/2307.09781)] 

**Guided Image Synthesis via Initial Image Editing in Diffusion Model** \
[[Website](https://arxiv.org/abs/2305.03382)] 

**Controllable Text-to-Image Generation with GPT-4** \
[[Website](https://arxiv.org/abs/2305.18583)] 

**It is all about where you start: Text-to-image generation with seed selection** \
[[Website](https://arxiv.org/abs/2304.14530)] 

**End-to-End Diffusion Latent Optimization Improves Classifier Guidance** \
[[Website](https://arxiv.org/abs/2303.13703)] 

**Stimulating the Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling** \
[[Website](https://arxiv.org/abs/2307.03992)] 

**If at First You Don’t Succeed, Try, Try Again:Faithful Diffusion-based Text-to-Image Generation by Selection** \
[[Website](https://arxiv.org/abs/2305.13308)] 
[[Code](https://github.com/ExplainableML/ImageSelect)]

**Compositional Text-to-Image Synthesis with Attention Map Control of Diffusion Models** \
[[Website](https://arxiv.org/abs/2305.13921)] 

**Norm-guided latent space exploration for text-to-image generation** \
[[Website](https://arxiv.org/abs/2306.08687)] 

**DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.14685)] 

**A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis** \
[[Website](https://arxiv.org/abs/2306.14544)] 

**Decompose and Realign: Tackling Condition Misalignment in Text-to-Image Diffusion Models** \
[[Website](https://arxiv.org/abs/2306.14408)] 




<!-- # Watermark
**The Stable Signature: Rooting Watermarks in Latent Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.15435)] 
[[Project](https://pierrefdz.github.io/publications/stablesignature/)] 

**A Recipe for Watermarking Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.10137)] 
[[Code](https://github.com/yunqing-me/watermarkdm)] -->


# Segmentation Detection Tracking
⭐⭐**odise: open-vocabulary panoptic segmentation with text-to-image diffusion modelss** \
[[CVPR 2023 Highlight](https://arxiv.org/abs/2303.04803)] 
[[Project](https://jerryxu.net/ODISE/)] 
[[Code](https://github.com/NVlabs/ODISE)]
[[Demo](https://huggingface.co/spaces/xvjiarui/ODISE)]

⭐**Personalize Segment Anything Model with One Shot** \
[[Website](https://arxiv.org/abs/2305.03048)] 
[[Code](https://github.com/ZrrSkywalker/Personalize-SAM)]

**DDP: Diffusion Model for Dense Visual Prediction**\
[[ICCV 2023]](https://arxiv.org/abs/2303.17559)     
[[Code]](https://github.com/JiYuanFeng/DDP)

**DiffusionSeg: Adapting Diffusion Towards Unsupervised Object Discovery** \
[[Website](https://arxiv.org/abs/2303.09813)] 


**DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models** \
[[Website](https://arxiv.org/abs/2303.11681)] 
[[Project](https://weijiawu.github.io/DiffusionMask/)] 

**MaskDiff: Modeling Mask Distribution with Diffusion Probabilistic Model for Few-Shot Instance Segmentation** \
[[Website](https://arxiv.org/abs/2303.05105)] 

**DiffusionDet: Diffusion Model for Object Detection** \
[[ICCV 2023](https://arxiv.org/abs/2211.09788)] 
[[Code](https://github.com/shoufachen/diffusiondet)]

**OVTrack: Open-Vocabulary Multiple Object Tracking** \
[[CVPR 2023](https://arxiv.org/abs/2304.08408)] 



# Few-Shot 
**Discriminative Diffusion Models as Few-shot Vision and Language Learners** \
[[Website](https://arxiv.org/abs/2305.10722)] 

**Few-shot Semantic Image Synthesis with Class Affinity Transfer** \
[[CVPR 2023](https://arxiv.org/abs/2304.02321)] 

**Few-Shot Diffusion Models** \
[[Website](https://arxiv.org/abs/2205.15463)] 
[[Code](https://github.com/georgosgeorgos/few-shot-diffusion-models)]

**DiffAlign : Few-shot learning using diffusion based synthesis and alignment** \
[[Website](https://arxiv.org/abs/2212.05404)] 

**Few-shot Image Generation with Diffusion Models** \
[[Website](https://arxiv.org/abs/2211.03264)] 

**Lafite2: Few-shot Text-to-Image Generation** \
[[Website](https://arxiv.org/abs/2210.14124)] 

<!-- # Restoration -->

# Drag Image Edit
**Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2305.10973)] 
[[Code](https://github.com/XingangPan/DragGAN)]
[[Project](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)] 


**DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing** \
[[Website](https://arxiv.org/abs/2306.14435)] 
[[Code](https://github.com/Yujun-Shi/DragDiffusion)]

**DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models** \
[[Website](https://arxiv.org/abs/2307.02421)] 
[[Code](https://github.com/MC-E/DragonDiffusion)]













# SD-inpaint
**Blended Diffusion for Text-driven Editing of Natural Images** \
[[Website](https://arxiv.org/abs/2111.14818)] 
[[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.html)] 
[[Project](https://omriavrahami.com/blended-diffusion-page/)]
[[Code](https://github.com/omriav/blended-diffusion)]

**Blended Latent Diffusion** \
[[SIGGRAPH 2023](https://arxiv.org/abs/2206.02779)] 
[[Code](https://github.com/omriav/blended-latent-diffusion)]
[[Project](https://omriavrahami.com/blended-latent-diffusion-page/)]

**Paint by Example: Exemplar-based Image Editing with Diffusion Models** \
[[Website](https://arxiv.org/abs/2211.13227)] 
[[Code](https://github.com/Fantasy-Studio/Paint-by-Example)]
[[Diffusers Doc](https://huggingface.co/docs/diffusers/api/pipelines/paint_by_example)] 
[[Diffusers Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/paint_by_example/pipeline_paint_by_example.py)] 


**GLIDE: Towards photorealistic image generation and editing with text-guided diffusion model** \
[[Website](https://arxiv.org/abs/2112.10741)] 
[[Code](https://github.com/openai/glide-text2im)]


**Reference-based Image Composition with Sketch via Structure-aware Diffusion Model** \
[[Website](https://arxiv.org/abs/2304.09748)] 
[[Code](https://github.com/kangyeolk/Paint-by-Sketch)]

**Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting** \
[[CVPR 2023](https://arxiv.org/abs/2212.06909)] 
[[Code](https://github.com/fenglinglwb/PSM)]

**Delving Globally into Texture and Structure for Image Inpainting**\
[[ACM MM 2022](https://arxiv.org/abs/2209.08217)] 
[[Code](https://github.com/htyjers/DGTS-Inpainting)]

**Image Inpainting via Iteratively Decoupled Probabilistic Modeling** \
[[Website](https://arxiv.org/abs/2212.02963)] 
[[Code](https://github.com/fenglinglwb/PSM)]

**Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models** \
[[Website](https://arxiv.org/abs/2304.03322)] 
[[Code](https://github.com/ucsb-nlp-chang/copaint)]

**SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model** \
[[Website](https://arxiv.org/abs/2212.05034)] 



## Document Layout Generation

⭐**LayoutDM: Discrete Diffusion Model for Controllable Layout Generation** \
[[CVPR 2023]](https://arxiv.org/abs/2303.08137) 
[[Code]](https://github.com/CyberAgentAILab/layout-dm) 
[[Project](https://cyberagentailab.github.io/layout-dm/)] 

**Unifying Layout Generation with a Decoupled Diffusion Model** \
[[CVPR 2023]](https://arxiv.org/abs/2303.05049) 

**PLay: Parametrically Conditioned Layout Generation using Latent Diffusion** \
[[ICML 2023]](https://arxiv.org/abs/2301.11529) 

**DLT: Conditioned layout generation with Joint Discrete-Continuous Diffusion Layout Transformer** \
[[Website]](https://arxiv.org/abs/2303.03755) 

**Diffusion-based Document Layout Generation** \
[[Website]](https://arxiv.org/abs/2303.10787) 

**LayoutDM: Transformer-based Diffusion Model for Layout Generation** \
[[CVPR 2023]](https://arxiv.org/abs/2305.02567) 

**LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models** \
[[Website]](https://arxiv.org/abs/2303.11589) 


## Super Resolution
⭐⭐⭐**Image Super-Resolution via Iterative Refinement** \
[[TPAMI]](https://arxiv.org/abs/2104.07636) 
[[Code]](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) 
[[Project](https://iterative-refinement.github.io/)] 

⭐**Exploiting Diffusion Prior for Real-World Image Super-Resolution** \
[[Website]](https://arxiv.org/abs/2305.07015) 
[[Code]](https://github.com/IceClear/StableSR) 
[[Project](https://iceclear.github.io/projects/stablesr/)] 

**ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting** \
[[Website]](https://arxiv.org/abs/2307.12348) 
[[Code]](https://github.com/zsyoaoa/resshift) 

**Solving Diffusion ODEs with Optimal Boundary Conditions for Better Image Super-Resolution** \
[[Website]](https://arxiv.org/abs/2305.15357) 

**Dissecting Arbitrary-scale Super-resolution Capability from Pre-trained Diffusion Generative Models** \
[[Website]](https://arxiv.org/abs/2306.00714) 
