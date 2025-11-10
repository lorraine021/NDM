import inspect
import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import logging
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from torch.optim.adam import Adam
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin, FromSingleFileMixin, IPAdapterMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from utils.ptp_utils import AttendExciteAttnProcessor, AttentionStore
from utils.attn_utils import fn_smoothing_func, fn_clean_mask, fn_get_otsu_mask
from tqdm import tqdm
import matplotlib.pyplot as plt
from text import get_word_sets
logging.basicConfig(format='%(asctime)s: %(message)s',level=logging.INFO)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionAttendAndExcitePipeline

        >>> pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ... ).to("cuda")


        >>> prompt = "a cat and a frog"

        >>> # use get_indices function to find out indices of the tokens you want to alter
        >>> pipe.get_indices(prompt)
        {0: '<|startoftext|>', 1: 'a</w>', 2: 'cat</w>', 3: 'and</w>', 4: 'a</w>', 5: 'frog</w>', 6: '<|endoftext|>'}

        >>> token_indices = [2, 5]
        >>> seed = 6141
        >>> generator = torch.Generator("cuda").manual_seed(seed)

        >>> images = pipe(
        ...     prompt=prompt,
        ...     token_indices=token_indices,
        ...     guidance_scale=7.5,
        ...     generator=generator,
        ...     num_inference_steps=50,
        ...     max_iter_to_alter=25,
        ... ).images

        >>> image = images[0]
        >>> image.save(f"../images/{prompt}_{seed}.png")
        ```
"""


class StableDiffusionNdmPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion and Attend-and-Excite and Latent Consistency Models.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True, 
    ):
        super().__init__()

        self.K = 1
        self.cross_attention_maps_cache = None

        if safety_checker is None and requires_safety_checker:
            logging.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None, #safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=None) #requires_safety_checker) 

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp : prompt_embeds, negative_prompt_embeds, unconditional_prompt_embeds
        prompt_embeds = torch.cat([prompt_embeds_tuple[2], prompt_embeds_tuple[0]]) #TODO:!!!第一个作为无条件，第二个才是prompt产生的
        neg_prompt_embeds = torch.cat([prompt_embeds_tuple[2], prompt_embeds_tuple[1]])
        return prompt_embeds, neg_prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
        """
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logging.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)


        #TODO:
        unconditional_prompt_embeds = None
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            uncond_tokens = [""] * batch_size
            if negative_prompt is None:
                negative_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                negative_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            unconditional_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            unconditional_prompt_embeds = unconditional_prompt_embeds[0]
            
            if negative_prompt is not None:
                negative_input = self.tokenizer(
                    negative_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask_neg = negative_input.attention_mask.to(device)
                else:
                    attention_mask_neg = None

                negative_prompt_embeds = self.text_encoder(
                    negative_input.input_ids.to(device),
                    attention_mask=attention_mask_neg,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]
            else:
                negative_prompt_embeds = unconditional_prompt_embeds 


        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = unconditional_prompt_embeds.shape[1]
            unconditional_prompt_embeds = unconditional_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            unconditional_prompt_embeds = unconditional_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            unconditional_prompt_embeds = unconditional_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            seq_len_neg = negative_prompt_embeds.shape[1] # 使用 negative_prompt_embeds 的 seq_len
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds, unconditional_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        has_nsfw_concept=None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        indices,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        indices_is_list_ints = isinstance(indices, list) and isinstance(indices[0], int)
        indices_is_list_list_ints = (
            isinstance(indices, list) and isinstance(indices[0], list) and isinstance(indices[0][0], int)
        )

        if not indices_is_list_ints and not indices_is_list_list_ints:
            raise TypeError("`indices` must be a list of ints or a list of a list of ints")

        if indices_is_list_ints:
            indices_batch_size = 1
        elif indices_is_list_list_ints:
            indices_batch_size = len(indices)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"indices batch size must be same as prompt batch size. indices batch size: {indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def fn_calc_kld_loss_func(self, log_var, mu):
        return torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp()), dim=0)
    
    def fn_compute_loss(
        self, 
        indices: List[int], 
        top3_indices_record: Optional[List[int]] = None,
        smooth_attentions: bool = True,
        K: int = 1,
        ) -> torch.Tensor:
       
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
           from_where=("up", "mid", "down"), is_cross=True)
        
        aggregate_cross_attention_maps = aggregate_cross_attention_maps.to('cuda')
        H, W, T = aggregate_cross_attention_maps.shape
        
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)
        indices = [index - 1 for index in indices]
       
        area_list = []
        area_list = []
        topk_coord_list_list = []

        for i in indices:
            attn_map = cross_attention_maps[:, :, i]
            if smooth_attentions:
                attn_map = fn_smoothing_func(attn_map)  

            flat_attn = attn_map.view(-1)  
            topk_values, topk_indices = torch.topk(flat_attn, k=K, dim=0)
            topk_coords = [(idx // W, idx % W) for idx in topk_indices]
            mask_otsu = fn_get_otsu_mask(attn_map)  
            
            flat_idx = topk_indices[0]           
            x0_tensor = flat_idx // W             
            y0_tensor = flat_idx % W             
            mask_clean = fn_clean_mask(mask_otsu, x0_tensor, y0_tensor)
            
            foreground = attn_map * mask_clean
            num_ones = foreground.sum() * 2.0
            area_list.append(num_ones)
            topk_coord_list_list.append(topk_coords)

        area_tensor = torch.stack(area_list)  # (len(indices),)

        if top3_indices_record is None:
            _, top3_local_idx = torch.topk(area_tensor, k=min(3, len(area_tensor)), largest=True)
            top3_local_idx = top3_local_idx.tolist()
            while len(top3_local_idx) < 3:
                top3_local_idx.append(top3_local_idx[0])
            cross_attn_loss = area_tensor[top3_local_idx].max()
            top3_indices_record = [indices[i] for i in top3_local_idx] 
        else:
            cross_attn_loss = area_tensor.max()

        joint_loss = cross_attn_loss
        return joint_loss, cross_attn_loss, top3_indices_record

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    
    def fn_ndm(
        self,
        latents: torch.Tensor,
        indices: List[int],
        text_embeddings: torch.Tensor,
        use_grad_checkpoint: bool = False,
        ndm_lr: float = 1e-2,
        max_step: int = 50,
        num_inference_steps: int = 50,
        device: str = "",
        denoising_step_for_loss: int = 1,
        guidance_scale: int = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        do_classifier_free_guidance: bool = False,
        top3_indices_record: Optional[List[int]] = None,
        alpha: float = 0.7,
    ):
        latents = latents.clone().detach()
        log_var, mu = torch.zeros_like(latents), torch.zeros_like(latents)
        log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        optimizer = Adam([log_var, mu], lr=ndm_lr, eps=1e-3)
        history = []  # 存储 log_var, mu, cross_attn_loss 的列表
        best_loss = float('inf')
        best_latents = None
       
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        initial_loss = 0.
        optimization_succeed = False
        for iteration in tqdm(range(max_step), desc="Optimizing initial noise"):
            optimized_latents = latents * (torch.exp(0.5 * log_var)) + mu
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            joint_loss_list, cross_attn_loss_list = [], []
            
            for i, t in enumerate(timesteps):
                
                if i >= denoising_step_for_loss: break
                if use_grad_checkpoint:
                    noise_pred_text = checkpoint.checkpoint(self.unet, optimized_latents, t, text_embeddings[1].unsqueeze(0), use_reentrant=False).sample
                else: noise_pred_text = self.unet(optimized_latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
                
                joint_loss, cross_attn_loss, top3_indices_record = self.fn_compute_loss(indices=indices, K=1, top3_indices_record=top3_indices_record)
                joint_loss_list.append(joint_loss), cross_attn_loss_list.append(cross_attn_loss)
                if iteration == 0:
                    initial_loss = joint_loss
                    # print("initial loss:", initial_loss)
                if denoising_step_for_loss > 1:
                    with torch.no_grad():
                        if use_grad_checkpoint:
                            noise_pred_uncond = checkpoint.checkpoint(self.unet, optimized_latents, t, text_embeddings[0].unsqueeze(0), use_reentrant=False).sample
                        else: noise_pred_uncond = self.unet(optimized_latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample

                    if do_classifier_free_guidance: noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # compute the previous noisy sample x_t -> x_t-1
                    optimized_latents = self.scheduler.step(noise_pred, t, optimized_latents, **extra_step_kwargs).prev_sample
            
            joint_loss      = sum(joint_loss_list) / denoising_step_for_loss
            cross_attn_loss = max(cross_attn_loss_list)
            
            history.append({
                'log_var': log_var.clone().detach(),
                'mu': mu.clone().detach(),
                'cross_attn_loss': joint_loss
            })
            
            if joint_loss < alpha * initial_loss: #0.85-coco, 0.5others
                optimization_succeed = True
                break
  
            self.unet.zero_grad()
            optimizer.zero_grad()
            joint_loss = joint_loss.mean()
            joint_loss.backward()
            optimizer.step()

        if not optimization_succeed:
            for item in history:
                if item['cross_attn_loss'] < best_loss:
                    best_loss = item['cross_attn_loss']
                    best_latents = latents * (torch.exp(0.5 * item['log_var'])) + item['mu']
            optimized_latents = best_latents.clone().detach()

        else:
            optimized_latents = (latents * (torch.exp(0.5 * log_var)) + mu).clone().detach()

        return optimized_latents, optimization_succeed, cross_attn_loss, top3_indices_record

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    
    def get_ca_layers(self):
        sub_nets = self.unet.named_children()
        ca_layers = []
        for net in sub_nets:
            if 'up' in net[0] or 'down' in net[0]:
                for block in net[1]:
                    if 'Cross' in block.__class__.__name__ :
                        for attn in block.attentions:
                            for transformer in attn.transformer_blocks:
                                ca_layers.append(transformer.attn2)
            if 'mid' in net[0]:
                for attn in net[1].attentions:
                    for transformer in attn.transformer_blocks:
                        ca_layers.append(transformer.attn2)
        return ca_layers

    
    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {i: tok for tok, i in zip(self.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
        return indices

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        token_indices: Union[List[int], List[List[int]]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        scale_factor: int = 20,
        attn_res: Optional[Tuple[int]] = (16, 16),
        clip_skip: Optional[int] = None,
        optimize_noise: bool = True,
        semantic_guidance=True, #new
        alpha=0.7, 
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            token_indices (`List[int]`):
                The token indices to alter with attend-and-excite.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            thresholds (`dict`, *optional*, defaults to `{0: 0.05, 10: 0.5, 20: 0.8}`):
                Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in.
            scale_factor (`int`, *optional*, default to 20):
                Scale factor to control the step size of each attend-and-excite update.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            a
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        
        self.cross_attention_maps_cache = None
        self.cross_attn_layers = self.get_ca_layers()
        # print("layers:",self.cross_attn_layers)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if semantic_guidance == False:
            negative_prompt = ""
            
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            token_indices,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, unconditional_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        
        if do_classifier_free_guidance:
            denoising_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = torch.cat([unconditional_prompt_embeds, negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()
        
        # default config for step size from original repo
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = scale_factor * np.sqrt(scale_range)

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt * 2 :] if do_classifier_free_guidance else prompt_embeds
        )

        if isinstance(token_indices[0], int):
            token_indices = [token_indices]

        indices = []
        for ind in token_indices:
            indices = indices + [ind] * num_images_per_prompt

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # 8. ndm
        if optimize_noise:
            top3_indices_record = None
            max_round = 1
            with torch.enable_grad():
                optimized_latents_pool = []
                for i in range(max_round):
                    optimized_latents, optimization_succeed, final_attn_loss, top3_indices_record = self.fn_ndm(
                        latents=latents,
                        indices=token_indices[0],
                        text_embeddings=prompt_embeds,
                        max_step=30,
                        num_inference_steps=num_inference_steps,
                        device=device,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        eta=eta,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        top3_indices_record=top3_indices_record,
                        alpha=alpha,
                    )
                    optimized_latents_pool.append((final_attn_loss, i, optimized_latents.clone(), latents.clone(), optimization_succeed))
                    if optimization_succeed: break
            
                    latents = self.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents=None,
                    )
                optimized_latents_pool.sort()
                latents = optimized_latents_pool[0][2]
            print("finish initial noise optimization!")
                  
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    
                    latents = latents.clone().detach().requires_grad_(True)
                    updated_latents = []
                    for latent, index, text_embedding in zip(latents, indices, text_embeddings):
                        latent = latent.unsqueeze(0)
                        text_embedding = text_embedding.unsqueeze(0)
                        
                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_embedding,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        self.unet.zero_grad()
                            
                        updated_latents.append(latent)
                    latents = torch.cat(updated_latents, dim=0)
                   
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=denoising_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                
                noise_pred_neg, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_neg + guidance_scale * (noise_pred_text - noise_pred_neg)
               
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, )

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)