
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from PIL import Image
import torch
import matplotlib.pyplot as plt

def savefig_pil(img, path):
    pil_img = convert_to_pil(img)       
    pil_img[0].save(path)
    
def convert_to_pil(img):
    pil_img = (img * 0.5 + 0.5).clamp(0, 1)
    pil_img = pil_img.detach().cpu().permute(0, 2, 3, 1).numpy()
    pil_img = numpy_to_pil(pil_img)
    return pil_img

def savefig_tiled_pil(dic_pils, ts_save_fig, path):
    
    from PIL import ImageDraw, ImageFont
    max_h = -1
    max_w = -1
    for _, list_pils in dic_pils.items():
        for pil_img in list_pils:
            w, h = pil_img[0].width, pil_img[0].height
            max_w = max(max_w, w)
            max_h = max(max_h, h)

    num_keys = len(dic_pils.keys())
    num_ts =   len(ts_save_fig)
    W, H = max_w * num_keys, max_h * num_ts
    results_image = Image.new('RGB', (W, H), (255, 255, 255))

    x = 0
    for key, list_pils in dic_pils.items():
        y = 0
        for pil_img in list_pils:
            
            results_image.paste(pil_img[0], (x, y))
            y += max_h
        x += max_w

    draw = ImageDraw.Draw(results_image)
    font = ImageFont.load_default()
    x, y = 5, 5
    for key in dic_pils.keys():
        draw.text((x, y), key, "white", font=font)
        x += max_w

    x, y = 5, 15
    for t in ts_save_fig:
        draw.text((x, y), str(t), "white", font=font)
        y += max_h

    # results_image.save(path)

    return results_image

def save_fig_sample_changes(t, sample, sample_bf_rlc, mask_class):
    with torch.no_grad():

        sample_difference = torch.ones_like(sample)
        indices_unmasked2unmasked = torch.where((sample_bf_rlc != mask_class) * (sample != mask_class ))
        indices_unmasked2masked   = torch.where((sample_bf_rlc != mask_class) * (sample == mask_class ))
        indices_masked2masked     = torch.where((sample_bf_rlc == mask_class) * (sample == mask_class ))
        indices_masked2unmasked   = torch.where((sample_bf_rlc == mask_class) * (sample != mask_class ))

        sample_difference[indices_unmasked2unmasked] = 0
        sample_difference[indices_unmasked2masked]   = 1
        sample_difference[indices_masked2masked]     = 2
        sample_difference[indices_masked2unmasked]   = 3

        image_sample_difference = torch.zeros(32, 32, 3, dtype=torch.uint8)
        color_map = {
                        0: [175, 175, 188],  #unmask -> unmask
                        1: [229, 130, 178],  #unmask -> mask
                        2: [219, 219, 223],  #mask -> mask
                        3: [89,  146, 230]  #mask-> unmask
                    }

        for i in range(32):
            for j in range(32):
                element_value = sample_difference[0, i * 32 + j].item()  
                color = color_map[element_value]
                image_sample_difference[i, j] = torch.tensor(color, dtype=torch.uint8)
                            
        image_sample_difference = torch.repeat_interleave(torch.repeat_interleave(image_sample_difference, 10, dim=0), 10, dim=1)

        pil_image = Image.fromarray(image_sample_difference.numpy())
        pil_image.save(f"./generated_img/sample_difference/sample_difference_{t}.png")

def plot_model_outputs(t, model_output, model_output_not_optimized):

    model_output = model_output.clamp(-70)
    model_output_not_optimized = model_output_not_optimized.clamp(-70)
    sorted_model_output, _ = torch.sort(model_output, dim=1, descending=True)
    sorted_model_output_not_optimized, _ = torch.sort(model_output_not_optimized, dim=1, descending=True)

    plt.figure(figsize=(15, 10))
    n_tokens = sorted_model_output.shape[2]
    for i in range(0, n_tokens, n_tokens // 50):
        plt.plot(sorted_model_output[0, :16, i].cpu().numpy(), linewidth=0.5)
    plt.ylabel("log probability")
    plt.title(f"sorted log probability, top 16 classes t = {t}")
    plt.savefig(f"./generated_img/sorted_log_prob_t_{t}_top16.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(15, 10))
    for i in range(0, n_tokens, n_tokens // 50):
        plt.plot(sorted_model_output_not_optimized[0, :16, i].cpu().numpy(), linewidth=0.5)
    plt.ylabel("log probability")
    plt.title(f"sorted log probability, top 16 classes t = {t}")
    plt.savefig(f"./generated_img/sorted_log_prob_t_{t}_top16_bf_optim.png", dpi=300, bbox_inches="tight")
