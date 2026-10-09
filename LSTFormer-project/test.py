import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.modules.LSTformer import LSTformer


def get_args():
    parser = argparse.ArgumentParser(description='Test LSTformer')
    
    parser.add_argument('--input_dir', type=str, default='images/',
                        help='Input directory containing low light images')
    parser.add_argument('--result_dir', type=str, default='./results/',
                        help='Directory to save results')
    
    parser.add_argument('--checkpoint', type=str,
                        default='/checkpoints/best_model.pth',
                        help='Path to best model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    
    parser.add_argument('--sampling_mode', type=str, default='conv',
                        choices=['dwt', 'conv'],
                        help='Down/Up sampling mode: dwt (Ours) or conv (Baseline)')
    parser.add_argument('--fusion_mode', type=str, default='concat',
                        choices=['cafb', 'concat'],
                        help='Fusion mode: cafb (Ours) or concat (Baseline)')
    parser.add_argument('--attn_mode', type=str, default='softmax',
                        choices=['lst', 'topk', 'softmax'],
                        help='Attention mode: lst (Ours), topk, or softmax')
    # ====================================================

    return parser.parse_args()


def save_image(tensor, filepath):
    
    img = tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255.0).round().astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filepath)


def main():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")

   
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        train_args = checkpoint.get('args', {}) or {}
    else:
        state_dict = checkpoint
        train_args = {}

  
    sampling_mode = train_args.get('sampling_mode', args.sampling_mode)
    fusion_mode = train_args.get('fusion_mode', args.fusion_mode)
    attn_mode = train_args.get('attn_mode', args.attn_mode)
    dim = train_args.get('dim', 32)

    print(f"Model config -> dim={dim}, sampling_mode={sampling_mode}, "
          f"fusion_mode={fusion_mode}, attn_mode={attn_mode}")

    
    model = LSTformer(
        inp_channels=3,
        out_channels=3,
        dim=dim,
        num_blocks=[2, 2, 2, 2],
        heads=[8, 8, 8, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        use_layer_scale=False,
        use_act=False,
        conv_bias=False,
       
        sampling_mode=sampling_mode,
        fusion_mode=fusion_mode,
        attn_mode=attn_mode
    ).to(device)

    
    model.load_state_dict(state_dict)
    model.eval()

  
    files = sorted(os.listdir(args.input_dir))
    print(f"Found {len(files)} images in {args.input_dir}")

    with torch.no_grad():
        for file in tqdm(files):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

           
            img_path = os.path.join(args.input_dir, file)
            img = Image.open(img_path).convert('RGB')

            
            w, h = img.size

           
            new_w = max(16, (w // 16) * 16)
            new_h = max(16, (h // 16) * 16)
            if new_w != w or new_h != h:
                img = img.resize((new_w, new_h), Image.BICUBIC)

            input_img = (np.array(img) / 255.0).astype(np.float32)
            input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)

          
            output = model(input_tensor)

            
            if output.shape[-2] != h or output.shape[-1] != w:
                output = F.interpolate(output, size=(h, w), mode='bicubic', align_corners=False)

           
            save_name = os.path.join(args.result_dir, file)
            save_image(output, save_name)

    print(f"Done! Results saved to {args.result_dir}")


if __name__ == '__main__':
    main()
