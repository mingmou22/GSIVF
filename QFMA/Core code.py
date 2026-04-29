import torch
from torchvision.models import vit_b_16 
import torchvision
from torchvision import transforms
from torch_geometric.nn import ChebConv 
import networkx as nx
import numpy as np
import cv2
from PIL import Image
from scipy import fftpack
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import torch.optim as optim

def load_image(img_path, img_size=640, device='cpu'):
    img = Image.open(img_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])
    tensor = tf(img).to(device)

    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor



def load_attention_model(device):
    model = vit_b_16(pretrained=True) 
    model.eval() 
    return model.to(device)


def get_attention_map(model, img):
    with torch.no_grad():
        img = img.unsqueeze(0)
        output = model(img)

        attention_map = output[0] if isinstance(output, tuple) else output
        attention_map = torch.softmax(attention_map, dim=1)  


        attention_map = attention_map.squeeze() 
        return attention_map


def mask_non_important_areas(attention_map, threshold):
    mask = attention_map > threshold 
    return mask.float()


class StructuralGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, K=2):  
        super().__init__()
        self.cheb1 = ChebConv(in_dim, hidden, K)  
        self.cheb2 = ChebConv(hidden, 1, K)  

    def forward(self, data):
        print(f"Input Feature Shape: {data.x.shape}") 
        x, edge_index = data.x, data.edge_index
        h = torch.relu(self.cheb1(x, edge_index))  
        score = self.cheb2(h, edge_index) 
        return score.flatten()  


def create_heat_mask(infrared_image, n_clusters=3):

    pixels = infrared_image.flatten().reshape(-1, 1)  
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    clustered = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()
    cluster_centers.sort()
    threshold = cluster_centers[-1]
   
    heat_mask = (infrared_image > threshold).astype(np.uint8)
    return heat_mask

def image_to_overlapping_patches(img, patch_size=16, stride=1, mask=None):

    C, H, W = img.shape  
    node_feats, idx_map = [], []
    group_ids = []
    gid = 0

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            take = True
            if mask is not None:
                m_patch = mask[i:i + patch_size, j:j + patch_size]
                if m_patch.sum() < patch_size * patch_size * 0.4:
                    take = False
            if take:
                patch = img[:, i:i + patch_size, j:j + patch_size]

                node_feats.append(patch.reshape(-1))
                idx_map.append((i, j, gid, patch_size))
                group_ids.append(gid)
                gid += 1 

    if len(node_feats) == 0:
        raise RuntimeError("patch，mask small！")

    node_feats = torch.stack(node_feats)
    print(f"Node Features Shape: {node_feats.shape}")

    edge_list = []
    coord2idx = {(i, j): idx for idx, (i, j, _, _) in enumerate(idx_map)}
    for idx, (i, j, _, _) in enumerate(idx_map):
        for dx, dy in [(-stride, 0), (0, -stride), (stride, 0), (0, stride)]:
            ni, nj = i + dx, j + dy
            if (ni, nj) in coord2idx:
                edge_list.append((idx, coord2idx[(ni, nj)]))
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=img.device).T if edge_list else torch.zeros((2, 0),
                                                                                                              dtype=torch.long,
                                                                                                              device=img.device)

    return node_feats, edge_index, idx_map, group_ids

def scatter_patches(adv_img, idx_map, patch_size, device, important_indices):
    """
    Scatters the image patches to random locations in the image, but only for important patches.

    Args:
        adv_img: The adversarial image (C, H, W).
        idx_map: List of tuples containing patch information (i, j, gid, patch_size).
        patch_size: Size of each patch.
        device: The device to run the operations on.
        important_indices: A list of indices indicating which patches are important.

    Returns:
        A new image with scattered patches.
    """
    C, H, W = adv_img.shape
    new_img = torch.zeros_like(adv_img, device=device)
    available_positions = []
    for i in range(0, H - patch_size + 1):
        for j in range(0, W - patch_size + 1):
            available_positions.append((i, j))

    random.shuffle(available_positions)
    scattered_count = 0
    for idx, (original_i, original_j, gid, scale) in enumerate(idx_map):
        if idx in important_indices:  
            patch = adv_img[:, original_i:original_i + scale, original_j:original_j + scale]
           
            if scattered_count < len(available_positions):
                new_i, new_j = available_positions[scattered_count]
                new_img[:, new_i:new_i + scale, new_j:new_j + scale] = patch
                scattered_count += 1
            else:
                
                break
        else:
           
            new_img[:, original_i:original_i + scale, original_j:original_j + scale] = adv_img[:, original_i:original_i + scale, original_j:original_j + scale]
    return new_img


def gnn_patch_attack_with_attention(img, target_img, structural_gnn, device, attention_model,
                                     patch_size=16, n_steps=20, adv_eps=0.08, lr=0.1,
                                     tv_lambda=1.0, verbose=True, mask=None, threshold=0.5):
   
    print(f"Before moving to device: img type: {type(img)}, img shape: {img.shape}")
    img = img.to(device)

    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32, device=device)

    try:  
        node_feats, edge_index, idx_map, group_ids = image_to_overlapping_patches(
            img, patch_size=patch_size, stride=max(1, patch_size // 2), mask=mask
        )
    except RuntimeError as e:
        print(f"Error in image_to_overlapping_patches: {e}")
        return None

    node_feats = node_feats.clone().detach().to(device)
    edge_index = edge_index.to(device)
    node_feats.requires_grad_(True)
    n_nodes = node_feats.shape[0]
    structural_gnn.eval()

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    edges = edge_index.cpu().numpy().T
    G.add_edges_from([tuple(e) for e in edges])
    centrality = torch.tensor(list(nx.betweenness_centrality(G).values()), device=device)
    imp_weight = centrality / (centrality.max() + 1e-6)

    momentum = torch.zeros_like(node_feats)
    momentum_decay = 1.0
    loss_trace = []


    attention_map = get_attention_map(attention_model, img)  

 
    expanded_attention = attention_map.unsqueeze(0).repeat(node_feats.shape[0], 1) 

    for step in range(n_steps):
        step_lr = lr * (0.5 ** (step / (n_steps // 2)))

        class GNNData:
            pass

        data = GNNData()
        data.x = node_feats
        data.edge_index = edge_index

        # GNN score calculation
        score = structural_gnn(data)
        loss_gnn = -score.mean() 

        # TV loss (smoothness constraint)
        if edge_index.shape[1] > 0:
            x_diffs = node_feats[edge_index[0]] - node_feats[edge_index[1]]
            loss_tv = x_diffs.abs().mean() 
        else:
            loss_tv = 0.0

        loss = loss_gnn + tv_lambda * loss_tv

        loss.backward()
        grad = node_feats.grad

       
        weighted_grad = torch.zeros_like(grad)
        for i in range(node_feats.shape[0]):  
            
            attention_per_feature = torch.ones(768, device=device) * (sum(expanded_attention[i]) / 768)
            weighted_grad[i] = grad[i] * attention_per_feature  

        momentum = momentum_decay * momentum + weighted_grad / (grad.abs().mean() + 1e-8)
        node_feats.data += step_lr * torch.sign(momentum)
        node_feats.data = torch.clamp(node_feats.data, 0, 1) 
        node_feats.grad.zero_()
        loss_trace.append((loss.item(), loss_gnn.item(), loss_tv.item()))

        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"[{step + 1:02d}/{n_steps}] loss={loss.item():.5f}, TV={loss_tv:.5f}, GNN={-loss_gnn:.5f}")

    adv_img = torch.zeros_like(img)
    counts = torch.zeros_like(img)

    for iidx, (i, j, gid, scale) in enumerate(idx_map):
        patch = node_feats[iidx].detach().reshape(img.shape[0], scale, scale)
        adv_img[:, i:i + scale, j:j + scale] += patch
        counts[:, i:i + scale, j:j + scale] += 1

    counts = torch.clamp(counts, min=1)

    if mask is not None:
        mask3d = mask.unsqueeze(0)  
        adv_img = img * (1 - mask3d) + (adv_img / counts) * mask3d
    else:
        adv_img = adv_img / counts

    adv_img = torch.clamp(adv_img, 0, 1)  


    num_important = min(5, n_nodes)  
    important_indices = torch.argsort(score, descending=True)[:num_important]
    important_indices = important_indices.cpu().numpy().tolist()


    adv_img = scatter_patches(adv_img, idx_map, patch_size, device, important_indices)

    perturb_out = adv_img - img

    print(f"img.shape: {img.shape}")
    print(f"perturb_out.shape: {perturb_out.shape}")

    if perturb_out.dim() == 3 and perturb_out.shape[0] == img.shape[1] and perturb_out.shape[1] == img.shape[2] and \
            perturb_out.shape[2] == img.shape[0]:![](16.jpg)![](17.jpg)
        perturb_out = perturb_out.permute(2, 0, 1)  
        print("Transposed perturb_out to (C, H, W)")
    elif perturb_out.shape != img.shape:
        raise ValueError(
            f"The shape of perturb_out is invalid. Expected shape: {perturb_out.shape}")

    perturb = torch.clamp(perturb_out, -adv_eps, adv_eps)  
    print(f"perturb min: {perturb.min()}, max: {perturb.max()}")
    adv_img = torch.clamp(img + perturb, 0, 1) 
    print(f"adv_img min: {adv_img.min()}, max: {adv_img.max()}")
    return adv_img

def load_image(image_path, img_size, device):![](v1.jpg)
    
    image = cv2.imread(image_path)  
    image = cv2.resize(image, (img_size, img_size))  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = image / 255.0  
    return torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1)  

def normalize_image(image):
    image = image.clip(0, 255)
    normalized_image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image.astype(np.uint8)

class ImageFusion:
    def dct_transform(self, img):
        dct_img = fftpack.dct(fftpack.dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_img -= dct_img.min()
        dct_img /= (dct_img.max() + 1e-6)  
        dct_img *= 255
        return dct_img

    def idct_transform(self, dct_img):
        return fftpack.idct(fftpack.idct(dct_img, axis=0, norm='ortho'), axis=1, norm='ortho')

    def generate_fusion_image(self, visible_image, infrared_image, perturbation_weights=None):
        print("fusing image...")


        visible_np = cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        infrared_np = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2RGB).astype(np.float32)

 
        dct_visible = self.dct_transform(visible_np)
        dct_infrared = self.dct_transform(infrared_np)


        low_freq_fusion = np.maximum(dct_visible, dct_infrared).clip(0, 255)


        print(f"DCT Visible Min: {dct_visible.min()}, Max: {dct_visible.max()}")
        print(f"DCT Infrared Min: {dct_infrared.min()}, Max: {dct_infrared.max()}")


        if perturbation_weights is None:
            perturbation_weights = np.zeros(visible_np.shape, dtype=np.float32)

 
        visible_high_freq = np.clip(visible_np + perturbation_weights, 0, 255)
        infrared_high_freq = np.clip(infrared_np + (1 - perturbation_weights), 0, 255)


        high_freq_fusion = (visible_high_freq * 0.5) + (infrared_high_freq * 0.5)
        high_freq_fusion = np.clip(high_freq_fusion, 0, 255)


        fusion_image_low = self.idct_transform(low_freq_fusion)


        final_fusion_image = np.clip(fusion_image_low + high_freq_fusion, 0, 255).astype(np.uint8)


        perturbation_image = np.clip(fusion_image_low + perturbation_weights, 0, 255).astype(np.uint8)
        final_fusion_image = np.clip(final_fusion_image + (perturbation_image - fusion_image_low), 0, 255)

        print(f"fused - Min: {final_fusion_image.min()}, Max: {final_fusion_image.max()}")

        if len(final_fusion_image.shape) == 2:  
            final_fusion_image = np.stack([final_fusion_image] * 3, axis=2)

        return final_fusion_image

def main(visible_image_path, infrared_image_path, structural_gnn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    visible_image = load_image(visible_image_path, img_size=224, device=device)
    infrared_image = load_image(infrared_image_path, img_size=224, device=device)

    attention_model = load_attention_model(device)

    visible_image_np = visible_image.permute(1, 2, 0).detach().cpu().numpy() * 255
    infrared_image_np = infrared_image.permute(1, 2, 0).detach().cpu().numpy() * 255


    image_fusion = ImageFusion()
    fused_image_no_perturbation = image_fusion.generate_fusion_image(visible_image_np, infrared_image_np)


    fused_image_no_perturbation = np.clip(fused_image_no_perturbation, 0, 255).astype(np.uint8)


    cv2.imshow("No Perturbation Fusion Image", fused_image_no_perturbation)
    cv2.imwrite("no_perturbation_fused_image.png", fused_image_no_perturbation)
    plt.close()


    infrared_image_np = infrared_image.permute(1, 2, 0).cpu().numpy()
    reshaped_infrared_image = infrared_image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
    kmeans.fit(reshaped_infrared_image)
    infrared_labels = kmeans.labels_.reshape(infrared_image_np.shape[:2])


    plt.imshow(infrared_labels, cmap='viridis', aspect='auto')
    plt.axis('off')
    plt.savefig("infrared_heatmap.png", bbox_inches='tight', pad_inches=0)
    plt.close()


    heat_mask = Image.open("infrared_heatmap.png").convert('L')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    heat_mask_resized = transform(heat_mask).squeeze(0).to(device)


    binary_mask = (heat_mask_resized > 0).float()
    mask_indices = torch.nonzero(binary_mask)

    print("Number of mask indices:", mask_indices.shape[0])
    if mask_indices.shape[0] == 0:
        print("No mask indices found. Returning original images.")
        final_visible_adv_img = (visible_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        final_infrared_adv_img = (infrared_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        cv2.imshow("Visible Image", final_visible_adv_img)
        cv2.imshow("Infrared Image", final_infrared_adv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
        return 

    adv_img_visible = gnn_patch_attack_with_attention(
        visible_image.clone().detach(),
        visible_image.clone().detach(),
        structural_gnn,
        device,
        attention_model, 
        threshold=0.5,  
        verbose=True
    )

    if adv_img_visible is None:
        print("Error: adv_img_visible is None. Check input values and mask.")
        return

    adv_img_infrared = gnn_patch_attack_with_attention(
        infrared_image.clone().detach(),
        infrared_image.clone().detach(),
        structural_gnn,
        device,
        attention_model, 
        threshold=0.5  
    )


    if adv_img_infrared is None:
        print("Error: adv_img_infrared is None.")
        return  


    final_visible_adv_img = (adv_img_visible.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    final_infrared_adv_img = (adv_img_infrared.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


    perturb_visible = (adv_img_visible - visible_image).detach().cpu() * 255
    perturb_visible = perturb_visible.numpy().squeeze().transpose(1, 2, 0)

    perturb_infrared = (adv_img_infrared - infrared_image).detach().cpu() * 255
    perturb_infrared = perturb_infrared.numpy().squeeze().transpose(1, 2, 0)


    cv2.imshow("Adversarial Visible Image", final_visible_adv_img)
    cv2.imwrite("adversarial_visible_image.png", final_visible_adv_img)

    cv2.imshow("Adversarial Infrared Image", final_infrared_adv_img)
    cv2.imwrite("adversarial_infrared_image.png", final_infrared_adv_img)

    cv2.waitKey(0)  # Maintain the window open until a key is pressed


    normalized_visible = normalize_image(perturb_visible)
    cv2.imshow("Perturbation for Visible Image", normalized_visible)
    cv2.imwrite("perturbation_visible_image.png", normalized_visible)

    normalized_infrared = normalize_image(perturb_infrared)
    cv2.imshow("Perturbation for Infrared Image", normalized_infrared)
    cv2.imwrite("perturbation_infrared_image.png", normalized_infrared)

    cv2.waitKey(0)  # Maintain the window open until a key is pressed


    heat_mask_np = (heat_mask_resized.cpu().numpy() * 255).astype(np.uint8)
    heat_mask_colored = cv2.applyColorMap(heat_mask_np, cv2.COLORMAP_JET)
    cv2.imshow("Heat Map", heat_mask_colored)
    cv2.imwrite("heat_map.png", heat_mask_colored)
    cv2.waitKey(0)  # Maintain the window open until a key is pressed


    image_fusion = ImageFusion()
    perturbation_weights = np.ones_like(final_visible_adv_img, dtype=np.float32) * 0.5
    fused_image = image_fusion.generate_fusion_image(final_visible_adv_img, final_infrared_adv_img, perturbation_weights)
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)


    if len(fused_image.shape) == 2: 
        fused_image_colored = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
    else:  
        fused_image_colored = fused_image

    cv2.imshow("Fused Image", fused_image_colored)
    cv2.imwrite("fused_image.png", fused_image_colored)
    cv2.waitKey(0)  # Maintain the window open until a key is pressed


    fused_image_tensor = torch.tensor(fused_image).permute(2, 0, 1).to(device).float() / 255
    adv_fused_image, _, perturb_out_fused = gnn_patch_attack_with_attention(
        fused_image_tensor.clone().detach(),
        fused_image_tensor.clone().detach(),
        structural_gnn,
        device,
        attention_model,  
        threshold=0.5  
    )


    if adv_fused_image is None:
        print("Error: adv_fused_image is None.")
        return  




    final_fused_adv_img = (adv_fused_image.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


    print(f"Final Fused Adversarial Image shape: {final_fused_adv_img.shape}")


    if len(final_fused_adv_img.shape) == 2:  

        final_fused_adv_img = np.stack([final_fused_adv_img] * 3, axis=-1)  

    
    elif final_fused_adv_img.shape[2] != 3:
        raise ValueError("The image is not in the expected shape of (H, W, 3) or (C, H, W)")


    if final_fused_adv_img.dtype != np.uint8:
        final_fused_adv_img = final_fused_adv_img.astype(np.uint8)


    if len(final_fused_adv_img.shape) == 3:
        final_fused_adv_img = final_fused_adv_img.transpose(2, 0, 1)  


    final_fused_adv_img_for_display = final_fused_adv_img.transpose(1, 2, 0) if len(
        final_fused_adv_img.shape) == 3 else final_fused_adv_img


    print(f"Final image for display shape: {final_fused_adv_img_for_display.shape}")


    cv2.imshow("Final Fused Adversarial Image", final_fused_adv_img_for_display)
    cv2.imwrite("final_fused_adversarial_image.png", final_fused_adv_img_for_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_mask_and_add_perturbation(image, mask_indices, perturbation):
    image = np.array(image)
    perturbation = np.array(perturbation)

    if perturbation.ndim == 3 and perturbation.shape[0] == 3:
        perturbation = np.transpose(perturbation, (1, 2, 0))

    print(f"Image shape: {image.shape}")
    print(f"Perturbation shape after transpose: {perturbation.shape}")

    if perturbation.ndim != 3 or perturbation.shape != (224, 224, 3):
        raise ValueError("Perturbation must be a valid 3D array with shape (224, 224, 3)")

    perturbed_image = image.copy()

    for i, j in zip(*mask_indices):
        if 0 <= i < perturbed_image.shape[0] and 0 <= j < perturbed_image.shape[1]:
            pixel_value = image[i, j]
            perturb_value = perturbation[i, j]

            print(f"Pixel value at ({i}, {j}): {pixel_value}")
            print(f"Perturbation value at ({i}, {j}): {perturb_value}")

            if pixel_value.ndim != 1 or perturb_value.ndim != 1:
                raise ValueError(f"Unexpected shape for pixel or perturbation at index ({i}, {j})")

            perturbed_image[i, j] = np.clip(pixel_value + perturb_value, 0, 255)

    return perturbed_image.astype(np.uint8)


if __name__ == "__main__":
    visible_image_path = ' .jpg' 
    infrared_image_path = ' .jpg'  


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structural_gnn = StructuralGNN(in_dim=768, hidden=128, K=2).to(device)

    main(visible_image_path, infrared_image_path, structural_gnn)  