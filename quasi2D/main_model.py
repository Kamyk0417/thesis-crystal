import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

import utils_gen2 as utils
from aux_model import SimpleCNN

from scipy.ndimage import affine_transform

def plot_to_tensor(data):
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    ax.imshow(data, cmap='PiYG')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    image_tensor = transform(image)
    plt.close(fig)

    return image_tensor


class CrystalData(torch.utils.data.Dataset):
    def __init__(self, size=1000, input_size=300):
        self.size = size
        self.input_size = input_size
        
        self.data_list = []
        generator = utils.RandomLattice(supercell=[25,25,1], max_rotation_angle=0)
        
        for _ in range(size):
            lattice, label = generator.generate_lattice(2)
            crystal_render = generator.render_lattice(lattice)
            
            transform_r0, transform_i0 = generator.fourier_trasform_(255-crystal_render[500:1500,500:1500])
            transform_r0, transform_i0 = transform_r0[350:650,350:650], transform_i0[350:650,350:650]

            data_real = torch.from_numpy(transform_r0).float()
            data_imag = torch.from_numpy(transform_i0).float()
            
            self.data_list.append((data_real, data_imag))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        real, imag = self.data_list[idx]
        return real + 1j * imag


def load_aux_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

class Encoder(nn.Module):
    def __init__(self, input_size=300):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8))
        )
        
        self.affine_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x_combined = torch.stack([x.real, x.imag], dim=1)
        features = self.conv_layers(x_combined)
        return self.affine_predictor(features)
    
class Transform(nn.Module):
    def __init__(self, size=300):
        super().__init__()
        self.size = size
        
    def forward(self, x, theta):
        theta = theta.view(-1, 2, 3)

        x_real = x.real.unsqueeze(1)
        x_imag = x.imag.unsqueeze(1)
        
        grid = F.affine_grid(theta, x_real.size(), align_corners=False)
        t_x_real = F.grid_sample(x_real, grid, align_corners=False)
        t_x_imag = F.grid_sample(x_imag, grid, align_corners=False)
        
        transformed_x = t_x_real + t_x_imag*1.j
        Phi = torch.angle(transformed_x/x.unsqueeze(1))
        
        Phi_plots = []
        for i in range(Phi.size(0)):
            Phi_plot = plot_to_tensor(Phi[i, 0].cpu().detach().numpy())
            Phi_plots.append(Phi_plot)
    
        return torch.stack(Phi_plots).to(x.device)
    
class UnsupervisedTransformModel(nn.Module):
    def __init__(self, input_size=300):
        super().__init__()
        self.encoder_affine = Encoder(input_size)
        self.spatial_transformer = Transform(input_size)
        
        self.auxiliary_discriminator = load_aux_model('./thesis crystal/quasi2D/simple_cnn_model.pth')
        
        for param in self.auxiliary_discriminator.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        theta = self.encoder_affine(x)
        phi = self.spatial_transformer(x, theta)
        
        with torch.no_grad():
            logits = self.auxiliary_discriminator(phi)
            probs = F.softmax(logits, dim=1)
            score = probs[:, 1:2]
        
        return {
            'theta': theta,
            'phi': phi,
            'score': score
        }
    

def reinforce_loss(scores, theta, lambda_reg=0.1):
    baseline = scores.mean().detach()
    advantages = scores - baseline
    
    loss = -torch.log(1e-8 + scores) * advantages
    
    theta = theta.view(-1, 2, 3)

    identity = torch.eye(2, 3).unsqueeze(0).repeat(theta.size(0), 1, 1)
    reg_loss = F.mse_loss(theta, identity.to(theta.device))
    
    total_loss = loss.mean() + lambda_reg * reg_loss
    return total_loss

def train_model(model, dataloader, device, epochs=100):
    optimizer = torch.optim.Adam(model.encoder_affine.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            
            loss = reinforce_loss(
                outputs['score'], 
                outputs['theta']
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 1 == 0:
            avg_score = outputs['score'].mean().item()
            print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}, '
                  f'Avg Score: {avg_score:.4f}')
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = 300
    batch_size = 16
    epochs = 20
    
    dataset = CrystalData(size=30, input_size=input_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )

    model = UnsupervisedTransformModel(input_size=input_size).to(device)
    
    print(f"Dataset prepared: {len(dataset)} samples")
    
    print("Starting training...")
    train_model(model, dataloader, epochs=epochs, device=device)
    
    model.eval()
    with torch.no_grad():
        test_sample = dataset[0].unsqueeze(0).to(device)
        output = model(test_sample)
        
        print(f"\nTest results:")
        print(f"Predicted affine matrix shape: {output['theta'].shape}")
        print(f"Affine matrix:\n{output['theta'][0].cpu().numpy()}")
        print(f"Score: {output['score'][0].item():.4f}")


if __name__ == "__main__":
    main()