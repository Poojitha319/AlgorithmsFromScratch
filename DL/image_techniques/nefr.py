import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

@torch.no_grad()
def visualize_rendered_image(near_plane, far_plane, dataset, chunk_size=10, image_idx=0, num_bins=192, height=400, width=400, model=None, device='cuda'):
    """
    Renders and saves an image by processing rays in chunks for memory efficiency.

    Args:
        near_plane (float): Near plane distance.
        far_plane (float): Far plane distance.
        dataset (torch.Tensor): Dataset containing ray origins and directions.
        chunk_size (int, optional): Number of rays to process per chunk. Defaults to 10.
        image_idx (int, optional): Index of the image to render. Defaults to 0.
        num_bins (int, optional): Number of bins for density estimation. Defaults to 192.
        height (int, optional): Height of the image. Defaults to 400.
        width (int, optional): Width of the image. Defaults to 400.
        model (nn.Module, optional): NeRF model to use for rendering. Must be provided. Defaults to None.
        device (str, optional): Device to perform computations on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        None
    """
    if model is None:
        raise ValueError("NeRF model must be provided for rendering.")

    # Extract ray origins and directions for the specified image
    start_idx = image_idx * height * width
    end_idx = (image_idx + 1) * height * width
    ray_origins = dataset[start_idx:end_idx, :3].to(device)
    ray_directions = dataset[start_idx:end_idx, 3:6].to(device)

    rendered_pixels = []  # List to store regenerated pixel values

    num_chunks = int(np.ceil(height / chunk_size))
    for chunk in range(num_chunks):
        # Define the range for the current chunk
        chunk_start = chunk * width * chunk_size
        chunk_end = min((chunk + 1) * width * chunk_size, ray_origins.shape[0])

        # Extract chunk of rays
        origins_chunk = ray_origins[chunk_start:chunk_end]
        directions_chunk = ray_directions[chunk_start:chunk_end]

        # Render pixels for the current chunk
        pixels_chunk = render_rays(model, origins_chunk, directions_chunk, near_plane=near_plane, far_plane=far_plane, num_bins=num_bins)
        rendered_pixels.append(pixels_chunk)

    # Concatenate all chunks and reshape to image dimensions
    image = torch.cat(rendered_pixels).cpu().numpy().reshape(height, width, 3)

    # Plot and save the rendered image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'novel_views/rendered_image_{image_idx}.png', bbox_inches='tight')
    plt.close()

class NeRF(nn.Module):
    def __init__(self, pos_encoding_dim=10, dir_encoding_dim=4, hidden_dim=256):
        """
        Initializes the NeRF model with positional and directional encoding.

        Args:
            pos_encoding_dim (int, optional): Dimensionality of positional encoding. Defaults to 10.
            dir_encoding_dim (int, optional): Dimensionality of directional encoding. Defaults to 4.
            hidden_dim (int, optional): Number of units in hidden layers. Defaults to 256.
        """
        super(NeRF, self).__init__()
        
        # Encoding dimensions
        self.pos_encoding_dim = pos_encoding_dim
        self.dir_encoding_dim = dir_encoding_dim

        # Block 1: Position Encoding to Hidden
        self.position_encoder = nn.Sequential(
            nn.Linear(pos_encoding_dim * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Block 2: Density Estimation
        self.density_estimator = nn.Sequential(
            nn.Linear(pos_encoding_dim * 6 + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )

        # Block 3: Color Estimation
        self.color_estimator = nn.Sequential(
            nn.Linear(dir_encoding_dim * 6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
        )
        self.color_output = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        # Activation function
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, num_freqs):
        """
        Applies positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3].
            num_freqs (int): Number of frequency bands.

        Returns:
            torch.Tensor: Positional encoded tensor.
        """
        encoding = [x]
        for i in range(num_freqs):
            for func in [torch.sin, torch.cos]:
                encoding.append(func((2 ** i) * x))
        return torch.cat(encoding, dim=1)

    def forward(self, positions, directions):
        """
        Forward pass of the NeRF model.

        Args:
            positions (torch.Tensor): Tensor of ray positions [batch_size, 3].
            directions (torch.Tensor): Tensor of ray directions [batch_size, 3].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted colors and densities.
        """
        # Positional Encoding
        pos_encoded = self.positional_encoding(positions, self.pos_encoding_dim)  # [batch_size, pos_encoding_dim * 6 + 3]
        dir_encoded = self.positional_encoding(directions, self.dir_encoding_dim)  # [batch_size, dir_encoding_dim * 6 + 3]

        # Position to Hidden
        hidden = self.position_encoder(pos_encoded)  # [batch_size, hidden_dim]

        # Density Estimation
        density_input = torch.cat((hidden, pos_encoded), dim=1)  # [batch_size, hidden_dim + pos_encoding_dim * 6 + 3]
        density_output = self.density_estimator(density_input)  # [batch_size, hidden_dim + 1]
        hidden_density, sigma = density_output[:, :-1], self.relu(density_output[:, -1:])  # [batch_size, hidden_dim], [batch_size, 1]

        # Color Estimation
        color_input = torch.cat((hidden_density, dir_encoded), dim=1)  # [batch_size, hidden_dim + dir_encoding_dim * 6 + 3]
        color_hidden = self.color_estimator(color_input)  # [batch_size, hidden_dim // 2]
        color = self.color_output(color_hidden)  # [batch_size, 3]

        return color, sigma

def compute_transmittance(alphas):
    """
    Computes the accumulated transmittance for each ray.

    Args:
        alphas (torch.Tensor): Alpha values [batch_size, num_bins].

    Returns:
        torch.Tensor: Accumulated transmittance [batch_size, num_bins].
    """
    accumulated_transmittance = torch.cumprod(alphas, dim=1)
    # Prepend ones to handle the first transmittance value
    return torch.cat((torch.ones((accumulated_transmittance.size(0), 1), device=alphas.device), 
                      accumulated_transmittance[:, :-1]), dim=1)

def render_rays(nerf_model, ray_origins, ray_directions, near_plane=0.0, far_plane=0.5, num_bins=192):
    """
    Renders pixels by sampling points along rays and integrating colors and densities.

    Args:
        nerf_model (nn.Module): Trained NeRF model.
        ray_origins (torch.Tensor): Ray origins [batch_size, 3].
        ray_directions (torch.Tensor): Ray directions [batch_size, 3].
        near_plane (float, optional): Near plane distance. Defaults to 0.0.
        far_plane (float, optional): Far plane distance. Defaults to 0.5.
        num_bins (int, optional): Number of samples per ray. Defaults to 192.

    Returns:
        torch.Tensor: Rendered pixel colors [batch_size, 3].
    """
    device = ray_origins.device

    # Generate evenly spaced sample points along each ray
    t_vals = torch.linspace(near_plane, far_plane, num_bins, device=device).unsqueeze(0).expand(ray_origins.size(0), num_bins)  # [batch_size, num_bins]

    # Stratified sampling with random perturbations
    mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])  # [batch_size, num_bins-1]
    lower = t_vals[:, :-1]
    upper = t_vals[:, 1:]
    u = torch.rand(t_vals.shape, device=device)
    t_samples = lower + (upper - lower) * u  # [batch_size, num_bins]

    # Compute deltas between consecutive samples
    deltas = torch.cat((t_samples[:, 1:] - t_samples[:, :-1], torch.full((t_samples.size(0), 1), 1e10, device=device)), dim=1)  # [batch_size, num_bins]

    # Compute 3D points along each ray
    points = ray_origins.unsqueeze(1) + t_samples.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, num_bins, 3]

    # Flatten points and directions for batch processing
    points_flat = points.view(-1, 3)  # [batch_size * num_bins, 3]
    directions_expanded = ray_directions.unsqueeze(1).expand(-1, num_bins, -1).reshape(-1, 3)  # [batch_size * num_bins, 3]

    # Pass points and directions through NeRF model
    colors_flat, densities_flat = nerf_model(points_flat, directions_expanded)  # [batch_size * num_bins, 3], [batch_size * num_bins, 1]

    # Reshape outputs back to [batch_size, num_bins, ...]
    colors = colors_flat.view(ray_origins.size(0), num_bins, 3)  # [batch_size, num_bins, 3]
    densities = densities_flat.view(ray_origins.size(0), num_bins)  # [batch_size, num_bins]

    # Compute alpha values
    alpha = 1.0 - torch.exp(-densities * deltas)  # [batch_size, num_bins]

    # Compute weights for each sample
    transmittance = compute_transmittance(1.0 - alpha)  # [batch_size, num_bins]
    weights = transmittance * alpha  # [batch_size, num_bins]

    # Compute final pixel colors as weighted sum of sampled colors
    pixel_colors = torch.sum(weights.unsqueeze(2) * colors, dim=1)  # [batch_size, 3]

    # Regularization for white background
    weight_sum = torch.sum(weights, dim=1, keepdim=True)  # [batch_size, 1]
    pixel_colors += (1.0 - weight_sum).clamp(0.0, 1.0)

    return pixel_colors

def train_nerf(nerf_model, optimizer, scheduler, data_loader, device='cuda', near_plane=0.0, far_plane=1.0,
              num_epochs=100000, num_bins=192, height=400, width=400, testing_dataset=None):
    """
    Trains the NeRF model using the provided data loader.

    Args:
        nerf_model (nn.Module): NeRF model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        data_loader (DataLoader): DataLoader for training data.
        device (str, optional): Device to perform computations on ('cuda' or 'cpu'). Defaults to 'cuda'.
        near_plane (float, optional): Near plane distance. Defaults to 0.0.
        far_plane (float, optional): Far plane distance. Defaults to 1.0.
        num_epochs (int, optional): Number of training epochs. Defaults to 100000.
        num_bins (int, optional): Number of samples per ray. Defaults to 192.
        height (int, optional): Height of the rendered images. Defaults to 400.
        width (int, optional): Width of the rendered images. Defaults to 400.
        testing_dataset (torch.Tensor, optional): Dataset for testing/rendering images. Defaults to None.

    Returns:
        list: List of training loss values.
    """
    training_losses = []
    progress_bar = tqdm(range(num_epochs), desc="Training NeRF")

    for epoch in progress_bar:
        for batch in data_loader:
            # Split batch into ray origins, directions, and ground truth pixel values
            ray_origins = batch[:, :3].to(device)          # [batch_size, 3]
            ray_directions = batch[:, 3:6].to(device)     # [batch_size, 3]
            ground_truth_colors = batch[:, 6:].to(device) # [batch_size, 3]

            # Render colors using the NeRF model
            predicted_colors = render_rays(nerf_model, ray_origins, ray_directions,
                                          near_plane=near_plane, far_plane=far_plane, num_bins=num_bins)  # [batch_size, 3]

            # Compute Mean Squared Error loss
            loss = torch.nn.functional.mse_loss(predicted_colors, ground_truth_colors)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())

        # Update learning rate
        scheduler.step()

        # Periodically render and save test images
        if testing_dataset is not None and epoch % 1000 == 0:
            for img_idx in range(200):
                visualize_rendered_image(near_plane, far_plane, testing_dataset, image_idx=img_idx,
                                         num_bins=num_bins, height=height, width=width, model=nerf_model, device=device)

    return training_losses

if __name__ == '__main__':
    # Configuration Parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAINING_DATA_PATH = 'training_data.pkl'
    TESTING_DATA_PATH = 'testing_data.pkl'
    HIDDEN_DIM = 256
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 1024
    NUM_EPOCHS = 16
    NEAR_PLANE = 2.0
    FAR_PLANE = 6.0
    NUM_BINS = 192
    IMAGE_HEIGHT = 400
    IMAGE_WIDTH = 400

    # Load datasets
    training_dataset = torch.from_numpy(np.load(TRAINING_DATA_PATH, allow_pickle=True)).float()
    testing_dataset = torch.from_numpy(np.load(TESTING_DATA_PATH, allow_pickle=True)).float()

    # Initialize NeRF model, optimizer, and scheduler
    nerf_model = NeRF(embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    # Create DataLoader for training
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Start training
    training_losses = train_nerf(nerf_model, optimizer, scheduler, training_loader, device=DEVICE,
                                  near_plane=NEAR_PLANE, far_plane=FAR_PLANE, num_epochs=NUM_EPOCHS,
                                  num_bins=NUM_BINS, height=IMAGE_HEIGHT, width=IMAGE_WIDTH,
                                  testing_dataset=testing_dataset)
