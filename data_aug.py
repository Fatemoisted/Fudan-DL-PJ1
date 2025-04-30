import numpy as np
def augment_data(images, labels, num_augmented_samples=30000):
    """
    Augment the training data by applying various transformations.
    
    Args:
        images: Original training images (N, C, H, W) for CNN or (N, H*W) for MLP
        labels: Original training labels
        num_augmented_samples: Number of augmented samples to generate
    
    Returns:
        Augmented images and corresponding labels
    """
    is_cnn = len(images.shape) == 4
    
    # Get original dimensions
    if is_cnn:
        N, C, H, W = images.shape
    else:
        N, D = images.shape
        H = W = int(np.sqrt(D))
    
    # Select random samples to augment
    indices = np.random.choice(N, num_augmented_samples, replace=True)
    
    # Initialize arrays for augmented data
    if is_cnn:
        aug_images = np.zeros((num_augmented_samples, C, H, W), dtype=images.dtype)
    else:
        aug_images = np.zeros((num_augmented_samples, H*W), dtype=images.dtype)
    
    aug_labels = np.zeros(num_augmented_samples, dtype=labels.dtype)
    
    for i, idx in enumerate(indices):
        img = images[idx]
        aug_labels[i] = labels[idx]
        
        # Convert to appropriate shape for transformations
        if is_cnn:
            img_reshaped = img[0]  # (H, W)
        else:
            img_reshaped = img.reshape(H, W)
        
        # Apply random transformation
        transform_type = np.random.randint(0, 2)
        
        if transform_type == 0:
            # Slight rotation (±10 degrees)
            angle = np.random.uniform(-10, 10)
            img_transformed = _rotate_image(img_reshaped, angle)
        elif transform_type == 1:
            # Slight translation (±2 pixels)
            dx, dy = np.random.randint(-2, 3, size=2)
            img_transformed = _translate_image(img_reshaped, dx, dy)
        elif transform_type == 2:
            # Add small amount of noise
            noise_level = np.random.uniform(0, 15)
            img_transformed = img_reshaped + np.random.normal(0, noise_level, img_reshaped.shape)
            img_transformed = np.clip(img_transformed, 0, 255)
        elif transform_type == 3:
            # Slight zoom (0.9-1.1x)
            zoom_factor = np.random.uniform(0.9, 1.1)
            img_transformed = _zoom_image(img_reshaped, zoom_factor)
        else:
            # Small elastic deformation
            img_transformed = _elastic_transform(img_reshaped, alpha=8, sigma=3)
        
        # Store the augmented image
        if is_cnn:
            aug_images[i, 0] = img_transformed
        else:
            aug_images[i] = img_transformed.flatten()
    
    return aug_images, aug_labels

def _rotate_image(image, angle):
    """Rotate an image by a given angle."""
    from scipy.ndimage import rotate
    return rotate(image, angle, reshape=False, mode='nearest')

def _translate_image(image, dx, dy):
    """Translate an image by dx, dy pixels."""
    from scipy.ndimage import shift
    return shift(image, [dy, dx], mode='nearest')

def _zoom_image(image, zoom_factor):
    """Zoom an image by a factor while preserving original dimensions."""
    from scipy.ndimage import zoom
    h, w = image.shape
    
    # Zoom the image
    if zoom_factor < 1:
        # Zoom out
        zoomed_small = zoom(image, zoom_factor, order=1)
        
        # Create an empty output image
        zoomed = np.zeros_like(image)
        
        # Calculate offsets precisely
        new_h, new_w = zoomed_small.shape
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        
        # Place the zoomed image into the center
        zoomed[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = zoomed_small
    else:
        # Zoom in - need to crop precisely to original dimensions
        zoomed_large = zoom(image, zoom_factor, order=1)
        
        # Calculate offsets precisely
        new_h, new_w = zoomed_large.shape
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        
        # Crop from center
        zoomed = zoomed_large[y_offset:y_offset+h, x_offset:x_offset+w]
        
        # Final size check
        if zoomed.shape != image.shape:
            # Handle any edge cases by resizing
            from skimage.transform import resize
            zoomed = resize(zoomed, image.shape, preserve_range=True).astype(image.dtype)
    
    return zoomed

def _elastic_transform(image, alpha=8, sigma=3):
    """Apply elastic deformation to an image while preserving dimensions."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    
    shape = image.shape
    
    # Create random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    # Create coordinate matrices
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # Add displacement fields to the coordinates
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    # Map the coordinates
    transformed = map_coordinates(image, indices, order=1, mode='reflect')
    
    # Ensure the shape is preserved
    result = transformed.reshape(shape)
    
    # Final size check
    if result.shape != image.shape:
        # This is a fallback though it should never happen with the above implementation
        from skimage.transform import resize
        result = resize(result, image.shape, preserve_range=True).astype(image.dtype)
    
    return result