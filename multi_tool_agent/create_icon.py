from PIL import Image
import os

def create_optimized_icon():
    """Create an optimized icon from the original matpropai.png"""
    try:
        # Open the original image
        original = Image.open("matpropai.png")
        
        # Convert to RGBA if not already
        if original.mode != 'RGBA':
            original = original.convert('RGBA')
        
        # Create different sizes for better compatibility
        sizes = [16, 32, 64, 128]
        
        for size in sizes:
            # Resize the image
            resized = original.resize((size, size), Image.Resampling.LANCZOS)
            
            # Save as PNG with optimization
            filename = f"matpropai_icon_{size}.png"
            resized.save(filename, "PNG", optimize=True, compress_level=9)
            print(f"Created {filename} ({os.path.getsize(filename)} bytes)")
        
        # Create a favicon.ico file with multiple sizes
        original.resize((32, 32), Image.Resampling.LANCZOS).save(
            "favicon.ico", 
            format='ICO', 
            sizes=[(16, 16), (32, 32)]
        )
        print(f"Created favicon.ico ({os.path.getsize('favicon.ico')} bytes)")
        
        print("Icon optimization complete!")
        
    except Exception as e:
        print(f"Error creating icon: {e}")

if __name__ == "__main__":
    create_optimized_icon()
