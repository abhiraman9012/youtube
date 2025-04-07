# Import configuration
from src.config import *

def generate_thumbnail(image_files, story_text, metadata):
    """
    Generates a video thumbnail using one of the generated images and adding text overlay.

    Args:
        image_files: List of images from the story
        story_text: The complete story text
        metadata: The SEO metadata dictionary

    Returns:
        Path to the generated thumbnail
    """
    print("⏳ Generating video thumbnail...")

    try:
        # Select the best image for thumbnail
        # Typically one of the first few images works well as they introduce the character
        if not image_files:
            print("⚠️ No images available for thumbnail generation")
            return None

        # Choose image based on availability - prioritize 2nd image if available (often shows main character clearly)
        thumbnail_base_img = image_files[min(1, len(image_files) - 1)]

        # Create a temporary file for the thumbnail
        thumbnail_path = os.path.join(os.path.dirname(thumbnail_base_img), "thumbnail.jpg")

        # Open the image using PIL
        from PIL import Image, ImageDraw, ImageFont

        # Open and resize the image to standard YouTube thumbnail size (1920x1080) for high quality
        # Then we'll downscale to 1280x720 for the final thumbnail with better quality
        img = Image.open(thumbnail_base_img)
        # First upscale if needed to ensure we have enough details
        if img.width < 1920 or img.height < 1080:
            img = img.resize((1920, 1080), PILImage.LANCZOS)

        # Ensure proper aspect ratio for YouTube thumbnail
        img = img.resize((1280, 720), PILImage.LANCZOS)

        # Create a drawing context
        draw = ImageDraw.Draw(img)

        # Try to load a font, with fallback to default
        try:
            # Try to find a suitable font
            font_path = None

            # List of common system fonts to try
            font_names = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
                '/System/Library/Fonts/Supplemental/Arial Bold.ttf',     # macOS
                'C:\\Windows\\Fonts\\arialbd.ttf',                       # Windows
                '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',  # Another Linux option
            ]

            for font_name in font_names:
                if os.path.exists(font_name):
                    font_path = font_name
                    break

            # Use the font if found, otherwise will use default
            if font_path:
                # Title font (large)
                title_font = ImageFont.truetype(font_path, 60)
                # Get the title from metadata
                title = metadata['title']

                # Measure text size to position it
                text_width = draw.textlength(title, font=title_font)

                # Add semi-transparent background for better readability
                # Draw a rectangle at the bottom for the title
                rectangle_height = 120
                draw.rectangle(
                    [(0, img.height - rectangle_height), (img.width, img.height)],
                    fill=(0, 0, 0, 180)  # Semi-transparent black
                )

                # Draw the title text
                draw.text(
                    (img.width / 2 - text_width / 2, img.height - rectangle_height / 2 - 30),
                    title,
                    font=title_font,
                    fill=(255, 255, 255)  # White color
                )

                # Add a small banner at the top for "Children's Story"
                draw.rectangle(
                    [(0, 0), (img.width, 80)],
                    fill=(0, 0, 0, 150)  # Semi-transparent black
                )

                # Use a smaller font for the banner
                banner_font = ImageFont.truetype(font_path, 40)
                banner_text = "Children's Story Animation"
                banner_width = draw.textlength(banner_text, font=banner_font)

                draw.text(
                    (img.width / 2 - banner_width / 2, 20),
                    banner_text,
                    font=banner_font,
                    fill=(255, 255, 255)  # White color
                )
            else:
                print("⚠️ Could not find a suitable font, using basic text overlay")
                # Use PIL's default font
                # Add semi-transparent black rectangles for text placement
                draw.rectangle(
                    [(0, img.height - 100), (img.width, img.height)],
                    fill=(0, 0, 0, 180)
                )
                draw.rectangle(
                    [(0, 0), (img.width, 80)],
                    fill=(0, 0, 0, 150)
                )

                # Add text - simplified when no font is available
                draw.text(
                    (40, img.height - 80),
                    metadata['title'][:50],
                    fill=(255, 255, 255)
                )
                draw.text(
                    (40, 30),
                    "Children's Story Animation",
                    fill=(255, 255, 255)
                )

        except Exception as font_error:
            print(f"⚠️ Error with font rendering: {font_error}")
            # Add basic text using default settings
            draw.rectangle(
                [(0, img.height - 100), (img.width, img.height)],
                fill=(0, 0, 0, 180)
            )
            draw.text(
                (40, img.height - 80),
                metadata['title'][:50],
                fill=(255, 255, 255)
            )

        # Save the thumbnail
        img.save(thumbnail_path, quality=95)
        print(f"✅ Thumbnail generated and saved to: {thumbnail_path}")

        return thumbnail_path

    except Exception as e:
        print(f"⚠️ Error generating thumbnail: {e}")
        return None
