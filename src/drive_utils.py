# Import configuration
from src.config import *
from src.seo_metadata import generate_seo_metadata

def save_to_drive(output_path, story_text, image_files, prompt_text, metadata, thumbnail_path):
    """
    Saves the video and related files to Google Drive if running in Colab.
    
    Args:
        output_path: Path to the video file
        story_text: The story text
        image_files: List of image files
        prompt_text: The original prompt
        metadata: The SEO metadata
        thumbnail_path: Path to the thumbnail
        
    Returns:
        Path to the saved folder or None if unsuccessful
    """
    print("\n--- Saving Video to Google Drive ---")
    try:
        # Mount Google Drive if in Colab
        try:
            from google.colab import drive
            drive_already_mounted = False
            try:
                # Check if drive is already mounted
                with open('/content/drive/MyDrive', 'r') as f:
                    drive_already_mounted = True
            except:
                drive.mount('/content/drive')

            # Create directory if it doesn't exist
            save_dir = '/content/drive/MyDrive/GeminiStories'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Generate a filename based on current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            drive_path = f"{save_dir}/gemini_story_{timestamp}.mp4"

            # Copy the file to Google Drive
            import shutil
            shutil.copy(output_path, drive_path)

            print(f"✅ Video saved to Google Drive: {drive_path}")

            # New functionality: Generate and save SEO metadata and thumbnail
            print("\n--- Generating SEO Metadata and Thumbnail ---")

            # Generate SEO metadata if not provided
            if not metadata:
                metadata = generate_seo_metadata(story_text, image_files, prompt_text)

            # Create a separate folder for each video with all its assets
            video_folder = f"{save_dir}/{timestamp}_story"
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)

            # Move the video to its dedicated folder
            video_in_folder_path = f"{video_folder}/video.mp4"
            shutil.copy(output_path, video_in_folder_path)

            # Save metadata files
            title_path = f"{video_folder}/title.txt"
            desc_path = f"{video_folder}/description.txt"
            tags_path = f"{video_folder}/tags.txt"

            # Write metadata to files
            with open(title_path, 'w') as f:
                f.write(metadata['title'])

            with open(desc_path, 'w') as f:
                f.write(metadata['description'])

            with open(tags_path, 'w') as f:
                f.write('\n'.join(metadata['tags']))

            # Save thumbnail to the folder if one was generated
            if thumbnail_path and os.path.exists(thumbnail_path):
                thumb_in_folder_path = f"{video_folder}/thumbnail.jpg"
                shutil.copy(thumbnail_path, thumb_in_folder_path)

            print(f"✅ All video assets saved to dedicated folder: {video_folder}")
            print(f"   - Video: video.mp4")
            print(f"   - Title: {metadata['title']}")
            print(f"   - Tags: {len(metadata['tags'])} tags")
            print(f"   - Description: {len(metadata['description'])} characters")
            if thumbnail_path:
                print(f"   - Thumbnail: thumbnail.jpg")
                
            return video_folder

        except ImportError:
            # This except block is for the "try" that attempts to import google.colab
            print("⚠️ Not running in Google Colab, cannot use Google Drive integration")
            print("💡 You can manually save the video from the temporary location:")
            print(f"   {output_path}")
            return None

    except Exception as e:
        print(f"⚠️ Could not save to Google Drive: {e}")
        print("💡 You can manually download the video from the temporary location:")
        print(f"   {output_path}")
        return None


def create_download_button(output_path):
    """
    Creates an HTML download button for the video if possible.
    
    Args:
        output_path: Path to the video file
    """
    try:
        print("\n--- Download Video ---")
        # Get file size in MB
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        if file_size_mb < 50:  # Only try data URL method for files under 50MB
            with open(output_path, "rb") as video_file:
                video_data = video_file.read()
                b64_data = base64.b64encode(video_data).decode()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                display(HTML(f"""
                <a href="data:video/mp4;base64,{b64_data}"
                   download="gemini_story_{timestamp}.mp4"
                   style="
                       display: inline-block;
                       padding: 10px 20px;
                       background-color: #4CAF50;
                       color: white;
                       text-decoration: none;
                       border-radius: 5px;
                       font-weight: bold;
                       margin-top: 10px;
                   ">
                   Download Video ({file_size_mb:.1f} MB)
                </a>
                """))
        else:
            print("⚠️ Video file is too large for direct download in notebook.")
            print(f"Video size: {file_size_mb:.1f} MB")
            print("Please download it from the location shown above.")
    except Exception as e:
        print(f"⚠️ Could not create download button: {e}")
        print("Please download the video from the path shown above.")
