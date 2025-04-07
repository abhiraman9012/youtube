# Import all modules
from src.config import *
from src.prompt_generator import generate_prompt
from src.story_utils import collect_complete_story
from src.seo_metadata import generate_seo_metadata, default_seo_metadata
from src.thumbnail_generator import generate_thumbnail
from src.video_generator import create_video
from src.drive_utils import save_to_drive, create_download_button

# Generate function
def generate(use_prompt_generator=True, prompt_input="Create a unique children's story with a different animal character, setting, and adventure theme."):
    try:
        client = genai.Client(
             api_key=os.environ.get("GEMINI_API_KEY"),
        )
        print("✅ Initializing client using genai.Client...")
    except AttributeError:
        print("🔴 FATAL ERROR: genai.Client is unexpectedly unavailable.")
        return
    except Exception as e:
        print(f"🔴 Error initializing client: {e}")
        return
    print("✅ Client object created successfully.")

    model = "gemini-2.0-flash-exp-image-generation"

    # --- Modified Prompt ---
    if use_prompt_generator:
        print("🧠 Using prompt generator model first...")
        generated_prompt = generate_prompt(prompt_input)
        if generated_prompt and generated_prompt.strip():
            prompt_text = generated_prompt
            print("✅ Using AI-generated prompt for story and image creation")
        else:
            print("⚠️ Prompt generation failed or returned empty, using default prompt")
            prompt_text = """Generate a story about a white baby goat named Pip going on an adventure in a farm in a highly detailed 3d cartoon animation style. For each scene, generate a high-quality, photorealistic image **in landscape orientation suitable for a widescreen (16:9 aspect ratio) YouTube video**. Ensure maximum detail, vibrant colors, and professional lighting."""
    else:
        prompt_text = """Generate a story about a white baby goat named Pip going on an adventure in a farm in a highly detailed 3d cartoon animation style. For each scene, generate a high-quality, photorealistic image **in landscape orientation suitable for a widescreen (16:9 aspect ratio) YouTube video**. Ensure maximum detail, vibrant colors, and professional lighting."""
    # --- End Modified Prompt ---

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=["image", "text"],
        response_mime_type="text/plain",
        safety_settings=safety_settings,
    )

    print(f"ℹ️ Using Model: {model}")
    print(f"📝 Using Prompt: {prompt_text}") # Show the modified prompt
    print(f"⚙️ Using Config (incl. safety): {generate_content_config}")
    print("⏳ Calling client.models.generate_content_stream...")

    try:
        # Create a temporary directory to store images and audio
        temp_dir = tempfile.mkdtemp()

        # Variables to collect story and images
        story_text = ""
        image_files = []

        try:
            # Flag to determine if we should use streaming or fallback approach
            use_streaming = True

            try:
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
            except json.decoder.JSONDecodeError as je:
                print(f"⚠️ JSON decoding error during stream creation: {je}")
                print("Trying fallback to non-streaming API call...")
                use_streaming = False

                # Fallback to non-streaming version
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    )

                    # Process the non-streaming response
                    print("Using non-streaming response instead")
                    image_found = False

                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_found = True
                                inline_data = part.inline_data
                                
                                # Handle both direct access and getter method patterns
                                if hasattr(inline_data, 'data'):
                                    image_data = inline_data.data
                                elif hasattr(inline_data, 'get_data') and callable(inline_data.get_data):
                                    image_data = inline_data.get_data()
                                else:
                                    # Try to access as dictionary-like object
                                    try:
                                        image_data = inline_data['data']
                                    except:
                                        print("⚠️ Could not extract image data from inline_data object")
                                        continue
                                
                                # Handle mime_type similarly
                                if hasattr(inline_data, 'mime_type'):
                                    mime_type = inline_data.mime_type
                                else:
                                    try:
                                        mime_type = inline_data['mime_type']
                                    except:
                                        mime_type = "image/jpeg"  # Default

                                # Save image to a temporary file
                                img_path = os.path.join(temp_dir, f"image_{len(image_files)}.jpg")
                                with open(img_path, "wb") as f:
                                    f.write(base64.b64decode(image_data) if isinstance(image_data, str) else image_data)
                                image_files.append(img_path)

                                print(f"\n\n🖼️ --- Image Received ({mime_type}) ---")
                                try:
                                    display(Image(data=base64.b64decode(image_data) if isinstance(image_data, str) else image_data))
                                except:
                                    print("⚠️ Could not display image, but it's saved to file")
                                print("--- End Image ---\n")
                            elif hasattr(part, 'text') and part.text:
                                print(part.text)
                                story_text += part.text

                    # Skip the streaming loop since we already processed the response
                    print("✅ Non-streaming processing complete.")
                    if not image_found:
                        print("⚠️ No images were found in the non-streaming response.")

                    # Continue with audio and video processing
                    image_found = True  # Set this to true to prevent early exit

                except Exception as e:
                    print(f"⚠️ Fallback API call also failed: {e}")
                    return

            except Exception as e:
                print(f"⚠️ Error creating stream: {e}")
                return

            # Only enter the streaming loop if we're using streaming
            if use_streaming:
                image_found = False
                print("--- Response Stream ---")

                # Track JSON parsing errors to decide when to fallback
                json_errors = 0
                max_json_errors = 5  # Allow up to 5 errors before giving up on streaming

                try:
                    for chunk in stream:
                        try:
                            # If we get a raw string instead of parsed content
                            if isinstance(chunk, str):
                                print(chunk, end="")
                                story_text += chunk
                                continue

                            # Check if chunk has candidates
                            if not hasattr(chunk, 'candidates') or not chunk.candidates:
                                # Try to extract as much as possible from the chunk
                                if hasattr(chunk, 'text') and chunk.text:
                                    print(chunk.text, end="")
                                    story_text += chunk.text
                                continue

                            if not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                                if hasattr(chunk, 'text') and chunk.text:
                                    print(chunk.text, end="")
                                    story_text += chunk.text
                                continue

                            part = chunk.candidates[0].content.parts[0]

                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_found = True
                                inline_data = part.inline_data
                                # Handle both direct access and getter method patterns
                                if hasattr(inline_data, 'data'):
                                    image_data = inline_data.data
                                elif hasattr(inline_data, 'get_data') and callable(inline_data.get_data):
                                    image_data = inline_data.get_data()
                                else:
                                    # Try to access as dictionary-like object
                                    try:
                                        image_data = inline_data['data']
                                    except:
                                        print("⚠️ Could not extract image data from inline_data object")
                                        continue
                                
                                # Handle mime_type similarly
                                if hasattr(inline_data, 'mime_type'):
                                    mime_type = inline_data.mime_type
                                else:
                                    try:
                                        mime_type = inline_data['mime_type']
                                    except:
                                        mime_type = "image/jpeg"  # Default

                                # Save image to a temporary file
                                img_path = os.path.join(temp_dir, f"image_{len(image_files)}.jpg")
                                with open(img_path, "wb") as f:
                                    f.write(base64.b64decode(image_data) if isinstance(image_data, str) else image_data)
                                image_files.append(img_path)

                                print(f"\n\n🖼️ --- Image Received ({mime_type}) ---")
                                try:
                                    display(Image(data=base64.b64decode(image_data) if isinstance(image_data, str) else image_data))
                                except:
                                    print("⚠️ Could not display image, but it's saved to file")
                                print("--- End Image ---\n")
                            elif hasattr(part,'text') and part.text:
                                print(part.text, end="")
                                story_text += part.text
                        except json.decoder.JSONDecodeError as je:
                            print(f"\n⚠️ JSON decoding error in chunk: {je}")
                            json_errors += 1
                            if json_errors >= max_json_errors:
                                print(f"Too many JSON errors ({json_errors}), falling back to non-streaming mode...")
                                # Try to extract any text that might be in the raw response
                                try:
                                    if hasattr(chunk, '_response') and hasattr(chunk._response, 'text'):
                                        raw_text = chunk._response.text
                                        # Extract text content between markdown or code blocks if possible
                                        story_text += re.sub(r'```.*?```', '', raw_text, flags=re.DOTALL)
                                        print(f"Extracted {len(raw_text)} characters from raw response")
                                except Exception:
                                    pass
                                break  # Exit the streaming loop and use the fallback
                            continue  # Skip this chunk and continue with next
                        except Exception as e:
                            print(f"\n⚠️ Error processing chunk: {e}")
                            continue  # Skip this chunk and continue with next
                except Exception as e:
                    print(f"\n⚠️ Error in stream processing: {e}")

                    # If streaming failed completely, try the non-streaming fallback
                    if not story_text.strip() and json_errors > 0:
                        print("Stream processing failed, trying non-streaming fallback...")
                        try:
                            response = client.models.generate_content(
                                model=model,
                                contents=contents,
                                config=generate_content_config,
                            )

                            if response.candidates and response.candidates[0].content:
                                for part in response.candidates[0].content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        story_text += part.text
                                        print(part.text)

                            print("✅ Non-streaming fallback successful")
                        except Exception as fallback_error:
                            print(f"⚠️ Non-streaming fallback also failed: {fallback_error}")
        except Exception as e:
            print(f"⚠️ Error in stream creation: {e}")
            return

        print("\n" + "-"*20)
        if not image_found:
             print("⚠️ No images were found in the stream.")
        print("✅ Stream processing complete.")

        # After generating story and images, create audio
        if story_text and image_files:
            print("\n--- Starting Text-to-Speech Generation with Kokoro ---")
            try:
                # First collect and clean the complete story
                complete_story = collect_complete_story(story_text)

                # Check if we have enough segments for a complete story
                story_segments = collect_complete_story(story_text, return_segments=True)
                print(f"Story has {len(story_segments)} segments")

                # Check if we have matching image count (each segment should have one image)
                segments_count = len(story_segments)
                images_count = len(image_files)

                print(f"Story segments: {segments_count}, Images: {images_count}")

                # If we don't have enough segments or have mismatched images, try to regenerate
                retry_count = 0
                max_retries = 3
                min_segments = 6  # Require at least 6 segments for a complete story

                # Define conditions for regeneration
                needs_regeneration = (segments_count < min_segments) or (images_count < segments_count)

                while needs_regeneration and retry_count < max_retries:
                    retry_count += 1

                    if segments_count < min_segments:
                        print(f"\n⚠️ Story has only {segments_count} segments, which is less than the required {min_segments}.")

                    if images_count < segments_count:
                        print(f"\n⚠️ Mismatch between story segments ({segments_count}) and images ({images_count}).")

                    print(f"Attempting to regenerate a more detailed story with complete images (attempt {retry_count}/{max_retries})...")

                    # Modify prompt to encourage a complete story with images for each segment
                    enhanced_prompt = prompt_text
                    if "with at least 6 detailed scenes" not in enhanced_prompt:
                        # Add more specific instructions to generate a longer story with images
                        enhanced_prompt = enhanced_prompt.replace(
                            "Generate a story about",
                            "Generate a detailed story with at least 6 scenes about"
                        )
                    if "with one image per scene" not in enhanced_prompt:
                        enhanced_prompt += " Please create one clear image for each scene in the story."

                    # Retry with the enhanced prompt
                    retry_contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=enhanced_prompt),
                            ],
                        ),
                    ]

                    # Clear previous results
                    story_text_retry = ""
                    image_files_retry = []

                    try:
                        # Try non-streaming for retries as it's more reliable
                        retry_response = client.models.generate_content(
                            model=model,
                            contents=retry_contents,
                            config=generate_content_config,
                        )

                        if retry_response.candidates and retry_response.candidates[0].content:
                            for part in retry_response.candidates[0].content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    inline_data = part.inline_data
                                    
                                    # Handle both direct access and getter method patterns
                                    if hasattr(inline_data, 'data'):
                                        image_data = inline_data.data
                                    elif hasattr(inline_data, 'get_data') and callable(inline_data.get_data):
                                        image_data = inline_data.get_data()
                                    else:
                                        # Try to access as dictionary-like object
                                        try:
                                            image_data = inline_data['data']
                                        except:
                                            print("⚠️ Could not extract image data from inline_data object")
                                            continue
                                    
                                    # Handle mime_type similarly
                                    if hasattr(inline_data, 'mime_type'):
                                        mime_type = inline_data.mime_type
                                    else:
                                        try:
                                            mime_type = inline_data['mime_type']
                                        except:
                                            mime_type = "image/jpeg"  # Default

                                    # Save image to a temporary file
                                    img_path = os.path.join(temp_dir, f"image_retry_{len(image_files_retry)}.jpg")
                                    with open(img_path, "wb") as f:
                                        f.write(base64.b64decode(image_data) if isinstance(image_data, str) else image_data)
                                    image_files_retry.append(img_path)

                                    print(f"\n\n🖼️ --- Retry Image Received ({mime_type}) ---")
                                    try:
                                        display(Image(data=base64.b64decode(image_data) if isinstance(image_data, str) else image_data))
                                    except:
                                        print("⚠️ Could not display image, but it's saved to file")
                                    print("--- End Image ---\n")
                                elif hasattr(part, 'text') and part.text:
                                    print(part.text)
                                    story_text_retry += part.text

                        # Check if the retry generated enough content AND enough images
                        if story_text_retry:
                            story_segments = collect_complete_story(story_text_retry, return_segments=True)
                            segments_count = len(story_segments)
                            images_count = len(image_files_retry)

                            print(f"Retry generated {segments_count} segments and {images_count} images")

                            # Verify that we have sufficient segments AND images
                            if segments_count >= min_segments and images_count >= segments_count * 0.8:  # Allow for some missing images (80% coverage)
                                story_text = story_text_retry
                                if image_files_retry:
                                    image_files = image_files_retry
                                complete_story = collect_complete_story(story_text)
                                print("✅ Successfully regenerated a more detailed story with images")
                                needs_regeneration = False
                            else:
                                print("⚠️ Regenerated story still doesn't meet requirements")

                                # If we have good segment count but poor image count, keep trying
                                if segments_count >= min_segments and images_count < segments_count * 0.8:
                                    print("Generated enough segments but not enough images. Retrying...")
                                    # We'll continue the loop to try again
                    except Exception as retry_error:
                        print(f"⚠️ Error during story regeneration: {retry_error}")

                print("⏳ Converting complete story to speech...")
                print("Story to be converted:", complete_story[:100] + "...")

                # Initialize Kokoro pipeline
                pipeline = KPipeline(lang_code='a')

                try:
                    # Generate audio for the complete story
                    print("Full story length:", len(complete_story), "characters")
                    generator = pipeline(complete_story, voice='af_heart')

                    # Save the complete audio file
                    audio_path = os.path.join(temp_dir, "complete_story.wav")

                    # Process and save all audio chunks
                    all_audio = []
                    for _, (gs, ps, audio) in enumerate(generator):
                        all_audio.append(audio)

                    # Combine all audio chunks
                    if all_audio:
                        combined_audio = np.concatenate(all_audio)
                        sf.write(audio_path, combined_audio, 24000)
                        print(f"✅ Complete story audio saved to: {audio_path}")
                        print("🔊 Playing complete story audio:")
                        display(Audio(data=combined_audio, rate=24000))

                except Exception as e:
                    print(f"⚠️ Error in text-to-speech generation: {e}")
                    return

            except Exception as e:
                print(f"⚠️ Kokoro TTS failed: {e}")

            bark_audio_success = False

            # Create video from images and audio
            output_path = create_video(temp_dir, story_text, image_files, audio_path)
            
            if output_path:
                # Generate SEO metadata
                metadata = generate_seo_metadata(story_text, image_files, prompt_text)
                
                # Generate thumbnail
                thumbnail_path = generate_thumbnail(image_files, story_text, metadata)
                
                # Save to Google Drive if in Colab
                save_to_drive(output_path, story_text, image_files, prompt_text, metadata, thumbnail_path)
                
                # Create download button
                create_download_button(output_path)

    except Exception as e:
        print(f"\n🛑 An error occurred during streaming or processing: {e}")
        import traceback
        traceback.print_exc()


# --- Run the function ---
if __name__ == "__main__":
    print("--- Starting generation (attempting 16:9 via prompt) ---")
    # You can set use_prompt_generator=True to enable the prompt generator model
    # You can also customize the prompt_input to guide the prompt generator
    generate(use_prompt_generator=True)
    print("--- Generation function finished ---")
