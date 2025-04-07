# Import configuration
from src.config import *

def generate_seo_metadata(story_text, image_files, prompt_text):
    """
    Generates SEO-friendly title, description, and tags for the video.

    Args:
        story_text: The complete story text
        image_files: List of image files created for the story
        prompt_text: The original prompt used to generate the story

    Returns:
        Dictionary containing title, description, and tags
    """
    try:
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        print("✅ Initializing SEO metadata generator client...")
    except Exception as e:
        print(f"🔴 Error initializing SEO metadata generator client: {e}")
        return default_seo_metadata(story_text, prompt_text)

    # Use the same model as prompt generation for metadata
    model = "gemini-2.0-flash-thinking-exp-01-21"

    # Extract the first 1000 characters to give the model a sense of the story
    story_preview = story_text[:1000] + "..." if len(story_text) > 1000 else story_text

    # Create prompt for SEO metadata generation
    seo_prompt = f"""
    I need to create SEO-friendly metadata for a children's story video.

    Here is a preview of the story:
    ```
    {story_preview}
    ```

    Original prompt that generated this story:
    ```
    {prompt_text}
    ```

    Please generate the following in JSON format:
    1. A catchy YouTube-style title (max 60 characters) that will attract families with children
    2. An engaging description (150-300 words) that describes the story, mentions key moments, and includes relevant keywords
    3. A list of 10-15 tags relevant to the content (children's stories, animation, etc.)

    Format your response ONLY as a valid JSON object with keys: "title", "description", and "tags" (as an array).
    """

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=seo_prompt),
            ],
        ),
    ]

    print("⏳ Generating SEO-friendly metadata...")

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
        )

        if response.candidates and response.candidates[0].content:
            response_text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    response_text += part.text

            # Extract the JSON data from the response
            # First, try to find JSON within markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no markdown code blocks, try to extract the entire response
                json_str = response_text

            # Parse the JSON data
            try:
                metadata = json.loads(json_str)
                # Validate the metadata
                if not all(key in metadata for key in ['title', 'description', 'tags']):
                    print("⚠️ Metadata is missing required fields, using fallback...")
                    return default_seo_metadata(story_text, prompt_text)

                print("✅ SEO metadata generated successfully")
                return metadata
            except json.JSONDecodeError:
                print("⚠️ Failed to parse metadata as JSON, using fallback...")
                return default_seo_metadata(story_text, prompt_text)
    except Exception as e:
        print(f"⚠️ Error generating SEO metadata: {e}")
        return default_seo_metadata(story_text, prompt_text)

def default_seo_metadata(story_text, prompt_text):
    """
    Creates default SEO metadata if the AI generation fails.

    Args:
        story_text: The complete story text
        prompt_text: The original prompt used to generate the story

    Returns:
        Dictionary with default title, description, and tags
    """
    # Extract character and setting from the prompt if possible
    import re
    char_setting = re.search(r'about\s+(.*?)\s+going\s+on\s+an\s+adventure\s+in\s+(.*?)(?:\s+in\s+a|\.)', prompt_text)

    character = "an animal"
    setting = "an adventure"

    if char_setting:
        character = char_setting.group(1)
        setting = char_setting.group(2)

    # Create a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

    # Create default metadata
    title = f"Adventure of {character} in {setting} | Children's Story"
    title = title[:60]  # Ensure title is not too long

    # Create a brief description from the beginning of the story
    story_preview = story_text[:500] + "..." if len(story_text) > 500 else story_text
    description = f"""
    Join {character} on an exciting adventure in {setting}!

    {story_preview}

    This animated children's story is perfect for bedtime reading, family story time, or whenever your child wants to explore magical worlds and learn valuable lessons. Watch as our character overcomes challenges and discovers new friends along the way.

    #ChildrensStory #Animation #KidsEntertainment

    Created: {timestamp}
    """

    # Default tags
    tags = [
        "children's story",
        "kids animation",
        "bedtime story",
        "animated story",
        character,
        setting,
        "family friendly",
        "kids entertainment",
        "story time",
        "animated adventure",
        "educational content",
        "preschool",
        "moral story",
        "3D animation",
        "storybook"
    ]

    print("✅ Created default SEO metadata")
    return {
        "title": title,
        "description": description,
        "tags": tags
    }
