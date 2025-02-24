import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import uuid
import torch
from diffusers import StableDiffusionPipeline

# Load environment variables (if any)
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Logo Generator",
    page_icon="🎨",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: url("https://images.pexels.com/photos/1166209/pexels-photo-1166209.jpeg?auto=compress&cs=tinysrgb&w=600") no-repeat center center fixed;
        background-size: cover;
        color: black;
    }
    .title {
        color: #010127;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
        background-image: url('generated/full image in the vintage style.png');
        background-size: cover;
        background-position: center;
    }
    .stButton button {
        background: #ff4b2b;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background: #000000;
    }


    .stButton>button {
        width: 100%;
        background-color: #4299e1;
        color: white;
        border-radius: 5px;
        padding: 0.75rem 1.5rem;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #3182ce;
    }
    .style-box {
        border: 2px solid #e2e8f0;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
    }
    .style-box:hover {
        border-color: #4299e1;
        background: #ebf8ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("AI Logo Generator 🎨")
st.markdown("Create unique logos using state-of-the-art diffusion technology")

# Check if the 'generated' folder exists, if not create it
if not os.path.exists('generated'):
    os.makedirs('generated')

# Input form
with st.form("logo_generator"):
    # Business name input
    business_name = st.text_input(
        "Business Name",
        placeholder="Enter your business name...",
        help="The name that will be included in the logo"
    )
    
    # Logo description input
    prompt = st.text_area(
        "Describe your logo",
        placeholder="Enter a detailed description of the logo you want to generate...",
        help="Be specific about colors, style, and elements you want in your logo"
    )

    # Aspect Ratio selection
    aspect_ratio = st.selectbox(
        "Select Aspect Ratio",
        options=["16:9", "4:3", "1:1"],
        help="Choose the aspect ratio for the logo"
    )

    # Background Type selection
    background_type = st.selectbox(
        "Select Background Type",
        options=["Solid Color", "Gradient", "Image"],
        help="Choose the type of background for the logo"
    )

    # Style selection

    st.subheader("Select Style")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        minimalist = st.checkbox("Minimalist", help="Clean, simple designs")
    with col2:
        modern = st.checkbox("Modern", help="Contemporary, trendy designs")
    with col3:
        vintage = st.checkbox("Vintage", help="Retro, classic designs")
    with col4:
        abstract = st.checkbox("Abstract", help="Non-representative designs")

    # Advanced options
    with st.expander("Advanced Options"):
        num_steps = st.slider("Generation Steps", 30, 100, 50, help="More steps = higher quality but slower")
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, help="Higher = more adherent to prompt")

    # Generate button
    submitted = st.form_submit_button("Generate Logo")

# Handle logo generation
if submitted:
    # Validate inputs
    if not business_name:
        st.error("Please enter your business name")
    elif not prompt:
        st.error("Please enter a description for your logo")
    elif not any([minimalist, modern, vintage, abstract]):
        st.error("Please select at least one style")

    else:
        try:
            # Get selected styles
            styles = []
            if minimalist: styles.append("minimalist")
            if modern: styles.append("modern")
            if vintage: styles.append("vintage")
            if abstract: styles.append("abstract")
            
            style_str = " and ".join(styles)
            
            # Create enhanced prompt
            enhanced_prompt = f"Create a logo according to{prompt} for my business called '{business_name}'. The logo should show contain the business name. The background should be {background_type}. The aspect ratio should be {aspect_ratio}. The style of the logo should be {style_str}. Make sure the design is visually appealing and reflects the essence of the business."



            # Show generation status
            with st.status("Generating your logo...") as status:
                st.write("Generating logo using local Hugging Face model...")

                # Load the pre-trained model from Hugging Face (ensure you have installed diffusers library)
                pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
                pipe = pipe.to("cpu")  # Using CPU


                # Generate the image
                image = pipe(prompt=enhanced_prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]

                # Save the image
                filename = f"logo-{uuid.uuid4()}.png"
                filepath = os.path.join('generated', filename)
                image.save(filepath)
                
                status.update(label="Logo generated!", state="complete")
                
                # Display the image
                st.subheader("Generated Logo")
                st.image(image, use_column_width=True)
                
                # Download button for the image
                with open(filepath, "rb") as file:
                    btn = st.download_button(
                        label="Download Logo",
                        data=file,
                        file_name=filename,
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Sidebar information
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This AI Logo Generator uses state-of-the-art diffusion models to create unique logos based on your description.
    
    **Tips for best results:**
    - Be specific about colors and shapes
    - Mention key visual elements
    - Specify the mood or feeling
    - Include industry context
    """)
    
    st.subheader("Examples")
    st.markdown("""
    Try these prompts:
    - "A minimalist mountain peak in blue and white"
    - "Abstract geometric shapes forming a leaf"
    - "Vintage coffee shop logo with warm colors"
    """)




