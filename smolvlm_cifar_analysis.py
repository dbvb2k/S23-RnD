import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import time
import os
from datetime import datetime
import csv

NUM_DASHES = 150

def transform_to_pil(tensor_image):
    """Convert a tensor image to PIL Image."""
    img = tensor_image.cpu().numpy().transpose((1, 2, 0))
    img = ((img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    return Image.fromarray(img)

def clean_text_for_csv(text):
    """Clean text for CSV format by removing newlines and extra spaces."""
    if text is None:
        return ""
    return ' '.join(str(text).replace('\n', ' ').split())

def main():
    print("\n=== SmolVLM2 CIFAR10 Image Analysis ===")
    print("Starting the analysis process...\n")

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"infer_{timestamp}.csv")
    
    print(f"Results will be saved to: {output_file}\n")

    # Load CIFAR10 dataset
    print("Step 1: Loading CIFAR10 dataset...")
    print("-"*NUM_DASHES)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    start_time = time.time()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    print(f"✓ Dataset loaded successfully! ({len(trainset)} images available)")
    print(f"✓ Available classes: {', '.join(trainset.classes)}\n")
    
    # Load SmolVLM2 model and processor
    print("Step 2: Loading SmolVLM2 model and processor...")
    print("-"*NUM_DASHES)
    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    
    try:
        print(f"• Loading processor from {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        print("✓ Processor loaded successfully!")
        
        print(f"• Loading model from {model_name}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"✓ Model loaded successfully! (Using device: {model.device})\n")
        
        # Sample 50 random images
        print("Step 3: Preparing image analysis...")
        print("-"*NUM_DASHES)
        sample_indices = random.sample(range(len(trainset)), 50)
        
        # Questions to ask about each image
        questions = [
            # Object Identification
            "What is the main object in this image?",
            "How many distinct objects can you identify in the image?",
            "Are there any secondary objects in the background?",
            "What is the approximate size of the main object?",
            "Is the main object complete or partially visible?",
            "Can you identify any brand names or logos?",
            "Are there any text or numbers visible in the image?",
            "What type of vehicle or transportation is shown, if any?",
            "Are there any animals or living creatures in the image?",
            "Can you identify any technological devices or equipment?",

            # Color and Lighting
            "What colors are present in the image?",
            "What is the dominant color in this image?",
            "How would you describe the lighting conditions?",
            "Are there any shadows visible in the image?",
            "Is this image bright or dark overall?",
            "Are there any reflective surfaces in the image?",
            "How would you describe the contrast in this image?",
            "Are the colors vibrant or muted?",
            "Is there any color gradient visible?",
            "How does the lighting affect the mood of the image?",

            # Composition and Layout
            "Describe the background of the image.",
            "Is the main subject centered in the frame?",
            "How would you describe the composition of this image?",
            "Is there a clear foreground and background separation?",
            "What perspective is this image taken from?",
            "Is the image symmetrical or asymmetrical?",
            "How much of the frame does the main subject occupy?",
            "Are there any leading lines in the composition?",
            "Is the image cluttered or minimalist?",
            "How would you describe the depth of field?",

            # Texture and Pattern
            "What textures can you identify in the image?",
            "Are there any recurring patterns visible?",
            "How would you describe the surface texture of the main object?",
            "Are there any natural or artificial patterns?",
            "Is there any visible grain or noise in the image?",
            "Can you identify any geometric shapes or patterns?",
            "Are there any interesting textural contrasts?",
            "How does texture contribute to the image's depth?",
            "Are there any smooth or rough surfaces visible?",
            "Can you identify any architectural patterns?",

            # Motion and Action
            "Is there any sense of movement in the image?",
            "Does the image appear to be in motion or static?",
            "Are there any action elements in the scene?",
            "How is motion portrayed in this image, if at all?",
            "Is there any implied direction of movement?",
            "Are there any motion blur effects?",
            "Does the composition suggest movement?",
            "Are there any frozen action moments?",
            "How does the image capture dynamic elements?",
            "Is there any suggestion of past or future movement?",

            # Environment and Context
            "What type of environment is shown in the image?",
            "Is this an indoor or outdoor scene?",
            "What time of day does this appear to be?",
            "What season is suggested by the image?",
            "Are there any weather conditions visible?",
            "What is the general setting of this image?",
            "Can you identify the location type?",
            "Are there any environmental elements visible?",
            "How would you describe the atmosphere?",
            "What context clues are present in the image?",

            # Technical Aspects
            "How would you rate the image quality?",
            "Is the image in focus or blurry?",
            "Are there any visible artifacts or distortions?",
            "How would you describe the resolution?",
            "Is there any visible noise or grain?",
            "How is the white balance in this image?",
            "Are there any lens effects visible?",
            "How would you describe the sharpness?",
            "Are there any exposure issues?",
            "Can you identify any post-processing effects?",

            # Emotional and Aesthetic
            "What mood does this image convey?",
            "How does this image make you feel?",
            "What is the overall aesthetic of the image?",
            "Is there any emotional content visible?",
            "What story does this image tell?",
            "Are there any symbolic elements?",
            "What is the intended message, if any?",
            "How would you describe the visual impact?",
            "What artistic elements are present?",
            "Does the image evoke any particular emotions?",

            # Unique Features
            "What is unique about this image?",
            "Are there any unusual elements present?",
            "What makes this image stand out?",
            "Are there any unexpected details?",
            "What is the most interesting aspect?",
            "Are there any rare or uncommon features?",
            "What catches your attention first?",
            "Is there anything surprising in the image?",
            "What distinguishes this image from others?",
            "Are there any special characteristics?",

            # Details and Specifics
            "What small details are visible in the image?",
            "Can you describe any intricate patterns?",
            "Are there any hidden elements?",
            "What subtle features can you identify?",
            "Are there any fine details worth noting?",
            "Can you spot any minute textures?",
            "What precise characteristics stand out?",
            "Are there any microscopic details visible?",
            "What specific features define this image?",
            "Can you identify any particular markings?"
        ]
        
        print(f"• Total available questions: {len(questions)}")
        print(f"• Will randomly select 5 questions per image")
        print(f"• Total operations: 50 * 5 = 250 image-question pairs\n")
        
        # Process images and save results
        print("Step 4: Starting image analysis...")
        print("-"*NUM_DASHES)
        successful_analyses = 0
        failed_analyses = 0
        
        # Prepare CSV headers
        headers = ['Image_Number', 'Dataset_Index', 'Class_Label']
        for i in range(1, 6):  # 5 questions
            headers.extend([f'Q{i}', f'A{i}'])
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Processing images", unit="image")):
                image_tensor, label = trainset[sample_idx]
                image = transform_to_pil(image_tensor)
                
                # Randomly select 5 questions for this image
                selected_questions = random.sample(questions, 5)
                
                # Prepare row data
                row_data = [
                    f"{idx + 1:02d}",  # Image Number
                    str(sample_idx),    # Dataset Index
                    trainset.classes[label]  # Class Label
                ]
                
                question_success = 0
                responses = []
                
                for qnum, question in enumerate(selected_questions, 1):
                    try:
                        # Create chat messages format
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "url": image},
                                    {"type": "text", "text": question},
                                ]
                            },
                        ]
                        
                        # Process inputs
                        inputs = processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        ).to(model.device, dtype=torch.bfloat16)
                        
                        # Generate response
                        generated_ids = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=64
                        )
                        
                        # Decode response
                        response = processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0]
                        
                        # Clean up response
                        response = clean_text_for_csv(response)
                        response = response.replace("User:", "").replace("Assistant:", "").strip()
                        response = response.replace(question, "").strip()
                        
                        row_data.extend([clean_text_for_csv(question), response])
                        question_success += 1
                        
                    except Exception as e:
                        error_msg = f"ERROR: {str(e)}"
                        row_data.extend([clean_text_for_csv(question), error_msg])
                        print(f"\n⚠️  Error on image {idx + 1}, Q{qnum}: {question[:30]}...")
                        print(f"   Error message: {str(e)}")
                        print(f"   Continuing with next question...")
                
                # Write the row to CSV
                writer.writerow(row_data)
                
                if question_success == len(selected_questions):
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                
                # Add a small delay between images to prevent potential rate limiting
                time.sleep(0.1)
        
        # Final statistics
        total_time = time.time() - start_time
        print("\n=== Analysis Complete ===")
        print("-"*NUM_DASHES)
        print(f"✓ Total images processed: 50")
        print(f"✓ Successful analyses: {successful_analyses}")
        print(f"⚠️  Failed analyses: {failed_analyses}")
        print(f"✓ Total time taken: {total_time:.2f} seconds")
        print(f"✓ Average time per image: {total_time/50:.2f} seconds")
        print(f"\nResults have been saved to: {output_file}")
        print("="*NUM_DASHES)
        
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        print("Please check if you have the correct model access and all required dependencies.")
        raise

if __name__ == "__main__":
    main() 