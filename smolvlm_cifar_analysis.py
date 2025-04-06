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
import argparse

NUM_DASHES = 150
FLUSH_INTERVAL = 100  # Flush to CSV file every 100 images

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

def get_last_processed_index(output_file):
    """Get the last processed image index from the CSV file."""
    if not os.path.exists(output_file):
        return -1
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            # Read all lines and get the last one
            lines = f.readlines()
            if not lines:
                return -1
            last_line = lines[-1].strip()
            if not last_line:
                return -1
            # Extract image number from the first column
            last_image_num = int(last_line.split(',')[0]) - 1  # Convert back to 0-based index
            return last_image_num
    except Exception as e:
        print(f"Warning: Error reading last processed index: {e}")
        return -1

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CIFAR10 Image Analysis with SmolVLM2')
    parser.add_argument('--resume', type=str, help='Path to the CSV file to resume from')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save output files (default: output)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n=== SmolVLM2 CIFAR10 Image Analysis ===")
    print("Starting the analysis process...\n")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output file
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume file not found: {args.resume}")
        output_file = args.resume
        print(f"Resuming analysis from: {output_file}")
    else:
        # Generate new filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"infer_test_{timestamp}.csv")
        print(f"Starting new analysis, output will be saved to: {output_file}")
    
    print()  # Empty line for readability

    # Load CIFAR10 dataset
    print("Step 1: Loading CIFAR10 test dataset...")
    print("-"*NUM_DASHES)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    start_time = time.time()
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    print(f"✓ Test dataset loaded successfully! ({len(testset)} images available)")
    print(f"✓ Available classes: {', '.join(testset.classes)}\n")
    
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
        
        # Process all test images
        print("Step 3: Preparing image analysis...")
        print("-"*NUM_DASHES)
        
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
        print(f"• Total operations: {len(testset) * 5} image-question pairs")
        print(f"• Estimated time: {(len(testset) * 5 * 4.0) / 60:4.0f} minutes (at 4.0s per question)")
        print(f"• Data will be flushed to CSV every {FLUSH_INTERVAL} images\n")
        
        # Process images and save results
        print("Step 4: Starting image analysis...")
        print("-"*NUM_DASHES)
        successful_analyses = 0
        failed_analyses = 0
        
        # Check if we need to resume from a previous run
        last_processed_idx = get_last_processed_index(output_file)
        start_idx = last_processed_idx + 1
        
        if start_idx > 0:
            print(f"Resuming from image {start_idx + 1} ({start_idx} images already processed)")
            print(f"Using existing file: {output_file}")
        else:
            print(f"Starting new analysis")
            print(f"Creating new file: {output_file}")
        
        # Show processing information
        remaining_images = len(testset) - start_idx
        estimated_time = (remaining_images * 5 * 4.0) / 60  # 4.0 seconds per question
        print(f"\nProcessing Information:")
        print(f"• Images remaining    : {remaining_images:,}")
        print(f"• Questions per image : 5")
        print(f"• Total operations    : {remaining_images * 5:,}")
        print(f"• Estimated time      : {estimated_time:4.0f} minutes")
        print(f"• Auto-save interval  : Every {FLUSH_INTERVAL} images")
        print("-"*NUM_DASHES)
        
        # Prepare CSV headers
        headers = ['Image_Number', 'Dataset_Index', 'Class_Label']
        for i in range(1, 6):  # 5 questions
            headers.extend([f'Q{i}', f'A{i}'])
        headers.extend(['Concat_Q', 'Concat_A'])  # Add new concatenated columns
        
        # Open file in append mode if resuming, write mode if starting fresh
        file_mode = 'a' if start_idx > 0 else 'w'
        with open(output_file, file_mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if file_mode == 'w':
                writer.writerow(headers)
            
            # Buffer for batch writing
            rows_buffer = []
            
            for idx in tqdm(range(start_idx, len(testset)), 
                          initial=start_idx, total=len(testset),
                          desc="Processing images", unit="image"):
                image_tensor, label = testset[idx]
                image = transform_to_pil(image_tensor)
                
                # Randomly select 5 questions for this image
                selected_questions = random.sample(questions, 5)
                
                # Prepare row data
                row_data = [
                    f"{idx + 1:05d}",  # Image Number (padded to 5 digits)
                    str(idx),          # Dataset Index
                    testset.classes[label]  # Class Label
                ]
                
                question_success = 0
                questions_list = []  # Store questions for concatenation
                answers_list = []    # Store answers for concatenation
                
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
                        
                        # Store cleaned question and response
                        cleaned_question = clean_text_for_csv(question)
                        questions_list.append(f'"{cleaned_question}"')
                        answers_list.append(f'"{response}"')
                        
                        row_data.extend([cleaned_question, response])
                        question_success += 1
                        
                    except Exception as e:
                        error_msg = f"ERROR: {str(e)}"
                        questions_list.append(f'"{clean_text_for_csv(question)}"')
                        answers_list.append(f'"ERROR"')
                        row_data.extend([clean_text_for_csv(question), error_msg])
                        print(f"\n⚠️  Error on image {idx + 1}, Q{qnum}: {question[:30]}...")
                        print(f"   Error message: {str(e)}")
                        print(f"   Continuing with next question...")
                
                # Add concatenated columns
                row_data.append(", ".join(questions_list))
                row_data.append(", ".join(answers_list))
                
                # Add row to buffer
                rows_buffer.append(row_data)
                
                # Update statistics
                if question_success == len(selected_questions):
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                
                # Flush buffer to CSV file every FLUSH_INTERVAL images
                if len(rows_buffer) >= FLUSH_INTERVAL:
                    writer.writerows(rows_buffer)
                    f.flush()  # Force write to disk
                    rows_buffer = []  # Clear buffer
                    print(f"\n✓ Progress saved at image {idx + 1}")
                
                # Add a small delay between images to prevent potential rate limiting
                time.sleep(0.1)
            
            # Write any remaining rows in buffer
            if rows_buffer:
                writer.writerows(rows_buffer)
                f.flush()
        
        # Final statistics
        total_time = time.time() - start_time
        print("\n=== Analysis Complete ===")
        print("-"*NUM_DASHES)
        print(f"✓ Total images processed: {len(testset)}")
        print(f"✓ Successful analyses: {successful_analyses}")
        print(f"⚠️  Failed analyses: {failed_analyses}")
        print(f"✓ Total time taken: {total_time:.2f} seconds")
        print(f"✓ Average time per image: {total_time/len(testset):.2f} seconds")
        print(f"\nResults have been saved to: {output_file}")
        print("="*NUM_DASHES)
        
    except Exception as e:
        # If we encounter an error, try to save any remaining buffered rows
        if 'rows_buffer' in locals() and 'f' in locals() and 'writer' in locals():
            try:
                writer.writerows(rows_buffer)
                f.flush()
                print(f"\n✓ Saved buffered data before error")
            except:
                pass
        
        print(f"\n❌ Error during execution: {str(e)}")
        if args.resume:
            print(f"\nTo resume from the last successful save, run:")
            print(f"python {os.path.basename(__file__)} --resume {output_file}")
        raise

if __name__ == "__main__":
    main() 