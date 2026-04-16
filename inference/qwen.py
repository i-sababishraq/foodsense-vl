import sys
from pathlib import Path

# Add project root so we can import evaluate
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import run_qwen2vl_inference
from config.prompts import USER_PROMPT


def main():
    image_dir = "data/Images"
    image_paths = [
        "0001_01lamiW2bWW0rXlllNHYMA.jpg",
        "0002_01zZeZBIFZ82S5XmA4GYJg.jpg",
        "0005_08Eu2m3RTrpssX9GIKtHtg.jpg",
        "0010_0dHJ9fque7joEy7J0UrHmA.jpg",
        "0015_0g2pruxDhqhh2E-cEoYOLA.jpg",
        "0020_0K5mp0ZyTPvTxQ7JOaDWkQ.jpg",
    ]

    print("Running Qwen2-VL on selected images...")
    results = run_qwen2vl_inference(
        image_paths=image_paths,
        prompt=USER_PROMPT,
        max_new_tokens=1024,
        image_dir=image_dir,
        model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        do_sample=False,
        temperature=0.0,
        local_files_only=False,
    )

    print("\n--- RESULTS ---")
    for r in results:
        print(f"Image: {r['image']}")
        print(f"Text:\n{r['text']}")
        print(f"Parsed Ratings: {r['ratings']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
