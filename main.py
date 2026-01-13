import logging
from peanut_butter import (
    PBMTConfig, 
    load_raw_wmt14, 
    PhraseExtractor, 
    TranslationModel, 
    Scorer
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Configuration
    logger.info("Initializing Peanut Butter Configuration...")
    config = PBMTConfig(
        weight_init="pretrained",  # Change to "random" to train from scratch
        training_type="lora",      # Change to "full" for full finetuning
        epochs=3,
        num_extraction_samples=15
    )

    # 2. Phase 1: Phrase Extraction
    # We always use a pretrained model here to create the dataset
    logger.info("--- STARTING EXTRACTION ---")
    raw_dataset = load_raw_wmt14(num_samples=config.num_extraction_samples)
    
    extractor = PhraseExtractor(config)
    phrase_table = extractor.extract(raw_dataset)
    logger.info(f"Extracted {len(phrase_table)} phrase pairs.")
    
    # Unload to save memory
    extractor.unload_model()

    # 3. Phase 2: Model Setup & Training
    logger.info("--- STARTING TRAINING ---")
    tm_model = TranslationModel(config)
    tm_model.setup()
    
    # Train the weights (either from scratch or fine-tuned)
    tm_model.train(phrase_table)

    # 4. Phase 3: Scoring & Inference
    logger.info("--- STARTING SCORING ---")
    scorer = Scorer(tm_model)

    # Demo Test Cases
    test_cases = [
        ("Das ist ein Haus", ["This is a house", "A building", "Peanut butter"]),
        ("Ich mag Brot", ["I like bread", "I enjoy loaf", "Peanut butter jelly"]),
    ]
    
    # PBMT Weights
    W_TM = 1.0
    W_LM = 0.5

    print("\n" + "="*60)
    print("PEANUT BUTTER PBMT RESULTS")
    print("="*60)

    for src, candidates in test_cases:
        print(f"\nSource: {src}")
        print(f"{'Candidate':<25} | {'TM (P(e|f))':<10} | {'LM (P(e))':<10} | {'Total':<8}")
        print("-" * 62)
        
        results = []
        for cand in candidates:
            tm, lm = scorer.get_scores(src, cand)
            total = (W_TM * tm) + (W_LM * lm)
            results.append((cand, tm, lm, total))
        
        # Sort by total score
        results.sort(key=lambda x: x[3], reverse=True)
        
        for r in results:
            print(f"{r[0]:<25} | {r[1]:.2f}       | {r[2]:.2f}       | {r[3]:.2f}")

if __name__ == "__main__":
    main()
