"""
Model 3: Whisper Fine-tuning for Uzbek ASR
OpenAI Whisper modelini o'zbek tili uchun fine-tuning qilish
"""

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate


class UzbekWhisperTrainer:
    """
    O'zbek tili uchun Whisper fine-tuning
    """
    
    def __init__(
        self,
        model_name="openai/whisper-small",  # tiny, base, small, medium, large
        language="uzbek",
        task="transcribe"
    ):
        """
        Args:
            model_name: Whisper model size
            language: Til kodi
            task: transcribe yoki translate
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        
        # Processor
        self.processor = WhisperProcessor.from_pretrained(
            model_name,
            language=language,
            task=task
        )
        
        # Model
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Model konfiguratsiyasi
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.config.use_cache = False  # Gradient checkpointing uchun
        
        # Metric
        self.metric = evaluate.load("wer")
    
    def prepare_dataset(self, batch):
        """Dataset preprocessing"""
        # Audio feature extraction
        audio = batch["audio"]
        
        # 30 sekundgacha qirqish
        batch["input_features"] = self.processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Tokenize text
        batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids
        
        return batch
    
    def compute_metrics(self, pred):
        """WER calculation"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 (padding)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Calculate WER
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir="./whisper-uzbek",
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-5,
        warmup_steps=500
    ):
        """Training"""
        
        @dataclass
        class DataCollatorSpeechSeq2SeqWithPadding:
            """Data collator for speech-to-text"""
            processor: Any
            
            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                # Input features
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
                
                # Labels
                label_features = [{"input_ids": feature["labels"]} for feature in features]
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
                
                # Replace padding with -100
                labels = labels_batch["input_ids"].masked_fill(
                    labels_batch.attention_mask.ne(1), -100
                )
                
                # Remove BOS token (Whisper adds it automatically)
                if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                    labels = labels[:, 1:]
                
                batch["labels"] = labels
                
                return batch
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch size = 16
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            logging_steps=100,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            predict_with_generate=True,
            generation_max_length=225,
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        trainer.train()
        
        return trainer
    
    def predict(self, audio_path, return_timestamps=False):
        """
        Inference
        
        Args:
            audio_path: Audio fayl yo'li
            return_timestamps: Vaqt belgilari qaytarilsinmi
        """
        import soundfile as sf
        import librosa
        
        # Load audio
        speech, sample_rate = sf.read(audio_path)
        
        # Resample
        if sample_rate != 16000:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        
        # Process
        input_features = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        # Generate with GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_features = input_features.to(device)
        self.model = self.model.to(device)
        
        # Generate
        with torch.no_grad():
            if return_timestamps:
                predicted_ids = self.model.generate(
                    input_features,
                    return_timestamps=True
                )
            else:
                predicted_ids = self.model.generate(input_features)
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def predict_long_audio(self, audio_path, chunk_length_s=30):
        """
        Uzun audio uchun chunk-chunk qilib tanish
        
        Args:
            audio_path: Audio fayl
            chunk_length_s: Chunk uzunligi (sekundlarda)
        """
        import soundfile as sf
        import librosa
        import numpy as np
        
        # Load audio
        speech, sample_rate = sf.read(audio_path)
        
        # Resample
        if sample_rate != 16000:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Chunk-larga bo'lish
        chunk_length_samples = chunk_length_s * sample_rate
        chunks = []
        
        for i in range(0, len(speech), chunk_length_samples):
            chunk = speech[i:i + chunk_length_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        # Har bir chunk uchun prediction
        transcriptions = []
        for chunk in chunks:
            input_features = self.processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_features = input_features.to(device)
            self.model = self.model.to(device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            transcriptions.append(transcription)
        
        # Combine
        full_transcription = " ".join(transcriptions)
        
        return full_transcription


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Initialize trainer
    trainer = UzbekWhisperTrainer(
        model_name="openai/whisper-small",  # yoki whisper-medium
        language="uzbek",
        task="transcribe"
    )
    
    print(f"Model: {trainer.model_name}")
    print(f"Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Example: Load dataset
    # dataset = load_dataset("your_uzbek_dataset")
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # # Preprocess
    # dataset = dataset.map(trainer.prepare_dataset, remove_columns=dataset.column_names['train'])
    
    # # Train
    # trained_model = trainer.train(
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     output_dir="./whisper-uzbek",
    #     num_epochs=10,
    #     batch_size=8
    # )
    
    # # Inference
    # text = trainer.predict("path/to/uzbek_audio.wav")
    # print(f"Transcription: {text}")
    
    # # Long audio
    # long_text = trainer.predict_long_audio("path/to/long_audio.wav", chunk_length_s=30)
    # print(f"Long transcription: {long_text}")


# ==========================================
# WHISPER MODEL O'LCHAMLARI
# ==========================================
"""
Model       Parameters  English-only    Multilingual    Required VRAM
tiny        39 M        ✓               ✓               ~1 GB
base        74 M        ✓               ✓               ~1 GB
small       244 M       ✓               ✓               ~2 GB
medium      769 M       ✓               ✓               ~5 GB
large       1550 M      ✗               ✓               ~10 GB

Tavsiya: 
- Dissertatsiya uchun: whisper-small yoki whisper-medium
- GPU yo'q bo'lsa: whisper-tiny yoki whisper-base
- Eng yaxshi natija: whisper-large (lekin juda sekin)
"""


# ==========================================
# EVALUATION HELPER
# ==========================================
def evaluate_model(model_path, test_dataset):
    """
    Model baholash
    
    Args:
        model_path: Fine-tuned model path
        test_dataset: Test dataset
    """
    from jiwer import wer, cer
    import time
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    predictions = []
    references = []
    inference_times = []
    
    for sample in test_dataset:
        # Feature extraction
        input_features = processor(
            sample["audio"]["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Predict
        start_time = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        inference_time = time.time() - start_time
        
        # Decode
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        predictions.append(transcription)
        references.append(sample["text"])
        inference_times.append(inference_time)
    
    # Metrics
    wer_score = wer(references, predictions) * 100
    cer_score = cer(references, predictions) * 100
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    print(f"WER: {wer_score:.2f}%")
    print(f"CER: {cer_score:.2f}%")
    print(f"Avg Inference Time: {avg_inference_time:.3f}s")
    
    return {
        "wer": wer_score,
        "cer": cer_score,
        "avg_inference_time": avg_inference_time,
        "predictions": predictions,
        "references": references
    }
